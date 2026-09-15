# P3b: export hand-written layer loop as fixed-shape ONNX (KV past/present, cache_position runtime)
import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
MAX_KV = 768
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
class GraphBDeployExport(torch.nn.Module):
    def __init__(self, base, max_kv):
        super().__init__()
        self.base = base
        self.max_kv = max_kv
    def forward(self, inputs_embeds, position_ids, cache_position, kv_k, kv_v):
        hidden = inputs_embeds
        pos_emb = self.base.rotary_emb(hidden, position_ids)
        kvk_out = kv_k.clone()
        kvv_out = kv_v.clone()
        for i in range(28):
            layer = self.base.layers[i]
            residual = hidden
            h = layer.input_layernorm(hidden)
            attn = layer.self_attn
            head_dim = attn.head_dim
            hs = h.shape[:-1]
            hidden_shape = (*hs, -1, head_dim)
            q = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1,2)
            k = attn.k_norm(attn.k_proj(h).view(hidden_shape)).transpose(1,2)
            v = attn.v_proj(h).view(hidden_shape).transpose(1,2)
            cos, sin = pos_emb
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling['mrope_section'], attn.rope_scaling['interleaved'])
            # slot-write current k/v into buffer at cp
            kvk_out[i, :, :, cache_position, :] = k[:, :, 0, :].unsqueeze(2)
            kvv_out[i, :, :, cache_position, :] = v[:, :, 0, :].unsqueeze(2)
            kv_groups = attn.num_key_value_groups
            k_all = kvk_out[i, :, :, :, :].repeat_interleave(kv_groups, dim=1)
            v_all = kvv_out[i, :, :, :, :].repeat_interleave(kv_groups, dim=1)
            positions = torch.arange(self.max_kv).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            amask = torch.where(positions <= cache_position, torch.tensor(0.0), torch.tensor(float('-inf')))
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_all, v_all, attn_mask=amask, dropout_p=0.0, is_causal=False)
            attn_out = attn_out.reshape(*hs, -1).contiguous()
            attn_out = attn.o_proj(attn_out)
            h = residual + attn_out
            residual2 = h
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h)
            hidden = residual2 + h
        hidden = self.base.norm(hidden)
        return hidden, kvk_out, kvv_out
core = GraphBDeployExport(base, MAX_KV).eval()
B,H,D,Cd = 1, 8, 128, 2048
emb = torch.zeros((1,1,Cd), dtype=torch.float32)
pos = torch.zeros((3,1,1), dtype=torch.float32)
pos[0,0,0]=62; pos[1,0,0]=62; pos[2,0,0]=62
cp_ = torch.tensor([62], dtype=torch.int64)
kk = torch.zeros((28,B,H,MAX_KV,D), dtype=torch.float32)
vv = torch.zeros((28,B,H,MAX_KV,D), dtype=torch.float32)
with torch.inference_mode():
    y1, yk, yv = core(emb, pos, cp_, kk, vv)
print('fwd ok', tuple(y1.shape), tuple(yk.shape), flush=True)
torch.onnx.export(
    core,
    (emb, pos, cp_, kk, vv),
    ART + '/graphb_deploy_hand.onnx',
    input_names=['inputs_embeds','position_ids','cache_position','kv_k','kv_v'],
    output_names=['hidden_states','kv_k_out','kv_v_out'],
    opset_version=18,
    dynamo=False,
)
print('GRAPHB_DEPLOY_HAND_EXPORT_OK', flush=True)
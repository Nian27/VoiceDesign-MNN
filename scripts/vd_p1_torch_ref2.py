# HTP-P1 torch reference with the SAME input the HTP dump run used (p1_traj step0).
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p1'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
MAX_KV = 768
emb = np.fromfile(D + '/vd_m5a_inputs/step0_emb.raw', dtype=np.float32).reshape(1,1,2048)
posv = np.fromfile(D + '/vd_m5a_inputs/step0_pos.raw', dtype=np.float32).reshape(3,1,1)
cp = int(np.fromfile(D + '/vd_m5a_inputs/step0_cp.raw', dtype=np.int32)[0])
kvk = np.fromfile(D + '/vd_m5a_inputs/step0_kvk.raw', dtype=np.float32).reshape(28,1,8,768,128)
kvv = np.fromfile(D + '/vd_m5a_inputs/step0_kvv.raw', dtype=np.float32).reshape(28,1,8,768,128)
print('cp', cp, 'emb maxabs %.4f' % np.abs(emb).max())
with torch.no_grad():
    hidden = torch.from_numpy(emb); kvk_t = torch.from_numpy(kvk); kvv_t = torch.from_numpy(kvv)
    pos_emb = base.rotary_emb(hidden, torch.from_numpy(posv))
    layer_outs = []
    for i in range(28):
        layer = base.layers[i]
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        hd = attn.head_dim; hs = h.shape[:-1]; shp = (*hs, -1, hd)
        q = attn.q_norm(attn.q_proj(h).view(shp)).transpose(1,2)
        k = attn.k_norm(attn.k_proj(h).view(shp)).transpose(1,2)
        v = attn.v_proj(h).view(shp).transpose(1,2)
        cosv, sinv = pos_emb
        q, k = apply_multimodal_rotary_pos_emb(q, k, cosv, sinv, attn.rope_scaling['mrope_section'], attn.rope_scaling['interleaved'])
        kvk_t[i,:,:,cp,:] = k[:,:,0,:]; kvv_t[i,:,:,cp,:] = v[:,:,0,:]
        g = attn.num_key_value_groups
        k_all = kvk_t[i].repeat_interleave(g, dim=1); v_all = kvv_t[i].repeat_interleave(g, dim=1)
        positions = torch.arange(MAX_KV).view(1,1,1,MAX_KV)
        amask = torch.where(positions <= cp, torch.tensor(0.0), torch.tensor(float('-inf')))
        ao = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=amask, dropout_p=0.0, is_causal=False)
        ao = attn.o_proj(ao.reshape(*hs,-1).contiguous())
        h2 = residual + ao; r2 = h2
        h2 = layer.post_attention_layernorm(h2)
        hidden = r2 + layer.mlp(h2)
        layer_outs.append(hidden[0,0].numpy().astype(np.float32).copy())
    final = base.norm(hidden)[0,0].numpy().astype(np.float32).copy()
np.save(OUT + '/torch_layers_step0.npy', np.stack(layer_outs))
np.save(OUT + '/torch_final_step0.npy', final)
print('final maxabs %.4f  f8 %s' % (np.abs(final).max(), final[:4]))
# also dump the input_layernorm(L0) input == emb, and L0 norm output for reference
print('TORCH_REF2_DONE')

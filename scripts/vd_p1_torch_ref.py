# HTP-P1: torch reference per-layer hidden for device step0, then compare with HTP dump.
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_p1'
os.makedirs(OUT, exist_ok=True)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
MAX_KV = 768

# --- build device step0 input exactly: emb = frame_emb([1138,1095,...]) ---
import onnxruntime as ort
so = ort.InferenceSession(OUT.replace('/vd_p1','/vd_m5b') + '/frame_emb.onnx', providers=['CPUExecutionProvider'])
frame0 = np.array([[1138, 1095, 1540, 826, 1714, 964, 520, 609, 1673, 1978, 966, 1854, 601, 424, 529, 179]], dtype=np.int64)
emb = so.run(None, {'codes': frame0})[0].astype(np.float32)     # (1,1,2048)
print('emb maxabs %.4f' % np.abs(emb).max())

kvk = np.fromfile(D + '/vd_m5b/prefill_kv_k.raw', dtype=np.float32).reshape(28,1,8,62,128)
kvv = np.fromfile(D + '/vd_m5b/prefill_kv_v.raw', dtype=np.float32).reshape(28,1,8,62,128)
LP = 62
kk = np.zeros((28,1,8,MAX_KV,128), np.float32); vv = np.zeros((28,1,8,MAX_KV,128), np.float32)
kk[:,:,:,:LP,:] = kvk; vv[:,:,:,:LP,:] = kvv

with torch.no_grad():
    hidden = torch.from_numpy(emb)
    kvk_t = torch.from_numpy(kk)
    kvv_t = torch.from_numpy(vv)
    cp = 62
    pos = torch.tensor([[[62.0]],[[62.0]],[[62.0]]])
    pos_emb = base.rotary_emb(hidden, pos)
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
        kvk_t[i,:,:,cp,:] = k[:,:,0,:]
        kvv_t[i,:,:,cp,:] = v[:,:,0,:]
        g = attn.num_key_value_groups
        k_all = kvk_t[i].repeat_interleave(g, dim=1)
        v_all = kvv_t[i].repeat_interleave(g, dim=1)
        positions = torch.arange(MAX_KV).view(1,1,1,MAX_KV)
        amask = torch.where(positions <= cp, torch.tensor(0.0), torch.tensor(float('-inf')))
        ao = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=amask, dropout_p=0.0, is_causal=False)
        ao = attn.o_proj(ao.reshape(*hs,-1).contiguous())
        h2 = residual + ao
        r2 = h2
        h2 = layer.post_attention_layernorm(h2)
        hidden = r2 + layer.mlp(h2)
        layer_outs.append(hidden[0,0].numpy().astype(np.float32).copy())
    final = base.norm(hidden)[0,0].numpy().astype(np.float32).copy()
np.save(OUT + '/torch_layers.npy', np.stack(layer_outs))
np.save(OUT + '/torch_final.npy', final)
print('torch layers', np.stack(layer_outs).shape)
print('L0 f8', layer_outs[0][:4])
print('L1 f8', layer_outs[1][:4])
print('L2 f8', layer_outs[2][:4])
print('final f8', final[:4], 'maxabs %.4f' % np.abs(final).max())
print('TORCH_REF_DONE')

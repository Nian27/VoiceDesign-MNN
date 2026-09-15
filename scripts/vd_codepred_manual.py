# Manual codepred forward (export-safe) + numeric check vs official cp.model, then AR vs captured.
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
import torch.nn.functional as F
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import apply_rotary_pos_emb
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp = talker.code_predictor
cpm = cp.model
print('attn scaling head0 =', getattr(cpm.layers[0].self_attn, 'scaling', 'MISSING'))
print('kv_groups head0 =', getattr(cpm.layers[0].self_attn, 'num_key_value_groups', 'MISSING'))

def manual_model(x):
    # x: (1,n,1024) -> (1,n,1024), causal
    n = x.shape[1]
    pos = torch.arange(n, dtype=torch.long).unsqueeze(0)
    cosv, sinv = cpm.rotary_emb(x, pos)
    h = x
    for layer in cpm.layers:
        residual = h
        hn = layer.input_layernorm(h)
        attn = layer.self_attn
        hs = hn.shape[:-1]; hd = attn.head_dim
        shp = (*hs, -1, hd)
        q = attn.q_norm(attn.q_proj(hn).view(shp)).transpose(1,2)
        k = attn.k_norm(attn.k_proj(hn).view(shp)).transpose(1,2)
        v = attn.v_proj(hn).view(shp).transpose(1,2)
        q, k = apply_rotary_pos_emb(q, k, cosv, sinv)
        g = attn.num_key_value_groups
        kk = k.repeat_interleave(g, dim=1); vv = v.repeat_interleave(g, dim=1)
        ao = F.scaled_dot_product_attention(q, kk, vv, dropout_p=0.0, is_causal=True)
        ao = ao.reshape(*hs, -1).contiguous()
        ao = attn.o_proj(ao)
        h = residual + ao
        r2 = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = r2 + h
    return cpm.norm(h)

# 1) numeric check of manual model vs official cpm
torch.manual_seed(0)
for n in (2, 5, 16):
    xt = torch.randn(1, n, 1024) * 0.5
    with torch.inference_mode():
        a = cpm(inputs_embeds=xt, use_cache=False).last_hidden_state
        b = manual_model(xt)
    print('manual_model n=%2d cos=%.8f' % (n, cos(a.numpy(), b.numpy())))

# 2) AR with manual model vs captured official logits
z = np.load(D + '/codepred_ar/official_codepred.npz')
code0_in=z['code0_in']; past_hidden=z['past_hidden']; codec_ids=z['codec_ids']
lm_marks=z['lm_marks']; lm_last=z['lm_last_logits']
codec_emb = talker.get_input_embeddings(); proj = cp.small_to_mtp_projection
allc=[]
for f in range(3):
    m = int(lm_marks[f-1]) if f>0 else 0
    seq = torch.cat([torch.from_numpy(past_hidden[f]), codec_emb(torch.tensor([[int(code0_in[f])]]))], dim=1)
    row=[]
    for g in range(15):
        with torch.inference_mode():
            h = manual_model(proj(seq))
            lg = cp.lm_head[g](h[:, -1:])[0,0].numpy()
        row.append(cos(lg, lm_last[m+g])); allc.append(row[-1])
        seq = torch.cat([seq, cp.model.codec_embedding[g](torch.tensor([[int(codec_ids[f,0,g+1])]]))], dim=1)
    print('AR-manual frame %d min=%.7f  %s' % (f, min(row), ' '.join('%.5f'%x for x in row)))
print('=== AR-manual OVERALL min = %.7f (n=%d) ===' % (min(allc), len(allc)))
print('MANUAL_CHECK_DONE')

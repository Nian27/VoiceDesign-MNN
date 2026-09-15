import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'; torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
at = core.layers[0].self_attn
print('q_proj', at.q_proj, )
print('k_proj', at.k_proj)
print('v_proj', at.v_proj)
print('o_proj', at.o_proj)
print('q_norm', at.q_norm, 'k_norm', at.k_norm)
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    hn = core.layers[0].input_layernorm(xp)
    qq = at.q_proj(hn); kk = at.k_proj(hn); vv = at.v_proj(hn)
    print('shapes  q_proj out', tuple(qq.shape), ' k_proj out', tuple(kk.shape), ' v_proj out', tuple(vv.shape))
    print('=> q heads =', qq.shape[-1]//128, ' k heads =', kk.shape[-1]//128, ' v heads =', vv.shape[-1]//128)
    qn = at.q_norm(qq.view(1,2,-1,128))
    print('q_norm out', tuple(qn.shape))
print('PROJ_DONE')

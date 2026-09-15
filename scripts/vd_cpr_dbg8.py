import os, math, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    o = core(inputs_embeds=xp, use_cache=True)
    pk = o.past_key_values
    KC = pk.key_cache if hasattr(pk,'key_cache') else [pk[i][0] for i in range(5)]
    VC = pk.value_cache if hasattr(pk,'value_cache') else [pk[i][1] for i in range(5)]
    print('cache K len', len(KC), 'V len', len(VC))
    for i in range(5):
        layer = core.layers[i]; at = layer.self_attn
        hn = layer.input_layernorm(xp if i == 0 else None) if i == 0 else None
        if i > 0:
            # need the running hidden; recompute quickly using the model's own layer outputs
            break
    # recompute running hidden from hidden_states
    o2 = core(inputs_embeds=xp, use_cache=True, output_hidden_states=True)
    hss = [t for t in o2.hidden_states]
    for i in range(5):
        layer = core.layers[i]; at = layer.self_attn
        hin = xp if i == 0 else hss[i]
        hn = layer.input_layernorm(hin)
        shp = (*hn.shape[:-1], -1, 128)
        v = at.v_proj(hn).view(shp).transpose(1,2)
        vmodel = VC[i][0] if VC[i].dim()==4 else VC[i]
        print('layer %d  V: model %s  mine(as v_proj) %s   cos=%.7f' % (
            i, tuple(vmodel.shape), tuple(v.shape), cos(vmodel.numpy(), v.numpy())))
        kk = KC[i][0] if KC[i].dim()==4 else KC[i]
        kraw = at.k_proj(hn).view(shp).transpose(1,2)
        print('        K raw (pre-rope) cos=%.7f' % cos(kk.numpy(), kraw.numpy()))
print('DBG8_DONE')

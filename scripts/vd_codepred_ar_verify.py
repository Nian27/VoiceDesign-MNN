# Verify my codepred AR reimplementation against official captured per-step logits.
# Official: prefix0 = [past_hidden, codec_emb(code0)] -> proj -> model -> lm_head[0] -> c1
#           prefix_g = prefix_{g-1} + proj(cp_emb[g-1](c_g)) -> lm_head[g] -> c_{g+1}
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
z = np.load(D + '/codepred_ar/official_codepred.npz')
code0_in = z['code0_in']; past_hidden = z['past_hidden']; codec_ids = z['codec_ids']
lm_marks = z['lm_marks']; lm_g = z['lm_g']; lm_last = z['lm_last_logits']
print('frames', len(code0_in), 'past_hidden', past_hidden.shape, 'codec_ids', codec_ids.shape, 'lm_last', lm_last.shape)
def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp = talker.code_predictor
cpm = cp.model
lm = cp.lm_head
cp_emb = cp.get_input_embeddings()
codec_emb = talker.get_input_embeddings()
proj = cp.small_to_mtp_projection
print('LOADED')
NF = 4
allcos = []
for f in range(NF):
    m = int(lm_marks[f])
    c0 = int(code0_in[f])
    ph = torch.from_numpy(past_hidden[f])
    c0e = codec_emb(torch.tensor([[c0]]))
    seq = torch.cat([ph, c0e], dim=1)   # (1,2,2048)
    row = []
    for g in range(15):
        with torch.inference_mode():
            s1024 = proj(seq)
            out = cpm(inputs_embeds=s1024, use_cache=False)
            lg = lm[g](out.last_hidden_state[:, -1:])[0,0].float().numpy()
        ref = lm_last[m+g]
        # verify head index matches
        if int(lm_g[m+g]) != g:
            print('  !! head mismatch at frame %d step %d: captured g=%d' % (f, g, int(lm_g[m+g])))
        c = cos(lg, ref)
        row.append(c)
        allcos.append(c)
        if g < 15:
            cg = int(codec_ids[f, g+1]) if g+1 < codec_ids.shape[1] else 0
            seq = torch.cat([seq, cp_emb[g](torch.tensor([[cg]]))], dim=1)
    print('frame %d min_cos=%.6f mean=%.6f  per-step=%s' % (f, min(row), sum(row)/len(row), ' '.join('%.4f'%x for x in row)))
print('=== OVERALL min cos = %.6f  (n=%d) ===' % (min(allcos), len(allcos)))
print('AR_VERIFY_DONE')

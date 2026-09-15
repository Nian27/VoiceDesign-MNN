# Verify AR v3: correct codec_ids indexing (f, 0, g). Full-refeed path.
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
z = np.load(D + '/codepred_ar/official_codepred.npz')
code0_in = z['code0_in']; past_hidden = z['past_hidden']; codec_ids = z['codec_ids']
lm_marks = z['lm_marks']; lm_last = z['lm_last_logits']
def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp = talker.code_predictor
cpm = cp.model; lm = cp.lm_head
cp_emb = cp.get_input_embeddings(); codec_emb = talker.get_input_embeddings()
proj = cp.small_to_mtp_projection
print('LOADED')
NF = 4
allcos = []
for f in range(NF):
    m = int(lm_marks[f-1]) if f > 0 else 0
    c0 = int(code0_in[f])
    seq = torch.cat([torch.from_numpy(past_hidden[f]), codec_emb(torch.tensor([[c0]]))], dim=1)
    row = []
    for g in range(15):
        with torch.inference_mode():
            out = cpm(inputs_embeds=proj(seq), use_cache=False)
            lg = lm[g](out.last_hidden_state[:, -1:])[0,0].float().numpy()
        row.append(cos(lg, lm_last[m+g])); allcos.append(row[-1])
        cg = int(codec_ids[f, 0, g+1])          # CORRECT: c_{g+1}
        seq = torch.cat([seq, cp_emb[g](torch.tensor([[cg]]))], dim=1)
    print('frame %d min=%.7f mean=%.7f  %s' % (f, min(row), sum(row)/len(row), ' '.join('%.5f'%x for x in row)))
print('=== OVERALL min cos = %.7f (n=%d) ===' % (min(allcos), len(allcos)))
print('AR_VERIFY3_DONE')

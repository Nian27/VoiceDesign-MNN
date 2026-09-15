# Decisive: run the OFFICIAL codepred path (wrapper forward + DynamicCache, generation_steps)
# and compare per-step logits vs captured. Then compare vs my full-refeed.
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
z = np.load(D + '/codepred_ar/official_codepred.npz')
code0_in = z['code0_in']; past_hidden = z['past_hidden']; codec_ids = z['codec_ids']
lm_marks = z['lm_marks']; lm_g = z['lm_g']; lm_last = z['lm_last_logits']
def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
from transformers.cache_utils import DynamicCache
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp = talker.code_predictor
cpm = cp.model; lm = cp.lm_head
cp_emb = cp.get_input_embeddings(); codec_emb = talker.get_input_embeddings()
proj = cp.small_to_mtp_projection
print('LOADED')

def official_path(f):
    m = int(lm_marks[f-1]) if f > 0 else 0
    c0 = int(code0_in[f])
    ph = torch.from_numpy(past_hidden[f])
    c0e = codec_emb(torch.tensor([[c0]]))
    seq = torch.cat([ph, c0e], dim=1)
    kv = DynamicCache()
    logs = []
    # prefill via wrapper forward (generation_steps None -> computed 0)
    with torch.inference_mode():
        out = cp(inputs_embeds=seq, past_key_values=kv, use_cache=True)
    logs.append(out.logits[0, -1].float().numpy())
    gs = int(out.generation_steps)
    for g in range(1, 15):
        cg = int(codec_ids[f, g])
        with torch.inference_mode():
            out = cp(input_ids=torch.tensor([[cg]]), generation_steps=gs, past_key_values=kv, use_cache=True)
        logs.append(out.logits[0, -1].float().numpy())
        gs = int(out.generation_steps)
    return m, logs

for f in range(2):
    m, logs = official_path(f)
    row = [cos(logs[g], lm_last[m+g]) for g in range(15)]
    print('OFFICIAL-PATH frame %d min=%.6f mean=%.6f  %s' % (f, min(row), sum(row)/len(row), ' '.join('%.4f'%x for x in row)))
print('OFFICIAL_PATH_TEST_DONE')

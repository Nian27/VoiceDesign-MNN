import os, torch, numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
G = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
print('loaded', flush=True)
gold = torch.load(G + '/g1_decode_golden.pt', map_location='cpu', weights_only=False)
st0 = gold['decode_steps'][0]
emb = st0['inputs_embeds'].float()
pos = st0['position_ids'].float()
cp0 = int(st0['cache_position'].item())
print('emb', tuple(emb.shape), flush=True)
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
with torch.inference_mode():
    out = talker.model(inputs_embeds=emb, position_ids=pos, past_key_values=kv, cache_position=torch.tensor([cp0]), use_cache=True)
hidden_pt = out.last_hidden_state.float()
print('hidden_pt', tuple(hidden_pt.shape), flush=True)
print('kv type:', type(kv), 'len layers:', len(kv.layers) if hasattr(kv, 'layers') else 'n/a', flush=True)
l0 = kv.layers[0] if hasattr(kv, 'layers') else kv
print('layer0 attrs:', [a for a in dir(l0) if not a.startswith('_')][:20], flush=True)
# KV stack via official API
try:
    kk = torch.stack([kv.layers[i].keys for i in range(28)], dim=0)
    print('kv stack keys ok', tuple(kk.shape), flush=True)
except Exception as e:
    print('kv keys stack ERR:', str(e)[:200], flush=True)
    kk = None
try:
    vv = torch.stack([kv.layers[i].values for i in range(28)], dim=0)
    print('kv stack values ok', tuple(vv.shape), flush=True)
except Exception as e:
    print('kv values stack ERR:', str(e)[:200], flush=True)
    vv = None
# codepred step0
pj = proj(hidden_pt)
print('proj out', tuple(pj.shape), flush=True)
hid_t = pj
codes = []
for g in range(15):
    with torch.inference_mode():
        o = cpm(inputs_embeds=hid_t, use_cache=False)
        lg = lm[g](o.last_hidden_state)
    code = int(torch.argmax(lg[0,0]))
    codes.append(code)
    hid_t = cp_emb[g](torch.tensor([[code]]))
print('codes15:', codes, flush=True)
codec_h = [hidden_pt] + [cp_emb[g](torch.tensor([[codes[g]]])) for g in range(15)]
print('codec_h shapes:', [tuple(x.shape) for x in codec_h], flush=True)
emb_next = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
print('emb_next', tuple(emb_next.shape), flush=True)
print('DONE', flush=True)
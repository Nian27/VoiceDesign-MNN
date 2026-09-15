import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
G = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
gold = torch.load(G + '/g1_decode_golden.pt', map_location='cpu', weights_only=False)
st0 = gold['decode_steps'][0]
emb = st0['inputs_embeds'].float()
pos = st0['position_ids'].float()
cp0 = int(st0['cache_position'].item())
print('emb', tuple(emb.shape), 'pos', tuple(pos.shape), 'cp', cp0, flush=True)
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
print('calling talker.model...', flush=True)
with torch.inference_mode():
    out = talker.model(inputs_embeds=emb, position_ids=pos, past_key_values=kv, cache_position=torch.tensor([cp0]), use_cache=True)
print('out.last_hidden_state', tuple(out.last_hidden_state.shape), flush=True)
print('DONE', flush=True)
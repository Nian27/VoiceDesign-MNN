import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
print('cp type:', type(cp).__name__)
print('cp.model type:', type(cp.model).__name__)
import inspect
print('cp.model forward sig:', str(inspect.signature(cp.model.forward))[:200])
# 直接试 cp.model 前向
ie = torch.randn((1,2,1024), dtype=torch.float32)
pos = torch.tensor([[0,1]], dtype=torch.int64)
with torch.inference_mode():
    out = cp.model(inputs_embeds=ie, position_ids=pos, use_cache=False)
print('cp.model fwd ok', tuple(out.last_hidden_state.shape))
lg = cp.lm_head(out.last_hidden_state)
print('logits', tuple(lg.shape))
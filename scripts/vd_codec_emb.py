import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
ce = gw.model.talker.model.codec_embedding.weight
print('talker codec_embedding', tuple(ce.shape))
ce.detach().float().contiguous().numpy().astype(np.float32).tofile(OUT + '/talker_codec_emb.f32')
print('bytes', os.path.getsize(OUT + '/talker_codec_emb.f32'))
print('CODEC_EMB_DONE')
# export talker codec embedding table for host-side lookup (removes MNN from the code0-emb path)
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
w = gw.model.talker.get_input_embeddings().weight.detach().numpy().astype(np.float32)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_cpr'
w.tofile(D + '/talker_codec_emb_weight.f32')
print('talker codec emb', w.shape, '->', D + '/talker_codec_emb_weight.f32')

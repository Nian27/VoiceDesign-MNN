import os, torch, inspect
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as M
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
a0 = gw.model.talker.model.layers[0].self_attn
print('rope_scaling =', a0.rope_scaling)
print('head_dim =', a0.head_dim)
print()
print('=== apply_multimodal_rotary_pos_emb source (head) ===')
src = inspect.getsource(M.apply_multimodal_rotary_pos_emb)
print(src[:1400])

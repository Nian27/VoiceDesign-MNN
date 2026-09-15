import os, inspect, torch, numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
import qwen_tts.core.models.modeling_qwen3_tts as MF
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
at = core.layers[0].self_attn
print('at.sliding_window =', getattr(at,'sliding_window','N/A'))
print('config.sliding_window =', getattr(core.config,'sliding_window','N/A'))
print('at.scaling =', at.scaling, ' 1/sqrt(128) =', 1.0/(128**0.5))
print('attn impl =', core.config._attn_implementation)
print()
print('=== eager_attention_forward source ===')
try:
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    src = inspect.getsource(eager_attention_forward)
except Exception:
    try:
        import transformers.models.qwen3.modeling_qwen3 as Q3
        src = inspect.getsource(Q3.eager_attention_forward)
    except Exception as e:
        src = 'IMPORT FAIL: ' + str(e)
print(src[:1200])

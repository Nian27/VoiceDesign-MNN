import os, inspect
os.environ['HF_HUB_OFFLINE'] = '1'
import qwen_tts.core.models.modeling_qwen3_tts as M
src = inspect.getsource(M.apply_multimodal_rotary_pos_emb)
i = src.find('if mrope_interleaved')
if i < 0:
    i = src.find('interleaved')
print(src[i-200:i+1200] if i > 0 else src[-1500:])

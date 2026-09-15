import os, inspect
os.environ['HF_HUB_OFFLINE'] = '1'
import qwen_tts.core.models.modeling_qwen3_tts as MF
src = inspect.getsource(MF.Qwen3TTSAttention.forward)
i = src.find('attn_weights = torch.matmul')
print(src[i-200:i+1400] if i > 0 else src[-1600:])

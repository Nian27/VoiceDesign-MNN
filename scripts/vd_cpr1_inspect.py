import os, torch, inspect
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
print('=== code_predictor type ===', type(cp).__name__)
print('=== top-level children ===')
for n, m in cp.named_children():
    print('  %-28s %s' % (n, type(m).__name__))
print()
print('=== cp.model config (core) ===')
c = cp.model.config
for k in ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size','max_position_embeddings','rope_scaling']:
    print('  %-24s %s' % (k, getattr(c, k, 'N/A')))
print()
print('=== small_to_mtp_projection ===', cp.small_to_mtp_projection)
print()
print('=== lm_head ===')
lh = cp.lm_head
print('  type:', type(lh).__name__, 'len:', len(lh) if hasattr(lh,'__len__') else 'n/a')
try:
    print('  [0]:', lh[0])
except Exception as e:
    print('  idx err:', e)
print()
print('=== codec_embedding ===')
ce = cp.model.codec_embedding
print('  type:', type(ce).__name__, 'len:', len(ce) if hasattr(ce,'__len__') else 'n/a')
try:
    print('  [0]:', ce[0])
except Exception as e:
    print('  idx err:', e)
print()
print('=== get_input_embeddings of cp ===')
try:
    print(' ', cp.get_input_embeddings())
except Exception as e:
    print('  err', e)

import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
t = gw.model.talker
cfg = t.config
for k in ['hidden_size','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','num_hidden_layers','vocab_size','text_hidden_size','text_vocab_size']:
    print(k, '=', getattr(cfg, k, 'MISSING'))
print('--- talker.model config ---')
mc = t.model.config
for k in ['hidden_size','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','num_hidden_layers']:
    print(k, '=', getattr(mc, k, 'MISSING'))
print('--- q_proj/o_proj shapes layer0 ---')
attn = t.model.layers[0].self_attn
print('q_proj', tuple(attn.q_proj.weight.shape))
print('k_proj', tuple(attn.k_proj.weight.shape))
print('v_proj', tuple(attn.v_proj.weight.shape))
print('o_proj', tuple(attn.o_proj.weight.shape))
print('head_dim attr', getattr(attn, 'head_dim', 'MISSING'))
print('DONE', flush=True)
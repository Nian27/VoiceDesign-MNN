import os, torch, numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
c = cp.config
print('=== codepred config ===')
for k in ['hidden_size','num_hidden_layers','vocab_size','num_code_groups','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','layer_types','max_position_embeddings','rope_scaling','rms_norm_eps','sliding_window']:
    print(' ', k, '=', getattr(c, k, 'MISSING'))
tc = gw.model.talker.config
print('=== talker config (subset) ===')
for k in ['hidden_size','num_hidden_layers','vocab_size','num_code_groups']:
    print(' ', k, '=', getattr(tc, k, 'MISSING'))
print('=== modules ===')
print(' cp.model.layers:', len(cp.model.layers))
print(' cp.lm_head:', len(cp.lm_head))
print(' cp.model.codec_embedding:', len(cp.model.codec_embedding))
print(' lm_head[0]:', cp.lm_head[0])
print(' codec_embedding[0]:', cp.model.codec_embedding[0])
print(' small_to_mtp_projection:', cp.small_to_mtp_projection)
n_params = sum(p.numel() for p in cp.parameters())
print(' codepred total params: %.2fM (%.1f MB fp16)' % (n_params/1e6, n_params*2/1e6))

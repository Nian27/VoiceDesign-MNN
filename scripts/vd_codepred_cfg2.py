import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
c = cp.config
print('=== codepred cfg ===')
for k in ['hidden_size','num_hidden_layers','vocab_size','num_code_groups','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','layer_types','max_position_embeddings','rms_norm_eps','sliding_window','rope_scaling','attention_dropout','num_logits_to_keep']:
    print(' ', k, '=', getattr(c, k, 'MISSING'))
print('layers:', len(cp.model.layers), 'lm_head:', len(cp.lm_head), 'emb:', len(cp.model.codec_embedding))
print('lm_head[0]:', cp.lm_head[0], '| emb[0]:', cp.model.codec_embedding[0])
print('proj:', cp.small_to_mtp_projection)
print('norm:', cp.model.norm)
L0 = cp.model.layers[0]
print('layer0 attn:', type(L0.self_attn).__name__, '| mlp:', type(L0.mlp).__name__)
print('attn has q_norm/k_norm:', hasattr(L0.self_attn,'q_norm'), hasattr(L0.self_attn,'k_norm'))
print('layer0 attention_type:', getattr(L0,'attention_type','MISSING'))
n = sum(p.numel() for p in cp.parameters())
print('params: %.1fM' % (n/1e6))
print('CFG_DONE')

import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
core = cp.model
print('=== core children ===')
for n, m in core.named_children():
    print('  %-20s %s' % (n, type(m).__name__))
print('=== core.layers[0] children ===')
for n, m in core.layers[0].named_children():
    print('  %-24s %s' % (n, type(m).__name__))
print('=== layer0.self_attn children ===')
for n, m in core.layers[0].self_attn.named_children():
    print('  %-20s %s' % (n, type(m).__name__))
print('=== mlp children ===')
for n, m in core.layers[0].mlp.named_children():
    print('  %-20s %s' % (n, type(m).__name__))
print('=== has rotary_emb ===', hasattr(core, 'rotary_emb'), ' has norm ===', hasattr(core, 'norm'))
a = core.layers[0].self_attn
print('head_dim', a.head_dim, 'num_kv_groups', a.num_key_value_groups, 'scaling', getattr(a,'scaling',None))
print('sliding_window', getattr(core.config,'sliding_window',None), 'rope_theta', getattr(core.config,'rope_theta',None))
with torch.no_grad():
    x = torch.randn(1,2,1024)*0.1
    out = core(inputs_embeds=x, use_cache=True)
    print('out keys', out.keys())
    print('last_hidden', tuple(out.last_hidden_state.shape))
    pk = out.past_key_values
    print('past_key_values type', type(pk).__name__, 'len', len(pk) if hasattr(pk,'__len__') else 'n/a')
    try:
        k0 = pk[0][0] if not hasattr(pk,'key_cache') else pk.key_cache[0]
        print('layer0 K shape', tuple(k0.shape))
    except Exception as e:
        print('kv inspect err', e)
print('INSPECT_CORE_DONE')

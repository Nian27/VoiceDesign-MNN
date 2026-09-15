import os, numpy as np, torch, functools
os.environ['HF_HUB_OFFLINE']='1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker; base = talker.model
print('--- talker model structure ---')
for n, p in base.named_parameters():
    if 'embed' in n or 'codec' in n.lower():
        print(' ', n, tuple(p.shape))
print('--- config ---')
cfg = base.config
for k in ['vocab_size','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','rope_scaling','sliding_window','max_position_embeddings']:
    v = getattr(cfg, k, None)
    print('  %s = %s' % (k, v))
print('--- 关键: 捕获的真实 prompt token ids ---')
ids_cap = {}
orig_emb = base.embed_tokens.forward if hasattr(base,'embed_tokens') else None
cap = {}
def patched_emb(ids):
    if 'ids' not in cap:
        cap['ids'] = ids.detach().cpu().numpy().astype(np.int64)
    return orig_emb(ids)
if orig_emb is not None:
    base.embed_tokens.forward = patched_emb
orig_fwd = base.forward
def pf(*a, **kw):
    return orig_fwd(*a, **kw)
base.forward = pf
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
try:
    gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True, max_new_tokens=1)
except Exception as e:
    print('  (early stop ok)')
if orig_emb is not None: base.embed_tokens.forward = orig_emb
if 'ids' in cap:
    ids = cap['ids'].reshape(-1)
    print('  prompt token count =', len(ids))
    print('  ids =', ids.tolist())
    print('  min=%d max=%d vocab_size=%s' % (ids.min(), ids.max(), cfg.vocab_size))
    cap_ids = ids
else:
    print('  embed_tokens 未被调用（可能走 inputs_embeds）')
print('--- 验证 prefill_emb 是否 = embed_tokens(ids) ---')
pe = np.fromfile(D+'/vd_m5b/prefill_emb.raw', dtype=np.float32).reshape(-1, 2048)
print('  captured prefill_emb shape =', pe.shape)
if 'cap_ids' in dir() and len(cap_ids) == pe.shape[0]:
    with torch.no_grad():
        e = base.embed_tokens(torch.tensor(cap_ids).unsqueeze(0)).float().numpy().reshape(-1,2048)
    a=pe.ravel().astype(np.float64); b=e.ravel().astype(np.float64)
    cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    print('  cos(prefill_emb, embed_tokens(ids)) = %.7f' % cos)
print('DONE')

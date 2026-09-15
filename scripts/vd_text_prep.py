import os, json, struct, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
os.makedirs(OUT, exist_ok=True)

# ---------- 1) GPT-2 byte -> unicode alphabet ----------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("\u00a1"), ord("\u00ac")+1)) + list(range(ord("\u00ae"), ord("\u00ff")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
B2U = bytes_to_unicode(); U2B = {v: k for k, v in B2U.items()}

vocab = json.load(open(MDL + '/vocab.json', encoding='utf-8'))
merges = []
with open(MDL + '/merges.txt', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line or line.startswith('#version'):
            continue
        a, b = line.split(' ')
        merges.append((a, b))
tcfg = json.load(open(MDL + '/tokenizer_config.json', encoding='utf-8'))
added = []
for sid, info in tcfg.get('added_tokens_decoder', {}).items():
    added.append((int(sid), info.get('content'), bool(info.get('special', False))))
added.sort()
print('vocab=%d merges=%d added=%d' % (len(vocab), len(merges), len(added)))
print('added tokens:', [(i, c) for i, c, s in added][:14])

# ---------- 2) tokenizer.bin ----------
def w_u32(f, v): f.write(struct.pack('<I', v))
def w_str(f, s):
    b = s.encode('utf-8'); f.write(struct.pack('<H', len(b))); f.write(b)
path = OUT + '/tokenizer.bin'
with open(path, 'wb') as f:
    f.write(b'KOTB'); w_u32(f, 1)
    w_u32(f, len(vocab)); w_u32(f, len(merges)); w_u32(f, len(added))
    # vocab 按 id 顺序
    inv = {}
    for tok, i in vocab.items():
        inv[i] = tok
    for i in range(len(vocab)):
        w_str(f, inv.get(i, ''))
    # merges: (a_id, b_id, new_id)
    for a, b in merges:
        w_u32(f, vocab.get(a, 0xFFFFFFFF)); w_u32(f, vocab.get(b, 0xFFFFFFFF)); w_u32(f, vocab.get(a+b, 0xFFFFFFFF))
    # added tokens
    for sid, content, sp in added:
        w_u32(f, sid); f.write(b'\x01' if sp else b'\x00'); w_str(f, content)
print('tokenizer.bin bytes =', os.path.getsize(path))

# ---------- 3) text_embedding / text_projection fp16 ----------
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
tp = gw.model.talker.text_projection
te = gw.model.talker.model.text_embedding.weight
print('text_embedding', tuple(te.shape), 'fc1', tuple(tp.linear_fc1.weight.shape), 'fc2', tuple(tp.linear_fc2.weight.shape))
te.detach().half().contiguous().numpy().tofile(OUT + '/text_embedding.fp16')
tp.linear_fc1.weight.detach().half().contiguous().numpy().tofile(OUT + '/text_proj_fc1.fp16')
tp.linear_fc2.weight.detach().half().contiguous().numpy().tofile(OUT + '/text_proj_fc2.fp16')
for n in ['text_embedding.fp16', 'text_proj_fc1.fp16', 'text_proj_fc2.fp16', 'tokenizer.bin']:
    print('  %-24s %d bytes' % (n, os.path.getsize(OUT + '/' + n)))
print('VD_TEXT_PREP_DONE')

import os, json, torch
os.environ['HF_HUB_OFFLINE']='1'
from qwen_tts import Qwen3TTSModel
MDL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_models'
gw = Qwen3TTSModel.from_pretrained(MDL, dtype=torch.float32, device_map='cpu')
tok = gw.processor
def enc(s):
    r = tok(text=s)
    ids = r['input_ids']
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]
CASES = [
 ('苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。',
  '小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。'),
 ('年轻女性，清脆明亮，语速偏快，带一点笑意。', '今天天气不错，我们出去走走吧。'),
 ('a deep male voice, slightly raspy, calm and authoritative', 'The quick brown fox jumps over the lazy dog.'),
 ('中年男性，温厚', "It's a test: 3 apples, 2 oranges... don't stop!"),
 ('温柔女性，沙哑', '第一行。\n第二行，换行测试。'),
]
with open(OUT + '/tok_cases.txt', 'w', encoding='utf-8') as f:
    for ins, txt in CASES:
        a = enc('<|im_start|>user\n' + ins + '<|im_end|>\n')
        b = enc('<|im_start|>assistant\n' + txt + '<|im_end|>\n<|im_start|>assistant\n')
        f.write(ins.replace('\n','\\n') + '\t' + txt.replace('\n','\\n') + '\t' + ','.join(map(str,a)) + '\t' + ','.join(map(str,b)) + '\n')
        print('  ins=%d asst=%d' % (len(a), len(b)))
print('TOK_CASES_DONE')

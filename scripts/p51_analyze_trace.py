import json
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
st = json.load(open(D + '/p51_trace.json'))
# 找 decode 起点和 EOS
dec = [s for s in st if s['phase'] == 'DECODE']
print('total decode:', len(dec))
print('first 5 decode:', [(s['cp'], s['gen_step'], s['emb_src']) for s in dec[:5]])
print('last 3 decode:', [(s['cp'], s['gen_step'], s['emb_src']) for s in dec[-3:]])
print('prefill calls:', len([s for s in st if s['phase']=='PREFILL']))
print('prefill emb shapes:', [(s['emb_shape']) for s in st if s['phase']=='PREFILL'][:3])
# cp 范围
cps = [s['cp'][0] for s in dec if s['cp']]
print('decode cp range:', min(cps), '..', max(cps))
print('decode count =', len(cps), '=> frames =', len(cps))
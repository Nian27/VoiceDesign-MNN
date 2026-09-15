import torch
G = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden'
gold = torch.load(G + '/g1_decode_golden.pt', map_location='cpu', weights_only=False)
st0 = gold['decode_steps'][0]
print('step0 keys:', list(st0.keys()))
for k in st0.keys():
    v = st0[k]
    if hasattr(v, 'shape'): print(k, 'tensor', tuple(v.shape))
    else: print(k, '=', str(v)[:100])
print('cp_start:', gold['cp_start'][0].shape if hasattr(gold['cp_start'][0], 'shape') else gold['cp_start'])
import torch
g = torch.load('E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden/g1_decode_golden.pt', map_location='cpu', weights_only=False)
st = g['decode_steps']
print('num steps:', len(st))
for i, s in enumerate(st):
    print('--- step', i, '---')
    print('emb', tuple(s['inputs_embeds'].shape), str(s['inputs_embeds'].dtype))
    print('pos', tuple(s['position_ids'].shape), str(s['position_ids'].dtype))
    print('cp', tuple(s['cache_position'].shape), str(s['cache_position'].dtype), 'val', s['cache_position'].tolist())
    print('mask_shape', s['mask_shape'])
    print('kv_shape', s['kv_shape'])
print('cp_start emb', tuple(g['cp_start'][0].shape), str(g['cp_start'][0].dtype))
print('cp_start max_new', g['cp_start'][1], 'dosample', g['cp_start'][2])
import torch
g = torch.load('E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden/g1_decode_golden.pt', map_location='cpu', weights_only=False)
print('top keys:', list(g.keys()) if isinstance(g, dict) else type(g))
def walk(d, pre='', depth=0):
    if depth > 3: return
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                print(pre + str(k), tuple(v.shape), str(v.dtype))
            else:
                print(pre + str(k), type(v).__name__)
                walk(v, pre + '  ', depth+1)
    elif isinstance(d, (list, tuple)):
        print(pre + '[list len=%d]' % len(d))
        if len(d) > 0: walk(d[0], pre + '  [0] ', depth+1)
walk(g)
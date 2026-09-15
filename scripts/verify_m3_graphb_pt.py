import os, torch, numpy as np, MNN, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/onnx'
GOLD = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden/g1_decode_golden.pt'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
g = torch.load(GOLD, map_location='cpu', weights_only=False)
step = None
if isinstance(g, dict):
    ds = g.get('decode_steps')
    if ds and len(ds) > 0: step = ds[0]
print('step keys:', list(step.keys()) if step else 'NONE', flush=True)
emb_t = step['inputs_embeds'].to(torch.float32)  # (1,1,2048) fp32
pos_t = step['position_ids'].to(torch.float32)   # (3,1,1)
cp_t = step['cache_position'].to(torch.int64)    # (1,)
T = 34
B,H,D,Cd = 1, 8, 128, 2048
emb = emb_t.numpy()
pos = pos_t.numpy()
cp_ = cp_t.numpy()
kvk = np.zeros((28,1,H,T,D), dtype=np.float32)
kvv = np.zeros((28,1,H,T,D), dtype=np.float32)
print('inputs:', emb.shape, pos.shape, cp_.shape, flush=True)
# ---- PyTorch reference (load talker base, fp32) ----
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
for i in range(28):
    kv.update(torch.zeros((1,H,T,D), dtype=torch.float32), torch.zeros((1,H,T,D), dtype=torch.float32), i)
with torch.inference_mode():
    out = base(inputs_embeds=emb_t.unsqueeze(0) if emb_t.dim()==3 else emb_t, position_ids=pos_t, past_key_values=kv, cache_position=cp_t, use_cache=True, output_hidden_states=False)
pt_hidden = out.last_hidden_state.detach().numpy()
print('PT hidden:', pt_hidden.shape, flush=True)
# ---- MNN ----
interp = MNN.Interpreter(ART + '/graphb_replay_step_fp32_nonan.mnn')
sess2 = interp.createSession()
interp.resizeTensor(interp.getSessionInput(sess2, 'inputs_embeds'), (1,1,Cd))
interp.resizeTensor(interp.getSessionInput(sess2, 'position_ids'), (3,1,1))
interp.resizeTensor(interp.getSessionInput(sess2, 'kv_k'), (28,1,H,T,D))
interp.resizeTensor(interp.getSessionInput(sess2, 'kv_v'), (28,1,H,T,D))
interp.resizeTensor(interp.getSessionInput(sess2, 'cache_position'), (1,))
interp.resizeSession(sess2)
def feed(name, arr, dtype, ndim):
    t = interp.getSessionInput(sess2, name)
    ht = MNN.Tensor(tuple(arr.shape), dtype, arr.ravel(), MNN.Tensor_DimensionType_Caffe)
    t.copyFromHostTensor(ht)
feed('inputs_embeds', emb, MNN.Halide_Type_Float)
feed('position_ids', pos, MNN.Halide_Type_Float)
feed('kv_k', kvk, MNN.Halide_Type_Float)
feed('kv_v', kvv, MNN.Halide_Type_Float)
feed('cache_position', cp_, MNN.Halide_Type_Int)
ot = interp.getSessionOutput(sess2, 'hidden_states')
t0 = time.time()
interp.runSession(sess2)
t1 = time.time()
osh = list(ot.getShape())
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
mnn_hidden = np.asarray(ho.getNumpyData()).reshape(osh)
print('MNN hidden:', mnn_hidden.shape, 'elapsed', round(t1-t0,2), flush=True)
# ---- compare ----
a = pt_hidden.astype(np.float64).ravel(); b = mnn_hidden.astype(np.float64).ravel()
n = min(len(a), len(b)); a=a[:n]; b=b[:n]
cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
mae = float(np.mean(np.abs(a-b))); mx = float(np.max(np.abs(a-b)))
print('M3-GRAPHB MNN vs PyTorch: cosine', round(cos,6), 'mae', round(mae,8), 'maxabs', round(mx,6), flush=True)
print('M3_GRAPHB_DONE', flush=True)
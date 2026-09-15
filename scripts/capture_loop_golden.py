import os, torch, json
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
# 构造 3 帧 decode 循环: 用固定 code 序列（g0 码手动给 3 个）模拟第一码，capture 每帧契约
# 帧0: input_ids=codec_bos? 用 golden 已有 decode_steps 形状: inputs_embeds (1,1,2048)
# 简化: 捕获 code_predictor 的 15 码 embedding 查表 + codec_hiddens 求和路径
try:
print('loading done', flush=True)
# 捕获 get_input_embeddings 行为（code0 帧 id -> 2048）
emb = talker.get_input_embeddings()
cp_emb = talker.code_predictor.get_input_embeddings()
print('talker emb type:', type(emb).__name__)
print('cp emb type:', type(cp_emb).__name__, 'len', len(cp_emb) if hasattr(cp_emb,'__len__') else 'n/a')
# code0 帧: 用 codec_bos_id=2149 试 embedding
ids = torch.tensor([[2149]], dtype=torch.long)
h = emb(ids)
print('code0 emb shape', tuple(h.shape), 'maxabs', float(h.abs().max()))
# cp embedding 第 0 个 codebook
if hasattr(cp_emb, '__len__') and len(cp_emb) > 0:
    c0 = cp_emb[0](ids)
    print('cp emb[0] shape', tuple(c0.shape))
# 15 码求和路径
seqs = torch.randint(0, 2048, (1, 15))
codec_hiddens = [h] + [cp_emb[i](seqs[:, i:i+1]) for i in range(15)]
summed = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
print('codec_hiddens summed shape', tuple(summed.shape), 'maxabs', float(summed.abs().max()))
torch.save({'code0_emb': h, 'seqs15': seqs, 'codec_sum': summed}, ART + '/loop_step0_golden.pt')
print('LOOP_GOLDEN_SAVED')
print('EXCEPT', flush=True)

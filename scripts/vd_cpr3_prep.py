# CP-R3 prep: past_hidden, code0_emb, rope16, forced_emb (14 forced tokens from the torch reference)
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'; torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
G3 = CR + '/g3'; os.makedirs(G3, exist_ok=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
CAP, HD = 16, 128
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
ew  = np.fromfile(CR + '/codec_emb_weight.f32', dtype=np.float32).reshape(15,2048,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
past.tofile(G3 + '/past_hidden.raw')
CODE0 = 1995
c0e = tew[CODE0].reshape(1,1,2048); c0e.tofile(G3 + '/code0_emb.raw')
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad(): c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
c.reshape(CAP, HD).tofile(G3 + '/cos16.raw'); si.reshape(CAP, HD).tofile(G3 + '/sin16.raw')
# forced tokens = torch reference codes (code1..code14 feed steps 0..13)
torch_codes = [496, 1114, 1363, 157, 881, 1579, 145, 1253, 455, 288, 586, 914, 779, 1219, 901]
forced = torch_codes[:14]
emb = np.stack([ew[g][forced[g]] for g in range(14)]).astype(np.float32)   # (14,2048)
emb.tofile(G3 + '/forced_emb.raw')
np.array(forced, np.int32).tofile(G3 + '/forced_tokens.i32')
print('forced tokens (feed steps 0..13) =', forced)
print('stage-0 sampled token (code1) =', torch_codes[0], ' ... code15 =', torch_codes[14])
print('shapes  cos16', c.shape, ' forced_emb', emb.shape)
print('CPR3_PREP_DONE')

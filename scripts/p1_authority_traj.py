# P1 authority trajectory: real 62-KV start, 20 steps, real emb/codes, save per-step state
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
B = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/p1_traj'
os.makedirs(OUT, exist_ok=True)
NUM_STEPS = 20
LOG = open(D + '/p1_authority.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
log('LOADED')
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
kvk62 = torch.from_numpy(np.load(B + '/kv_k_decode.npy')).float()
kvv62 = torch.from_numpy(np.load(B + '/kv_v_decode.npy')).float()
for i in range(28):
    kv.update(kvk62[i], kvv62[i], i)
z = np.load(B + '/step_decode_inputs.npz')
emb = torch.from_numpy(z['emb']).float()
pos = torch.from_numpy(z['pos']).float()
cp0 = int(z['cp'][0])
log('start kv62 cp ' + str(cp0))
steps_meta = []
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    try:
        with torch.inference_mode():
            out = base(inputs_embeds=emb, position_ids=pos, past_key_values=kv, cache_position=torch.tensor([cp_val]), use_cache=True, output_hidden_states=True)
        hidden = out.last_hidden_state.detach().float()
        kk = torch.stack([kv.layers[i].keys.detach().float() for i in range(28)], 0)
        vv = torch.stack([kv.layers[i].values.detach().float() for i in range(28)], 0)
        # codepred 15 codes (parallel, same proj input)
        with torch.no_grad():
            pj = proj(hidden).clone()
        codes = []
        hid_t = pj
        for g in range(15):
            with torch.inference_mode():
                o = cpm(inputs_embeds=hid_t, use_cache=False)
                lg = lm[g](o.last_hidden_state)
            codes.append(int(torch.argmax(lg[0,0])))
        log('step ' + str(step) + ' cp ' + str(cp_val) + ' kvlen ' + str(kk.shape[3]) + ' codes ' + str(codes[:4]) + '...')
        # save per-step state: emb_in, pos, cp, kv_k/vv (full current), hidden, codes
        np.savez(OUT + '/step' + str(step) + '.npz', emb_in=emb.detach().float().numpy(), pos=pos.detach().float().numpy(), cp=np.array([cp_val]), kvk=kk.numpy(), kvv=vv.numpy(), hidden=hidden.numpy(), codes=np.array(codes))
        steps_meta.append({'step': step, 'cp': cp_val, 'kv_len': int(kk.shape[3]), 'codes': codes})
        # next emb from codec_hiddens sum (parallel 15)
        codec_h = [hidden] + [cp_emb[g](torch.tensor([[codes[g]]])) for g in range(15)]
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
    except Exception as ex:
        log('STEP ERR: ' + str(ex)[:400])
        traceback.print_exc(file=LOG)
        raise
with open(OUT + '/traj.json', 'w') as f:
    f.write(json.dumps({'steps': steps_meta, 'start_kv_len': 62}, indent=2))
log('P1_AUTHORITY_DONE ' + str(len(steps_meta)) + ' steps')
print('P1_AUTHORITY_DONE', len(steps_meta), flush=True)
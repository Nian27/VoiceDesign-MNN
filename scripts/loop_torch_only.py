# Gate4 torch-authoritative trajectory (20 steps, no MNN)
import os, json, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
G = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden'
NUM_STEPS = 20
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
LOG = open(D + '/loop_torch_diag.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
log('LOADED')
gold = torch.load(G + '/g1_decode_golden.pt', map_location='cpu', weights_only=False)
st0 = gold['decode_steps'][0]
emb = st0['inputs_embeds'].float()
pos = st0['position_ids'].float()
cp0 = int(st0['cache_position'].item())
log('emb ' + str(tuple(emb.shape)) + ' cp0 ' + str(cp0))
import os as _os
_os.makedirs(D + '/loop_pt', exist_ok=True)
from transformers.cache_utils import DynamicCache
kv = DynamicCache()
steps = []
for step in range(NUM_STEPS):
    cp_val = cp0 + step
    pos = torch.tensor([[[float(cp_val)]],[[float(cp_val)]],[[float(cp_val)]]])
    try:
        with torch.inference_mode():
            out = talker.model(inputs_embeds=emb, position_ids=pos, past_key_values=kv, cache_position=torch.tensor([cp_val]), use_cache=True)
    except Exception as ex:
        log('TORCH ERR step ' + str(step) + ': ' + str(ex)[:300])
        raise
    hidden_pt = out.last_hidden_state.float()
    log('step ' + str(step) + ' hidden ' + str(tuple(hidden_pt.shape)) + ' max ' + str(round(float(hidden_pt.abs().max()),4)))
    kk = torch.stack([kv.layers[i].keys for i in range(28)], dim=0)
    vv = torch.stack([kv.layers[i].values for i in range(28)], dim=0)
    log('  kv ' + str(tuple(kk.shape)))
    with torch.no_grad():
        pj = proj(hidden_pt).clone()
    log('  proj ok ' + str(tuple(pj.shape)))
    codes = []; top2s = []
    hid_t = pj
    for g in range(15):
        log('  cp g' + str(g) + ' hid ' + str(tuple(hid_t.shape)))
        try:
            with torch.inference_mode():
                o = cpm(inputs_embeds=hid_t, use_cache=False)
                lg = lm[g](o.last_hidden_state)
        except Exception as ex:
            log('  CP ERR g' + str(g) + ': ' + str(ex)[:300])
            raise
        lgv = lg[0,0]
        code = int(torch.argmax(lgv))
        codes.append(code)
        top2s.append([int(x) for x in torch.argsort(lgv, descending=True)[:2].tolist()])
    log('  codes ' + str(codes))
    steps.append({'step': step, 'cache_position': cp_val, 'codes': codes, 'top2': top2s, 'hidden_max': round(float(hidden_pt.abs().max()),4)})
    try:
        np.savez(D + '/loop_pt/step' + str(step) + '.npz', emb=emb.detach().numpy().astype(np.float32), pos=pos.detach().numpy().astype(np.float32), kvk=kk.detach().numpy().astype(np.float32), kvv=vv.detach().numpy().astype(np.float32), hidden=hidden_pt.detach().cpu().numpy().astype(np.float32), cp=np.array([cp_val]))
        log('  npz ok')
    except Exception as ex:
        log('  NPZ ERR: ' + str(ex)[:200])
    codec_h = [hidden_pt.detach()] + [cp_emb[g](torch.tensor([[codes[g]]])) for g in range(15)]
    try:
        emb = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
        log('  emb_next ' + str(tuple(emb.shape)))
    except Exception as ex:
        log('  EMB ERR: ' + str(ex)[:200])
        raise
    log('  emb_next ' + str(tuple(emb.shape)))
traj = {'steps': steps, 'meta': {'golden_sha': 'DF390E0EB3AADD9A1B4A2E0A45F9DE8DC7D2D77E93E25B68898FFDA91AB0FACF', 'num_steps': NUM_STEPS}}
with open(D + '/loop_traj_torch.json', 'w') as f:
    f.write(json.dumps(traj, indent=2))
log('TORCH_TRAJ_SAVED ' + str(len(steps)) + ' steps')
print('TORCH_TRAJ_SAVED', len(steps), flush=True)
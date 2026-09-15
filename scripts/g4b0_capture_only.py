# G4-B0 capture-only: hook first base call, save prefill boundary, exit immediately
import os, json, numpy as np, torch, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
OUT = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/g4b_boundary'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/capture.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
log('LOADED')
state = {'done': False}
def hk(mod, *hook_args, **hook_kwargs):
    # compat: new-style (args, kwargs, output) or old-style (input, output)
    if len(hook_args) == 3:
        args, kwargs, output = hook_args
    elif len(hook_args) == 2:
        args, output = hook_args
        kwargs = {}
    else:
        args = hook_args[0] if hook_args else None; kwargs = {}; output = None
    if state['done']:
        return None
    state['done'] = True
    try:
        log('HOOK raw args n=' + str(len(args) if args else 0) + ' kw_keys=' + str(list(kwargs.keys()) if kwargs else []))
        emb = kwargs.get('inputs_embeds', args[0] if args else None)
        pos = kwargs.get('position_ids')
        cp = kwargs.get('cache_position')
        pkv = kwargs.get('past_key_values')
        log('HOOK emb ' + str(tuple(emb.shape)))
        log('HOOK pos ' + (str(tuple(pos.shape)) if pos is not None else 'None'))
        log('HOOK cp ' + (str(cp.tolist()) if cp is not None else 'None'))
        if hasattr(pkv, 'layers') and len(pkv.layers) > 0:
            l0 = pkv.layers[0]
            log('HOOK kv layers ' + str(len(pkv.layers)) + ' l0.keys ' + (str(tuple(l0.keys.shape)) if l0.keys is not None else 'None'))
            kk = torch.stack([pkv.layers[i].keys for i in range(28)], 0)
            vv = torch.stack([pkv.layers[i].values for i in range(28)], 0)
            log('HOOK kv stacked ' + str(tuple(kk.shape)))
            np.save(OUT + '/kv_k.npy', kk.float().numpy().astype(np.float32))
            np.save(OUT + '/kv_v.npy', vv.float().numpy().astype(np.float32))
            np.savez(OUT + '/step0_inputs.npz', emb=emb.float().numpy().astype(np.float32), pos=pos.float().numpy().astype(np.float32) if pos is not None else np.array([]), cp=cp.long().numpy() if cp is not None else np.array([]))
            with open(OUT + '/manifest.json', 'w') as f:
                json.dump({'kv_shape': list(kk.shape), 'emb_shape': list(emb.shape), 'pos_shape': list(pos.shape) if pos is not None else None, 'cp': cp.tolist() if cp is not None else None, 'kv_len': kk.shape[3]}, f, indent=2)
            log('BOUNDARY_SAVED kv_len=' + str(kk.shape[3]))
        else:
            log('HOOK pkv empty (pre-prefill?) layers=' + str(len(pkv.layers) if hasattr(pkv,'layers') else 'n/a'))
    except Exception as e:
        log('HOOK ERR: ' + str(e)[:300])
        traceback.print_exc(file=LOG)
    # signal stop
    state['stop'] = True
    return None
h = base.register_forward_hook(hk)
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
try:
    import threading
    # run generate in thread, main polls state; stop after hook captures
    res = [None]
    def run_gen():
        try:
            res[0] = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
        except Exception as e:
            log('GEN ERR: ' + str(e)[:200])
    t = threading.Thread(target=run_gen, daemon=True)
    t.start()
    import time
    deadline = time.time() + 120
    while not state.get('done') and time.time() < deadline:
        time.sleep(0.2)
    log('DONE state=' + str(state))
except Exception as e:
    log('MAIN ERR: ' + str(e)[:200])
h.remove()
print('G4B0_CAPTURE_DONE', flush=True)
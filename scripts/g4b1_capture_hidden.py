# G4-B0 capture via monkey-patched base.forward (no hook arg issues)
import os, json, numpy as np, torch, traceback, time
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
orig_fwd = base.forward
state = {'done': False}
def patched_fwd(*args, **kwargs):
    state.setdefault('count', 0)
    state['count'] += 1
    if state['count'] == 2 and not state.get('done2'):
        state['done2'] = True
        state['done'] = True
        try:
            emb = kwargs.get('inputs_embeds', args[0] if args else None)
            pos = kwargs.get('position_ids')
            cp = kwargs.get('cache_position')
            pkv = kwargs.get('past_key_values')
            log('PATCH emb ' + (str(tuple(emb.shape)) if emb is not None else 'None'))
            log('PATCH pos ' + (str(tuple(pos.shape)) if pos is not None else 'None'))
            log('PATCH cp ' + (str(cp.tolist()) if cp is not None else 'None'))
            if hasattr(pkv, 'layers') and len(pkv.layers) > 0:
                kk = torch.stack([pkv.layers[i].keys for i in range(28)], 0)
                vv = torch.stack([pkv.layers[i].values for i in range(28)], 0)
                log('PATCH kv ' + str(tuple(kk.shape)))
                np.save(OUT + '/kv_k_decode.npy', kk.float().numpy().astype(np.float32))
                np.save(OUT + '/kv_v_decode.npy', vv.float().numpy().astype(np.float32))
                np.savez(OUT + '/step_decode_inputs.npz', emb=emb.float().numpy().astype(np.float32) if emb is not None else np.array([]), pos=pos.float().numpy().astype(np.float32) if pos is not None else np.array([]), cp=cp.long().numpy() if cp is not None else np.array([]))
                with open(OUT + '/manifest.json', 'w') as f:
                    json.dump({'kv_shape': list(kk.shape), 'emb_shape': list(emb.shape) if emb is not None else None, 'pos_shape': list(pos.shape) if pos is not None else None, 'cp': cp.tolist() if cp is not None else None, 'kv_len': kk.shape[3]}, f, indent=2)
                log('BOUNDARY_SAVED kv_len=' + str(kk.shape[3]))
            else:
                log('PATCH pkv empty')
        except Exception as e:
            log('PATCH ERR: ' + str(e)[:300])
    out = orig_fwd(*args, **kwargs)
    if state.get('count') == 2 and not state.get('saved_out'):
        state['saved_out'] = True
        try:
            hs = out.last_hidden_state if hasattr(out, 'last_hidden_state') else None
            if hs is not None:
                np.save(OUT + '/decode_step0_hidden.npy', hs.detach().float().numpy().astype(np.float32))
                log('SAVED decode hidden ' + str(tuple(hs.shape)))
            else:
                log('no last_hidden_state on output')
        except Exception as e:
            log('SAVE OUT ERR: ' + str(e)[:200])
    return out
base.forward = patched_fwd
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
t0 = time.time()
try:
    wavs, sr = gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True)
    log('GEN OK sr ' + str(sr) + ' secs ' + str(round(time.time()-t0,1)) + ' nwav ' + str(len(wavs)))
except Exception as e:
    log('GEN ERR: ' + str(e)[:300])
log('DONE state=' + str(state['done']))
print('G4B0_MONKEY_DONE', flush=True)
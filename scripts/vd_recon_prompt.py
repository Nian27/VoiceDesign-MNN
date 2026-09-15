import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
m = gw.model; t = m.talker; tc = m.config.talker_config
print('TTS specials: bos=%s eos=%s pad=%s' % (m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id))
print('text_vocab=%s text_hidden=%s' % (t.model.text_embedding.weight.shape[0], t.model.text_embedding.weight.shape[1]))
instruct = '苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。'
text = '小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。'
instr_txt = '<|im_start|>user\n' + instruct + '<|im_end|>\n'
asst_txt  = '<|im_start|>assistant\n' + text + '<|im_end|>\n<|im_start|>assistant\n'
tok = gw.processor
iid = tok(text=instr_txt, return_tensors='pt')['input_ids']
aid = tok(text=asst_txt,  return_tensors='pt')['input_ids']
print('instruct_ids len=%d  assistant_ids len=%d' % (iid.shape[1], aid.shape[1]))
print('  iid[:6]=', iid[0,:6].tolist(), '... tail=', iid[0,-3:].tolist())
print('  aid[:6]=', aid[0,:6].tolist(), '... tail=', aid[0,-6:].tolist())
TXT_EMB = t.get_text_embeddings(); CD_EMB = t.get_input_embeddings(); PROJ = t.text_projection
def te(ids):  return PROJ(TXT_EMB(torch.tensor([ids])))
def ce(ids):  return CD_EMB(torch.tensor([ids]))
with torch.no_grad():
    bos, eos, pad = te([m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]).chunk(3, dim=1)
    ins = te(iid[0].tolist())
    lang = tc.codec_language_id['chinese']
    c0 = ce([tc.codec_think_id, tc.codec_think_bos_id, lang, tc.codec_think_eos_id])
    c1 = ce([tc.codec_pad_id, tc.codec_bos_id])
    codec = torch.cat([c0, c1], dim=1)
    role = te(aid[0,:3].tolist())
    main = torch.cat([pad.expand(-1, codec.shape[1]-2, -1), bos], dim=1) + codec[:, :-1]
    head = torch.cat([role, main], dim=1)
    body_ids = aid[0, 3:-5].tolist()
    body = torch.cat((te(body_ids), eos), dim=1) + ce([tc.codec_pad_id]*(len(body_ids)+1))
    tail = pad + ce([tc.codec_bos_id])
    full = torch.cat([ins, head, body, tail], dim=1)
    got = full[0].numpy().astype(np.float32)
want = np.fromfile(D+'/vd_m5b/prefill_emb.raw', dtype=np.float32).reshape(-1,2048)
print()
print('reconstructed shape =', got.shape, ' captured =', want.shape)
if got.shape == want.shape:
    a=want.ravel().astype(np.float64); b=got.ravel().astype(np.float64)
    print('GLOBAL cos = %.8f' % float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)))
    per = [float(np.dot(want[i].astype(np.float64), got[i].astype(np.float64))/
                 (np.linalg.norm(want[i])*np.linalg.norm(got[i])+1e-12)) for i in range(got.shape[0])]
    print('per-token cos: min=%.6f max=%.6f' % (min(per), max(per)))
    bad=[i for i,c in enumerate(per) if c<0.999]
    print('tokens with cos<0.999:', bad if bad else 'NONE')
    print('exact_equal =', bool(np.array_equal(want, got)), ' maxabsdiff=%.3e' % float(np.abs(want-got).max()))
else:
    print('LENGTH MISMATCH -> 拼装逻辑还需修正')
print('RECON_DONE')

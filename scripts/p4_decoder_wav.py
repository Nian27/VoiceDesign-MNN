# P4 decoder: 20-step codes -> tokenizer-decoder MNN -> PCM wav
import os, json, numpy as np, MNN, time
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
LOG = open(D + '/p4_decoder.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
# load 20-step codes (15 per step)
traj = json.load(open(D + '/p4_closure_codes.json'))
steps = traj['steps']
log('steps ' + str(len(steps)))
# build decoder input codes (1,16,300): 16 codebooks x 300 frames
CODEC0 = 2149  # codec_bos_id placeholder for code0
codes = np.full((1, 16, 300), 0, dtype=np.int32)
# code0 row: placeholder; rows 1..15 from trajectory
for t, st in enumerate(steps[:300]):
    codes[0, 0, t] = CODEC0
    for g in range(15):
        codes[0, g+1, t] = st['codes15'][g]
log('decoder input shape ' + str(codes.shape) + ' finite ' + str(bool(np.isfinite(codes).all())))
codes.tofile(D + '/p4_dec_codes_i32.raw')
# run decoder MNN (static_t300)
interp = MNN.Interpreter(D + '/tokenizer_decoder_static_t300.mnn')
s2 = interp.createSession()
t_in = interp.getSessionInput(s2, 'codes')
ht = MNN.Tensor((1,16,300), MNN.Halide_Type_Int, codes.ravel(), MNN.Tensor_DimensionType_Caffe)
t_in.copyFromHostTensor(ht)
ot = interp.getSessionOutput(s2, 'waveform')
t0 = time.time()
interp.runSession(s2)
t1 = time.time()
osh = [d if d > 0 else 1 for d in ot.getShape()]
ho = MNN.Tensor(osh, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
ot.copyToHostTensor(ho)
pcm = np.asarray(ho.getNumpyData()).reshape(osh).astype(np.float32)
log('decoder ms ' + str(round((t1-t0)*1000,1)) + ' pcm ' + str(pcm.shape))
pcm_flat = pcm.ravel()
log('pcm finite ' + str(bool(np.isfinite(pcm_flat).all())) + ' maxabs ' + str(round(float(np.abs(pcm_flat).max()),4)) + ' nonzero ' + str(bool(np.count_nonzero(pcm_flat) > 0)) + ' rms ' + str(round(float(np.sqrt(np.mean(pcm_flat.astype(np.float64)**2))),6)))
# save wav (24kHz)
import soundfile as sf
sr = 24000
wav_path = D + '/p4_reference_candidate.wav'
sf.write(wav_path, pcm_flat, sr)
log('WAV_SAVED ' + wav_path + ' dur ' + str(round(len(pcm_flat)/sr, 3)) + 's')
print('P4_DECODER_DONE', bool(np.isfinite(pcm_flat).all()), bool(np.count_nonzero(pcm_flat) > 0), flush=True)
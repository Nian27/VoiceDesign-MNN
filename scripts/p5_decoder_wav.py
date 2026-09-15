# P5 decoder: full 500-frame 16-code -> decoder MNN (T300 first 300) -> PCM wav
import os, json, numpy as np, MNN, time
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
LOG = open(D + '/p5_decoder.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
codes = np.load(D + '/p5_full_codes.npy')  # [500,16]
log('codes shape ' + str(codes.shape) + ' range ' + str(int(codes.min())) + '..' + str(int(codes.max())))
# first 300 frames (decoder T300 cap)
N = min(300, codes.shape[0])
codes_in = codes[:N].T  # [16, N]
# decoder expects (1,16,300): pad to 300
codes_pad = np.zeros((1,16,300), dtype=np.int32)
codes_pad[0,:,:N] = codes_in
codes_pad.tofile(D + '/p5b_dec_codes_i32.raw')
interp = MNN.Interpreter(D + '/tokenizer_decoder_static_t300.mnn')
s2 = interp.createSession()
t_in = interp.getSessionInput(s2, 'codes')
ht = MNN.Tensor((1,16,300), MNN.Halide_Type_Int, codes_pad.ravel(), MNN.Tensor_DimensionType_Caffe)
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
log('finite ' + str(bool(np.isfinite(pcm_flat).all())) + ' maxabs ' + str(round(float(np.abs(pcm_flat).max()),4)) + ' nonzero ' + str(bool(np.count_nonzero(pcm_flat) > 0)) + ' rms ' + str(round(float(np.sqrt(np.mean(pcm_flat.astype(np.float64)**2))),6)))
import soundfile as sf
sr = 24000
wav_path = D + '/p5_reference_v2.wav'
sf.write(wav_path, pcm_flat, sr)
log('WAV_SAVED ' + wav_path + ' dur ' + str(round(len(pcm_flat)/sr, 3)) + 's (frames ' + str(N) + ')')
print('P5_DECODER_DONE', bool(np.isfinite(pcm_flat).all()), bool(np.count_nonzero(pcm_flat) > 0), flush=True)
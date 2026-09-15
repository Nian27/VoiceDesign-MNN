import wave, numpy as np, os
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_m5b_out'
p = D + '/reference_m5b_10f.wav'
w = wave.open(p)
print('channels', w.getnchannels(), 'rate', w.getframerate(), 'frames', w.getnframes(), 'sampwidth', w.getsampwidth())
d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
print('dur %.3fs  rms %.5f  peak %.4f  zero-frac %.3f' % (len(d)/w.getframerate(), float(np.sqrt((d**2).mean())), float(np.abs(d).max()), float((np.abs(d) < 1e-4).mean())))
c = np.fromfile(D + '/codes_10f_i32.raw', dtype=np.int32).reshape(-1, 16)
print('codes', c.shape)
print(c[:5])

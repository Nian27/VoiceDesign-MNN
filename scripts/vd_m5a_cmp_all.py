import numpy as np, subprocess, os
ADB = r'C:/Users/Administrator/AppData/Local/Android/Sdk/platform-tools/adb.exe'
DEV = '/data/local/tmp/qwen3tts/vd_m5a'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
os.makedirs(D + '/vd_m5a_out', exist_ok=True)
steps = [0,1,5,10,19]
allpass = True
for s in steps:
    local = D + '/vd_m5a_out/step%d_hidden_dev.raw' % s
    subprocess.run([ADB,'-s','A3TE025B03003242','pull','%s/outs%d/hidden_states.raw'%(DEV,s), local],
                   capture_output=True)
    a = np.fromfile(local, dtype=np.float32).astype(np.float64)
    b = np.fromfile(D + '/vd_m5a_inputs/step%d_hidden_ref.raw' % s, dtype=np.float32).astype(np.float64)
    c = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    ok = c >= 0.999
    allpass = allpass and ok
    print('step %2d  cos=%.7f  maxdiff=%.5f  devMax=%.3f refMax=%.3f  %s' % (
        s, c, float(np.abs(a-b).max()), float(np.abs(a).max()), float(np.abs(b).max()), 'PASS' if ok else 'FAIL'))
print('VD_M5A_2_ALL_PASS =', allpass)

// g2a_loop.cpp - G2-A: 20-step FORCED identical trajectory, host-owned KV
// usage: g2a_loop MODEL EMBSEQ COSSEQ SINSEQ MASKSEQ SLOTSEQ INIT_KVK INIT_KVV OUTDIR [cpu|htp] [N] [CP0]
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
using namespace MNN;
using namespace MNN::Express;
static std::vector<char> readFile(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(2); }
    fclose(f);
    return v;
}
static void wf(const char* p, const float* d, size_t n) {
    FILE* f = fopen(p, "wb"); if (!f) return; fwrite(d, 4, n, f); fclose(f);
}
int main(int argc, char** argv) {
    if (argc < 11) { fprintf(stderr, "usage: %s MODEL EMBSEQ COSSEQ SINSEQ MASKSEQ SLOTSEQ KVK KVV OUTDIR [cpu|htp] [N] [CP0]\n", argv[0]); return 2; }
    const char* model = argv[1];
    std::vector<char> embB = readFile(argv[2]), cosB = readFile(argv[3]), sinB = readFile(argv[4]);
    std::vector<char> maskB = readFile(argv[5]), slotB = readFile(argv[6]);
    std::vector<char> kvkB = readFile(argv[7]), kvvB = readFile(argv[8]);
    const char* outDir = argv[9];
    const char* backend = argc > 10 ? argv[10] : "cpu";
    int N = argc > 11 ? atoi(argv[11]) : 20;
    int CP0 = argc > 12 ? atoi(argv[12]) : 62;
    const int KV = 768;
    const size_t KVONE = (size_t)28*8*KV*128;
    if (kvkB.size() != KVONE*4) { fprintf(stderr, "bad kv size %zu\n", kvkB.size()); return 2; }
    std::vector<float> kvk(KVONE), kvv(KVONE);
    memcpy(kvk.data(), kvkB.data(), KVONE*4);
    memcpy(kvv.data(), kvvB.data(), KVONE*4);
    BackendConfig bc; bc.precision = BackendConfig::Precision_High; bc.power = BackendConfig::Power_High; bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if (strcmp(backend, "htp") == 0) { sched.type = MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon\n"); }
    else { sched.type = MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    sched.numThread = 4; sched.backendConfig = &bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    Module::Config mcfg; mcfg.shapeMutable = true;
    std::shared_ptr<Module> mod(Module::load(
        {"inputs_embeds","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"},
        {"hidden_states","cur_k","cur_v"}, model, rtm, &mcfg), Module::destroy);
    if (!mod) { fprintf(stderr, "load failed\n"); return 3; }
    VARP vEmb  = _Input({1,1,2048}, NCHW, halide_type_of<float>());
    VARP vK    = _Input({28,1,8,KV,128}, NCHW, halide_type_of<float>());
    VARP vV    = _Input({28,1,8,KV,128}, NCHW, halide_type_of<float>());
    VARP vCos  = _Input({1,1,128}, NCHW, halide_type_of<float>());
    VARP vSin  = _Input({1,1,128}, NCHW, halide_type_of<float>());
    VARP vMask = _Input({1,1,1,KV}, NCHW, halide_type_of<float>());
    VARP vSlot = _Input({1,1,KV,1}, NCHW, halide_type_of<float>());
    double tot = 0;
    for (int n = 0; n < N; n++) {
        int cp = CP0 + n;
        memcpy(vEmb->writeMap<float>(), embB.data() + (size_t)n*2048*4, 2048*4);
        memcpy(vK->writeMap<float>(), kvk.data(), KVONE*4);
        memcpy(vV->writeMap<float>(), kvv.data(), KVONE*4);
        memcpy(vCos->writeMap<float>(), cosB.data() + (size_t)n*128*4, 128*4);
        memcpy(vSin->writeMap<float>(), sinB.data() + (size_t)n*128*4, 128*4);
        memcpy(vMask->writeMap<float>(), maskB.data() + (size_t)n*KV*4, KV*4);
        memcpy(vSlot->writeMap<float>(), slotB.data() + (size_t)n*KV*4, KV*4);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<VARP> out = mod->onForward({vEmb, vK, vV, vCos, vSin, vMask, vSlot});
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        tot += ms;
        const float* h = out[0]->readMap<float>();
        const float* ck = out[1]->readMap<float>();
        const float* cv = out[2]->readMap<float>();
        double hm = 0; bool fin = true;
        for (int i = 0; i < 2048; i++) { float x = h[i]; if (x!=x||x>1e37f||x<-1e37f) fin=false; double a=x<0?-x:x; if (a>hm) hm=a; }
        // host-side KV write at slot cp
        for (int l = 0; l < 28; l++) for (int hh = 0; hh < 8; hh++) {
            size_t off = ((size_t)(l*8+hh)*KV + cp) * 128;
            memcpy(&kvk[off], ck + ((size_t)(l*8+hh))*128, 128*4);
            memcpy(&kvv[off], cv + ((size_t)(l*8+hh))*128, 128*4);
        }
        char p[256];
        snprintf(p, sizeof(p), "%s/step%02d_hidden.raw", outDir, n); wf(p, h, 2048);
        snprintf(p, sizeof(p), "%s/step%02d_curk.raw", outDir, n);   wf(p, ck, 28*8*128);
        snprintf(p, sizeof(p), "%s/step%02d_curv.raw", outDir, n);   wf(p, cv, 28*8*128);
        printf("STEP %02d cp=%3d ms=%7.1f hidden_maxabs=%.5f finite=%d\n", n, cp, ms, hm, fin?1:0);
        fflush(stdout);
    }
    printf("TOTAL_MS=%.0f  AVG_MS=%.1f\n", tot, tot/N);
    {
        FILE* sf = fopen("/proc/self/status","r"); char ln[256];
        if (sf) { while (fgets(ln,sizeof(ln),sf)) { if (!strncmp(ln,"VmRSS",5)||!strncmp(ln,"VmHWM",5)) printf("MEM %s", ln); } fclose(sf); }
    }
    printf("G2A_DONE\n");
    return 0;
}

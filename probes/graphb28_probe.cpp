// graphb28_probe.cpp - 28L GraphB v6: hidden_states + cur_k/cur_v (G1/G2 gate)
// usage: graphb28_probe MODEL EMB KVK KVV COS SIN MASK SLOT OUT_DIR [cpu|htp] [REPEATS]
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static void writeRaw(const char* p, const float* d, size_t n) {
    FILE* f = fopen(p, "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", p); return; }
    fwrite(d, sizeof(float), n, f); fclose(f);
}
static uint32_t crc32f(const float* d, size_t n) {
    const uint8_t* p = (const uint8_t*)d; uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n*4; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1))); }
    return ~c;
}
int main(int argc, char** argv) {
    if (argc < 11) { fprintf(stderr, "usage: %s MODEL EMB KVK KVV COS SIN MASK SLOT OUT_DIR [cpu|htp] [REPEATS]\n", argv[0]); return 2; }
    const char* model = argv[1];
    std::vector<std::vector<char>> raws;
    for (int i = 2; i <= 8; i++) raws.push_back(readFile(argv[i]));
    const char* outDir = argv[9];
    const char* backend = argc > 10 ? argv[10] : "cpu";
    int repeats = argc > 11 ? atoi(argv[11]) : 1;
    BackendConfig bc; bc.precision = BackendConfig::Precision_High; bc.power = BackendConfig::Power_High; bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if (strcmp(backend, "htp") == 0) { sched.type = MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon\n"); }
    else { sched.type = MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    sched.numThread = 4; sched.backendConfig = &bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    if (!rtm) { fprintf(stderr, "rtm failed\n"); return 3; }
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    std::vector<std::string> ins = {"inputs_embeds","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"};
    std::vector<std::string> outs = {"hidden_states","cur_k","cur_v"};
    Module::Config mcfg; mcfg.shapeMutable = true;
    std::shared_ptr<Module> mod(Module::load(ins, outs, model, rtm, &mcfg), Module::destroy);
    if (!mod) { fprintf(stderr, "load failed\n"); return 3; }
    const Module::Info* info = mod->getInfo();
    printf("N_INPUTS=%zu\n", info->inputs.size());
    std::vector<VARP> inputs;
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        auto order = ii.order == NC4HW4 ? NCHW : ii.order;
        VARP v = _Input(ii.dim, order, ii.type);
        memcpy(v->writeMap<float>(), raws[i].data(), raws[i].size());
        inputs.push_back(v);
    }
    double best = 1e9, worst = 0;
    std::vector<VARP> out;
    for (int r = 0; r < repeats + 1; r++) {
        auto t0 = std::chrono::steady_clock::now();
        out = mod->onForward(inputs);
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        if (r == 0) printf("WARM_MS=%.0f\n", ms);
        else { if (ms < best) best = ms; if (ms > worst) worst = ms; printf("RUN[%d]_MS=%.0f\n", r, ms); }
    }
    printf("LAT min=%.0f max=%.0f\n", best, worst);
    for (size_t i = 0; i < out.size(); i++) {
        auto inf = out[i]->getInfo();
        size_t elems = 1; for (int d : inf->dim) elems *= d > 0 ? d : 1;
        const float* d = out[i]->readMap<float>();
        double mx = 0; bool fin = true;
        for (size_t k = 0; k < elems; k++) { float x=d[k]; if (x!=x||x>1e37f||x<-1e37f) fin=false; double a=x<0?-x:x; if (a>mx&&a<1e37) mx=a; }
        printf("OUT[%zu] %-13s elems=%zu maxabs=%.6f finite=%d crc32=%08x\n", i, outs[i].c_str(), elems, mx, fin?1:0, crc32f(d, elems));
        std::string f = std::string(outDir) + "/" + outs[i] + ".raw";
        writeRaw(f.c_str(), d, elems);
    }
    {
        FILE* sf = fopen("/proc/self/status", "r"); char ln[256];
        if (sf) { while (fgets(ln, sizeof(ln), sf)) { if (!strncmp(ln,"VmRSS",5)||!strncmp(ln,"VmHWM",5)) printf("MEM %s", ln); } fclose(sf); }
    }
    printf("DONE\n");
    return 0;
}

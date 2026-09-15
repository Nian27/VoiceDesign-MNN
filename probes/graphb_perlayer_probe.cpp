// graphb_perlayer_probe.cpp - dump per-layer hidden for CPU vs HTP first-divergence
// usage: graphb_perlayer_probe MODEL CP EMB KVK KVV POS OUT_DIR [cpu|htp]
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
static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
int main(int argc, char** argv) {
    if (argc < 9) { fprintf(stderr, "usage: %s MODEL CP EMB KVK KVV POS OUT_DIR [cpu|htp]\n", argv[0]); return 2; }
    const char* model = argv[1];
    std::vector<std::vector<char>> raws;
    for (int i = 2; i <= 6; i++) raws.push_back(readFile(argv[i]));
    const char* outDir = argv[7];
    const char* backend = argc > 8 ? argv[8] : "cpu";
    BackendConfig bc; bc.precision = BackendConfig::Precision_High; bc.power = BackendConfig::Power_High; bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if (strcmp(backend, "htp") == 0) { sched.type = MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon\n"); }
    else { sched.type = MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    sched.numThread = 4; sched.backendConfig = &bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    if (!rtm) { fprintf(stderr, "rtm failed\n"); return 3; }
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    std::vector<std::string> ins = {"cache_position","inputs_embeds","kv_k","kv_v","position_ids"};
    std::vector<std::string> outs = {"layer_hidden","hidden_states","kv_k_out","kv_v_out"};
    Module::Config mcfg; mcfg.shapeMutable = true;
    std::shared_ptr<Module> mod(Module::load(ins, outs, model, rtm, &mcfg), Module::destroy);
    if (!mod) { fprintf(stderr, "load failed\n"); return 3; }
    auto order = [](const Module::Info* info, size_t i) {
        auto o = info->inputs[i].order; return o == NC4HW4 ? NCHW : o;
    };
    const Module::Info* info = mod->getInfo();
    std::vector<VARP> inputs;
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        VARP v = _Input(ii.dim, order(info, i), ii.type);
        if (ii.type == halide_type_of<int32_t>()) memcpy(v->writeMap<int32_t>(), raws[i].data(), raws[i].size());
        else memcpy(v->writeMap<float>(), raws[i].data(), raws[i].size());
        inputs.push_back(v);
    }
    double t0 = nowMs();
    std::vector<VARP> out = mod->onForward(inputs);
    printf("FORWARD_MS=%.0f\n", nowMs() - t0);
    for (size_t i = 0; i < out.size() && i < 4; i++) {
        auto inf = out[i]->getInfo();
        if (!inf) { fprintf(stderr, "out %zu no info\n", i); continue; }
        size_t elems = 1; for (int d : inf->dim) elems *= d > 0 ? d : 1;
        const float* d = out[i]->readMap<float>();
        if (!d) { fprintf(stderr, "out %zu no map\n", i); continue; }
        double mx = 0, sum = 0;
        for (size_t k = 0; k < elems; k++) { double a = d[k] < 0 ? -d[k] : d[k]; if (a > mx) mx = a; sum += d[k]; }
        printf("OUT[%zu] elems=%zu maxabs=%.6f mean=%.6f\n", i, elems, mx, sum / (double)elems);
        std::string f = std::string(outDir) + "/out" + std::to_string(i) + ".raw";
        writeRaw(f.c_str(), d, elems);
    }
    {
        FILE* sf = fopen("/proc/self/status", "r"); char ln[256];
        if (sf) { while (fgets(ln, sizeof(ln), sf)) { if (!strncmp(ln,"VmRSS",5)) printf("MEM %s", ln); } fclose(sf); }
    }
    printf("DONE\n");
    return 0;
}

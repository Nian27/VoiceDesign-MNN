// graphb_module_probe.cpp - Express Module API, 5-in / 3-out GraphB runner
// (known-good path: qwen3tts_module_probe produced cos=1.0 on device)
// usage: graphb_module_probe MODEL CP EMB KVK KVV POS OUT_DIR REPEATS
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
static unsigned int crc32buf(const unsigned char* d, size_t n) {
    unsigned int c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) { c ^= d[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u))); }
    return ~c;
}
static void writeRaw(const char* p, const float* d, size_t elems) {
    FILE* f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", p); return; }
    fwrite(d, sizeof(float), elems, f);
    fclose(f);
}
int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s MODEL CP EMB KVK KVV POS OUT_DIR REPEATS\n", argv[0]);
        return 2;
    }
    std::vector<std::vector<char>> raws;
    for (int i = 2; i <= 6; i++) raws.push_back(readFile(argv[i]));
    const char* outDir = argv[7];
    int repeats = atoi(argv[8]);
    const char* backendStr = argc > 9 ? argv[9] : "cpu";
    for (int i = 0; i < 5; i++)
        fprintf(stderr, "raw[%d] bytes=%zu crc32=%08x\n", i, raws[i].size(), crc32buf((const unsigned char*)raws[i].data(), raws[i].size()));

    std::vector<std::string> inputNames = {"cache_position","inputs_embeds","kv_k","kv_v","position_ids"};
    std::vector<std::string> outputNames = {"hidden_states","kv_k_out","kv_v_out"};
    BackendConfig bc;
    bc.precision = BackendConfig::Precision_High;
    bc.power = BackendConfig::Power_High;
    bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if (strcmp(backendStr, "htp") == 0) { sched.type = MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon\n"); }
    else if (strcmp(backendStr, "opencl") == 0) { sched.type = MNN_FORWARD_OPENCL; printf("BACKEND=opencl\n"); }
    else { sched.type = MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    sched.numThread = 4;
    sched.backendConfig = &bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    if (!rtm) { fprintf(stderr, "createRuntimeManager failed\n"); return 3; }
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    Module::Config mcfg;
    mcfg.shapeMutable = true;
    std::shared_ptr<Module> module(Module::load(inputNames, outputNames, argv[1], rtm, &mcfg), Module::destroy);
    if (!module) { fprintf(stderr, "module load failed\n"); return 3; }
    const Module::Info* info = module->getInfo();
    if (!info) { fprintf(stderr, "no module info\n"); return 3; }
    printf("inputs from model: %zu\n", info->inputs.size());
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        printf("  in[%zu] dims=[", i);
        for (int d : ii.dim) printf("%d,", d);
        printf("] order=%d bits=%d code=%d\n", (int)ii.order, ii.type.bits, ii.type.code);
    }
    std::vector<VARP> inputs;
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        auto order = ii.order == NC4HW4 ? NCHW : ii.order;
        VARP v = _Input(ii.dim, order, ii.type);
        if (!v->getInfo()) { fprintf(stderr, "input alloc failed %zu\n", i); return 3; }
        if (ii.type == halide_type_of<int32_t>())
            memcpy(v->writeMap<int32_t>(), raws[i].data(), raws[i].size());
        else
            memcpy(v->writeMap<float>(), raws[i].data(), raws[i].size());
        inputs.push_back(v);
    }
    std::vector<VARP> outputs;
    double sumMs = 0, minMs = 1e9;
    for (int r = -1; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        outputs = module->onForward(inputs);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r < 0) printf("warm_ms=%.1f\n", ms);
        else { printf("run_ms=%.1f\n", ms); sumMs += ms; if (ms < minMs) minMs = ms; }
    }
    if (repeats > 0) printf("LAT avg_ms=%.1f min_ms=%.1f\n", sumMs / repeats, minMs);
    if (outputs.size() < 1) { fprintf(stderr, "no output\n"); return 3; }
    const char* onames[3] = {"hidden_states","kv_k_out","kv_v_out"};
    for (size_t oi = 0; oi < outputs.size() && oi < 3; oi++) {
        VARP out = outputs[oi];
        auto inf = out->getInfo();
        if (!inf) { fprintf(stderr, "out %zu info failed\n", oi); continue; }
        size_t elems = 1;
        for (int d : inf->dim) elems *= d > 0 ? d : 1;
        const float* data = out->readMap<float>();
        if (!data) { fprintf(stderr, "out %zu read failed\n", oi); continue; }
        double mx = 0;
        for (size_t i = 0; i < elems; i++) { double a = data[i] < 0 ? -data[i] : data[i]; if (a > mx) mx = a; }
        printf("OUT %s elems=%zu maxabs=%.6f crc32=%08x\n", onames[oi], elems, mx, crc32buf((const unsigned char*)data, elems*sizeof(float)));
        std::string f = std::string(outDir) + "/" + onames[oi] + ".raw";
        writeRaw(f.c_str(), data, elems);
    }
    {
        FILE* sf = fopen("/proc/self/status", "r");
        if (sf) { char ln[256];
            while (fgets(ln, sizeof(ln), sf)) {
                if (strncmp(ln, "VmHWM", 5) == 0 || strncmp(ln, "VmRSS", 5) == 0 || strncmp(ln, "VmPeak", 6) == 0) printf("MEM %s", ln);
            }
            fclose(sf);
        }
    }
    printf("DONE\n");
    return 0;
}

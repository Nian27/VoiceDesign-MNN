// codepred_module_probe.cpp - code_predictor single-step 1-input runner
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
    size_t rd = fread(v.data(), 1, n, f);
    if ((long)rd != n) { fprintf(stderr, "short read %s\n", p); exit(2); }
    fclose(f);
    return v;
}
static unsigned int crc32(const unsigned char* data, size_t n) {
    unsigned int crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        crc ^= data[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MODEL EMB OUT REPEATS\n", argv[0]);
        return 2;
    }
    std::vector<char> embB = readFile(argv[2]);
    printf("emb bytes=%zu crc32=%08x\n", embB.size(), crc32((const unsigned char*)embB.data(), embB.size()));
    int repeats = atoi(argv[4]);
    std::vector<std::string> inputNames = {"inputs_embeds"};
    std::vector<std::string> outputNames = {"logits"};
    BackendConfig bc;
    bc.precision = BackendConfig::Precision_High;
    bc.power = BackendConfig::Power_High;
    bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    sched.type = MNN_FORWARD_CPU;
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
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        printf("  in[%zu] dims=[", i);
        for (int d : ii.dim) printf("%d,", d);
        printf("] order=%d bits=%d\n", (int)ii.order, (int)ii.type.bits);
    }
    const auto& i0 = info->inputs[0];
    VARP v0 = _Input(i0.dim, i0.order == NC4HW4 ? NCHW : i0.order, i0.type);
    memcpy(v0->writeMap<float>(), embB.data(), embB.size());
    std::vector<VARP> inputs = {v0};
    std::vector<VARP> outputs;
    for (int r = -1; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        outputs = module->onForward(inputs);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf(r < 0 ? "warm_ms=%.1f\n" : "run_ms=%.1f\n", ms);
    }
    if (outputs.empty()) { fprintf(stderr, "no output\n"); return 3; }
    VARP ov = outputs[0];
    auto oi = ov->getInfo();
    if (!oi) { fprintf(stderr, "output info failed\n"); return 3; }
    size_t elems = 1;
    for (int d : oi->dim) elems *= d > 0 ? d : 1;
    const float* data = ov->readMap<float>();
    if (!data) { fprintf(stderr, "output read failed\n"); return 3; }
    double mx = 0; size_t finite = 0; double sum = 0;
    for (size_t i = 0; i < elems; i++) { float v = data[i]; double av = v < 0 ? -v : v; if (av > mx) mx = av; sum += v; finite++; }
    FILE* f = fopen(argv[3], "wb");
    fwrite(data, sizeof(float), elems, f);
    fclose(f);
    printf("elems=%zu finite=%zu maxabs=%.6f crc=%08x\n", elems, finite, mx, crc32((const unsigned char*)data, elems * sizeof(float)));
    return 0;
}
// qwen3tts_module_probe.cpp - GraphB via Express Module API, MNN input order
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
    if (argc < 9) {
        fprintf(stderr, "usage: %s MODEL CP EMB KVK KVV POS OUT REPEATS\n", argv[0]);
        return 2;
    }
    // argv order = MNN model input order: cache_position, inputs_embeds, kv_k, kv_v, position_ids
    fprintf(stderr, "S1 argc ok\n");
    std::vector<std::vector<char>> raws;
    for (int i = 2; i <= 6; i++) raws.push_back(readFile(argv[i]));
    fprintf(stderr, "S2 raws read\n");
    fprintf(stderr, "S2a pre-crc\n");
    for (int i = 0; i < 5; i++)
        fprintf(stderr, "raw[%d] bytes=%zu crc32=%08x\n", i, raws[i].size(), crc32((const unsigned char*)raws[i].data(), raws[i].size()));
    fprintf(stderr, "S2b post-crc\n");
    int repeats = atoi(argv[8]);
    fprintf(stderr, "S2c post-atoi\n");
    fprintf(stderr, "S3 before load\n");
    std::vector<std::string> inputNames = {"cache_position","inputs_embeds","kv_k","kv_v","position_ids"};
    std::vector<std::string> outputNames = {"hidden_states"};
    fprintf(stderr, "S3a bc init\n");
    BackendConfig bc;
    bc.precision = BackendConfig::Precision_High;
    bc.power = BackendConfig::Power_High;
    bc.memory = BackendConfig::Memory_Normal;
    fprintf(stderr, "S3b sched init\n");
    ScheduleConfig sched;
    sched.type = MNN_FORWARD_CPU;
    sched.numThread = 4;
    sched.backendConfig = &bc;
    fprintf(stderr, "S3c createRuntimeManager\n");
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    fprintf(stderr, "S3d rtm done\n");
    if (!rtm) { fprintf(stderr, "createRuntimeManager failed\n"); return 3; }
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    Module::Config mcfg;
    mcfg.shapeMutable = true;
    fprintf(stderr, "S4 calling load\n");
    std::shared_ptr<Module> module(Module::load(inputNames, outputNames, argv[1], rtm, &mcfg), Module::destroy);
    fprintf(stderr, "S5 load done\n");
    if (!module) { fprintf(stderr, "module load failed\n"); return 3; }
    const Module::Info* info = module->getInfo();
    if (!info) { fprintf(stderr, "no module info\n"); return 3; }
    printf("input count from model: %zu\n", info->inputs.size());
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        printf("  in[%zu] dims=[", i);
        for (int d : ii.dim) printf("%d,", d);
        printf("] order=%d bits=%d lanes=%d code=%d\n", (int)ii.order, (int)ii.type.bits, (int)ii.type.lanes, (int)ii.type.code);
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
        printf("fed[%zu] name=%s\n", i, inputNames[i].c_str());
    }
    std::vector<VARP> outputs;
    for (int r = -1; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        outputs = module->onForward(inputs);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf(r < 0 ? "warm_ms=%.1f\n" : "run_ms=%.1f\n", ms);
    }
    if (outputs.empty()) { fprintf(stderr, "no output\n"); return 3; }
    VARP out = outputs[0];
    auto oi = out->getInfo();
    if (!oi) { fprintf(stderr, "output info failed\n"); return 3; }
    size_t elems = 1;
    for (int d : oi->dim) elems *= d > 0 ? d : 1;
    const float* data = out->readMap<float>();
    if (!data) { fprintf(stderr, "output read failed\n"); return 3; }
    double mx = 0; size_t finite = 0; double sum = 0;
    for (size_t i = 0; i < elems; i++) { float v = data[i]; double av = v < 0 ? -v : v; if (av > mx) mx = av; sum += v; finite++; }
    FILE* f = fopen(argv[7], "wb");
    fwrite(data, sizeof(float), elems, f);
    fclose(f);
    printf("elems=%zu finite=%zu maxabs=%.6f mean=%.6f\n", elems, finite, mx, sum / (double)elems);
    printf("CRC32_OUT=%08x\n", crc32((const unsigned char*)data, elems * sizeof(float)));
    return 0;
}
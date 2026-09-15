// graphb_diag_probe_v5.cpp - 8 inputs / 14 outputs (scatter replaced by broadcast add)
// usage: graphb_diag_probe_v5 MODEL EMB CP KVK KVV COS SIN MASK SLOT OUT_DIR [cpu|htp]
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
int main(int argc, char** argv) {
    if (argc < 12) { fprintf(stderr, "usage: %s MODEL EMB CP KVK KVV COS SIN MASK SLOT OUT_DIR [cpu|htp]\n", argv[0]); return 2; }
    const char* model = argv[1];
    std::vector<std::vector<char>> raws;
    for (int i = 2; i <= 9; i++) raws.push_back(readFile(argv[i]));
    const char* outDir = argv[10];
    const char* backend = argc > 11 ? argv[11] : "cpu";
    BackendConfig bc; bc.precision = BackendConfig::Precision_High; bc.power = BackendConfig::Power_High; bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if (strcmp(backend, "htp") == 0) { sched.type = MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon\n"); }
    else { sched.type = MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    sched.numThread = 4; sched.backendConfig = &bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    if (!rtm) { fprintf(stderr, "rtm failed\n"); return 3; }
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    std::vector<std::string> ins = {"inputs_embeds","cache_position","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"};
    std::vector<std::string> outs = {"norm_out","q_rope","k_rope","k_all","k_T","qk_logits","qk_masked","attn_probs","attn_out","layer0_out",
                                     "kv_k_slot","kv_v_slot","kv_k_old0","kv_v_old0"};
    Module::Config mcfg; mcfg.shapeMutable = true;
    std::shared_ptr<Module> mod(Module::load(ins, outs, model, rtm, &mcfg), Module::destroy);
    if (!mod) { fprintf(stderr, "load failed\n"); return 3; }
    const Module::Info* info = mod->getInfo();
    std::vector<VARP> inputs;
    for (size_t i = 0; i < info->inputs.size(); i++) {
        const auto& ii = info->inputs[i];
        auto order = ii.order == NC4HW4 ? NCHW : ii.order;
        VARP v = _Input(ii.dim, order, ii.type);
        if (ii.type == halide_type_of<int32_t>()) memcpy(v->writeMap<int32_t>(), raws[i].data(), raws[i].size());
        else memcpy(v->writeMap<float>(), raws[i].data(), raws[i].size());
        inputs.push_back(v);
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<VARP> out = mod->onForward(inputs);
    auto t1 = std::chrono::steady_clock::now();
    printf("FORWARD_MS=%.0f n_out=%zu\n", std::chrono::duration<double,std::milli>(t1-t0).count(), out.size());
    for (size_t i = 0; i < out.size(); i++) {
        auto inf = out[i]->getInfo();
        if (!inf) { printf("OUT[%zu] NOINFO\n", i); continue; }
        size_t elems = 1; for (int d : inf->dim) elems *= d > 0 ? d : 1;
        const float* d = out[i]->readMap<float>();
        if (!d) { printf("OUT[%zu] NOMAP\n", i); continue; }
        double mx = 0;
        for (size_t k = 0; k < elems; k++) { double a = d[k] < 0 ? -d[k] : d[k]; if (a > mx && a < 1e37) mx = a; }
        printf("OUT[%zu] %-11s elems=%zu maxabs=%.6f\n", i, i<outs.size()?outs[i].c_str():"?", elems, mx);
        std::string f = std::string(outDir) + "/o" + std::to_string(i) + "_" + (i<outs.size()?outs[i]:"x") + ".raw";
        writeRaw(f.c_str(), d, elems);
    }
    printf("DONE\n");
    return 0;
}

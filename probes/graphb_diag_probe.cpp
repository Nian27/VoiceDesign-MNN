// graphb_diag_probe.cpp - dump ALL graph outputs (first-divergence probe)
// usage: graphb_diag_probe MODEL CP EMB KVK KVV POS OUT_DIR [cpu|htp]
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
    std::vector<std::string> outs = {"norm_out","q_raw","k_raw","v_raw","q_norm","k_norm","q_rope","k_rope",
                                     "attn_out","o_proj_out","res1","post_norm","mlp_out","layer0_out","kv_k_out","kv_v_out"};
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
        double mx = 0; bool fin = true;
        for (size_t k = 0; k < elems; k++) { double a = d[k] < 0 ? -d[k] : d[k]; if (a > mx) mx = a; if (!(d[k]==d[k])) fin=false; }
        printf("OUT[%zu] %-11s elems=%zu maxabs=%.6f finite=%d\n", i, i < outs.size() ? outs[i].c_str() : "?", elems, mx, fin?1:0);
        std::string f = std::string(outDir) + "/o" + std::to_string(i) + "_" + (i < outs.size() ? outs[i] : "x") + ".raw";
        writeRaw(f.c_str(), d, elems);
    }
    printf("DONE\n");
    return 0;
}

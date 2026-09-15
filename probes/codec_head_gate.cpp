// codec_head_gate.cpp - F1 functional gate: CPU-hidden vs HTP-hidden decision surface
// usage: codec_head_gate CODECHEAD_HIDDEN_A CODECHEAD_HIDDEN_B OUT_TXT [cpu|htp]
// A = device-CPU GraphB hidden, B = device-HTP GraphB hidden
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
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
static double cosv(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return d / (sqrt(na)*sqrt(nb) + 1e-30);
}
static unsigned int g_rng;
static inline unsigned int xr() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static inline double rndu() { return (double)((xr() >> 8) & 0xFFFFFF) / 16777216.0; }
// official policy: temp=0.9, top_k=50, top_p=1.0, suppress [2048,3072) except EOS 2150
static int sampleCode0(const double* lg, int vocab, double temp, int topk, double u, std::vector<double>* probsOut) {
    std::vector<std::pair<double,int>> v;
    for (int i = 0; i < vocab; i++) {
        if (i >= 2048 && i != 2150) continue;
        v.push_back({ (double)lg[i] / temp, i });
    }
    std::sort(v.begin(), v.end(), [](const std::pair<double,int>&a, const std::pair<double,int>&b){ return a.first > b.first; });
    if ((int)v.size() > topk) v.resize(topk);
    double mx = v[0].first, s = 0;
    std::vector<double> pr(v.size());
    for (size_t i = 0; i < v.size(); i++) { pr[i] = exp(v[i].first - mx); s += pr[i]; }
    for (size_t i = 0; i < v.size(); i++) pr[i] /= s;
    if (probsOut) *probsOut = pr;
    double acc = 0;
    for (size_t i = 0; i < v.size(); i++) { acc += pr[i]; if (u <= acc) return v[i].second; }
    return v.back().second;
}
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL HID_A HID_B OUT_TXT [cpu|htp]\n", argv[0]); return 2; }
    const char* model = argv[1];
    std::vector<char> hA = readFile(argv[2]), hB = readFile(argv[3]);
    const char* outTxt = argv[4];
    const char* backend = argc > 5 ? argv[5] : "cpu";
    BackendConfig bc; bc.precision = BackendConfig::Precision_High; bc.power = BackendConfig::Power_High; bc.memory = BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if (strcmp(backend, "htp") == 0) { sched.type = MNN_FORWARD_HEXAGON; } else { sched.type = MNN_FORWARD_CPU; }
    sched.numThread = 4; sched.backendConfig = &bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    Module::Config mcfg; mcfg.shapeMutable = true;
    std::shared_ptr<Module> mod(Module::load({"hidden_states"}, {"logits"}, model, rtm, &mcfg), Module::destroy);
    if (!mod) { fprintf(stderr, "load failed\n"); return 3; }
    FILE* fo = fopen(outTxt, "w");
    auto runOne = [&](std::vector<char>& raw, std::vector<double>& lg) {
        VARP in = _Input({1,1,2048}, NCHW, halide_type_of<float>());
        memcpy(in->writeMap<float>(), raw.data(), raw.size());
        std::vector<VARP> out = mod->onForward({in});
        const float* d = out[0]->readMap<float>();
        auto inf = out[0]->getInfo();
        int vocab = 1; for (int x : inf->dim) vocab *= x > 0 ? x : 1;
        lg.assign(d, d + vocab);
    };
    std::vector<double> lA, lB;
    runOne(hA, lA); runOne(hB, lB);
    int vocab = (int)lA.size();
    int top1A = 0, top1B = 0;
    for (int i = 0; i < vocab; i++) { if (lA[i] > lA[top1A]) top1A = i; if (lB[i] > lB[top1B]) top1B = i; }
    auto topkIdx = [&](std::vector<double>& l, int k) {
        std::vector<int> idx(vocab); for (int i = 0; i < vocab; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), [&](int a, int b){ return l[a] > l[b]; });
        idx.resize(k); return idx;
    };
    std::vector<int> t50A = topkIdx(lA, 50), t50B = topkIdx(lB, 50);
    int overlap = 0; for (int a : t50A) for (int b : t50B) if (a == b) { overlap++; break; }
    auto rankOf = [&](std::vector<double>& l, int id) { int r = 1; for (int i = 0; i < vocab; i++) if (l[i] > l[id]) r++; return r; };
    double dmax = 0, na = 0, nb = 0, dn = 0;
    for (int i = 0; i < vocab; i++) { double dd = fabs(lA[i]-lB[i]); if (dd > dmax) dmax = dd; dn += dd*dd; na += lA[i]*lA[i]; nb += lB[i]*lB[i]; }
    double relL2 = sqrt(dn) / (sqrt(na) + 1e-30);
    // shared-RNG sampling over several fixed uniforms
    printf("VOCAB=%d TOP1_A=%d TOP1_B=%d TOP1_MATCH=%d\n", vocab, top1A, top1B, top1A==top1B);
    printf("LOGITS_COS=%.9f MAXABS_DIFF=%.6f REL_L2=%.6f\n", cosv(lA, lB), dmax, relL2);
    printf("TOP50_OVERLAP=%d/50\n", overlap);
    printf("EOS_2150 logitA=%.5f rankA=%d | logitB=%.5f rankB=%d\n", lA[2150], rankOf(lA,2150), lB[2150], rankOf(lB,2150));
    fprintf(fo, "VOCAB=%d TOP1_A=%d TOP1_B=%d TOP1_MATCH=%d\n", vocab, top1A, top1B, top1A==top1B);
    fprintf(fo, "LOGITS_COS=%.9f MAXABS_DIFF=%.6f REL_L2=%.6f\n", cosv(lA, lB), dmax, relL2);
    fprintf(fo, "TOP50_OVERLAP=%d/50\n", overlap);
    fprintf(fo, "EOS_2150 logitA=%.5f rankA=%d logitB=%.5f rankB=%d\n", lA[2150], rankOf(lA,2150), lB[2150], rankOf(lB,2150));
    // 32 fixed uniforms
    int same = 0, diff = 0;
    std::vector<double> pA, pB;
    printf("--- shared-RNG sampled code0 (temp=0.9, top_k=50, u=16-bit grid) ---\n");
    for (int t = 0; t < 32; t++) {
        double u = (double)(t * 65537 % 1048576) / 1048576.0;
        int cA = sampleCode0(lA.data(), vocab, 0.9, 50, u, &pA);
        int cB = sampleCode0(lB.data(), vocab, 0.9, 50, u, &pB);
        if (cA == cB) same++; else { diff++; printf("  u=%.6f  CPU=%d HTP=%d  <== DIFF\n", u, cA, cB); }
        fprintf(fo, "u=%.6f cpu=%d htp=%d %s\n", u, cA, cB, cA==cB?"same":"DIFF");
    }
    printf("SHARED_RNG_SAME=%d/32 DIFF=%d\n", same, diff);
    {
        std::vector<double> p1, p2, p3, p4;
        sampleCode0(lA.data(), vocab, 0.9, 50, 0.5, &p1);
        sampleCode0(lB.data(), vocab, 0.9, 50, 0.5, &p2);
        double c = 0, s1 = 0, s2 = 0, md = 0;
        for (size_t i = 0; i < p1.size(); i++) { c += p1[i]*p2[i]; s1 += p1[i]*p1[i]; s2 += p2[i]*p2[i]; double dd = fabs(p1[i]-p2[i]); if (dd>md) md=dd; }
        printf("TOPK50_PROBS cos=%.9f maxdiff=%.8f\n", c/(sqrt(s1)*sqrt(s2)+1e-30), md);
        fprintf(fo, "TOPK50_PROBS cos=%.9f maxdiff=%.8f\n", c/(sqrt(s1)*sqrt(s2)+1e-30), md);
        printf("P_TOP1_A=%.6f P_TOP1_B=%.6f\n", p1[0], p2[0]);
    }
    fclose(fo);
    printf("F1_DONE\n");
    return 0;
}

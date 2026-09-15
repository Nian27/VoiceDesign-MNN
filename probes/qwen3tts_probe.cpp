// qwen3tts_probe.cpp - GraphB (talker 28L decode-step) 5-input minimal runner
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
using namespace MNN;
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
int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s MODEL EMB POS KVK KVV CP OUT REPEATS\n", argv[0]);
        return 2;
    }
    std::vector<char> embB = readFile(argv[2]);
    std::vector<char> posB = readFile(argv[3]);
    std::vector<char> kvkB = readFile(argv[4]);
    std::vector<char> kvvB = readFile(argv[5]);
    std::vector<char> cpB  = readFile(argv[6]);
    int repeats = atoi(argv[8]);
    auto interp = std::shared_ptr<Interpreter>(Interpreter::createFromFile(argv[1]));
    if (!interp) { fprintf(stderr, "load failed\n"); return 3; }
    std::string wpath(argv[1]);
    wpath += ".weight";
    interp->setExternalFile(wpath.c_str());
    ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    auto sess = interp->createSession(cfg);
    auto inEmb = interp->getSessionInput(sess, "inputs_embeds");
    auto inPos = interp->getSessionInput(sess, "position_ids");
    auto inK   = interp->getSessionInput(sess, "kv_k");
    auto inV   = interp->getSessionInput(sess, "kv_v");
    auto inCp  = interp->getSessionInput(sess, "cache_position");
    interp->resizeTensor(inEmb, {1,1,2048});
    interp->resizeTensor(inPos, {3,1,1});
    interp->resizeTensor(inK, {28,1,8,35,128});
    interp->resizeTensor(inV, {28,1,8,35,128});
    interp->resizeTensor(inCp, {1});
    interp->resizeSession(sess);
    auto feedT = [&](Tensor* t, std::vector<char>& b) {
        Tensor host(t, Tensor::CAFFE);
        memcpy(host.host<void>(), b.data(), b.size());
        t->copyFromHostTensor(&host);
    };
    feedT(inEmb, embB); feedT(inPos, posB); feedT(inK, kvkB); feedT(inV, kvvB); feedT(inCp, cpB);
    auto out = interp->getSessionOutput(sess, "hidden_states");
    for (int r = -1; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        interp->runSession(sess);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r < 0) printf("warm_ms=%.1f\n", ms); else printf("run_ms=%.1f\n", ms);
    }
    Tensor hostOut(out, Tensor::CAFFE);
    out->copyToHostTensor(&hostOut);
    auto dims = hostOut.shape();
    size_t elems = 1;
    for (int d : dims) elems *= d > 0 ? d : 1;
    const float* data = hostOut.host<float>();
    double mx = 0; size_t finite = 0; double sum = 0;
    for (size_t i = 0; i < elems; i++) { float v = data[i]; double av = v < 0 ? -v : v; if (av > mx) mx = av; sum += v; finite++; }
    FILE* f = fopen(argv[7], "wb");
    fwrite(data, sizeof(float), elems, f);
    fclose(f);
    printf("elems=%zu finite=%zu maxabs=%.6f mean=%.6f\n", elems, finite, mx, sum / (double)elems);
    return 0;
}
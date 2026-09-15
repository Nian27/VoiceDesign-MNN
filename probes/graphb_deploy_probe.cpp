// graphb_deploy_probe v2 - type-safe feeding (fixes int64 cache_position UB)
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
        fprintf(stderr, "usage: %s MODEL EMB POS KVK KVV CP OUT_DIR REPEATS [cpu|htp] [NUMTHREADS]\n", argv[0]);
        return 2;
    }
    const char* model = argv[1];
    std::vector<char> embB = readFile(argv[2]);
    std::vector<char> posB = readFile(argv[3]);
    std::vector<char> kvkB = readFile(argv[4]);
    std::vector<char> kvvB = readFile(argv[5]);
    std::vector<char> cpB  = readFile(argv[6]);
    const char* outDir = argv[7];
    int repeats = atoi(argv[8]);
    const char* backendStr = argc > 9 ? argv[9] : "cpu";
    int numThread = argc > 10 ? atoi(argv[10]) : 4;

    printf("INPUT_CRC emb=%08x pos=%08x kvk=%08x kvv=%08x cp=%08x\n",
        crc32buf((const unsigned char*)embB.data(), embB.size()),
        crc32buf((const unsigned char*)posB.data(), posB.size()),
        crc32buf((const unsigned char*)kvkB.data(), kvkB.size()),
        crc32buf((const unsigned char*)kvvB.data(), kvvB.size()),
        crc32buf((const unsigned char*)cpB.data(), cpB.size()));

    auto interp = std::shared_ptr<Interpreter>(Interpreter::createFromFile(model));
    if (!interp) { fprintf(stderr, "load failed\n"); return 3; }
    std::string wpath(model); wpath += ".weight";
    interp->setExternalFile(wpath.c_str());

    ScheduleConfig cfg;
    if (strcmp(backendStr, "htp") == 0) { cfg.type = MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon\n"); }
    else { cfg.type = MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    cfg.numThread = numThread;
    auto sess = interp->createSession(cfg);
    if (!sess) { fprintf(stderr, "createSession failed\n"); return 3; }

    const char* names[5] = {"inputs_embeds","position_ids","kv_k","kv_v","cache_position"};
    std::vector<char>* raws[5] = {&embB, &posB, &kvkB, &kvvB, &cpB};
    Tensor* tens[5];
    for (int i = 0; i < 5; i++) {
        tens[i] = interp->getSessionInput(sess, names[i]);
        if (!tens[i]) { fprintf(stderr, "input %s missing\n", names[i]); return 3; }
        auto t = tens[i]->getType();
        printf("IN %-14s dims=%d type_code=%d bits=%d bytes=%d raw=%zu\n",
               names[i], tens[i]->dimensions(), t.code, t.bits, tens[i]->size(), raws[i]->size());
    }
    for (int i = 0; i < 5; i++) {
        Tensor* t = tens[i];
        auto ty = t->getType();
        Tensor host(t, Tensor::CAFFE);
        size_t hostBytes = (size_t)t->size();
        size_t rawBytes = raws[i]->size();
        if (ty.bits == 64 && ty.code == halide_type_int && rawBytes * 2 == hostBytes) {
            int32_t v = 0; memcpy(&v, raws[i]->data(), 4);
            int64_t v64 = (int64_t)v;
            memset(host.host<void>(), 0, hostBytes);
            memcpy(host.host<void>(), &v64, sizeof(v64));
            printf("  widened %s int32->int64 value=%lld\n", names[i], (long long)v64);
        } else {
            size_t n = rawBytes < hostBytes ? rawBytes : hostBytes;
            memcpy(host.host<void>(), raws[i]->data(), n);
        }
        t->copyFromHostTensor(&host);
    }

    auto outH = interp->getSessionOutput(sess, "hidden_states");
    auto outK = interp->getSessionOutput(sess, "kv_k_out");
    auto outV = interp->getSessionOutput(sess, "kv_v_out");

    double sumMs = 0; double minMs = 1e9;
    for (int r = -1; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        interp->runSession(sess);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r < 0) printf("warm_ms=%.1f\n", ms);
        else { printf("run_ms=%.1f\n", ms); sumMs += ms; if (ms < minMs) minMs = ms; }
    }
    if (repeats > 0) printf("LAT avg_ms=%.1f min_ms=%.1f\n", sumMs / repeats, minMs);

    auto copyOut = [](Tensor* t) -> std::vector<float> {
        Tensor host(t, Tensor::CAFFE);
        t->copyToHostTensor(&host);
        auto dims = host.shape();
        size_t elems = 1;
        for (int d : dims) elems *= (size_t)(d > 0 ? d : 1);
        const float* data = host.host<float>();
        return std::vector<float>(data, data + elems);
    };
    std::vector<float> h = copyOut(outH);
    std::vector<float> k = copyOut(outK);
    std::vector<float> v = copyOut(outV);
    double mh = 0; for (float x : h) { double a = x < 0 ? -x : x; if (a > mh) mh = a; }
    printf("OUT hidden elems=%zu maxabs=%.6f crc32=%08x\n", h.size(), mh, crc32buf((const unsigned char*)h.data(), h.size()*sizeof(float)));
    writeRaw((std::string(outDir) + "/hidden_out.raw").c_str(), h.data(), h.size());
    writeRaw((std::string(outDir) + "/kv_k_out.raw").c_str(), k.data(), k.size());
    writeRaw((std::string(outDir) + "/kv_v_out.raw").c_str(), v.data(), v.size());
    printf("OUT kv_k_out elems=%zu kv_v_out elems=%zu\n", k.size(), v.size());
    printf("DONE\n");
    return 0;
}

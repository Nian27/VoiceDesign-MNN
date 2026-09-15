// vd_m5b_chain.cpp - SM8850 CPU full VoiceDesign self-driven chain
// prefill -> loop[ codec_head -> codec_emb -> codepred_ar -> frame_emb -> GraphB ] -> decoder -> wav
// usage: vd_m5b_chain GRAPHB CODECHEAD CODECEMB CODEPRED_AR FRAMEEMB DECODER
//        PREFILL_HIDDEN PREFILL_KVK PREFILL_KVV OUTDIR MAXFRAMES
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
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
static void writeFile(const char* p, const void* d, size_t bytes) {
    FILE* f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", p); return; }
    fwrite(d, 1, bytes, f);
    fclose(f);
}
static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
struct Mod { std::shared_ptr<Module> m; };
static Mod loadMod(const char* path, std::vector<std::string> ins, std::vector<std::string> outs,
                   std::shared_ptr<Executor::RuntimeManager> rtm) {
    Module::Config cfg; cfg.shapeMutable = true;
    Mod r;
    r.m = std::shared_ptr<Module>(Module::load(ins, outs, path, rtm, &cfg), Module::destroy);
    if (!r.m) { fprintf(stderr, "load failed: %s\n", path); exit(3); }
    return r;
}
static VARP inF(const std::vector<int>& dim, const float* d, size_t n) {
    VARP v = _Input(dim, NCHW, halide_type_of<float>());
    memcpy(v->writeMap<float>(), d, n * sizeof(float));
    return v;
}
static VARP inI(const std::vector<int>& dim, const int32_t* d, size_t n) {
    VARP v = _Input(dim, NCHW, halide_type_of<int32_t>());
    memcpy(v->writeMap<int32_t>(), d, n * sizeof(int32_t));
    return v;
}
// ---- sampling (official policy: temp/top-k + repetition penalty) ----
static unsigned int g_rng = 20260826u;
static inline unsigned int xr() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static inline float rndu() { return (float)((xr() >> 8) & 0xFFFFFF) * (1.0f / 16777216.0f); }
static int sampleFrom(const float* logits, int n, float temp, int topk, bool suppress2048) {
    std::vector<std::pair<float,int>> v; v.reserve(n);
    for (int i = 0; i < n; i++) {
        if (suppress2048 && i >= 2048 && i != 2150) continue;
        v.push_back({ logits[i] / temp, i });
    }
    if (topk > 0 && (int)v.size() > topk) {
        std::partial_sort(v.begin(), v.begin()+topk, v.end(),
                          [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });
        v.resize(topk);
    }
    float mx = v[0].first; for (auto& p : v) if (p.first > mx) mx = p.first;
    std::vector<double> pr(v.size()); double s = 0;
    for (size_t i = 0; i < v.size(); i++) { pr[i] = exp((double)(v[i].first - mx)); s += pr[i]; }
    double r = (double)rndu() * s, acc = 0;
    for (size_t i = 0; i < v.size(); i++) { acc += pr[i]; if (r <= acc) return v[i].second; }
    return v.back().second;
}
static void writeWav16(const char* path, const std::vector<float>& x, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int n = (int)x.size();
    int dataBytes = n * 2;
    unsigned char hdr[44] = {0};
    auto put32 = [&](int off, unsigned v){ hdr[off]=v&0xff; hdr[off+1]=(v>>8)&0xff; hdr[off+2]=(v>>16)&0xff; hdr[off+3]=(v>>24)&0xff; };
    auto put16 = [&](int off, unsigned v){ hdr[off]=v&0xff; hdr[off+1]=(v>>8)&0xff; };
    memcpy(hdr, "RIFF", 4); put32(4, 36 + dataBytes); memcpy(hdr+8, "WAVE", 4);
    memcpy(hdr+12, "fmt ", 4); put32(16, 16); put16(20, 1); put16(22, 1);
    put32(24, sr); put32(28, sr*2); put16(32, 2); put16(34, 16);
    memcpy(hdr+36, "data", 4); put32(40, dataBytes);
    fwrite(hdr, 1, 44, f);
    std::vector<short> pcm(n);
    for (int i = 0; i < n; i++) { float v = x[i]; if (v>1) v=1; if (v<-1) v=-1; pcm[i] = (short)lrintf(v*32767.0f); }
    fwrite(pcm.data(), 2, n, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 12) {
        fprintf(stderr, "usage: %s GRAPHB CODECHEAD CODECEMB CODEPRED_AR FRAMEEMB DECODER PREFILL_HIDDEN PREFILL_KVK PREFILL_KVV OUTDIR MAXFRAMES\n", argv[0]);
        return 2;
    }
    const char* fGraphB=argv[1]; const char* fCodecHead=argv[2]; const char* fCodecEmb=argv[3];
    const char* fCodepred=argv[4]; const char* fFrameEmb=argv[5]; const char* fDecoder=argv[6];
    const char* fPreH=argv[7]; const char* fPreK=argv[8]; const char* fPreV=argv[9];
    const char* outDir=argv[10]; int maxFrames=atoi(argv[11]);
    setvbuf(stdout, NULL, _IOLBF, 0);
    const int KV=768, NFRAME=300, SPF=1920, SR=24000;

    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High; bc.memory=BackendConfig::Memory_Normal;
    ScheduleConfig sched; sched.type=MNN_FORWARD_CPU; sched.numThread=4; sched.backendConfig=&bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched), Executor::RuntimeManager::destroy);
    if (!rtm) { fprintf(stderr,"rtm failed\n"); return 3; }
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER, 4);

    double t0 = nowMs();
    Mod mGB = loadMod(fGraphB, {"cache_position","inputs_embeds","kv_k","kv_v","position_ids"}, {"hidden_states","kv_k_out","kv_v_out"}, rtm);
    Mod mCH = loadMod(fCodecHead, {"hidden_states"}, {"logits"}, rtm);
    Mod mCE = loadMod(fCodecEmb, {"codes"}, {"emb"}, rtm);
    Mod mCP = loadMod(fCodepred, {"past_hidden","code0_emb"}, {"logits15"}, rtm);
    Mod mFE = loadMod(fFrameEmb, {"codes"}, {"emb"}, rtm);
    Mod mDE = loadMod(fDecoder, {"codes"}, {"waveform"}, rtm);
    printf("MODELS_LOADED ms=%.0f\n", nowMs()-t0);

    std::vector<char> hB=readFile(fPreH), kB=readFile(fPreK), vB=readFile(fPreV);
    int LP = (int)(kB.size() / (size_t)(28*8*128*sizeof(float)));
    printf("PREFILL_LP=%d\n", LP);
    const size_t KVONE = (size_t)28*8*KV*128;
    std::vector<float> kvk(KVONE, 0.f), kvv(KVONE, 0.f);
    for (int L=0; L<28; L++) for (int H=0; H<8; H++) {
        memcpy(&kvk[((size_t)L*8+H)*KV*128], (const float*)kB.data() + ((size_t)L*8+H)*LP*128, (size_t)LP*128*sizeof(float));
        memcpy(&kvv[((size_t)L*8+H)*KV*128], (const float*)vB.data() + ((size_t)L*8+H)*LP*128, (size_t)LP*128*sizeof(float));
    }
    std::vector<float> hidden((const float*)hB.data(), (const float*)hB.data()+2048);

    // persistent input VARPs (reused every frame -> MNN keeps its plan; creating new ones forces a full re-plan)
    VARP vCH  = _Input({1,1,2048}, NCHW, halide_type_of<float>());
    VARP vCE  = _Input({1,1}, NCHW, halide_type_of<int32_t>());
    VARP vCPa = _Input({1,1,2048}, NCHW, halide_type_of<float>());
    VARP vCPb = _Input({1,1,2048}, NCHW, halide_type_of<float>());
    VARP vFE  = _Input({1,16}, NCHW, halide_type_of<int32_t>());
    VARP vCp  = _Input({1}, NCHW, halide_type_of<int32_t>());
    VARP vEmb = _Input({1,1,2048}, NCHW, halide_type_of<float>());
    VARP vK   = _Input({28,1,8,KV,128}, NCHW, halide_type_of<float>());
    VARP vV   = _Input({28,1,8,KV,128}, NCHW, halide_type_of<float>());
    VARP vPos = _Input({3,1,1}, NCHW, halide_type_of<float>());
    memcpy(vK->writeMap<float>(), kvk.data(), KVONE*sizeof(float));
    memcpy(vV->writeMap<float>(), kvv.data(), KVONE*sizeof(float));
    std::vector<int32_t> frames;
    std::vector<int> seenCode0;
    int T=0, stopReason=-1;
    double tCH=0,tCP=0,tFE=0,tGB=0;
    for (int n=0; n<maxFrames; n++) {
        int cp = LP + n;
        double a=nowMs();
        memcpy(vCH->writeMap<float>(), hidden.data(), 2048*sizeof(float));
        std::vector<VARP> o1 = mCH.m->onForward({ vCH });
        const float* lg = o1[0]->readMap<float>();
        std::vector<float> lgP(lg, lg+3072);
        for (int c : seenCode0) { float x = lgP[c]; lgP[c] = (x > 0) ? x/1.05f : x*1.05f; }   // repetition_penalty 1.05
        int code0 = sampleFrom(lgP.data(), 3072, 0.9f, 50, true);
        tCH += nowMs()-a;
        if (code0==2150) { stopReason=0; printf("EOS_AT_FRAME %d cp=%d\n", n, cp); fflush(stdout); break; }
        seenCode0.push_back(code0);
        a=nowMs();
        { int32_t c32 = code0; memcpy(vCE->writeMap<int32_t>(), &c32, sizeof(int32_t)); }
        std::vector<VARP> o2 = mCE.m->onForward({ vCE });
        const float* c0e = o2[0]->readMap<float>();
        std::vector<float> c0e_buf(c0e, c0e+2048);
        memcpy(vCPa->writeMap<float>(), hidden.data(), 2048*sizeof(float));
        memcpy(vCPb->writeMap<float>(), c0e_buf.data(), 2048*sizeof(float));
        std::vector<VARP> o3 = mCP.m->onForward({ vCPa, vCPb });
        const float* lg15 = o3[0]->readMap<float>();                            // (1,15,2048)
        std::vector<int32_t> frame(16);
        frame[0]=code0;
        for (int g=0; g<15; g++){ const float* row=lg15+(size_t)g*2048; frame[g+1]=sampleFrom(row, 2048, 0.9f, 50, false); }
        tCP += nowMs()-a;
        a=nowMs();
        memcpy(vFE->writeMap<int32_t>(), frame.data(), 16*sizeof(int32_t));
        std::vector<VARP> o4 = mFE.m->onForward({ vFE });
        const float* emb = o4[0]->readMap<float>();
        std::vector<float> emb_buf(emb, emb+2048);
        if (getenv("VD_DUMP_EMB")) { char ep[256]; snprintf(ep, sizeof(ep), "%s/emb_%03d.raw", outDir, n); writeFile(ep, emb_buf.data(), 2048*sizeof(float)); }
        tFE += nowMs()-a;
        a=nowMs();
        float posv[3] = {(float)cp,(float)cp,(float)cp};
        { int32_t cp32 = cp; memcpy(vCp->writeMap<int32_t>(), &cp32, sizeof(int32_t)); }
        memcpy(vEmb->writeMap<float>(), emb_buf.data(), 2048*sizeof(float));
        memcpy(vPos->writeMap<float>(), posv, 3*sizeof(float));
        std::vector<VARP> o5 = mGB.m->onForward({ vCp, vEmb, vK, vV, vPos });
        const float* hn = o5[0]->readMap<float>();
        memcpy(hidden.data(), hn, 2048*sizeof(float));
        {
            // only slot cp changed -> copy that slot (28*8*128 floats) instead of the whole 88MB tensor
            const float* kout = o5[1]->readMap<float>();
            const float* vout = o5[2]->readMap<float>();
            float* kbuf = vK->writeMap<float>();
            float* vbuf = vV->writeMap<float>();
            for (int L = 0; L < 28; L++)
                for (int H = 0; H < 8; H++) {
                    size_t off = ((size_t)(L*8+H)*KV + cp) * 128;
                    memcpy(kbuf + off, kout + off, 128*sizeof(float));
                    memcpy(vbuf + off, vout + off, 128*sizeof(float));
                }
        }
        tGB += nowMs()-a;
        frames.insert(frames.end(), frame.begin(), frame.end());
        T++;
        if (n % 5 == 0 || n < 3) { printf("frame %3d cp=%3d code0=%d codes15=[%d,%d,%d...]\n", n, cp, code0, frame[1], frame[2], frame[3]); fflush(stdout); }
    }
    printf("GEN_DONE T=%d stop=%s\n", T, stopReason==0?"EOS":"MAXFRAMES");
    printf("LAT codec_head=%.0fms codepred_ar=%.0fms frame_emb=%.0fms graphb=%.0fms total=%.0fms\n",
           tCH, tCP, tFE, tGB, tCH+tCP+tFE+tGB);
    if (T == 0) { printf("NO_FRAMES\n"); return 0; }

    // pack codes (1,16,300) int32: index g*300 + t ; pad by repeating last frame
    std::vector<int32_t> codes((size_t)16*NFRAME, 0);
    for (int t=0; t<NFRAME; t++) {
        int src = t < T ? t : T-1;
        for (int g=0; g<16; g++) codes[(size_t)g*NFRAME + t] = frames[(size_t)src*16 + g];
    }
    double a=nowMs();
    std::vector<VARP> od = mDE.m->onForward({ inI({1,16,NFRAME}, codes.data(), codes.size()) });
    double tDE = nowMs()-a;
    const float* wv = od[0]->readMap<float>();
    size_t wavAll = 576000;
    size_t wavLen = (size_t)T * SPF; if (wavLen > wavAll) wavLen = wavAll;
    std::vector<float> wave(wv, wv + wavLen);
    printf("DECODER ms=%.0f wav_samples=%zu secs=%.2f\n", tDE, wavLen, (double)wavLen/SR);
    double mn=1e9, mx=-1e9, ss=0; size_t nan=0;
    for (float v : wave) { if (!std::isfinite(v)) { nan++; continue; } if (v<mn)mn=v; if (v>mx)mx=v; ss+=v*v; }
    printf("WAV finite=%zu/%zu min=%.4f max=%.4f rms=%.5f\n", wave.size()-nan, wave.size(), mn, mx, sqrt(ss/wave.size()));
    std::string s1 = std::string(outDir)+"/codes_i32.raw"; writeFile(s1.c_str(), frames.data(), frames.size()*4);
    std::string s2 = std::string(outDir)+"/waveform_f32.raw"; writeFile(s2.c_str(), wave.data(), wave.size()*4);
    std::string s3 = std::string(outDir)+"/reference.wav"; writeWav16(s3.c_str(), wave, SR);
    printf("SAVED %s\n", s3.c_str());
    {
        FILE* sf=fopen("/proc/self/status","r"); char ln[256];
        if (sf){ while(fgets(ln,sizeof(ln),sf)){ if(!strncmp(ln,"VmHWM",5)||!strncmp(ln,"VmRSS",5)) printf("MEM %s", ln); } fclose(sf); }
    }
    printf("CHAIN_DONE\n");
    return 0;
}

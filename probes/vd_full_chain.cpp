// vd_full_chain.cpp - standalone VoiceDesign full chain (subprocess form for the app)
// usage: vd_full_chain MODEL_DIR OUT_WAV [cpu|htp|opencl] [MAX_FRAMES]
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
using namespace MNN;
using namespace MNN::Express;
#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while (0)

static const int CAP=16, HD=128, NL=28, NKV=8, TKV=768, CPNL=5, CPKV=8;
static unsigned long long g_seed = 123456789ull;
static double nextU(){ g_seed = g_seed*6364136223846793005ull + 1442695040888963407ull;
                       return (double)((g_seed>>11)&((1ull<<53)-1))/(double)(1ull<<53); }
static int sampleTopK(const float* lg,int vocab,double temp,int topk,bool suppress,double u){
    std::vector<std::pair<double,int>> v; v.reserve(vocab);
    for(int i=0;i<vocab;i++){ if(suppress && i>=2048 && i!=2150) continue; v.push_back({(double)lg[i]/temp,i}); }
    std::sort(v.begin(),v.end(),[](const std::pair<double,int>&a,const std::pair<double,int>&b){return a.first>b.first;});
    if((int)v.size()>topk) v.resize(topk);
    double mx=v[0].first,s=0; std::vector<double> pr(v.size());
    for(size_t i=0;i<v.size();i++){ pr[i]=exp(v[i].first-mx); s+=pr[i]; }
    for(size_t i=0;i<v.size();i++) pr[i]/=s;
    double acc=0;
    for(size_t i=0;i<v.size();i++){ acc+=pr[i]; if(u<=acc) return v[i].second; }
    return v.back().second;
}
static bool readF(const std::string& p, void* d, size_t n){
    FILE* f=fopen(p.c_str(),"rb"); if(!f) return false;
    size_t r=fread(d,1,n,f); fclose(f); return r==n;
}
static std::vector<float> readVec(const std::string& p, size_t n){
    std::vector<float> v(n); if(!readF(p,v.data(),n*4)) v.clear(); return v;
}
static void* mapFile(const std::string& p, size_t bytes){
    int fd=open(p.c_str(),O_RDONLY); if(fd<0) return nullptr;
    void* q=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    return (q==MAP_FAILED)?nullptr:q;
}
static std::shared_ptr<Module> loadM(const std::string& path, std::vector<std::string> i, std::vector<std::string> o,
                                     std::shared_ptr<Executor::RuntimeManager> rtm){
    Module::Config c; c.shapeMutable=true;
    std::shared_ptr<Module> m(Module::load(i,o,path.c_str(),rtm,&c),Module::destroy);
    if(!m) LOGI("load fail: %s", path.c_str());
    return m;
}
static void writeWav16(const std::string& path, const std::vector<float>& x, int sr){
    FILE* f=fopen(path.c_str(),"wb"); if(!f){ LOGI("cannot write wav"); return; }
    int n=(int)x.size(), db=n*2; unsigned char h[44]={0};
    auto p32=[&](int o,unsigned v){h[o]=v&0xff;h[o+1]=(v>>8)&0xff;h[o+2]=(v>>16)&0xff;h[o+3]=(v>>24)&0xff;};
    auto p16=[&](int o,unsigned v){h[o]=v&0xff;h[o+1]=(v>>8)&0xff;};
    memcpy(h,"RIFF",4); p32(4,36+db); memcpy(h+8,"WAVE",4);
    memcpy(h+12,"fmt ",4); p32(16,16); p16(20,1); p16(22,1);
    p32(24,sr); p32(28,sr*2); p16(32,2); p16(34,16);
    memcpy(h+36,"data",4); p32(40,db);
    fwrite(h,1,44,f);
    std::vector<short> pcm(n);
    for(int i=0;i<n;i++){ float v=x[i]; if(v>1)v=1; if(v<-1)v=-1; pcm[i]=(short)lrintf(v*32767.0f); }
    fwrite(pcm.data(),2,n,f); fclose(f);
}
int main(int argc,char** argv){
    if(argc<3){ fprintf(stderr,"usage: %s MODEL_DIR OUT_WAV [cpu|htp|opencl] [MAX_FRAMES]\n",argv[0]); return 2; }
    std::string d=argv[1]; if(d.back()!='/') d+="/";
    std::string out=argv[2];
    std::string backend = argc>3?argv[3]:"cpu";
    int maxFrames = argc>4?atoi(argv[4]):300;
    LOGI("MODEL_DIR=%s", d.c_str());
    LOGI("BACKEND=%s", backend.c_str());
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High; bc.memory=BackendConfig::Memory_Normal;
    ScheduleConfig sc; sc.numThread=4; sc.backendConfig=&bc;
    if(backend=="htp") sc.type=MNN_FORWARD_HEXAGON;
    else if(backend=="opencl") sc.type=MNN_FORWARD_OPENCL;
    else sc.type=MNN_FORWARD_CPU;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sc),Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER,4);
    auto t0=std::chrono::steady_clock::now();
    auto mGB =loadM(d+"graphb28_v6_fp16.mnn",{"inputs_embeds","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"},{"hidden_states","cur_k","cur_v"},rtm);
    auto mCH =loadM(d+"codec_head_fp16.mnn",{"hidden_states"},{"logits"},rtm);
    auto mCE =loadM(d+"codec_emb_fp16.mnn",{"codes"},{"emb"},rtm);
    auto mFE =loadM(d+"frame_emb_fp16.mnn",{"codes"},{"emb"},rtm);
    auto mCPF=loadM(d+"codepred_prefill_fp16.mnn",{"past_hidden","code0_emb","rope_cos","rope_sin","slot_mask0","slot_mask1","attn_mask"},{"hidden_last","kv_k_out","kv_v_out"},rtm);
    auto mCPS=loadM(d+"codepred_step_fp16.mnn",{"token_emb","kv_k","kv_v","rope_cos","rope_sin","slot_mask","attn_mask"},{"hidden_last","kv_k_out","kv_v_out"},rtm);
    auto mDEC=loadM(d+"tokenizer_decoder_static_t300.mnn",{"codes"},{"waveform"},rtm);
    if(!mGB||!mCH||!mCE||!mFE||!mCPF||!mCPS||!mDEC){ LOGI("MODEL_LOAD_FAILED"); return 3; }
    const float* lmW=(const float*)mapFile(d+"lm_head_weight.f32",(size_t)15*2048*1024*sizeof(float));
    const float* cpW=(const float*)mapFile(d+"codec_emb_weight.f32",(size_t)15*2048*2048*sizeof(float));
    auto trC=readVec(d+"talker_rope_cos.f32",384*HD); auto trS=readVec(d+"talker_rope_sin.f32",384*HD);
    auto crC=readVec(d+"codepred_rope_cos.f32",CAP*HD); auto crS=readVec(d+"codepred_rope_sin.f32",CAP*HD);
    if(!lmW||!cpW||trC.empty()||trS.empty()||crC.empty()||crS.empty()){ LOGI("TABLE_LOAD_FAILED"); return 3; }
    FILE* pf=fopen((d+"prompt_emb.f32").c_str(),"rb");
    if(!pf){ LOGI("PROMPT_EMB_MISSING"); return 3; }
    fseek(pf,0,SEEK_END); long nb=ftell(pf); fseek(pf,0,SEEK_SET);
    int L=(int)(nb/(2048*4));
    std::vector<float> pe(L*2048);
    if(fread(pe.data(),4,pe.size(),pf)!=pe.size()){ fclose(pf); LOGI("PROMPT_EMB_SHORT"); return 3; }
    fclose(pf);
    LOGI("LOAD_OK in %.0f ms (prompt tokens=%d)", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(), L);
    const size_t TKVONE=(size_t)NL*NKV*TKV*HD, CPKVONE=(size_t)CPNL*CPKV*CAP*HD;
    std::vector<float> tK(TKVONE,0.f), tV(TKVONE,0.f), sm(TKV,0.f), am(TKV,0.f);
    std::vector<float> cpK(CPKVONE,0.f), cpV(CPKVONE,0.f), csm(CAP,0.f), cam(CAP,0.f);
    VARP vEmb=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vK=_Input({NL,1,NKV,TKV,HD},NCHW,halide_type_of<float>());
    VARP vV=_Input({NL,1,NKV,TKV,HD},NCHW,halide_type_of<float>());
    VARP vCos=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSin=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vMask=_Input({1,1,1,TKV},NCHW,halide_type_of<float>());
    VARP vSlot=_Input({1,1,TKV,1},NCHW,halide_type_of<float>());
    auto gbStep=[&](const float* emb,int cp)->std::vector<float>{
        memcpy(vEmb->writeMap<float>(),emb,2048*4);
        memcpy(vK->writeMap<float>(),tK.data(),TKVONE*4);
        memcpy(vV->writeMap<float>(),tV.data(),TKVONE*4);
        memcpy(vCos->writeMap<float>(),&trC[(size_t)cp*HD],HD*4);
        memcpy(vSin->writeMap<float>(),&trS[(size_t)cp*HD],HD*4);
        for(int i=0;i<TKV;i++) am[i]=(i<=cp)?0.f:-1e4f;
        memcpy(vMask->writeMap<float>(),am.data(),TKV*4);
        sm.assign(TKV,0.f); sm[cp]=1.f; memcpy(vSlot->writeMap<float>(),sm.data(),TKV*4);
        std::vector<VARP> o=mGB->onForward({vEmb,vK,vV,vCos,vSin,vMask,vSlot});
        const float* h=o[0]->readMap<float>();
        std::vector<float> hb(h,h+2048);
        const float* ck=o[1]->readMap<float>(); const float* cv=o[2]->readMap<float>();
        for(int l=0;l<NL;l++) for(int hh=0;hh<NKV;hh++){
            size_t off=((size_t)(l*NKV+hh)*TKV+cp)*HD;
            memcpy(&tK[off],ck+((size_t)(l*NKV+hh))*HD,HD*4);
            memcpy(&tV[off],cv+((size_t)(l*NKV+hh))*HD,HD*4);
        }
        return hb;
    };
    auto tp=std::chrono::steady_clock::now();
    std::vector<float> hidden;
    for(int cp=0;cp<L;cp++) hidden=gbStep(&pe[(size_t)cp*2048],cp);
    double prefillMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count();
    LOGI("PREFILL_DONE %d tokens in %.0f ms", L, prefillMs);
    std::vector<float> lg(3072),lg2(2048),c0e(2048),cpH(1024),femb(2048);
    std::vector<int32_t> allCodes; int T=0,stopReason=-1; double decMs=0;
    VARP vCH=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vCE=_Input({1,1},NCHW,halide_type_of<int32_t>());
    VARP vPFa=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vPFb=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vPFc=_Input({1,1,2,HD},NCHW,halide_type_of<float>());
    VARP vPFs=_Input({1,1,2,HD},NCHW,halide_type_of<float>());
    VARP vPFm0=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vPFm1=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vPFam=_Input({1,1,2,CAP},NCHW,halide_type_of<float>());
    VARP vSTk=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vSTk_=_Input({CPNL,1,CPKV,CAP,HD},NCHW,halide_type_of<float>());
    VARP vSTv=_Input({CPNL,1,CPKV,CAP,HD},NCHW,halide_type_of<float>());
    VARP vSTc=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSTs=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSTm=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vSTam=_Input({1,1,1,CAP},NCHW,halide_type_of<float>());
    VARP vFEi=_Input({1,16},NCHW,halide_type_of<int32_t>());
    for(int n=0;n<maxFrames;n++){
        int cp=L+n;
        auto td=std::chrono::steady_clock::now();
        memcpy(vCH->writeMap<float>(),hidden.data(),2048*4);
        std::vector<VARP> oc=mCH->onForward({vCH});
        memcpy(lg.data(),oc[0]->readMap<float>(),3072*4);
        int code0=sampleTopK(lg.data(),3072,0.9,50,true,nextU());
        if(code0==2150){ stopReason=0; break; }
        { int32_t c=code0; memcpy(vCE->writeMap<int32_t>(),&c,4); }
        std::vector<VARP> oe=mCE->onForward({vCE});
        memcpy(c0e.data(),oe[0]->readMap<float>(),2048*4);
        memcpy(vPFa->writeMap<float>(),hidden.data(),2048*4);
        memcpy(vPFb->writeMap<float>(),c0e.data(),2048*4);
        memcpy(vPFc->writeMap<float>(),crC.data(),2*HD*4);
        memcpy(vPFs->writeMap<float>(),crS.data(),2*HD*4);
        csm.assign(CAP,0.f); csm[0]=1.f; memcpy(vPFm0->writeMap<float>(),csm.data(),CAP*4);
        csm.assign(CAP,0.f); csm[1]=1.f; memcpy(vPFm1->writeMap<float>(),csm.data(),CAP*4);
        { std::vector<float> a2((size_t)2*CAP,0.f);
          for(int r=0;r<2;r++) for(int i=0;i<CAP;i++) a2[(size_t)r*CAP+i]=(i<=r)?0.f:-1e4f;
          memcpy(vPFam->writeMap<float>(),a2.data(),(size_t)2*CAP*4); }
        std::vector<VARP> op=mCPF->onForward({vPFa,vPFb,vPFc,vPFs,vPFm0,vPFm1,vPFam});
        memcpy(cpH.data(),op[0]->readMap<float>(),1024*4);
        memcpy(cpK.data(),op[1]->readMap<float>(),CPKVONE*4);
        memcpy(cpV.data(),op[2]->readMap<float>(),CPKVONE*4);
        int32_t codes[16]; codes[0]=code0;
        for(int o2=0;o2<2048;o2++){ double acc=0; const float* w=&lmW[(size_t)o2*1024]; for(int i=0;i<1024;i++) acc+=(double)w[i]*cpH[i]; lg2[o2]=(float)acc; }
        for(int g=0;g<15;g++){
            codes[g+1]=sampleTopK(lg2.data(),2048,0.9,50,false,nextU());
            if(g==14) break;
            const float* e=&cpW[((size_t)g*2048+codes[g+1])*2048];
            int ccp=2+g;
            memcpy(vSTk->writeMap<float>(),e,2048*4);
            memcpy(vSTk_->writeMap<float>(),cpK.data(),CPKVONE*4);
            memcpy(vSTv->writeMap<float>(),cpV.data(),CPKVONE*4);
            memcpy(vSTc->writeMap<float>(),&crC[(size_t)ccp*HD],HD*4);
            memcpy(vSTs->writeMap<float>(),&crS[(size_t)ccp*HD],HD*4);
            csm.assign(CAP,0.f); csm[ccp]=1.f; memcpy(vSTm->writeMap<float>(),csm.data(),CAP*4);
            for(int i=0;i<CAP;i++) cam[i]=(i<=ccp)?0.f:-1e4f;
            memcpy(vSTam->writeMap<float>(),cam.data(),CAP*4);
            std::vector<VARP> os_=mCPS->onForward({vSTk,vSTk_,vSTv,vSTc,vSTs,vSTm,vSTam});
            memcpy(cpH.data(),os_[0]->readMap<float>(),1024*4);
            memcpy(cpK.data(),os_[1]->readMap<float>(),CPKVONE*4);
            memcpy(cpV.data(),os_[2]->readMap<float>(),CPKVONE*4);
            for(int o2=0;o2<2048;o2++){ double acc=0; const float* w=&lmW[(size_t)(g+1)*2048*1024+(size_t)o2*1024]; for(int i=0;i<1024;i++) acc+=(double)w[i]*cpH[i]; lg2[o2]=(float)acc; }
        }
        memcpy(vFEi->writeMap<int32_t>(),codes,16*4);
        std::vector<VARP> of=mFE->onForward({vFEi});
        memcpy(femb.data(),of[0]->readMap<float>(),2048*4);
        hidden=gbStep(femb.data(),cp);
        decMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-td).count();
        allCodes.insert(allCodes.end(),codes,codes+16);
        T++;
        if(n%10==0) LOGI("frame %3d code0=%4d", n, code0);
    }
    LOGI("GEN_DONE steps=%d stop=%s decode=%.0f ms", T, stopReason==0?"EOS":"MAXFRAMES", decMs);
    if(T==0){ LOGI("NO_FRAMES"); return 4; }
    const int NFR=300, SPF=1920, SR=24000;
    std::vector<int32_t> packed((size_t)16*NFR,0);
    for(int t=0;t<NFR;t++){ int src=t<T?t:T-1; for(int g=0;g<16;g++) packed[(size_t)g*NFR+t]=allCodes[(size_t)src*16+g]; }
    VARP vDec=_Input({1,16,NFR},NCHW,halide_type_of<int32_t>());
    memcpy(vDec->writeMap<int32_t>(),packed.data(),packed.size()*4);
    auto tdec=std::chrono::steady_clock::now();
    std::vector<VARP> od=mDEC->onForward({vDec});
    double decdMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tdec).count();
    const float* wv=od[0]->readMap<float>();
    size_t wl=(size_t)T*SPF; if(wl>576000) wl=576000;
    std::vector<float> wave(wv,wv+wl);
    writeWav16(out,wave,SR);
    double mn=1e9,mx=-1e9; size_t nan=0;
    for(float v:wave){ if(!std::isfinite(v)){nan++;continue;} if(v<mn)mn=v; if(v>mx)mx=v; }
    LOGI("WAV_SAVED %s samples=%zu dur=%.2fs finite=%s min=%.4f max=%.4f decoder=%.0f ms",
         out.c_str(), wl, (double)wl/SR, nan==0?"yes":"NO", mn, mx, decdMs);
    { FILE* sf=fopen("/proc/self/status","r"); char ln[256];
      if(sf){ while(fgets(ln,sizeof(ln),sf)){ if(!strncmp(ln,"VmRSS",5)||!strncmp(ln,"VmHWM",5)) { printf("%s",ln); fflush(stdout);} } fclose(sf);} }
    LOGI("CHAIN_DONE");
    return 0;
}

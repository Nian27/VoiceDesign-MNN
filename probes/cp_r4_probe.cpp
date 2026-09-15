// cp_r4_probe.cpp - CP-R4: self-driven shared-RNG 16-code trajectory on the NEW codepred
// usage: cp_r4_probe CODECHEAD CODECEMB PREFILL STEP LMHEAD_TBL EMB_TBL PAST COS16 SIN16 OUTDIR [cpu|htp] [NFRAMES]
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
static std::vector<char> rf(const char* p){FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",p);exit(2);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n){fprintf(stderr,"short %s\n",p);exit(2);} fclose(f); return v;}
static double nowMs(){using namespace std::chrono; return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();}
static unsigned long long g_seed=123456789ull;
static double nextU(){ g_seed=g_seed*6364136223846793005ull+1442695040888963407ull; return (double)((g_seed>>11)&((1ull<<53)-1))/(double)(1ull<<53);}
struct Pick{int idx;double lo,hi;};
static Pick sampleTopK(const float* lg,int vocab,double temp,int topk,bool suppress,double u){
    std::vector<std::pair<double,int>> v;
    for(int i=0;i<vocab;i++){ if(suppress&&i>=2048&&i!=2150) continue; v.push_back({(double)lg[i]/temp,i}); }
    std::sort(v.begin(),v.end(),[](const std::pair<double,int>&a,const std::pair<double,int>&b){return a.first>b.first;});
    if((int)v.size()>topk) v.resize(topk);
    double mx=v[0].first,s=0; std::vector<double> pr(v.size());
    for(size_t i=0;i<v.size();i++){pr[i]=exp(v[i].first-mx);s+=pr[i];}
    for(size_t i=0;i<v.size();i++) pr[i]/=s;
    double acc=0; Pick r{v.back().second,0.0,1.0};
    for(size_t i=0;i<v.size();i++){double lo=acc;acc+=pr[i]; if(u<=acc){r.idx=v[i].second;r.lo=lo;r.hi=acc;break;}}
    return r;
}
struct Mod{std::shared_ptr<Module> m;};
static Mod load(const char* p,std::vector<std::string> i,std::vector<std::string> o,std::shared_ptr<Executor::RuntimeManager> rtm){
    Module::Config c; c.shapeMutable=true; Mod r;
    r.m=std::shared_ptr<Module>(Module::load(i,o,p,rtm,&c),Module::destroy);
    if(!r.m){fprintf(stderr,"load fail %s\n",p);exit(3);} return r;
}
int main(int argc,char** argv){
    if(argc<11){fprintf(stderr,"usage: %s CODECHEAD CODECEMB PREFILL STEP LMHEAD_TBL EMB_TBL PAST COS SIN OUTDIR [cpu|htp] [N]\n",argv[0]);return 2;}
    setvbuf(stdout,NULL,_IOLBF,0);
    const char* fCH=argv[1]; const char* fCE=argv[2]; const char* fPF=argv[3]; const char* fST=argv[4];
    std::vector<char> lhB=rf(argv[5]), emB=rf(argv[6]), pastB=rf(argv[7]), cosB=rf(argv[8]), sinB=rf(argv[9]);
    const char* outDir=argv[10];
    const char* backend=argc>11?argv[11]:"cpu";
    int N=argc>12?atoi(argv[12]):1;
    const int CAP=16,HD=128,NL=5,NKV=8;
    if(lhB.size()!= (size_t)15*2048*1024*4){fprintf(stderr,"lm_head tbl size %zu\n",lhB.size());return 2;}
    if(emB.size()!= (size_t)15*2048*2048*4){fprintf(stderr,"emb tbl size %zu\n",emB.size());return 2;}
    const float* LW=(const float*)lhB.data();
    const float* EW=(const float*)emB.data();
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High; bc.memory=BackendConfig::Memory_Normal;
    ScheduleConfig sc; sc.numThread=4; sc.backendConfig=&bc;
    if(!strcmp(backend,"htp")){sc.type=MNN_FORWARD_HEXAGON;printf("BACKEND=hexagon\n");}else{sc.type=MNN_FORWARD_CPU;printf("BACKEND=cpu\n");}
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sc),Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER,4);
    Mod mCH=load(fCH,{"hidden_states"},{"logits"},rtm);
    Mod mCE=load(fCE,{"codes"},{"emb"},rtm);
    Mod mPF=load(fPF,{"past_hidden","code0_emb","rope_cos","rope_sin","slot_mask0","slot_mask1","attn_mask"},{"hidden_last","kv_k_out","kv_v_out"},rtm);
    Mod mST=load(fST,{"token_emb","kv_k","kv_v","rope_cos","rope_sin","slot_mask","attn_mask"},{"hidden_last","kv_k_out","kv_v_out"},rtm);
    VARP vCH=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vCE=_Input({1,1},NCHW,halide_type_of<int32_t>());
    VARP vPast=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vC0=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vCos2=_Input({1,1,2,HD},NCHW,halide_type_of<float>());
    VARP vSin2=_Input({1,1,2,HD},NCHW,halide_type_of<float>());
    VARP vSm0=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vSm1=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vAm2=_Input({1,1,2,CAP},NCHW,halide_type_of<float>());
    VARP vTok=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vK=_Input({NL,1,NKV,CAP,HD},NCHW,halide_type_of<float>());
    VARP vV=_Input({NL,1,NKV,CAP,HD},NCHW,halide_type_of<float>());
    VARP vCos1=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSin1=_Input({1,1,1,HD},NCHW,halide_type_of<float>());
    VARP vSm=_Input({1,1,CAP,1},NCHW,halide_type_of<float>());
    VARP vAm=_Input({1,1,1,CAP},NCHW,halide_type_of<float>());
    const size_t KVONE=(size_t)NL*NKV*CAP*HD;
    std::vector<float> kvk(KVONE),kvv(KVONE),sm(CAP),am(CAP),c0e(2048),hidden(1024),lg(2048);
    char rp[256]; snprintf(rp,sizeof(rp),"%s/r4_records.txt",outDir); FILE* fr=fopen(rp,"w");
    std::vector<float> past((const float*)pastB.data(),(const float*)pastB.data()+2048);
    double tFrame=0;
    for(int f=0; f<N; f++){
        double tf0=nowMs();
        // 1) code0 from codec_head(past)
        memcpy(vCH->writeMap<float>(), past.data(), 2048*4);
        std::vector<VARP> oc=mCH.m->onForward({vCH});
        const float* lgp=oc[0]->readMap<float>();
        std::vector<float> lg0(lgp, lgp+3072);
        double u0=nextU(); Pick p0=sampleTopK(lg0.data(),3072,0.9,50,true,u0);
        int code0=p0.idx;
        if(code0==2150){ fprintf(fr,"frame %d STOP_EOS\n",f); printf("frame %d EOS\n",f); break; }
        // 2) talker codec embedding for code0
        { int32_t c=code0; memcpy(vCE->writeMap<int32_t>(), &c, 4); }
        std::vector<VARP> oe=mCE.m->onForward({vCE});
        memcpy(c0e.data(), oe[0]->readMap<float>(), 2048*4);
        // 3) prefill
        memcpy(vPast->writeMap<float>(), past.data(), 2048*4);
        memcpy(vC0->writeMap<float>(), c0e.data(), 2048*4);
        memcpy(vCos2->writeMap<float>(), cosB.data(), 2*HD*4);
        memcpy(vSin2->writeMap<float>(), sinB.data(), 2*HD*4);
        sm.assign(CAP,0.f); sm[0]=1.f; memcpy(vSm0->writeMap<float>(), sm.data(), CAP*4);
        sm.assign(CAP,0.f); sm[1]=1.f; memcpy(vSm1->writeMap<float>(), sm.data(), CAP*4);
        { std::vector<float> a2((size_t)2*CAP,0.f);
          for(int r=0;r<2;r++) for(int i=0;i<CAP;i++) a2[(size_t)r*CAP+i]=(i<=r)?0.f:-1e4f;
          memcpy(vAm2->writeMap<float>(), a2.data(), (size_t)2*CAP*4); }
        std::vector<VARP> op=mPF.m->onForward({vPast,vC0,vCos2,vSin2,vSm0,vSm1,vAm2});
        memcpy(hidden.data(), op[0]->readMap<float>(), 1024*4);
        memcpy(kvk.data(), op[1]->readMap<float>(), KVONE*4);
        memcpy(kvv.data(), op[2]->readMap<float>(), KVONE*4);
        int codes[16]; codes[0]=code0; double us[16]; us[0]=u0; double los[16],his[16]; los[0]=p0.lo; his[0]=p0.hi;
        // 4) lm_head[0]
        for(int o=0;o<2048;o++){ double s=0; const float* w=LW + (size_t)0*2048*1024 + (size_t)o*1024; for(int i=0;i<1024;i++) s+=(double)w[i]*hidden[i]; lg[o]=(float)s; }
        for(int g=0; g<15; g++){
            double u=nextU(); Pick p=sampleTopK(lg.data(),2048,0.9,50,false,u);
            codes[g+1]=p.idx; us[g+1]=u; los[g+1]=p.lo; his[g+1]=p.hi;
            if(g==14) break;
            const float* e = EW + (size_t)g*2048*2048 + (size_t)p.idx*2048;
            int cp=2+g;
            memcpy(vTok->writeMap<float>(), e, 2048*4);
            memcpy(vK->writeMap<float>(), kvk.data(), KVONE*4);
            memcpy(vV->writeMap<float>(), kvv.data(), KVONE*4);
            memcpy(vCos1->writeMap<float>(), cosB.data()+(size_t)cp*HD*4, HD*4);
            memcpy(vSin1->writeMap<float>(), sinB.data()+(size_t)cp*HD*4, HD*4);
            sm.assign(CAP,0.f); sm[cp]=1.f; memcpy(vSm->writeMap<float>(), sm.data(), CAP*4);
            for(int i=0;i<CAP;i++) am[i]=(i<=cp)?0.f:-1e4f;
            memcpy(vAm->writeMap<float>(), am.data(), CAP*4);
            std::vector<VARP> os_=mST.m->onForward({vTok,vK,vV,vCos1,vSin1,vSm,vAm});
            memcpy(hidden.data(), os_[0]->readMap<float>(), 1024*4);
            memcpy(kvk.data(), os_[1]->readMap<float>(), KVONE*4);
            memcpy(kvv.data(), os_[2]->readMap<float>(), KVONE*4);
            for(int o=0;o<2048;o++){ double s=0; const float* w=LW + (size_t)(g+1)*2048*1024 + (size_t)o*1024; for(int i=0;i<1024;i++) s+=(double)w[i]*hidden[i]; lg[o]=(float)s; }
        }
        double tf=nowMs()-tf0; tFrame+=tf;
        fprintf(fr,"frame %d code0 %d", f, code0);
        for(int g=0;g<16;g++) fprintf(fr," %d", codes[g]);
        fprintf(fr," | u"); for(int g=0;g<16;g++) fprintf(fr," %.9f", us[g]);
        fprintf(fr," | cdf"); for(int g=0;g<16;g++) fprintf(fr," %.6f %.6f", los[g], his[g]);
        fprintf(fr," | frame_ms %.1f\n", tf);
        printf("FRAME %d code0=%4d codes=[%d %d %d ... %d] ms=%.1f\n", f, code0, codes[1],codes[2],codes[3],codes[15], tf);
        // next frame uses this frame's last hidden as past (self-driven across frames)
        memcpy(past.data(), hidden.data(), 1024*4);
    }
    fprintf(fr,"FRAMES %d TOTAL_MS %.1f\n", N, tFrame); fclose(fr);
    printf("FRAME_SUM_MS=%.1f AVG=%.1f\n", tFrame, tFrame/N);
    {FILE* sf=fopen("/proc/self/status","r"); char ln[256];
     if(sf){while(fgets(ln,sizeof(ln),sf)){if(!strncmp(ln,"VmRSS",5)||!strncmp(ln,"VmHWM",5))printf("MEM %s",ln);}fclose(sf);}}
    printf("CPR4_DONE\n");
    return 0;
}

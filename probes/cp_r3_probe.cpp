// cp_r3_probe.cpp - CP-R3: prefill(A) + forced-prefix 14 steps(B), no sampling on device
// usage: cp_r3_probe PREFILL_MNN STEP_MNN PAST_HIDDEN CODE0_EMB COS16 SIN16 FORCED_EMB OUTDIR [cpu|htp] [NSTEPS]
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
using namespace MNN;
using namespace MNN::Express;
static std::vector<char> rf(const char* p){FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",p);exit(2);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n){fprintf(stderr,"short %s\n",p);exit(2);} fclose(f); return v;}
static void wf(const char* p,const float* d,size_t n){FILE* f=fopen(p,"wb"); if(!f)return; fwrite(d,4,n,f); fclose(f);}
static double nowMs(){using namespace std::chrono; return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();}
struct Mod{std::shared_ptr<Module> m;};
static Mod load(const char* p, std::vector<std::string> i, std::vector<std::string> o, std::shared_ptr<Executor::RuntimeManager> rtm){
    Module::Config c; c.shapeMutable=true; Mod r;
    r.m=std::shared_ptr<Module>(Module::load(i,o,p,rtm,&c),Module::destroy);
    if(!r.m){fprintf(stderr,"load fail %s\n",p);exit(3);} return r;
}
int main(int argc, char** argv){
    if(argc<10){fprintf(stderr,"usage: %s PREFILL STEP PAST CODE0E COS SIN FORCED OUTDIR [cpu|htp] [NSTEPS]\n",argv[0]);return 2;}
    setvbuf(stdout,NULL,_IOLBF,0);
    const char* fp=argv[1]; const char* fs=argv[2];
    std::vector<char> pastB=rf(argv[3]), c0eB=rf(argv[4]), cosB=rf(argv[5]), sinB=rf(argv[6]), forcedB=rf(argv[7]);
    const char* outDir=argv[8];
    const char* backend=argc>9?argv[9]:"cpu";
    int NS=argc>10?atoi(argv[10]):14;
    const int CAP=16, HD=128, NL=5, NKV=8;
    if(cosB.size()!=CAP*HD*4){fprintf(stderr,"bad cos size %zu\n",cosB.size());return 2;}
    if(forcedB.size()<(size_t)NS*2048*4){fprintf(stderr,"forced_emb too small: %zu need %zu\n",forcedB.size(),(size_t)NS*2048*4);return 2;}
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High; bc.memory=BackendConfig::Memory_Normal;
    ScheduleConfig sc; sc.numThread=4; sc.backendConfig=&bc;
    if(!strcmp(backend,"htp")){sc.type=MNN_FORWARD_HEXAGON;printf("BACKEND=hexagon\n");} else {sc.type=MNN_FORWARD_CPU;printf("BACKEND=cpu\n");}
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sc),Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER,4);
    Mod mp=load(fp,{"past_hidden","code0_emb","rope_cos","rope_sin","slot_mask0","slot_mask1","attn_mask"},{"hidden_last","kv_k_out","kv_v_out"},rtm);
    Mod ms=load(fs,{"token_emb","kv_k","kv_v","rope_cos","rope_sin","slot_mask","attn_mask"},{"hidden_last","kv_k_out","kv_v_out"},rtm);
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
    std::vector<float> kvk(KVONE,0.f), kvv(KVONE,0.f), sm(CAP,0.f), am(CAP,0.f);
    char p[256];
    // ---- CP-R3A: prefill ----
    memcpy(vPast->writeMap<float>(), pastB.data(), 2048*4);
    memcpy(vC0->writeMap<float>(), c0eB.data(), 2048*4);
    memcpy(vCos2->writeMap<float>(), cosB.data(), 2*HD*4);
    memcpy(vSin2->writeMap<float>(), sinB.data(), 2*HD*4);
    sm.assign(CAP,0.f); sm[0]=1.f; memcpy(vSm0->writeMap<float>(), sm.data(), CAP*4);
    sm.assign(CAP,0.f); sm[1]=1.f; memcpy(vSm1->writeMap<float>(), sm.data(), CAP*4);
    { std::vector<float> am2((size_t)2*CAP, 0.f);
      for(int r=0;r<2;r++) for(int i=0;i<CAP;i++) am2[(size_t)r*CAP+i] = (i<=r)?0.f:-1e4f;
      memcpy(vAm2->writeMap<float>(), am2.data(), (size_t)2*CAP*4); }
    for(int r=0;r<2;r++){
        double t0=nowMs();
        std::vector<VARP> o=mp.m->onForward({vPast,vC0,vCos2,vSin2,vSm0,vSm1,vAm2});
        double ms=nowMs()-t0;
        const float* h=o[0]->readMap<float>();
        memcpy(kvk.data(), o[1]->readMap<float>(), KVONE*4);
        memcpy(kvv.data(), o[2]->readMap<float>(), KVONE*4);
        if(r==0){printf("PREFILL warm_ms=%.2f\n",ms);} else {
            printf("PREFILL ms=%.2f\n",ms);
            double mx=0; for(int i=0;i<1024;i++){double a=h[i]<0?-h[i]:h[i]; if(a>mx)mx=a;}
            printf("PREFILL hidden maxabs=%.6f\n",mx);
            snprintf(p,sizeof(p),"%s/prefill_hidden.raw",outDir); wf(p,h,1024);
            snprintf(p,sizeof(p),"%s/prefill_kvk.raw",outDir); wf(p,kvk.data(),KVONE);
            snprintf(p,sizeof(p),"%s/prefill_kvv.raw",outDir); wf(p,kvv.data(),KVONE);
        }
    }
    // ---- CP-R3B: forced-prefix steps ----
    double tot=0, tmin=1e9, tmax=0;
    for(int g=0; g<NS; g++){
        int cp=2+g;
        memcpy(vTok->writeMap<float>(), forcedB.data()+(size_t)g*2048*4, 2048*4);
        memcpy(vK->writeMap<float>(), kvk.data(), KVONE*4);
        memcpy(vV->writeMap<float>(), kvv.data(), KVONE*4);
        memcpy(vCos1->writeMap<float>(), cosB.data()+(size_t)cp*HD*4, HD*4);
        memcpy(vSin1->writeMap<float>(), sinB.data()+(size_t)cp*HD*4, HD*4);
        sm.assign(CAP,0.f); sm[cp]=1.f; memcpy(vSm->writeMap<float>(), sm.data(), CAP*4);
        for(int i=0;i<CAP;i++) am[i]=(i<=cp)?0.f:-1e4f;
        memcpy(vAm->writeMap<float>(), am.data(), CAP*4);
        double t0=nowMs();
        std::vector<VARP> o=ms.m->onForward({vTok,vK,vV,vCos1,vSin1,vSm,vAm});
        double t=nowMs()-t0;
        const float* h=o[0]->readMap<float>();
        memcpy(kvk.data(), o[1]->readMap<float>(), KVONE*4);
        memcpy(kvv.data(), o[2]->readMap<float>(), KVONE*4);
        if(g>0){ tot+=t; if(t<tmin)tmin=t; if(t>tmax)tmax=t; } else { printf("STEP0 warm_ms=%.2f\n",t); tot+=t; tmin=t; tmax=t; }
        snprintf(p,sizeof(p),"%s/step%02d_hidden.raw",outDir,g); wf(p,h,1024);
        snprintf(p,sizeof(p),"%s/step%02d_kvk.raw",outDir,g); wf(p,kvk.data(),KVONE);
        snprintf(p,sizeof(p),"%s/step%02d_kvv.raw",outDir,g); wf(p,kvv.data(),KVONE);
        printf("STEP %02d cp=%2d ms=%.2f\n",g,cp,t);
    }
    printf("STEP_SUM_MS=%.2f AVG_MS=%.2f MIN=%.2f MAX=%.2f (n=%d)\n",tot,tot/NS,tmin,tmax,NS);
    {FILE* sf=fopen("/proc/self/status","r"); char ln[256];
     if(sf){while(fgets(ln,sizeof(ln),sf)){if(!strncmp(ln,"VmRSS",5)||!strncmp(ln,"VmHWM",5))printf("MEM %s",ln);}fclose(sf);}}
    printf("CPR3_DONE\n");
    return 0;
}

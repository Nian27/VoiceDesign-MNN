// g2b_loop2.cpp - G2-B corrected: official decode order
//   code0 = sample(codec_head(hiddenPrev)); codes15 = codepred(hiddenPrev, codec_emb(code0));
//   emb = frame_emb([code0]+codes15); hidden = GraphB(emb, KV, cp); hiddenPrev = hidden
// usage: g2b_loop2 GRAPHB CODECHEAD CODECEMB CODEPRED FRAMEEMB COS SIN KVK KVV PREFILL_HIDDEN OUTDIR [cpu|htp] N CP0
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
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",p);exit(2);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n){fprintf(stderr,"short read %s\n",p);exit(2);} fclose(f); return v;
}
static void wf(const char* p,const float* d,size_t n){FILE* f=fopen(p,"wb"); if(!f)return; fwrite(d,4,n,f); fclose(f);}
static unsigned long long g_seed=123456789ull;
static double nextU(){ g_seed=g_seed*6364136223846793005ull+1442695040888963407ull; return (double)((g_seed>>11)&((1ull<<53)-1))/(double)(1ull<<53); }
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
static Mod load(const char* path,std::vector<std::string> i,std::vector<std::string> o,std::shared_ptr<Executor::RuntimeManager> rtm){
    Module::Config c; c.shapeMutable=true; Mod r;
    r.m=std::shared_ptr<Module>(Module::load(i,o,path,rtm,&c),Module::destroy);
    if(!r.m){fprintf(stderr,"load fail %s\n",path);exit(3);} return r;
}
int main(int argc,char** argv){
    setvbuf(stdout,NULL,_IOLBF,0);
    if(argc<14){fprintf(stderr,"usage: %s GRAPHB CODECHEAD CODECEMB CODEPRED FRAMEEMB COS SIN KVK KVV PREHID OUTDIR [cpu|htp] N CP0\n",argv[0]);return 2;}
    const char* fGB=argv[1];const char* fCH=argv[2];const char* fCE=argv[3];const char* fCP=argv[4];const char* fFE=argv[5];
    std::vector<char> cosB=readFile(argv[6]),sinB=readFile(argv[7]),kvkB=readFile(argv[8]),kvvB=readFile(argv[9]),preB=readFile(argv[10]);
    const char* outDir=argv[11];
    const char* backend=argc>12?argv[12]:"cpu";
    int N=argc>13?atoi(argv[13]):20; int CP0=argc>14?atoi(argv[14]):62;
    const int KV=768; const size_t KVONE=(size_t)28*8*KV*128;
    std::vector<float> kvk(KVONE),kvv(KVONE);
    memcpy(kvk.data(),kvkB.data(),KVONE*4); memcpy(kvv.data(),kvvB.data(),KVONE*4);
    if(preB.size()!=2048*4){fprintf(stderr,"bad prefill hidden size %zu\n",preB.size());return 2;}
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High; bc.memory=BackendConfig::Memory_Normal;
    ScheduleConfig sched;
    if(!strcmp(backend,"htp")){sched.type=MNN_FORWARD_HEXAGON;printf("BACKEND=hexagon\n");}else{sched.type=MNN_FORWARD_CPU;printf("BACKEND=cpu\n");}
    sched.numThread=4; sched.backendConfig=&bc;
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched),Executor::RuntimeManager::destroy);
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER,4);
    Mod mGB=load(fGB,{"inputs_embeds","kv_k","kv_v","rope_cos","rope_sin","attn_mask","slot_mask"},{"hidden_states","cur_k","cur_v"},rtm);
    Mod mCH=load(fCH,{"hidden_states"},{"logits"},rtm);
    Mod mCE=load(fCE,{"codes"},{"emb"},rtm);
    Mod mCP=load(fCP,{"past_hidden","code0_emb"},{"logits15"},rtm);
    Mod mFE=load(fFE,{"codes"},{"emb"},rtm);
    VARP vEmb=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vK=_Input({28,1,8,KV,128},NCHW,halide_type_of<float>());
    VARP vV=_Input({28,1,8,KV,128},NCHW,halide_type_of<float>());
    VARP vCos=_Input({1,1,128},NCHW,halide_type_of<float>());
    VARP vSin=_Input({1,1,128},NCHW,halide_type_of<float>());
    VARP vMask=_Input({1,1,1,KV},NCHW,halide_type_of<float>());
    VARP vSlot=_Input({1,1,KV,1},NCHW,halide_type_of<float>());
    VARP vCH=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vCE=_Input({1,1},NCHW,halide_type_of<int32_t>());
    VARP vCPa=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vCPb=_Input({1,1,2048},NCHW,halide_type_of<float>());
    VARP vFE=_Input({1,16},NCHW,halide_type_of<int32_t>());
    std::vector<float> hiddenPrev((const float*)preB.data(), (const float*)preB.data()+2048);
    std::vector<float> maskv(KV,0.f),slotv(KV,0.f),c0e(2048),lg(3072),lg15(15*2048),emb(2048);
    char recp[256]; snprintf(recp,sizeof(recp),"%s/records.txt",outDir); FILE* fr=fopen(recp,"w");
    double tGB=0,tCH=0,tCP=0,tFE=0; int T=0,stopReason=-1;
    for(int n=0;n<N;n++){
        int cp=CP0+n;
        // 1) code0 from hiddenPrev
        auto t1=std::chrono::steady_clock::now();
        memcpy(vCH->writeMap<float>(),hiddenPrev.data(),2048*4);
        std::vector<VARP> o2=mCH.m->onForward({vCH});
        memcpy(lg.data(),o2[0]->readMap<float>(),3072*4);
        tCH+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t1).count();
        double u0=nextU(); Pick p0=sampleTopK(lg.data(),3072,0.9,50,true,u0);
        int code0=p0.idx;
        if(code0==2150){stopReason=0;fprintf(fr,"step %d cp %d STOP_EOS u0 %.9f\n",n,cp,u0);printf("EOS at step %d\n",n);break;}
        // 2) codepred
        auto t2=std::chrono::steady_clock::now();
        { int32_t c=code0; memcpy(vCE->writeMap<int32_t>(),&c,4); }
        std::vector<VARP> o3=mCE.m->onForward({vCE});
        memcpy(c0e.data(),o3[0]->readMap<float>(),2048*4);
        memcpy(vCPa->writeMap<float>(),hiddenPrev.data(),2048*4);
        memcpy(vCPb->writeMap<float>(),c0e.data(),2048*4);
        std::vector<VARP> o4=mCP.m->onForward({vCPa,vCPb});
        memcpy(lg15.data(),o4[0]->readMap<float>(),15*2048*4);
        tCP+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t2).count();
        int codes[16]; codes[0]=code0; double us[16]; us[0]=u0; double los[16],his[16]; los[0]=p0.lo; his[0]=p0.hi;
        double stageTop[15];
        for(int g=0;g<15;g++){
            double ug=nextU(); us[g+1]=ug;
            Pick pg=sampleTopK(lg15.data()+(size_t)g*2048,2048,0.9,50,false,ug);
            codes[g+1]=pg.idx; los[g+1]=pg.lo; his[g+1]=pg.hi;
            const float* row=lg15.data()+(size_t)g*2048; int bi=0; for(int i=1;i<2048;i++) if(row[i]>row[bi]) bi=i;
            stageTop[g]=(double)bi;
        }
        // 3) frame_emb
        auto t3=std::chrono::steady_clock::now();
        std::vector<int32_t> fr16(codes,codes+16);
        memcpy(vFE->writeMap<int32_t>(),fr16.data(),16*4);
        std::vector<VARP> o5=mFE.m->onForward({vFE});
        memcpy(emb.data(),o5[0]->readMap<float>(),2048*4);
        tFE+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t3).count();
        // 4) GraphB
        for(int i=0;i<KV;i++){maskv[i]=(i<=cp)?0.f:-1e4f; slotv[i]=(i==cp)?1.f:0.f;}
        memcpy(vEmb->writeMap<float>(),emb.data(),2048*4);
        memcpy(vK->writeMap<float>(),kvk.data(),KVONE*4);
        memcpy(vV->writeMap<float>(),kvv.data(),KVONE*4);
        memcpy(vCos->writeMap<float>(),cosB.data()+(size_t)n*128*4,128*4);
        memcpy(vSin->writeMap<float>(),sinB.data()+(size_t)n*128*4,128*4);
        memcpy(vMask->writeMap<float>(),maskv.data(),KV*4);
        memcpy(vSlot->writeMap<float>(),slotv.data(),KV*4);
        auto t0=std::chrono::steady_clock::now();
        std::vector<VARP> o1=mGB.m->onForward({vEmb,vK,vV,vCos,vSin,vMask,vSlot});
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        tGB+=ms;
        const float* hid=o1[0]->readMap<float>();
        const float* ck=o1[1]->readMap<float>();
        const float* cv=o1[2]->readMap<float>();
        std::vector<float> hidb(hid,hid+2048);
        for(int l=0;l<28;l++) for(int h=0;h<8;h++){
            size_t off=((size_t)(l*8+h)*KV+cp)*128;
            memcpy(&kvk[off],ck+((size_t)(l*8+h))*128,128*4);
            memcpy(&kvv[off],cv+((size_t)(l*8+h))*128,128*4);
        }
        char p[256];
        snprintf(p,sizeof(p),"%s/step%03d_hidden.raw",outDir,n); wf(p,hidb.data(),2048);
        snprintf(p,sizeof(p),"%s/step%03d_logits.raw",outDir,n); wf(p,lg.data(),3072);
        snprintf(p,sizeof(p),"%s/step%03d_c0e.raw",outDir,n); wf(p,c0e.data(),2048);
        snprintf(p,sizeof(p),"%s/step%03d_emb.raw",outDir,n); wf(p,emb.data(),2048);
        fprintf(fr,"step %d cp %d gb_ms %.1f code0 %d",n,cp,ms,code0);
        for(int g=0;g<16;g++) fprintf(fr," %d",codes[g]);
        fprintf(fr," | u"); for(int g=0;g<16;g++) fprintf(fr," %.9f",us[g]);
        fprintf(fr," | cdf"); for(int g=0;g<16;g++) fprintf(fr," %.6f %.6f",los[g],his[g]);
        fprintf(fr," | stageTop"); for(int g=0;g<15;g++) fprintf(fr," %d",(int)stageTop[g]);
        { char lp[256]; snprintf(lp,sizeof(lp),"%s/step%03d_lg15.raw",outDir,n); wf(lp,lg15.data(),15*2048); }
        fprintf(fr,"\n");
        printf("STEP %03d cp=%3d code0=%4d codes=[%d %d %d ...] gb_ms=%.1f\n",n,cp,code0,codes[1],codes[2],codes[3],ms);
        fflush(stdout); fflush(fr);
        memcpy(hiddenPrev.data(),hidb.data(),2048*4);
        T++;
    }
    fprintf(fr,"TOTAL steps %d stop %d\n",T,stopReason); fclose(fr);
    printf("GEN steps=%d stop=%s\n",T,stopReason==0?"EOS":"MAXN");
    printf("LAT gb=%.0f ch=%.0f cp=%.0f fe=%.0f total=%.0f\n",tGB,tCH,tCP,tFE,tGB+tCH+tCP+tFE);
    {FILE* sf=fopen("/proc/self/status","r"); char ln[256];
     if(sf){while(fgets(ln,sizeof(ln),sf)){if(!strncmp(ln,"VmRSS",5)||!strncmp(ln,"VmHWM",5))printf("MEM %s",ln);}fclose(sf);}}
    printf("G2B_DONE\n");
    return 0;
}

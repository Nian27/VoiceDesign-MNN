// small_probe.cpp - small-graph backend comparison: cpu vs opencl (GPU) vs hexagon
// usage: small_probe MODE MODEL IN_RAW OUT_RAW [cpu|opencl|htp] REPEATS
//   ch: hidden_states(1,1,2048) f32 -> logits(1,1,3072)
//   ce: codes(1,1) i32           -> emb(1,1,2048)
//   fe: codes(1,16) i32          -> emb(1,1,2048)
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
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",p);exit(2);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n){fprintf(stderr,"short read %s\n",p);exit(2);} fclose(f); return v;
}
static uint32_t crc32b(const float* d,size_t n){const uint8_t*p=(const uint8_t*)d;uint32_t c=0xFFFFFFFFu;
 for(size_t i=0;i<n*4;i++){c^=p[i];for(int k=0;k<8;k++)c=(c>>1)^(0xEDB88320u&(uint32_t)(-(int32_t)(c&1)));}return ~c;}
int main(int argc,char** argv){
    if(argc<6){fprintf(stderr,"usage: %s MODE MODEL IN_RAW OUT_RAW [cpu|opencl|htp] REPEATS\n",argv[0]);return 2;}
    setvbuf(stdout,NULL,_IOLBF,0);
    std::string mode=argv[1];
    const char* model=argv[2];
    std::vector<char> inRaw=readFile(argv[3]);
    const char* outPath=argv[4];
    const char* backend=argv[5];
    int repeats=argc>6?atoi(argv[6]):10;
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High; bc.memory=BackendConfig::Memory_Normal;
    ScheduleConfig sched; sched.numThread=4; sched.backendConfig=&bc;
    if(!strcmp(backend,"opencl")){ sched.type=MNN_FORWARD_OPENCL; printf("BACKEND=opencl(GPU)\n"); }
    else if(!strcmp(backend,"htp")){ sched.type=MNN_FORWARD_HEXAGON; printf("BACKEND=hexagon(NPU)\n"); }
    else { sched.type=MNN_FORWARD_CPU; printf("BACKEND=cpu\n"); }
    std::shared_ptr<Executor::RuntimeManager> rtm(Executor::RuntimeManager::createRuntimeManager(sched),Executor::RuntimeManager::destroy);
    if(!rtm){fprintf(stderr,"rtm failed\n");return 3;}
    rtm->setHint(Interpreter::INIT_THREAD_NUMBER,4);
    Module::Config mcfg; mcfg.shapeMutable=true;
    std::shared_ptr<Module> mod;
    std::vector<int> inDim; bool isInt=false;
    if(mode=="ch"){ mod=std::shared_ptr<Module>(Module::load({"hidden_states"},{"logits"},model,rtm,&mcfg),Module::destroy); inDim={1,1,2048}; }
    else if(mode=="ce"){ mod=std::shared_ptr<Module>(Module::load({"codes"},{"emb"},model,rtm,&mcfg),Module::destroy); inDim={1,1}; isInt=true; }
    else if(mode=="fe"){ mod=std::shared_ptr<Module>(Module::load({"codes"},{"emb"},model,rtm,&mcfg),Module::destroy); inDim={1,16}; isInt=true; }
    else { fprintf(stderr,"bad mode\n"); return 2; }
    if(!mod){fprintf(stderr,"load failed\n");return 3;}
    VARP vin = isInt ? _Input(inDim,NCHW,halide_type_of<int32_t>()) : _Input(inDim,NCHW,halide_type_of<float>());
    size_t inBytes = isInt ? (size_t)(inDim[0]*inDim[1])*4 : (size_t)(inDim[0]*inDim[1]*inDim[2])*4;
    if(inRaw.size()>=inBytes) memcpy(vin->writeMap<int8_t>(), inRaw.data(), inBytes);
    double best=1e9,worst=0,sum=0; std::vector<VARP> out;
    double tLoad=0;
    for(int r=0;r<=repeats;r++){
        auto t0=std::chrono::steady_clock::now();
        out=mod->onForward({vin});
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        if(r==0){ printf("WARM_MS=%.2f\n",ms); }
        else { if(ms<best)best=ms; if(ms>worst)worst=ms; sum+=ms; }
    }
    auto inf=out[0]->getInfo();
    size_t elems=1; for(int d:inf->dim) elems*= d>0?d:1;
    const float* d=out[0]->readMap<float>();
    double mx=0; bool fin=true;
    for(size_t i=0;i<elems;i++){ float x=d[i]; if(x!=x||x>1e37f||x<-1e37f) fin=false; double a=x<0?-x:x; if(a>mx&&a<1e37) mx=a; }
    printf("OUT elems=%zu maxabs=%.6f finite=%d crc32=%08x\n",elems,mx,fin?1:0,crc32b(d,elems));
    printf("LAT avg=%.2f min=%.2f max=%.2f (n=%d)\n",sum/repeats,best,worst,repeats);
    FILE* f=fopen(outPath,"wb"); if(f){fwrite(d,4,elems,f);fclose(f);printf("SAVED %s\n",outPath);}
    {FILE* sf=fopen("/proc/self/status","r"); char ln[256];
     if(sf){while(fgets(ln,sizeof(ln),sf)){if(!strncmp(ln,"VmRSS",5))printf("MEM %s",ln);}fclose(sf);}}
    printf("SMALL_DONE\n");
    return 0;
}

#include <cstdlib>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include "data.h"
#include "Matrixcuda.h"
#include "MatrixStdDouble.h"
#include "BlockPlanner.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace vmatrix
{
    void resetMem(double* y, int n)
    {
         for(int i=0;i<n;i++){
             y[i] = 0;
         }
    }
    void copyCuda(int blocks[], int blockrows, int blocksize, int bl[], int bu[])
    {
        int i,j,w;
        int memCount = 0;
          
        int blockmin = blockrows;
        int blockmax = 0;
            for (i = 0; i < blockrows*blockrows; i++)
            {
                if (blocks[i] > 0)
                {
                    memCount++;
                    if(blocks[i]>blockmax)
                        blockmax = blocks[i];
                    if(blocks[i]<blockmin)
                        blockmin = blocks[i];
                }
            }
//        std::cout<<"input total: "<<memCount<<" input start: "<<blockmin<<" input end: "<<blockmax<<"prepare cuda"<<std::endl;
        int* blockptrs = (int*)malloc(data::blockstorage.size() * sizeof(int));
        for(i=0;i<data::blockstorage.size();i++)
             blockptrs[i] = 0;
        int count = 1;
        int umax = 0;
        int uptrmax = 0;
        int umaxstage = 0;
        int uminstage = 10000;
        for (i = 0; i < blockrows*blockrows; i++) {
                if (blocks[i] > 0 && blockptrs[blocks[i]] ==0) {
                    blockptrs[blocks[i]] = count;
                    count++;
               if(blocks[i]>umax) umax=blocks[i];
               if(blockptrs[blocks[i]]>uptrmax) uptrmax = blockptrs[blocks[i]];
               if(data::laststage[blocks[i]] > umaxstage) umaxstage = data::laststage[blocks[i]];
               if(data::laststage[blocks[i]] < uminstage) uminstage = data::laststage[blocks[i]];
                }
        }
       std::cout<<" input count: "<<count<<" storage:" <<count/32<<"MB umax: "<<umax<<" uptrmax: "
                     <<uptrmax<<" maxlaststage:"<<umaxstage<<"minlaststage:"<<uminstage<<std::endl;

        int blcount = 0;
        int bxcount = 0;
        for (i = 0; i < blockrows*blockrows; i++) {
                if (bl[i] > 0) {
                    blcount++;
                }
                if (bu[i] > 0) {
                    bxcount++;
                }
        }
        bxcount += count;
        for (i = 0; i < blockrows*blockrows; i++) {
                if (bl[i] > 0 && blockptrs[bl[i]] ==0) {
                    blockptrs[bl[i]] = count;
                    count++;
                }
        }
        if(count<bxcount){
            for (i = 0; i < blockrows*blockrows; i++) {
                if (bu[i] > 0 && blockptrs[bu[i]] ==0) {
                    blockptrs[bu[i]] = count;
                    count++;
                }
                if(count>bxcount) break;
            }
        }
        umaxstage = 0;
        for (operation* o : data::seq)
        {
            if(o->stage>umaxstage) umaxstage = o->stage;
            if(o->src>0 && blockptrs[o->src] == 0){
                blockptrs[o->src] = count;
                count++;
            }
            if(o->src2>0 && blockptrs[o->src2] == 0){
                blockptrs[o->src2] = count;
                count++;
            }

            if(o->result>0 && blockptrs[o->result] == 0){
                blockptrs[o->result] = count;
                count++;
            }
            if(o->result2>0 && blockptrs[o->result2] == 0){
                blockptrs[o->result2] = count;
                count++;
            }
        }
        std::cout<<"host storage: "<< data::storageCount<<"  cuda storage: "<<count<<" = " <<data::blockstorage.size()
                     <<"  input count: "<<memCount<<" max stage: "<<umaxstage<<std::endl;

        blockmax = 0;
        for (i = 0; i < blockrows*blockrows; i++)
        {
            if (blocks[i] > 0 && blockptrs[blocks[i]] > 0){
                if(blockmax < blockptrs[blocks[i]]) 
                     blockmax = blockptrs[blocks[i]];
            }
        }
        blockmax += 1;
        std::cout<<"input blockmax: "<<blockmax<<" malloc: "<<sizeof(double)*blockmax * blocksize * blocksize<<std::endl;

        int blockmaxsrc = blockmax;
        blockmax = std::max(blockmaxsrc,std::max(bxcount,blcount)) + 1;

        double* hostsrc = (double*)malloc(sizeof(double)*blockmax * blocksize * blocksize);
std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
        for(i=0;i<blockmaxsrc * blocksize * blocksize;i++) hostsrc[i] = 0;
        for (i = 0; i < blockrows*blockrows; i++)
        {
            if (blocks[i] > 0 && blockptrs[blocks[i]] > 0){
                double* blocki = hostsrc + blockptrs[blocks[i]] * blocksize * blocksize;
//           std::cout<<"index: "<<i<<" new ptr: "<<blockptrs[blocks[i]]<< " old ptr: "<<data::blockstorage[blocks[i]]<<std::endl;
   //             std::memcpy(blocki, data::blockstorage[blocks[i]], sizeof(double)* blocksize * blocksize);
                for(j=0;j<blocksize * blocksize;j++){
                   blocki[j] = data::blockstorage[blocks[i]][j]; 
                }
            }
        }
        for(j=0;j<blocksize * blocksize;j++){
                   hostsrc[j] = 0;
        }
std::chrono::duration<double> time_clear =
                 std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout << "copying time: "<<time_clear.count() <<" host allocate "<<blockmax/32<<'\n';

        int* stat = (int*) malloc(sizeof(int)*count*3); // count, stage, op
        for(i=0;i<count*3;i++)
             stat[i] = -1;
        int maxstage = 0;

        for (operation* o : data::seq)
        {
            if(o->result>0){
                int loc = blockptrs[o->result]*3;
                if(stat[loc] >0 ) stat[loc] += 1;
                else   stat[loc] = 1;
                if(stat[loc+1] == -1) stat[loc+1] = o->stage;
                if(stat[loc+1] != o->stage){
              //       std::cout<<"result: "<<loc/3<<" updated at multiple stage: "<<stat[loc+1]<<" "<<o->stage<<" op "<<o->op<<std::endl;
                }
                if(maxstage<o->stage) maxstage=o->stage;
                if(stat[loc+2] == -1) stat[loc+2] = o->op;
                if(stat[loc+2] != o->op){
                     std::cout<<"result: "<<loc/3<<" updated at multiple op: "<<stat[loc+2]<<" "<<o->op<<std::endl;
                }
            }
            if(o->result2>0){
                int loc = blockptrs[o->result2]*3;
                if(stat[loc] >0 ) stat[loc] += 1;
                else   stat[loc] = 1;
                if(stat[loc+1] == -1) stat[loc+1] = o->stage;
                if(stat[loc+1] != o->stage){
               //      std::cout<<"result: "<<loc/3<<" updated at multiple stage: "<<stat[loc+1]<<" "<<o->stage<<" op "<<o->op<<std::endl;
                }
                if(maxstage<o->stage) maxstage=o->stage;
                if(stat[loc+2] == -1) stat[loc+2] = o->op;
                if(stat[loc+2] != o->op){
                     std::cout<<"result: "<<loc/3<<" updated at multiple op: "<<stat[loc+2]<<" "<<o->op<<std::endl;
                }
            }
        }

        int updatecount = 0;
        int maxupdate = 0;
        for(i=0;i<count*3;i=i+3){
            if(stat[i] == -1) continue;
            updatecount++;
            if(stat[i] == 1) continue;
            if(stat[i] > maxupdate) maxupdate = stat[i];
      //      std::cout<<"storage: "<<i/3<<" udate: "<<stat[i]<<" times at stage: "<<stat[i+1] << " op "<<stat[i+2]<<std::endl;
        }
        std::cout<<"result count: "<<updatecount<<" op count: "<<data::seq.size()<<" max updated: "<<maxupdate<<std::endl;
        std::cout<<"host storage: "<< data::storageCount<<"  cuda storage: "<<count<<"  input count: "<<memCount<<std::endl;
   //     std::cout<<"max stage: "<<maxstage<<std::endl;

        int* stagerange = (int*) malloc((maxstage*2 + 2)*sizeof(int));
        for(i=0;i<maxstage*2 + 2;i+=2){
            stagerange[i] = data::seq.size();
            stagerange[i+1] = 0;
        }
        for(i=0;i<data::seq.size();i++){
            int st = data::seq[i]->stage;
            if(stagerange[st*2] > i) stagerange[st*2] = i;
            if(stagerange[st*2+1] < i) stagerange[st*2+1] = i;
        }

        int gpumax = data::gpuMB * 32;
        int* stagemax = (int*) malloc((maxstage + 1)*sizeof(int));
        std::vector<int> tbd;
        tbd.reserve(10000);
        if(count>gpumax) {
            std::cout<<"cuda memory: "<< count/32 <<"MB allocate: "<<gpumax/32<<" MB"<<std::endl;
            count=gpumax;
        }
        gpumax -= 1;
 
        for(w=0;w<maxstage*2+2;w+=2)
        {
            stagemax[w/2] = 0;
            for(int s=stagerange[w]; s<=stagerange[w+1]; s++){
                operation* o2 = data::seq[s];
                if(o2->stage * 2 != w) continue;
                if(o2->src>0 && blockptrs[o2->src] > stagemax[w/2]){
                    stagemax[w/2] = blockptrs[o2->src];
                }
                if(o2->src2>0 && blockptrs[o2->src2]  > stagemax[w/2]){
                    stagemax[w/2] = blockptrs[o2->src2];
                }

                if(o2->result>0 && blockptrs[o2->result] >stagemax[w/2]){
                    stagemax[w/2] = blockptrs[o2->result];
                }
                if(o2->result2>0 && blockptrs[o2->result2] >stagemax[w/2]){
                    stagemax[w/2] = blockptrs[o2->result2];
                }
            }
        }

        int blocksizesq= blocksize * blocksize;
        unsigned long mem_size = sizeof(double) * count * blocksize * blocksize;
        if(count > 200000){
        }

        double* srcbuf;
        cudaHostAlloc((void**)&srcbuf, sizeof(double)* blocksize * blocksize, cudaHostAllocDefault);
        double* srcbuf2;
        cudaHostAlloc((void**)&srcbuf2, sizeof(double)* blocksize * blocksize, cudaHostAllocDefault);
        double* resultbuf;
        cudaHostAlloc((void**)&resultbuf, sizeof(double)* blocksize * blocksize, cudaHostAllocDefault);
        double* resultbuf2;
        cudaHostAlloc((void**)&resultbuf2, sizeof(double)* blocksize * blocksize, cudaHostAllocDefault);

        int* host_list;
        cudaHostAlloc((void**)&host_list, NUMSTREAM*sizeof(int)* 400, cudaHostAllocDefault);

        double *d_mem;
   //     double *d_t;
        cudaError_t error;
        error = cudaMalloc((void **) &d_mem, mem_size);
        //gpuErrchk(error);
        if(error != cudaSuccess){
            std::cout<<"out of memory: "<<mem_size/1024/1024<<" MB"<<std::endl;
            exit(0);
        }
        cudaStream_t cpustream;
        cudaStream_t gstream[NUMSTREAM];
        //cudaStreamCreateWithFlags(&cpustream,cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&cpustream,cudaStreamNonBlocking);
        for(i=0;i<NUMSTREAM;i++){
            cudaStreamCreateWithFlags(&gstream[i],cudaStreamNonBlocking);
        }
                int tbc[12];
        cudaMemset(d_mem,0,mem_size);
 /*               for(int uu=0;uu<count;uu+=12){
                    for(w = 0; w<12; w++)
                        tbc[w] = 0;
                    for(w=0;w<12 && uu+w<count; w += 1){
                        tbc[w] = uu+w;
                    }
                        cudaresetparam4(d_mem, tbc[0],tbc[1], tbc[2], tbc[3],tbc[4], tbc[5],
                                           tbc[6],tbc[7], tbc[8], tbc[9],tbc[10], tbc[11],gstream[(uu/12)%NUMSTREAM]);
                }
  */             cudaDeviceSynchronize();
    //    error = cudaMalloc((void **) &d_t, NUMSTREAM*sizeof(double)* blocksize * blocksize);
        int *d_mullist;
        error = cudaMalloc((void **) &d_mullist, NUMSTREAM*sizeof(int)* 400);

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
//        cudaFuncSetSharedMemConfig(kernel_func_name, cudaSharedMemBankSizeEightByte);

        cudaEvent_t cycleDone;
        cudaEventCreate(&cycleDone);
//            cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
//    MyKernel <<<100, 512, 0, stream[i] >>> (outputDevPtr + i * size, inputDevPtr + i * size, size);
 //   cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);

    //    cudaHostRegister(hostsrc, sizeof(double)*blockmax * blocksize * blocksize, 0);
        error = cudaMemcpyAsync(d_mem,hostsrc,sizeof(double)*blockmax * blocksize * blocksize,cudaMemcpyHostToDevice,cpustream);
                       cudaEventRecord(cycleDone, cpustream);
                       cudaEventSynchronize(cycleDone);

        cudaEvent_t launch[NUMSTREAM];
        for(i=0;i<NUMSTREAM;i++)
            cudaEventCreate(&(launch[i]));

        blockptrs[0] = 0;
        int lucounter = 0;
        int lower =0;
        int upper = 0;
        int mulp = 0;
        int muln = 0;
        int sub = 0;
        int blk = 0;
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(start, cpustream);
        cudaEventSynchronize(start);
        int mlist[400];
        int mlistneg[400];
        int mlistsub[400];

        for(i=0;i<maxstage*2+2;i+=2)
       {
//           std::cout<<i/2<<": start: "<<stagerange[i] <<" end: "<<stagerange[i+1]<<std::endl;
               cudaDeviceSynchronize();

            if(stagemax[i/2] >= gpumax){
                std::cout<<"clearing: "<<tbd.size()<<" blocks"<<" stage: "<<i/2<<" maxstage: "<<maxstage<<std::endl;
                std::sort(tbd.begin(),tbd.end());//,std::greater<std::vector<int>>());
                int tbdcur = 0;
                for(int uu=0;uu<tbd.size();uu++){
                    if(tbd[uu]==tbdcur){
                        tbd[uu] = 0;
                    }
                    else{

                        tbdcur = tbd[uu];
                    }
                }
                std::sort(tbd.begin(),tbd.end());//,std::greater<std::vector<int>>());
                for(int uu=0;uu<tbd.size();uu++){
                    if(tbd[uu] == 0) tbdcur = uu;
                }
                tbdcur += 1;
                std::cout<<tbdcur<<" before: "<<tbd[tbdcur-3]<<"  "<<tbd[tbdcur-2]<<"  "<<tbd[tbdcur-1]<<"  "<<" after: "
                            << tbd[tbdcur]<<"  "<< tbd[tbdcur + 1]<<"  "<< tbd[tbdcur + 2]<<" last:  "
                            << tbd[tbd.size()-1]<<" gpumax: "<<gpumax<<" ";
                for(int uu=tbdcur;uu<tbd.size();uu+=12){
                    for(w = 0; w<12; w++)
                        tbc[w] = 0;
                    for(w=0;w<12 && uu+w<tbd.size(); w += 1){
                        tbc[w] = tbd[uu+w];
                    }
                        cudaresetparam4(d_mem, tbc[0],tbc[1], tbc[2], tbc[3],tbc[4], tbc[5],
                                           tbc[6],tbc[7], tbc[8], tbc[9],tbc[10], tbc[11],gstream[(uu/12)%NUMSTREAM]);
                }
                cudaDeviceSynchronize();
                
                 int rcount = 0;
                 for (operation* o : data::seq)
                 {
                     if(o->stage * 2 < i) continue;
                     if(o->src>0 && blockptrs[o->src] >= gpumax){
                         
              //           std::cout<<" not here stage: "<<o->stage<<" src "<<tbdcur+rcount<<std::endl;
                         blockptrs[o->src] = tbd[tbdcur+rcount];
                         tbd[tbdcur+rcount] = 0;
                         rcount++;
                     }
                     if(o->src2>0 && blockptrs[o->src2] >= gpumax && o->op != blockOp::lu &&
                                 o->op != blockOp::lowerInv && o->op != blockOp::upperInv){
               //          std::cout<<" not here stage: "<<o->stage<<" src "<<tbdcur+rcount<<std::endl;
                         blockptrs[o->src2] = tbd[tbdcur+rcount];
                         tbd[tbdcur+rcount] = 0;
                         rcount++;
                     }

                     if(o->result>0 && blockptrs[o->result] >= gpumax){
                         blockptrs[o->result] = tbd[tbdcur+rcount];
                         tbd[tbdcur+rcount] = 0;
                         rcount++;
                     }
                     if(o->result2>0 && blockptrs[o->result2] >= gpumax &&  o->op == blockOp::lu){
                         blockptrs[o->result2] = tbd[tbdcur+rcount];
                         tbd[tbdcur+rcount] = 0;
                         rcount++;
                     }
                     if(rcount>=tbd.size()-tbdcur){
                         std::cout<<" fixed to stage: "<<o->stage<<std::endl;
                         break;
                     }
                 }
                 std::cout<<"updated: "<<rcount <<" blocks  "<<"  tbd unique : "<< tbd.size()-tbdcur<<" gpumax: "<<gpumax<<std::endl;
                     tbd.clear();
  //           std::cout<<"stage max before update: ["<<i/2<<"]: "<<stagemax[i/2]<<" next "<<stagemax[i/2+1] <<" next "<<stagemax[i/2+2]<<" next "<<stagemax[i/2+3] <<" tdb: "<<tbd.size()<<std::endl;
        for(w=i;w<maxstage*2+2;w+=2)
        {
            stagemax[w/2] = 0;
            for(int s=stagerange[w]; s<=stagerange[w+1]; s++){
                operation* o1 = data::seq[s];
                if(o1->stage * 2 != w) continue;
                if(o1->src>0 && blockptrs[o1->src] > stagemax[w/2]){
                    stagemax[w/2] = blockptrs[o1->src];
                }
                if(o1->src2>0 && blockptrs[o1->src2]  > stagemax[w/2] && o1->op != blockOp::lu && 
                                 o1->op != blockOp::lowerInv && o1->op != blockOp::upperInv){
                    stagemax[w/2] = blockptrs[o1->src2];
                }

                if(o1->result>0 && blockptrs[o1->result] >stagemax[w/2]){
                    stagemax[w/2] = blockptrs[o1->result];
                }
                if(o1->result2>0 && blockptrs[o1->result2] >stagemax[w/2] && o1->op == blockOp::lu){
                    stagemax[w/2] = blockptrs[o1->result2];
                }
            }
        }
   //          std::cout<<"cur stage max after: "<<stagemax[i/2]<<" next "<<stagemax[i/2+1] <<" next "<<stagemax[i/2+2]<<" next "<<stagemax[i/2+3] <<std::endl;
               if(stagemax[i/2] >= gpumax){
                   std::cout<<"out of memory: "<<mem_size/1024/1024<<" MB at stage:"<<i/2<<std::endl;
                   exit(0);
               }

//               cudaDeviceSynchronize();
            }


            int stagecpu = 0;
            int stagegpu = 0;
            for(int s=stagerange[i]; s<=stagerange[i+1]; s++){
               operation* p = data::seq[s];
               if(p->stage * 2 != i ) continue;
               blk = p->result % NUMSTREAM;
               int scanned = 0;
               int cudaMulCountsub = 0;
               while(cudaMulCountsub<100 ){
                 switch (p->op)
                 {
                   case blockOp::sub:

                       mlistsub[ cudaMulCountsub * 4 ] = 3;
                       mlistsub[ cudaMulCountsub * 4 + 1] = blockptrs[p->src2];
                       mlistsub[ cudaMulCountsub * 4 + 2] = blockptrs[p->src];
                       mlistsub[ cudaMulCountsub * 4 + 3] = blockptrs[p->result];
                       cudaMulCountsub++;
                       sub++;
                       stagegpu += 1;
                       break;
                   case blockOp::lowerInv:
                       cudainvlower(d_mem, blockptrs[p->src], blockptrs[p->result], gstream[p->result % NUMSTREAM]);
/*
                       cudaMemcpyAsync(srcbuf, d_mem+blockptrs[p->src]*blocksizesq,sizeof(double)*blocksizesq,cudaMemcpyDeviceToHost, cpustream);
                       cudaEventRecord(cycleDone, cpustream);
                       cudaEventSynchronize(cycleDone);
                       resetMem(resultbuf, blocksizesq);
                       MatrixStdDouble::inv_lower(srcbuf, data::blockSize, resultbuf);
                       cudaMemcpyAsync(d_mem+blockptrs[p->result]*blocksizesq,resultbuf,sizeof(double)*blocksizesq,cudaMemcpyHostToDevice, cpustream);
            /* */          lower++;
                       stagecpu++;
                       break;
                   case blockOp::upperInv:
                       cudainvupper(d_mem, blockptrs[p->src], blockptrs[p->result], gstream[p->result % NUMSTREAM]);
/*
                       cudaMemcpyAsync(srcbuf,d_mem+blockptrs[p->src]*blocksizesq,sizeof(double)*blocksizesq,cudaMemcpyDeviceToHost, cpustream);
                       cudaEventRecord(cycleDone, cpustream);
                       cudaEventSynchronize(cycleDone);
                       resetMem(resultbuf, blocksizesq);
                       MatrixStdDouble::inv_upper(srcbuf, data::blockSize, resultbuf);
                       cudaMemcpyAsync(d_mem+blockptrs[p->result]*blocksizesq,resultbuf,sizeof(double)*blocksizesq,cudaMemcpyHostToDevice, cpustream);
            /* */           upper++;
                       stagecpu++;
                       break;
         //          case blockOp::lu:
         //              cudaLU64(d_mem,blockptrs[p->src],blockptrs[p->result],blockptrs[p->result2],gstream[p->result % NUMSTREAM]);
         //              break;
                   default:
                       break;
                 }
                       scanned++;
                       if(s + scanned>stagerange[i+1])
                           break;
                       p = data::seq[s+scanned];
               };

               int* sslist = mlistsub;
               int  cudasslC = cudaMulCountsub;
               if(cudasslC>0){
                    while(cudasslC>=8){
                        opparam param;
                        for(int pp=0;pp<8;pp++)
                        {
                            param.a[pp] = sslist[pp*4+1];
                            param.b[pp] = sslist[pp*4+2];
                            param.c[pp] = sslist[pp*4+3];
                        }
                        cudasubparam8(d_mem,param,gstream[blk]);
                        cudasslC -= 8;
                        sslist += 32;
                    };
                    if(cudasslC>=4){
                        opparam param;
                        for(int pp=0;pp<4;pp++)
                        {
                            param.a[pp] = sslist[pp*4+1];
                            param.b[pp] = sslist[pp*4+2];
                            param.c[pp] = sslist[pp*4+3];
                        }
                        cudasubparam4(d_mem,param,gstream[blk]);
                        cudasslC -= 4;
                        sslist += 16;
                    }
                    if(cudasslC == 1){
                        cudasubparam1(d_mem, sslist[1], sslist[2], sslist[3], gstream[blk]);
                    }
                    else if(cudasslC == 2){
                        opparam param;
                        for(int pp=0;pp<2;pp++)
                        {
                            param.a[pp] = sslist[pp*4+1];
                            param.b[pp] = sslist[pp*4+2];
                            param.c[pp] = sslist[pp*4+3];
                        }
                        cudasubparam2(d_mem,param,gstream[blk]);
                    }
                    else if(cudasslC == 3){
                        opparam param;
                        for(int pp=0;pp<3;pp++)
                        {
                            param.a[pp] = sslist[pp*4+1];
                            param.b[pp] = sslist[pp*4+2];
                            param.c[pp] = sslist[pp*4+3];
                        }
                        cudasubparam3(d_mem,param,gstream[blk]);
                    }
               }

               s = s + scanned - 1;
            }
            for(int s=stagerange[i]; s<=stagerange[i+1]; s++){
               operation* p = data::seq[s];
               if(p->stage * 2 != i ) continue;
               blk = p->result % NUMSTREAM;
               int scanned = 0;
               int cudaMulCount = 0;
               int cudaMulCountneg = 0;
               while((p->result % NUMSTREAM) == blk && cudaMulCount<100 
                                   && cudaMulCountneg<100 ){
                 switch (p->op)
                 {
                   case blockOp::mul:
                       mlist[ cudaMulCount * 4 ] = 1;
                       mlist[ cudaMulCount * 4 + 1] = blockptrs[p->src];
                       mlist[ cudaMulCount * 4 + 2] = blockptrs[p->src2];
                       mlist[ cudaMulCount * 4 + 3] = blockptrs[p->result];
                       cudaMulCount++;
                       mulp++;
                       stagegpu += 1;
                       break;
                   case blockOp::mulneg:
                       mlistneg[ cudaMulCountneg * 4 ] = 2;
                       mlistneg[ cudaMulCountneg * 4 + 1] = blockptrs[p->src];
                       mlistneg[ cudaMulCountneg * 4 + 2] = blockptrs[p->src2];
                       mlistneg[ cudaMulCountneg * 4 + 3] = blockptrs[p->result];
                       cudaMulCountneg++;
                       muln++;
                       stagegpu += 1;
                       break;
                   default:
                       break;
                 }
                 if (p->src > 0)
                {
                    if(data::laststage[p->src] == p->stage)
                    {
                        tbd.push_back(blockptrs[p->src]);
                    }
                }
                if (p->src2 > 0)
                {
                    if (data::laststage[p->src2] == p->stage){
                    if(p->op == blockOp::sub || p->op == blockOp::mul || p->op == blockOp::mulneg)
                    {
                        tbd.push_back(blockptrs[p->src2]);
                    }
                    else{
                         std::cout<<"**** not here stage: "<<p->stage<<" src2 "<<p->src2<<" "<<p->op<<std::endl;
                    }
                    }
                }
                       scanned++;
                       if(s + scanned>stagerange[i+1])
                           break;
                       p = data::seq[s+scanned];
               };

               int* mmlist = mlist;
               int  cudaMulC = cudaMulCount;
               if(cudaMulC>0){
                    while(cudaMulC>=8){
                        opparam param;
                        for(int pp=0;pp<8;pp++)
                        {
                            param.a[pp] = mmlist[pp*4+1];
                            param.b[pp] = mmlist[pp*4+2];
                            param.c[pp] = mmlist[pp*4+3];
                        }
                        cudamulparam8(d_mem,param,gstream[blk]);
                        cudaMulC -= 8;
                        mmlist += 32;
                    };
                    if(cudaMulC>=4){
                        opparam param;
                        for(int pp=0;pp<4;pp++)
                        {
                            param.a[pp] = mmlist[pp*4+1];
                            param.b[pp] = mmlist[pp*4+2];
                            param.c[pp] = mmlist[pp*4+3];
                        }
                        cudamulparam4(d_mem,param,gstream[blk]);
                        cudaMulC -= 4;
                        mmlist += 16;
                    }
                    if(cudaMulC == 1){
                        cudamulparam1(d_mem, mmlist[1], mmlist[2], mmlist[3], gstream[blk]);
                          
                    } 
                    else if(cudaMulC == 2){
                        opparam param;
                        for(int pp=0;pp<2;pp++)
                        {
                            param.a[pp] = mmlist[pp*4+1];
                            param.b[pp] = mmlist[pp*4+2];
                            param.c[pp] = mmlist[pp*4+3];
                        }
                        cudamulparam2(d_mem,param,gstream[blk]);
                    }else if(cudaMulC == 3){
                        opparam param;
                        for(int pp=0;pp<3;pp++)
                        {
                            param.a[pp] = mmlist[pp*4+1];
                            param.b[pp] = mmlist[pp*4+2];
                            param.c[pp] = mmlist[pp*4+3];
                        }
                        cudamulparam3(d_mem,param,gstream[blk]);
                    }
               }

               int* nnlist = mlistneg;
               int  cudaMulCneg = cudaMulCountneg;
               if(cudaMulCneg>0){
                    while(cudaMulCneg>=8){
                        opparam param;
                        for(int pp=0;pp<8;pp++)
                        {
                            param.a[pp] = nnlist[pp*4+1];
                            param.b[pp] = nnlist[pp*4+2];
                            param.c[pp] = nnlist[pp*4+3];
                        }
                        cudamulnegparam8(d_mem,param,gstream[blk]);
                        cudaMulCneg -= 8;
                        nnlist += 32;
                    };
                    if(cudaMulCneg>=4){
                        opparam param;
                        for(int pp=0;pp<4;pp++)
                        {
                            param.a[pp] = nnlist[pp*4+1];
                            param.b[pp] = nnlist[pp*4+2];
                            param.c[pp] = nnlist[pp*4+3];
                        }
                        cudamulnegparam4(d_mem,param,gstream[blk]);
                        cudaMulCneg -= 4;
                        nnlist += 16;
                    }
                    if(cudaMulCneg == 1){
                        cudamulnegparam1(d_mem, nnlist[1], nnlist[2], nnlist[3], gstream[blk]);

                    }
                    else if(cudaMulCneg == 2){
                        opparam param;
                        for(int pp=0;pp<2;pp++)
                        {
                            param.a[pp] = nnlist[pp*4+1];
                            param.b[pp] = nnlist[pp*4+2];
                            param.c[pp] = nnlist[pp*4+3];
                        }
                        cudamulnegparam2(d_mem,param,gstream[blk]);
                    }else if(cudaMulCneg == 3){
                        opparam param;
                        for(int pp=0;pp<3;pp++)
                        {
                            param.a[pp] = nnlist[pp*4+1];
                            param.b[pp] = nnlist[pp*4+2];
                            param.c[pp] = nnlist[pp*4+3];
                        }
                        cudamulnegparam3(d_mem,param,gstream[blk]);
                    }
               }


               s = s + scanned - 1;
            }

            for(int s=stagerange[i]; s<=stagerange[i+1]; s++){
               operation* p = data::seq[s];
               if(p->stage * 2 != i ) continue;
               switch (p->op)
               {
                   case blockOp::lowerInv:
                   //    cudainvlower(d_mem, blockptrs[p->src], blockptrs[p->result], gstream[p->result % NUMSTREAM]);
/*
                       cudaMemcpyAsync(srcbuf, d_mem+blockptrs[p->src]*blocksizesq,sizeof(double)*blocksizesq,cudaMemcpyDeviceToHost, cpustream);
                       cudaEventRecord(cycleDone, cpustream);
                       cudaEventSynchronize(cycleDone);
                       resetMem(resultbuf, blocksizesq);
                       MatrixStdDouble::inv_lower(srcbuf, data::blockSize, resultbuf);
                       cudaMemcpyAsync(d_mem+blockptrs[p->result]*blocksizesq,resultbuf,sizeof(double)*blocksizesq,cudaMemcpyHostToDevice, cpustream);
            /* */          lower++;
                       stagecpu++;
                       break;
                   case blockOp::upperInv:
                    //   cudainvupper(d_mem, blockptrs[p->src], blockptrs[p->result], gstream[p->result % NUMSTREAM]);
/*
                       cudaMemcpyAsync(srcbuf,d_mem+blockptrs[p->src]*blocksizesq,sizeof(double)*blocksizesq,cudaMemcpyDeviceToHost, cpustream);
                       cudaEventRecord(cycleDone, cpustream);
                       cudaEventSynchronize(cycleDone);
                       resetMem(resultbuf, blocksizesq);
                       MatrixStdDouble::inv_upper(srcbuf, data::blockSize, resultbuf);
                       cudaMemcpyAsync(d_mem+blockptrs[p->result]*blocksizesq,resultbuf,sizeof(double)*blocksizesq,cudaMemcpyHostToDevice, cpustream);
            /* */           upper++;
                       stagecpu++;
                       break;
                   case blockOp::lu:
                 //      cudaLU64(d_mem,blockptrs[p->src],blockptrs[p->result],blockptrs[p->result2],gstream[p->result % NUMSTREAM]);
                       cudaMemcpyAsync(srcbuf,d_mem+blockptrs[p->src]*blocksizesq,sizeof(double)*blocksizesq,cudaMemcpyDeviceToHost, cpustream);
                       cudaEventRecord(cycleDone, cpustream);
                       cudaEventSynchronize(cycleDone);
                       resetMem(resultbuf, blocksizesq);
                       resetMem(resultbuf2, blocksizesq);
                       MatrixStdDouble::ludcmpSimple(srcbuf, data::blockSize, resultbuf, resultbuf2);
                       cudaMemcpyAsync(d_mem+blockptrs[p->result]*blocksizesq,resultbuf,sizeof(double)*blocksizesq,cudaMemcpyHostToDevice, cpustream);
                       cudaMemcpyAsync(d_mem+blockptrs[p->result2]*blocksizesq,resultbuf2,sizeof(double)*blocksizesq,cudaMemcpyHostToDevice, cpustream);
            /* */           lucounter++;
                       stagecpu++;
                       break;
                   default:
                       break;
               }
            }
//        cudaEventRecord(stop, cpustream);
//        cudaEventSynchronize(stop);
 //       float msec = 0.0f;
  //      cudaEventElapsedTime(&msec, start, stop);
   //         printf("stage: %d     cpu: %d      gpu: %d    mul: %d  nmul: %d,  sub: %d  upper: %d  lower: %d   lu:%d  time: %g (msec)\n",
    //               i/2, stagecpu, stagegpu,mulp,muln, sub, upper, lower, lucounter, msec);
        }
        cudaEventRecord(stop, cpustream);
        cudaEventSynchronize(stop);
        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);
        double msecPerMatrixMul = msecTotal / (mulp + muln);
        double flopsPerMatrixMul = 2.0 * (double)data::blockSize * (double)data::blockSize * (double)data::blockSize;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        double memaccess = 8.0 * 4.0 * (double)data::blockSize * (double)data::blockSize;
        double gbps = memaccess * 1.0e-6f / msecPerMatrixMul;
    printf(
        "Performance= %.2f GFlop/s, unit Time= %.2f usec, cuda total time=%.3f s, Size= %.0f Ops, bandwith=%.2f GB/s\n",
        gigaFlops,
        msecPerMatrixMul*1000, msecTotal/1000,
        flopsPerMatrixMul, gbps
        );
std::cout<<"lu: "<< lucounter<<" lower: "<<lower<<" upper: "<<upper<<" sub: "<< sub<<" mul: "<<mulp<<" nmul: "<<muln<<std::endl;
       
       int bumin = count;
       int bucount = 0;
       int bumax =0;
       int buptrmax = 0;
       int bumaxstage=0;
       int buminstage=count;
       for (i = 0; i < blockrows*blockrows; i++) {
           if (bl[i] > 0 && blockptrs[bl[i]] >0){
               if(bumin > blockptrs[bl[i]])
                   bumin = blockptrs[bl[i]];
               if(data::laststage[bl[i]] < maxstage){
                   std::cout<<bl[i]<<"not here "<<i<<" "<<data::laststage[bl[i]]<<" "<<maxstage<<std::endl;
               }
               bucount++;
               if(bl[i]>bumax) bumax=bl[i];
               if(blockptrs[bl[i]]>buptrmax) buptrmax = blockptrs[bl[i]];
               if(data::laststage[bl[i]] > bumaxstage) bumaxstage = data::laststage[bl[i]];
               if(data::laststage[bl[i]] < buminstage) buminstage = data::laststage[bl[i]];
           }
       }
       std::cout<<"bumin: "<<bumin<<" bucount: "<<bucount<<" bu storage:" <<bucount/32<<"MB bumax: "<<bumax<<" buptrmax: "
                     <<buptrmax<<" bumaxlaststage: "<<bumaxstage<<" buminlaststage: "<<buminstage<<" transfer: "<<(buptrmax-bumin+1)/32<<std::endl;
        if(bxcount+blcount<buptrmax-bumin+1){
            std::cout<<"device memory out of order: "<<(bxcount+blcount)/1024/1024<<" MB"<<std::endl;
            exit(0);
        }
   //     cudaHostAlloc((void**)&host_mem, mem_size, cudaHostAllocDefault);

       cudaMemcpyAsync(hostsrc, 
d_mem + bumin*blocksizesq,(buptrmax-bumin+1)*blocksizesq*sizeof(double),cudaMemcpyDeviceToHost,cpustream);
       cudaDeviceSynchronize();

       int dsrc[16];
       double* dest[16];
       int srccount = 0;
       for(j=0;j<16;j++) dsrc[j] = 0;
       for (i = 0; i < blockrows*blockrows; i++) {
           
           if (bl[i] > 0){
               dsrc[srccount] = bl[i];
               dest[srccount] = data::newalignedblock();
               data::blockstorage[bl[i]] = dest[srccount];
               srccount++;
           }
               
           if((srccount>=16 || i== blockrows*blockrows-1)&&srccount>0){
#pragma omp parallel for
             for(int jj=0;jj<srccount;jj++){
               double* ptr = hostsrc + (blockptrs[dsrc[jj]]-bumin)*blocksizesq;
               std::memcpy(dest[jj], ptr,sizeof(double)*blocksizesq);
           //    for(int ii=0;ii<data::blockSize * data::blockSize;ii++){
            //                dmp[ii] = ptr[ii];
             //  }
             }
             
       srccount = 0;
       for(j=0;j<16;j++) dsrc[j] = 0;
           }
       }

       count = bumin+1;
       int copyCount = 0;
       opparam param;
       for(int pp=0;pp<12;pp++)
       {
           param.a[pp] = 0;
           param.b[pp] = 0;
       }
       for (i = 0; i < blockrows*blockrows; i++) {
           if (bu[i] > 0 && blockptrs[bu[i]] >0){
               param.a[copyCount] = blockptrs[bu[i]];
               param.b[copyCount] = count;
               blockptrs[bu[i]] = count;
               copyCount++;
               count++;
           }
           if(copyCount>0 && (copyCount>=12 || i == blockrows*blockrows -1)){
               opparam par;
               for(int pp=0;pp<12;pp++) 
               {   
                   par.a[pp] = param.a[pp]; 
                   par.b[pp] = param.b[pp];
               }
               cudacopyparam12(d_mem, par, gstream[par.a[0]%NUMSTREAM]);
               for(int pp=0;pp<12;pp++)
               {
                   param.a[pp]=0;
                   param.b[pp]=0;
               }
               copyCount=0;
           }
       }
       cudaDeviceSynchronize();

       buptrmax = bumin;
       bumin = count;
       bucount = 0;
       for (i = 0; i < blockrows*blockrows; i++) {
           if (bu[i] > 0 && blockptrs[bu[i]] >0){
               if(bumin > blockptrs[bu[i]])
                   bumin = blockptrs[bu[i]];
               if(data::laststage[bu[i]] < maxstage){
                   std::cout<<"not here "<<i<<" "<<data::laststage[bu[i]]<<" "<<bu[i]<<" "<<maxstage<<std::endl;
               }
               bucount++;
               if(bu[i]>bumax) bumax=bu[i];
               if(blockptrs[bu[i]]>buptrmax) buptrmax = blockptrs[bu[i]];
               if(data::laststage[bu[i]] > bumaxstage) bumaxstage = data::laststage[bu[i]];
               if(data::laststage[bu[i]] < buminstage) buminstage = data::laststage[bu[i]];
           }
       }
       std::cout<<"bumin: "<<bumin<<" bucount: "<<bucount<<" bu storage:" <<bucount/32<<"MB bumax: "<<bumax<<" buptrmax: "
                     <<buptrmax<<" bumaxlaststage: "<<bumaxstage<<" buminlaststage: "<<buminstage<<
                     " transfer: "<<(buptrmax-bumin+1)/32<<std::endl;

       cudaMemcpyAsync(hostsrc, 
d_mem + bumin*blocksizesq,(buptrmax-bumin+1)*blocksizesq*sizeof(double),cudaMemcpyDeviceToHost,cpustream);
       cudaDeviceSynchronize();
    //    cudaHostUnregister(hostsrc);

       srccount = 0;
       for(j=0;j<16;j++) dsrc[j] = 0;
       for (i = 0; i < blockrows*blockrows; i++) {
           
           if (bu[i] > 0){
               dsrc[srccount] = bu[i];
               dest[srccount] = data::newalignedblock();
               data::blockstorage[bu[i]] = dest[srccount];
               srccount++;
           }
               
           if((srccount>=16 || i== blockrows*blockrows-1)&&srccount>0){
#pragma omp parallel for
             for(int jj=0;jj<srccount;jj++){
               double* ptr = hostsrc + (blockptrs[dsrc[jj]]-bumin)*blocksizesq;
               std::memcpy(dest[jj], ptr,sizeof(double)*blocksizesq);
           //    for(int ii=0;ii<data::blockSize * data::blockSize;ii++){
            //                dmp[ii] = ptr[ii];
             //  }
             }
             
       srccount = 0;
       for(j=0;j<16;j++) dsrc[j] = 0;
           }
       }
/*
       for (i = 0; i < blockrows*blockrows; i++) {
           
           if (bu[i] > 0){
               double* dmp = data::newalignedblock();
               double* ptr = hostsrc + (blockptrs[bu[i]]-bumin)*blocksizesq;
               std::memcpy(dmp, ptr,sizeof(double)*blocksizesq);
            //   for(int ii=0;ii<data::blockSize * data::blockSize;ii++){
             //               dmp[ii] = ptr[ii];
              // }
               data::blockstorage[bu[i]] = dmp;
           }
       }
*/
        cudaFreeHost(srcbuf);
        cudaFreeHost(srcbuf2);
        cudaFreeHost(resultbuf);
        cudaFreeHost(resultbuf2);
        free(hostsrc);

        cudaStreamDestroy(cpustream);
        cudaEventDestroy(cycleDone);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        for(i=0;i<NUMSTREAM;i++)
            cudaStreamDestroy(gstream[i]);
       cudaFree(d_mem);
       free(blockptrs);
       free(stat);
       free(stagerange);
      
    }
}


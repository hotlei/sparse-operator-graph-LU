/*
    This file is part of sparse-operator-graph-LU.
    Copyright (C) 2020, 2021 Lei Yan (yan_lei@hotmail.com)

    sparse-operator-graph-LU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sparse-operator-graph-LU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sparse-operator-graph-LU.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <sys/mman.h>
#include "operation.h"
#include "matrix.h"
#include "data.h"
#include "memutil.h"
#include "MatrixStdDouble.h"
#include "BlockPlanner.h"
#include "GPSOrder.h"
#include "config.h"

#define NUMTEAM 700
#define ALIGN64 64
namespace SOGLU
{
        ushort BlockPlanner::mask[16] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u,
                256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u};
        std::vector<int> BlockPlanner::tbd = {};
        long BlockPlanner::stream[NUMSTREAM*8192];
        int  BlockPlanner::streamcount[NUMSTREAM];
        std::vector<matrix*> BlockPlanner::blockstorageL2 = {};
        int BlockPlanner::storageCountL2 =0;
        std::vector<operation*> BlockPlanner::graphL2 = {};
        
        bool BlockPlanner::checkSymmetric(matrix* mat, int n, int blocksize, int* indexi, int* indexj, double* vals, int valcount)
        {
            double tiny = 2.0e-10;
            int expmask = data::blockSize-1;
            int exp = __builtin_popcount(data::blockSize-1);
	    for (int i = 0; i < data::valcount; i++) {
		int bi = data::indexi[i] >> exp;  
		int bj = data::indexj[i] >> exp; 
                uint64_t idx = mat[0][bj * data::blockRows + bi];
        	if(idx == 0) {
                    return false;
		}
		int ri = data::indexi[i] & expmask; 
		int rj = data::indexj[i] & expmask; 
                double * pb = data::blockstorage[idx];
                double diff = *(pb + ri * BLOCKCOL + rj) - data::vals[i];
                if(diff > tiny || diff<-tiny) {
                    return false;
                }
            }
            return true;
        }

        int addInput(matrix* a, int* lastuse)
        {
            int count = 0;
            if(a->level == 0 && a->blockindex >0){
                data::stage[a->blockindex] = 1;
                data::laststage[a->blockindex] = 1;
                lastuse[a->blockindex] = 0;
                count = 1;
            }else if(a->level > 0){
                for(int i=0; i<4; i++)
                    if(a->submatrix[i] != NULL)
                        count += addInput(a->submatrix[i], lastuse);
            }
            return count;
        }
        int addOutput(matrix* a, int* lastuse, int marker)
        {
            int count = 0;
            if(a->level == 0 && a->blockindex >0){
                data::laststage[a->blockindex] = marker;
                lastuse[a->blockindex] = marker;
                count = 1;
            }else if(a->level > 0){
                for(int i=0; i<4; i++)
                    if(a->submatrix[i] != NULL)
                        count += addOutput(a->submatrix[i], lastuse, marker);
            }
            return count;
        }
 
        void printOP()
        {
            for(int i = 0; i<data::graph.size(); i++)
            {
                operation* o = data::graph[i];
                printf("stage %8d  %8d  %8d src %8d %8d op %4d result %8d %8d \n",
                     o->stage, o->groupNum, o->sequenceNum, o->src, o->src2, o->op, o->result, o->result2);
            }
        }

        void BlockPlanner::blockPlan(matrix* saved, matrix* savedu, matrix* lhs)
        {
            long i, cnt = 0;

            int *lastuse = new int[data::storageCount];
            for(i=0;i<data::storageCount; i++){
                lastuse[i] = 0;
            }

            data::stage.reset(new int[data::storageCount]);
            data::laststage.reset(new int[data::storageCount]);
            int maxx = data::storageCount;
            int maxcnt = 0;
            for(i=0;i<data::storageCount;i++){
                lastuse[i] = 0;
                data::stage[i] = 0;
                data::laststage[i] = 0;
            }

            data::stage[0] = 1;
            cnt = addInput(data::blocks, lastuse);

            if(lhs != NULL) {
                cnt += addInput(lhs, lastuse);
            }

            if (saved != NULL) {
                maxcnt += addOutput(saved, lastuse, maxx);
            }

            if (savedu != NULL) {
                maxcnt += addOutput(savedu, lastuse, maxx);
            }

            for (i = data::graph.size() - 1; i >= 0; i--)
            {
                operation* o = data::graph[i];

                int stg = 0;
                if (o->result > 0)
                {
                    stg = o->result;
                    if (o->result2 > 0 && o->result2 > stg)
                    {
                        stg = o->result2;
                    }
                }

                if (o->result > 0 && o->result2 > 0)
                {
                    if (lastuse[o->result] == 0 && lastuse[o->result2] == 0)
                        continue;
                }

                if (o->result > 0 && o->result2 == 0)
                {
                    if (lastuse[o->result] == 0)
                        continue;
                }

                if (stg > 0)
                {
                    if (o->src > 0){
                        if (lastuse[o->src] < stg)
                            lastuse[o->src] = stg;
                    }
                    if (o->src2 > 0)
                    {
                        if(o->src2>= data::storageCount){
                           continue;
                        }
                        if (lastuse[o->src2] < stg)
                            lastuse[o->src2] = stg;
                    }
                }
            }

            int wastecount = 0;

            for (i = 0; i < data::graph.size(); i++)
            {
                operation* o = data::graph[i];
                if (lastuse[o->result] > 0) continue;
                if (o->result2 > 0 && lastuse[o->result2] > 0) continue;
                ++wastecount;
            }
        if(data::PlanL2 || wastecount>(data::graph.size()/25)){ // /25
            int bufcount = data::graph.size() - wastecount;
            operation *buf = new operation[bufcount];
            int bufcounter = 0;
            for (i = 0; i < data::graph.size(); i++)
            {
                operation* o = data::graph[i];
                
                if (lastuse[o->result] > 0)
                {
                    buf[bufcounter].sett(o->src, o->src2, o->op, o->result, o->result2);
                    buf[bufcounter].sequenceNum = o->sequenceNum;
                    buf[bufcounter].groupNum = o->groupNum;
                    bufcounter++;
                    continue;
                }
                if (o->result2 > 0 && lastuse[o->result2] > 0)
                {
                    buf[bufcounter].sett(o->src, o->src2, o->op, o->result, o->result2);
                    buf[bufcounter].sequenceNum = o->sequenceNum;
                    buf[bufcounter].groupNum = o->groupNum;
                    bufcounter++;
                    continue;
                }
                
            }

            data::clearOperations();
            for(i=0; i< bufcount;i++){
                operation* nop= memutil::newoperation(buf[i].src,buf[i].src2, buf[i].op, buf[i].result, buf[i].result2,0,buf[i].groupNum);
                nop->sequenceNum = buf[i].sequenceNum; 
                data::graph.push_back(nop);
            }
            delete[] buf;
        }

            std::cout<<"reduced ops to: " + std::to_string(data::graph.size())<<std::endl;

            int maxstg = 1;
            for (i = 0; i < data::graph.size(); i++)
            {
                operation* o = data::graph[i];
                int stg = 0;
                if (o->src > 0)
                {
                    stg = data::stage[o->src] + 1;
                    if(stg < 2) stg = 2;
                    if (o->src2 > 0)
                    {
                        if (data::stage[o->src2] + 1 > stg)
                        {
                            stg = data::stage[o->src2] + 1;
                        }
                    }
                }else{

                    if (o->src2 > 0)
                    {
                        if (data::stage[o->src2] + 1 > stg)
                        {
                            stg = data::stage[o->src2] + 1;
                        }
                    }
                    if(stg < 2) stg = 2;
                }

                if (stg > 1)
                {
                    o->stage = stg;
                    if (o->result > 0 && data::stage[o->result] < stg)
                    {
                        data::stage[o->result] = stg;
                    }
                    if (o->result2 > 0 && data::stage[o->result2] < stg)
                    {
                        data::stage[o->result2] = stg;
                    }
                    if (stg > maxstg)
                        maxstg = stg;
                }
            }

            // fix stg
            for(i=data::graph.size()-1; i>=0; i--){
                operation* o = data::graph[i];
                if(o->result2 > 0) continue;
                if(o->result > 0 && data::stage[o->result] > o->stage){
                    o->stage = data::stage[o->result];
                }
            }

            std::vector<std::vector<operation*>> stagelist;
            int  liststage[NUMTEAM];
            long listptr[NUMTEAM];
            for(int si = 0;si<NUMTEAM;si++){
                 std::vector<operation*> onelist;
                 stagelist.push_back(onelist);
            }
            for(long i=0;i< data::graph.size(); i++){
                 operation* o = data::graph[i];
                 stagelist[o->stage%NUMTEAM].push_back(o);
            }
     
#pragma omp parallel for
        for(int si=0;si<NUMTEAM; si++){
             std::sort (stagelist[si].begin(), stagelist[si].end(), operationCompare);
        }
         
            data::graph.clear();
            int curstream = -1;
            int curstage = maxstg+1;
            for(int si = 0;si<NUMTEAM;si++){
                liststage[si] = maxstg+1;
                if(stagelist[si].size()>0)
                    liststage[si] = stagelist[si][0]->stage;
                listptr[si] = 0;
                if(liststage[si] < curstage){
                    curstage = liststage[si];
                    curstream = si;
                }
            }
            while(curstage <= maxstg){
                long li = 0;
                for(li = listptr[curstream]; li< stagelist[curstream].size(); li++){
                    operation* o = stagelist[curstream][li];
                    if(o->stage == curstage){
                        data::graph.push_back(o);
                    }else{
                        liststage[curstream] = o->stage;
                        listptr[curstream] = li;
                        break;
                    }
                }
                listptr[curstream] = li;
                curstream = -1;curstage = maxstg+1;
                for(int si = 0;si<NUMTEAM;si++){
                    if(listptr[si] >= stagelist[si].size()) continue;
                    if(liststage[si] < curstage){
                        curstage = liststage[si];
                        curstream = si;
                    }
                }
            };

            for(int si = 0;si<NUMTEAM;si++){
                 stagelist[si].clear();
            }
	    
            int jump = 0;
            int band = 0;
            int stgc = 0;
/* */
            for (i = 0; i < data::graph.size(); i++)
            {
                operation* o = data::graph[i];
                if(o->stage > stgc){
                    stgc = o->stage;
                    band = 0;
                }

                if(band > 8000){
                    jump++;
                    band = 0;
                }
                
                o->stage += jump;
                band++;
            }
/* */
            for (i = 0; i < data::graph.size(); i++)
            {
                operation* o = data::graph[i];
                int stg = o->stage;

                if (o->src > 0 && data::stage[o->src] >=0)  // != 1)   //????
                {
                    if (data::laststage[o->src] < stg)
                        data::laststage[o->src] = stg;
                }
                    if (o->src2 > 0 && data::stage[o->src2] >=0 )
                    {
                        if (data::laststage[o->src2] < stg)
                            data::laststage[o->src2] = stg;
                    }
            }

            delete [] lastuse;
        }

          void BlockPlanner::calculate()
        {
            int count = 0;
         
            int curStage = -1;
            int blk;


            int maxstage = data::graph[data::graph.size()-1]->stage;
            long* stagerange = (long*) malloc((maxstage*2 + 2)*sizeof(long));
            for(long i=0;i<maxstage*2 + 2;i+=2){
                stagerange[i] = data::graph.size();
                stagerange[i+1] = 0;
            }
            for(long li=0;li<data::graph.size();li++){
                int st = data::graph[li]->stage;
                if(stagerange[st*2] > li) stagerange[st*2] = li;
                if(stagerange[st*2+1] < li) stagerange[st*2+1] = li;
            }



        for(int w=0;w<maxstage*2+2;w+=2){

            for(int si=0; si<NUMSTREAM; si++)
                streamcount[si] = 0;
            int lustream = 0;
            for (long i=stagerange[w]; i<=stagerange[w+1]; i++){
                operation* p = data::graph[i];
                if(p->stage * 2 != w ) continue;
                if(p->skip) continue;

                blk = p->result % NUMSTREAM;
                if(p->op == blockOp::lu) {
                    blk = lustream;
                    lustream++;
                    lustream = lustream % NUMSTREAM;
                }
              
                stream[blk * 8192 + streamcount[blk]] = i;
                streamcount[blk] += 1;
            }
            for(int si=0; si<NUMSTREAM; si++)
                if(streamcount[si] >= 8192) 
                     std::cout<<"cache overflow -- stage "<<w<<" range "<<stagerange[w+1] - stagerange[w]<<" max "<<streamcount[si] <<std::endl;
            
//#pragma omp parallel 
{ 
#pragma omp parallel for schedule(dynamic) 
        for(int si=0;si<NUMSTREAM; si++){
            int tid = omp_get_thread_num();
            for(int ci=0;ci<streamcount[si];ci++){
                operation o = *(data::graph[stream[si*8192+ci]]);
                int multimul = 1;
  //     printf("%8d  %8d  %8d src %8d %8d op %4d result %8d %8d thread %4d \n",
  //          o.stage, o.groupNum, o.sequenceNum, o.src, o.src2, o.op, o.result, o.result2, tid);
                if (o.result > 0)
                {
                    if (data::blockstorage[o.result] == NULL)
                    {
                        double* dmp = (double *) memutil::newalignedblock(tid,8);
 
                        if(dmp == NULL)
                                 std::cout<<"outof memory"<<std::endl;
                        
                        if(o.op == blockOp::mult || o.op == blockOp::llt)
                            resetOneBlock(dmp);
                        else
                            resetOneBlock(dmp);
                        
                        appendBlockStorageAgain(dmp, o.result);
                    }
                }
                if(o.op == blockOp::lu)
                if (o.result2 > 0)
                {
                    if (data::blockstorage[o.result2] == NULL)
                    {
                        double* dmp = (double *) memutil::newalignedblock(tid,8);
                        resetOneBlock(dmp);
                        appendBlockStorageAgain(dmp, o.result2);
                    }
                }

                switch (o.op)
                {
                    case blockOp::inv:
                        MatrixStdDouble::mat_inv(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        break;
                    case blockOp::lowerInv:
                        MatrixStdDouble::inv_lower(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        break;
                    case blockOp::mul:
                        MatrixStdDouble::blockMulOneAvxBlock(data::blockstorage[o.src], data::blockstorage[o.src2], 
                                data::blockstorage[o.result], 0, tid);
                        break;
                    case blockOp::mult:
                        if(o.src2 == o.src)
                            MatrixStdDouble::mat_mult(data::blockstorage[o.src], 
                                data::blockstorage[o.result], 0, tid);
                        else
                            MatrixStdDouble::mat_mult(data::blockstorage[o.src], data::blockstorage[o.src2],
                                data::blockstorage[o.result], 0, tid);
                        break;

                    case blockOp::mulneg:
                            MatrixStdDouble::blockMulOneAvxBlockNeg(data::blockstorage[o.src], data::blockstorage[o.src2], 
                                data::blockstorage[o.result], 0, tid);
                        break;
                    case blockOp::sub:
                        if (o.src2 > 0 && data::blockstorage[o.src2] != NULL && o.src > 0 && data::blockstorage[o.src] != NULL)
                        {
                            MatrixStdDouble::mat_sub(data::blockstorage[o.src2], data::blockstorage[o.src],
                                                         data::blockSize, data::blockstorage[o.result], tid);
                            break;
                        }

                        if(o.src2 > 0 && data::blockstorage[o.src2] != NULL){
                            MatrixStdDouble::mat_copy(data::blockstorage[o.src2], 
                                                         data::blockSize, data::blockstorage[o.result], tid);
                            break;
                        }
                       
                        if (o.src > 0 && data::blockstorage[o.src] != NULL)
                        {
                             MatrixStdDouble::mat_neg(data::blockstorage[o.src],
                                                         data::blockSize, data::blockstorage[o.result], tid);
                        }
                        break;
                    case blockOp::upperInv:
                        MatrixStdDouble::inv_upper(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        if(!MatrixStdDouble::inv_check_diag(data::blockstorage[o.src],data::blockstorage[o.result],data::blockSize)){
                            std::cout<<" upper out of tolerance "<<std::endl;
                        }
                        break;
                    case blockOp::lu:
                        MatrixStdDouble::mat_clean(data::blockstorage[o.src], BLOCK64);
                        MatrixStdDouble::ludcmpSimple(data::blockstorage[o.src], data::blockSize,
                                                     data::blockstorage[o.result], data::blockstorage[o.result2]);
                        break;

                    case blockOp::llt:
                        MatrixStdDouble::mat_clean(data::blockstorage[o.src], BLOCK64);
                        MatrixStdDouble::lltdcmpSimple(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        break;

                    default:
                        break;
                }
            }
        }
}

            for (long i=stagerange[w]; i<=stagerange[w+1]; i++){

                operation o = *(data::graph[i]);
                if (o.skip)
                    continue;
                count++;

                if(o.stage != curStage)
                {
                    if(tbd.size()>1)
                    std::sort (tbd.begin(), tbd.end());
                    int ddx = -1;
                    for (int idx = 0;idx <tbd.size(); idx++)
                    {
                        if(tbd[idx] == ddx)
                            continue;
                        ddx = tbd[idx];
                        if(data::blockstorage[tbd[idx]] != NULL){
                            memutil::freeblock(ddx%6,data::blockstorage[tbd[idx]]);
                            data::blockstorage[tbd[idx]] = NULL;
                        }
                    }
                    tbd.clear();
                    curStage = o.stage;
                }
                if (o.src > 0)
                {
                    if(data::laststage[o.src] == o.stage) {
                        tbd.push_back(o.src);
                    }
                }
                if (o.src2 > 0)
                {
                    if (data::laststage[o.src2] == o.stage) {
                        tbd.push_back(o.src2);
                    }
                }
            }
        }
            if(tbd.size()>1){
                    std::sort (tbd.begin(), tbd.end());
                    int ddx = -1;
                    for (int idx = 0;idx <tbd.size(); idx++) {
                        if(tbd[idx] == ddx)
                            continue;
                        ddx = tbd[idx];
                        if(data::blockstorage[tbd[idx]] != NULL){
                            memutil::freeblock(ddx%6,data::blockstorage[tbd[idx]]);
                            data::blockstorage[tbd[idx]] = NULL;
                        }
                    }
            }

            free(stagerange);
        }

         double* transposeblock(double* b)
        {
            double* rtn = memutil::newalignedblock(0, 8);
            std::memset(rtn, 0, ALLOCBLOCK);
            uint16_t *metab = (uint16_t*) (b + DETAILOFFSET);
            for(int i=0;i<BLOCK64;i++){
                for(int j=0;j<BLOCK64;j++){
                    uint16_t metai = metab[DETAILSKIPSHORT * (i/32) + i%32];
                    if(metai & BlockPlanner::mask[j/8]){
                        rtn[j*BLOCKCOL + i] = b[i*BLOCKCOL + j];
                    }
                }
            }
            return rtn;
        }

        void updateVector(matrix* bl, double* b, double* y, int n)
        {
            if(bl->level == 0){
                if(bl->blockindex > 0){
                    double* ldata = data::blockstorage[bl->blockindex];
                    uint16_t *metab = (uint16_t*) (ldata + DETAILOFFSET);
                    for(int i = 0; i < BLOCK64; i++)
                    {
                        uint16_t metai = metab[DETAILSKIPSHORT * (i/32) + i%32];
                        double sum8 = 0;
                        for(int j8 = 0; j8<BLOCK64/8; j8++){
                            if(metai & BlockPlanner::mask[j8]){
                                double sum = 0;
                                for(int k=0;k<8;k++){
                                    sum += ldata[i*BLOCKCOL+j8*8+k] * y[j8*8+k];
                                }
                                sum8 += sum;
                            }
                        }
                        b[i] -= sum8;
                    }
                }
                return;
            }
            int n2 = n / 2;
            if(bl->submatrix[0] != NULL)
                updateVector(bl->submatrix[0], b, y, n2);
            if(bl->submatrix[1] != NULL)
                updateVector(bl->submatrix[1], b, y+n2, n2);
            if(bl->submatrix[2] != NULL)
                updateVector(bl->submatrix[2], b+n2, y, n2);
            if(bl->submatrix[3] != NULL)
                updateVector(bl->submatrix[3], b+n2, y+n2, n2);
        }
        void updateVectorT(matrix* bl, double* b, double* y, int n)
        {
            if(bl->level == 0){
                if(bl->blockindex > 0){
                    double* ldata = transposeblock(data::blockstorage[bl->blockindex]);
                    for(int i = 0; i < BLOCK64; i++)
                    {
                        double sum8 = 0;
                        for(int j8 = 0; j8<BLOCK64/8; j8++){
                                double sum = 0;
                                for(int k=0;k<8;k++){
                                    sum += ldata[i*BLOCKCOL+j8*8+k] * y[j8*8+k];
                                }
                                sum8 += sum;
                        }
                        b[i] -= sum8;
                    }
                    memutil::freeblock(0,ldata);
                }
                return;
            }
            int n2 = n / 2;
            if(bl->submatrix[0] != NULL)
                updateVectorT(bl->submatrix[0], b, y, n2);
            if(bl->submatrix[2] != NULL)
                updateVectorT(bl->submatrix[2], b, y+n2, n2);
            if(bl->submatrix[1] != NULL)
                updateVectorT(bl->submatrix[1], b+n2, y, n2);
            if(bl->submatrix[3] != NULL)
                updateVectorT(bl->submatrix[3], b+n2, y+n2, n2);
        }
        void lowerSolver(matrix* bl, double* b, double* y, int n)
        {
            if(bl->level == 0){
                if(bl->blockindex > 0){
                    double* ldata = data::blockstorage[bl->blockindex];
                    uint16_t *metab = (uint16_t*) (ldata + DETAILOFFSET);
                    for(int i = 0; i < BLOCK64; i++)
                    {
                        uint16_t metai = metab[DETAILSKIPSHORT * (i/32) + i%32];
                        double sum8 = 0;
                        for(int j8 = 0; j8<=i/8; j8++){
                            if(metai & BlockPlanner::mask[j8]){
                                double sum = 0;
                                for(int k=0;k<8;k++){
                                    if(j8*8+k<i)
                                    sum += ldata[i*BLOCKCOL+j8*8+k] * y[j8*8+k];
                                }
                                sum8 += sum;
                            }
                        }
                        b[i] -= sum8;
                        y[i] = b[i] / ldata[i*BLOCKCOL+i];
                    }
                    
                }
                return;
            }
            int n2 = n / 2;
            lowerSolver(bl->submatrix[0], b, y, n2);
            if(bl->submatrix[2] != NULL)
                updateVector(bl->submatrix[2], b+n2, y, n2);
            lowerSolver(bl->submatrix[3], b+n2, y+n2, n2);
        }
        void upperSolverT(matrix* bu, double* b, double* x, int n)
        {
            if(bu->level == 0){
                if(bu->blockindex > 0){
                    double* ldata = transposeblock(data::blockstorage[bu->blockindex]);
                    for(int i = BLOCK64-1; i >= 0; i--)
                    {
                        double sum8 = 0;
                        for(int j8 = i/8; j8<BLOCK64/8; j8++){
                                double sum = 0;
                                for(int k=0;k<8;k++){
                                    if(j8*8+k>i)
                                    sum += ldata[i*BLOCKCOL+j8*8+k] * x[j8*8+k];
                                }
                                sum8 += sum;
                        }
                        b[i] -= sum8;
                        x[i] = b[i] / ldata[i*BLOCKCOL+i];
                    }
                    memutil::freeblock(0,ldata);
                }
                return;
            }
            int n2 = n / 2;
            upperSolverT(bu->submatrix[3], b+n2, x+n2, n2);
            if(bu->submatrix[2] != NULL)
                updateVectorT(bu->submatrix[2], b, x+n2, n2);
            upperSolverT(bu->submatrix[0], b, x, n2);
        }
        void upperSolver(matrix* bu, double* b, double* x, int n)
        {
            if(bu->level == 0){
                if(bu->blockindex > 0){
                    double* ldata = data::blockstorage[bu->blockindex];
                    uint16_t *metab = (uint16_t*) (ldata + DETAILOFFSET);
                    for(int i = BLOCK64-1; i >= 0; i--)
                    {
                        uint16_t metai = metab[DETAILSKIPSHORT * (i/32) + i%32];
                        double sum8 = 0;
                        for(int j8 = i/8; j8<BLOCK64/8; j8++){
                            if(metai & BlockPlanner::mask[j8]){
                                double sum = 0;
                                for(int k=0;k<8;k++){
                                    if(j8*8+k>i)
                                    sum += ldata[i*BLOCKCOL+j8*8+k] * x[j8*8+k];
                                }
                                sum8 += sum;
                            }
                        }
                        b[i] -= sum8;
                        x[i] = b[i] / ldata[i*BLOCKCOL+i];
                    }

                }
                return;
            }
            int n2 = n / 2;
            upperSolver(bu->submatrix[3], b+n2, x+n2, n2);
            if(bu->submatrix[1] != NULL)
                updateVector(bu->submatrix[1], b, x+n2, n2);
            upperSolver(bu->submatrix[0], b, x, n2);
        }
        void BlockPlanner::solve(matrix* bl, matrix* bu, double* b, int n)
        {
            double* x = (double *) aligned_alloc (64, n * data::blockSize *sizeof(double));
            double* y = (double *) aligned_alloc (64, n * data::blockSize *sizeof(double));
            for(int i=0;i<n* data::blockSize;i++){
                y[i] = 0;
                x[i] = 0;
            }
            lowerSolver(bl, b, y, n);
            if(data::symmetric)
                upperSolverT(bl, y, x, n);
            else
                upperSolver(bu, y, x, n);

            int nancount = 0;
            
            data::x = (double*)memutil::getSmallMem(1, sizeof(double) * data::mSize);
            for (int i = 0; i < data::mSize; i++) {
                data::x[i] = x[i];
                if (std::isnan(x[i])) {
                    nancount++;
                }
            }

            free(x);
            free(y);
            if(nancount>0)
                std::cout<<"found NaN:  " + std::to_string(nancount)<<std::endl;
        }

        void BlockPlanner::blockMatrixLU(matrix* a, matrix* l, matrix* u, int n, matrix* l3, matrix* u3, int group, matrix* x, matrix* y, matrix* lhs)
        {
            int hn = n / 2;
            double *tmp = NULL;
            if(a->level == 0){
                if(a->blockindex > 0){
                    if(l->blockindex <= 0){
                        l->blockindex = appendBlockStorage(tmp);
                    }
                    if(u->blockindex <= 0){
                        u->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(a->blockindex, 0, blockOp::lu, l->blockindex, u->blockindex, 0, group));
                }
                return;
            }
            matrix* ha = a->submatrix[0];
            matrix* l1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* u1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* l2 = l3;
            matrix* u2 = u3;
            if(l2 == NULL){
                l2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                u2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            }
            matrix* hl1 = NULL;
            matrix* hu1 = NULL;
            if(a->level>1){
                hl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
                hu1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
            }
            blockMatrixLU(ha,l1,u1,n/2,hl1,hu1,group,x->submatrix[0],y->submatrix[0],lhs->submatrix[0]);
            blockMatrixInvLower(l1,l2,n/2,hl1,group);
            blockMatrixInvUpper(u1,u2,n/2,hu1,group);
            l->submatrix[0] = l1;                // l2 ????
            u->submatrix[0] = u1;
           
            if(a->level == 1)
            {
                blockMatrixMul(l2,lhs->submatrix[0],y->submatrix[0],hn,group);
                blockMatrixMul(u2,y->submatrix[0],x->submatrix[0],hn,group);
            }

            matrix* aua1b = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* b = a->submatrix[1];
            if(b != NULL){
                matrix* a1b = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                blockMatrixMul(l2,b,aua1b,hn,group);
                u->submatrix[1] = aua1b;
                blockMatrixMulNeg(aua1b,x->submatrix[2],y->submatrix[0],hn,group);
            }

            matrix* dsub = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* d = a->submatrix[3];

            matrix* c = a->submatrix[2];
            if(c != NULL){
                matrix* ca1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                matrix* ca1al = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                blockMatrixMul(c,u2,ca1al,hn,group);
                l->submatrix[2] = ca1al;

        //    matrix* y2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
        //    matrix* b2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
        //    blockMatrixMul(ca1al,y->submatrix[0],y2,hn,group);
        //    blockMatrixSub(lhs->submatrix[2],y2,b2,hn,group);
                blockMatrixMulNeg(ca1al,y->submatrix[0],lhs->submatrix[2],hn,group);

                matrix* ca1b = memutil::newmatrix(0,a->blockrows2/2,a->level-1);

                blockMatrixMul(ca1al, aua1b, ca1b, hn,group);
                blockMatrixSub(d, ca1b, dsub, hn,group);
            } else {
                dsub = d;
            }

            l1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            u1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* dl1 = NULL;
            matrix* du1 = NULL;
            if(a->level>1){
                dl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
                du1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
            }
            blockMatrixLU(dsub,l1,u1,hn,dl1,du1,group,x->submatrix[2],y->submatrix[2],lhs->submatrix[2] );
            l->submatrix[3] = l1;
            u->submatrix[3] = u1;

            if(a->level == 1)
            {
             //   hl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
             //   hu1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
                blockMatrixInvLower(l1,l2,n/2,NULL,group);
                blockMatrixMul(l2,lhs->submatrix[2],y->submatrix[2],hn,group);
                blockMatrixInvUpper(u1,u2,n/2,NULL,group);
                blockMatrixMul(u2,y->submatrix[2],x->submatrix[2],hn,group);
            }
        }
        void BlockPlanner::blockMatrixLU(matrix* a, matrix* l, matrix* u, int n, matrix* l3, matrix* u3, int group)
        {
            int hn = n / 2;
            double *tmp = NULL;
            if(a->level == 0){
                if(a->blockindex > 0){
                    if(l->blockindex <= 0){
                        l->blockindex = appendBlockStorage(tmp);
                    }
                    if(u->blockindex <= 0){
                        u->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(a->blockindex, 0, blockOp::lu, l->blockindex, u->blockindex, 0, group));
                }
                return;
            }
            matrix* ha = a->submatrix[0];
            matrix* l1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* u1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* l2 = l3;
            matrix* u2 = u3;
            if(l2 == NULL){
                l2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                u2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            }
            matrix* hl1 = NULL;
            matrix* hu1 = NULL;
            if(a->level>1){
                hl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
                hu1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
            }
            blockMatrixLU(ha,l1,u1,n/2,hl1,hu1,group);
            blockMatrixInvLower(l1,l2,n/2,hl1,group);
            blockMatrixInvUpper(u1,u2,n/2,hu1,group);
            l->submatrix[0] = l1;                // l2 ????
            u->submatrix[0] = u1;

            matrix* aua1b = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* b = a->submatrix[1];
            if(b != NULL){
                matrix* a1b = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                blockMatrixMul(l2,b,aua1b,hn,group);
                u->submatrix[1] = aua1b;
            }

            matrix* dsub = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* d = a->submatrix[3];

            matrix* c = a->submatrix[2];
            if(c != NULL){
                matrix* ca1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                matrix* ca1al = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                blockMatrixMul(c,u2,ca1al,hn,group);
                l->submatrix[2] = ca1al;

                matrix* ca1b = memutil::newmatrix(0,a->blockrows2/2,a->level-1);

                blockMatrixMul(ca1al, aua1b, ca1b, hn,group);
                blockMatrixSub(d, ca1b, dsub, hn,group);
            } else {
                dsub = d;
            }

            l1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            u1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* dl1 = NULL;
            matrix* du1 = NULL;
            if(a->level>1){
                dl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
                du1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
            }
            blockMatrixLU(dsub,l1,u1,hn,dl1,du1,group);
            l->submatrix[3] = l1;
            u->submatrix[3] = u1;
        }

        void BlockPlanner::blockMatrixLLT(matrix* a, matrix* l, int n, matrix* l3,int group)
        {
            int hn = n / 2;
            double *tmp = NULL;
            if(a->level == 0){
                if(a->blockindex > 0){
                    if(l->blockindex <= 0){
                        l->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(a->blockindex, 0, blockOp::llt, l->blockindex, 0, 0, group));
                }
                return;
            }

            matrix* ha = a->submatrix[0];
            matrix* l1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* l2 = l3;
            if(l3 == NULL)
                l2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* hl1 = NULL;
            if(a->level>1){
                hl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
            }
            blockMatrixLLT(ha,l1,hn,hl1,group);
            blockMatrixInvLower(l1,l2,hn,hl1,group);
            l->submatrix[0] = l1;

            matrix* dsub = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* d = a->submatrix[3];

            matrix* c = a->submatrix[2];
            if(c!=NULL){
                matrix* ca1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                matrix* ca2 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                blockMatrixMulT(c, l2, ca1, hn,group);
                l->submatrix[2] = ca1;

                blockMatrixMulT(ca1,ca1,ca2,hn,group);
                blockMatrixSub(d,ca2,dsub,hn,group);
            } else {
                dsub = d;
            }
            l1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            matrix* dl1 = NULL;
            if(hn>1)
                dl1 = memutil::newmatrix(0,a->blockrows2/4,a->level-2);
            blockMatrixLLT(dsub,l1,hn,dl1,group);
            l->submatrix[3] = l1;
        }

        void BlockPlanner::blockMatrixInv(matrix* a, matrix* y, int n, int group)
        {
            int hn = n / 2;
            double *tmp = NULL;
            if(a->level == 0){
                if(a->blockindex > 0){
                    y->blockindex = appendBlockStorage(tmp);
                    data::graph.push_back(memutil::newoperation(a->blockindex, 0, blockOp::inv, y->blockindex, 0,0,group));
                }
                return;
            }

            matrix* l = memutil::newmatrix(0,a->blockrows2,a->level);
            matrix* u = memutil::newmatrix(0,a->blockrows2,a->level);
            matrix* l1 = memutil::newmatrix(0,a->blockrows2,a->level);
            matrix* u1 = memutil::newmatrix(0,a->blockrows2,a->level);
            matrix* hl1 = NULL;
            matrix* hu1 = NULL;
            if(hn>1){
                hl1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
                hu1 = memutil::newmatrix(0,a->blockrows2/2,a->level-1);
            }
            blockMatrixLU(a, l, u, n, hl1, hu1,group);
            blockMatrixInvLower(l, l1, n, hl1,group);
            blockMatrixInvUpper(u, u1, n, hu1,group);
            blockMatrixMul(u1, l1, y, n,group);
        }

        void BlockPlanner::blockMatrixInvLower(matrix* l, matrix* y, int n, matrix* a3,int group)
        {
            int hn = n / 2;
            double *tmp = NULL;
            if(l->level == 0){
                if(l->blockindex > 0){
                    y->blockindex = appendBlockStorage(tmp);
                    data::graph.push_back(memutil::newoperation(l->blockindex, 0, blockOp::lowerInv, y->blockindex, 0,0,group));
                }
                return;
            }
            matrix* a = l->submatrix[0];
            matrix* c = l->submatrix[2];
            matrix* d = l->submatrix[3];

            matrix* hd1 = NULL;
            matrix* ha1 = NULL;

            matrix* d1 = memutil::newmatrix(0,a->blockrows2,a->level);
            blockMatrixInvLower(d, d1, hn, hd1, group);
            y->submatrix[3] = d1;

            matrix* d1c = memutil::newmatrix(0,a->blockrows2,a->level);
            blockMatrixMul(d1, c, d1c, hn, group);

            matrix* a1 = a3;
            if(a1 == NULL){
                a1 = memutil::newmatrix(0,a->blockrows2,a->level);
                blockMatrixInvLower(a, a1, hn, ha1, group);
            }
            y->submatrix[0] = a1;
 
            matrix* c21 =  memutil::newmatrix(0,a->blockrows2,a->level);
            blockMatrixMulNeg(d1c, a1, c21, hn, group);
            y->submatrix[2] = c21;
        }

        void BlockPlanner::blockMatrixInvUpper(matrix* u, matrix* y, int n, matrix* a3, int group)
        {
            int hn = n / 2;
            double *tmp = NULL;
            if(u->level == 0){
                if(u->blockindex > 0){
                    y->blockindex = appendBlockStorage(tmp);
                    data::graph.push_back(memutil::newoperation(u->blockindex, 0, blockOp::upperInv, y->blockindex, 0, 0, group));
                }
                return;
            }
            matrix* a = u->submatrix[0];
            matrix* b = u->submatrix[1];
            matrix* d = u->submatrix[3];

            matrix* hd1 = NULL;
            matrix* ha1 = NULL;

            matrix* a1 = memutil::newmatrix(0,a->blockrows2,a->level);
            matrix* a2 = a3;
            if(a2 == NULL){
                a2 = memutil::newmatrix(0,a->blockrows2,a->level);
                blockMatrixInvUpper(a, a2, hn, ha1, group);
            }
            y->submatrix[0] = a2;

            matrix* a1b = memutil::newmatrix(0,a->blockrows2,a->level);
            blockMatrixMul(a2,b,a1b,hn, group);
 
            matrix* d1 = memutil::newmatrix(0,a->blockrows2,a->level);
            blockMatrixInvUpper(d,d1,hn,hd1, group);
            y->submatrix[3] = d1;
            
            matrix* a1bd1 = memutil::newmatrix(0,a->blockrows2,a->level);
            blockMatrixMulNeg(a1b,d1,a1bd1,hn, group);
            y->submatrix[1] = a1bd1;
        }

        int BlockPlanner::blockScanNAN(matrix* a, int n)
        {
            int rtn = 0;
            if(a->level == 0){
                if(a->blockindex > 0){
                    double *ele = data::blockstorage[a->blockindex];
                    for(int i=0; i<BLOCK64; i++)
                        for(int j=0; j<BLOCK64; j++){
                            if(std::isnan(ele[i*BLOCK64+j]))
                                rtn++;
                        }
                }
                return rtn;
            } 
            for(int i=0;i<4;i++) {
                if(a->submatrix[i] != NULL){
                    rtn += blockScanNAN(a->submatrix[i], n/2);
                }   
            }
            return rtn;
        }
        void BlockPlanner::printInt(matrix* a, int n)
        {
               int line = 8;
               for(int i=0;i<n*n;i++){
                  int e = a[0][i];
                  if(e == 0)
                      std::cout<<"   "<<"  \b";
                  else
                      std::cout<<e<<"  \b";
                  if((i%line)==line-1)
                     std::cout<<std::endl;
               }
  
        }

        void BlockPlanner::blockMatrixSub(matrix* a, matrix*b, matrix* c, int n, int group)
        {
            double *tmp = NULL;
            if(a->level == 0){
                if(a->blockindex > 0 || b->blockindex > 0){
                    if(c->blockindex <= 0){
                        c->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(b->blockindex, a->blockindex, blockOp::sub, c->blockindex, 0, 0, group));
                }
                return;
            } 
            for(int i=0;i<4;i++)
            {
                if(a->submatrix[i] != NULL || b->submatrix[i] != NULL){
                    if(c->submatrix[i] == NULL){
                        c->submatrix[i] = memutil::newmatrix(0, c->blockrows2 / 2, c->level-1);
                    }
                }

                if(a->submatrix[i] != NULL && b->submatrix[i] != NULL){
                    blockMatrixSub(a->submatrix[i],b->submatrix[i],c->submatrix[i], n/2, group);
                }   
                else if(a->submatrix[i] != NULL && b->submatrix[i] == NULL){
                    blockMatrixCopy(a->submatrix[i],c->submatrix[i], n/2, group);
                }   
                else if(a->submatrix[i] == NULL && b->submatrix[i] != NULL){
                    blockMatrixNeg(b->submatrix[i],c->submatrix[i], n/2, group);
                }   
            }
        }

        void BlockPlanner::blockMatrixCopy(matrix* a, matrix* c, int n, int group)
        {
            double *tmp = NULL;
            if(a->level == 0){
                if(a->blockindex > 0){
                    if(c->blockindex <= 0){
                        c->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(0, a->blockindex, blockOp::sub, c->blockindex, 0, 0, group));
                }
                return;
            } 
            for(int i=0;i<4;i++)
            {
                if(a->submatrix[i] != NULL){
                    if(c->submatrix[i] == NULL){
                        c->submatrix[i] = memutil::newmatrix(0, c->blockrows2 / 2, c->level-1);
                    }
                    blockMatrixCopy(a->submatrix[i],c->submatrix[i], n/2, group);
                }
            }
        }
        void BlockPlanner::blockMatrixNeg(matrix* b, matrix* c, int n, int group)
        {
            double *tmp = NULL;
            if(b->level == 0){
                if(b->blockindex > 0){
                    if(c->blockindex <= 0){
                        c->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(b->blockindex, 0, blockOp::sub, c->blockindex, 0, 0, group));
                }
                return;
            } 
            for(int i=0;i<4;i++)
            {
                if(b->submatrix[i] != NULL){
                    if(c->submatrix[i] == NULL){
                        c->submatrix[i] = memutil::newmatrix(0, c->blockrows2 / 2, c->level-1);
                    }
                    blockMatrixNeg(b->submatrix[i],c->submatrix[i], n/2, group);
                }
            }
        }

        void BlockPlanner::blockMatrixMulT(matrix* a, matrix*b, matrix* c, int n, int group)
        {
            double *tmp = NULL;
            if(a == NULL || b == NULL) return;
            if(a->level == 0){
                if(a->blockindex > 0 && b->blockindex > 0){
                    if(c->blockindex <= 0){
                        c->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(a->blockindex, b->blockindex, blockOp::mult, c->blockindex, 0, 0, group));
                }
                return;
            } 
            for(int i=0;i<2;i++)
                for(int j=0;j<2;j++)
                    for(int k=0;k<2;k++)
                    {
                        if(a->submatrix[i*2+k] != NULL && b->submatrix[j*2+k] != NULL){
                            if(c->submatrix[i*2+j] == NULL){
                                c->submatrix[i*2+j] = memutil::newmatrix(2);
                                c->submatrix[i*2+j]->blockrows2 = c->blockrows2 / 2;
                                c->submatrix[i*2+j]->level = c->level-1;
                            }
                            blockMatrixMulT(a->submatrix[i*2+k],b->submatrix[j*2+k],c->submatrix[i*2+j], n/2, group);
                        }   
                    }
        }
        void BlockPlanner::blockMatrixMulNeg(matrix* a, matrix*b, matrix* c, int n, int group)
        {
            double *tmp = NULL;
            if(a == NULL || b == NULL) return;
            if(a->level == 0){
                if(a->blockindex > 0 && b->blockindex > 0){
                    if(c->blockindex <= 0){
                        c->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(a->blockindex, b->blockindex, blockOp::mulneg, c->blockindex, 0, 0, group));
                }
                return;
            } 
            for(int i=0;i<2;i++)
                for(int j=0;j<2;j++)
                    for(int k=0;k<2;k++)
                    {
                        if(a->submatrix[i*2+k] != NULL && b->submatrix[k*2+j] != NULL){
                            if(c->submatrix[i*2+j] == NULL){
                                c->submatrix[i*2+j] = memutil::newmatrix(2);
                                c->submatrix[i*2+j]->blockrows2 = c->blockrows2 / 2;
                                c->submatrix[i*2+j]->level = c->level-1;
                            }
                            blockMatrixMulNeg(a->submatrix[i*2+k],b->submatrix[k*2+j],c->submatrix[i*2+j], n/2,group);
                        }   
                    }
        }
        void BlockPlanner::blockMatrixMul(matrix* a, matrix*b, matrix* c, int n, int group)
        {
            double *tmp = NULL;
            if(a == NULL || b == NULL) return;
            if(a->level == 0){
                if(a->blockindex > 0 && b->blockindex > 0){
                    if(c->blockindex <= 0){
                        c->blockindex = appendBlockStorage(tmp);
                    }
                    data::graph.push_back(memutil::newoperation(a->blockindex, b->blockindex, blockOp::mul, c->blockindex, 0, 0, group));
                }
                return;
            } 
            for(int i=0;i<2;i++)
                for(int j=0;j<2;j++)
                    for(int k=0;k<2;k++)
                    {
                        if(a->submatrix[i*2+k] != NULL && b->submatrix[k*2+j] != NULL){
                            if(c->submatrix[i*2+j] == NULL){
                                c->submatrix[i*2+j] = memutil::newmatrix(2);
                                c->submatrix[i*2+j]->blockrows2 = c->blockrows2 / 2;
                                c->submatrix[i*2+j]->level = c->level-1;
                            }
                            blockMatrixMul(a->submatrix[i*2+k],b->submatrix[k*2+j],c->submatrix[i*2+j], n/2, group);
                        }   
                    }
        }

        void BlockPlanner::resetOneBlockMeta(double t[])
	{
	    int columns = 8;
            uint *metab = (uint*) (t + METAOFFSET);
            uint *metay = (uint*) (t + METAOFFSET);
            unsigned short *metad = (unsigned short *) (t + DETAILOFFSET);

            std::memset(metay, 0, METASIZE);
            for(int j = 0; j < 4; j++){
                for(int i = 0; i<CACHE64/sizeof(unsigned short); i++) 
                    metad[i] = 0;
                metad += DETAILSKIPSHORT;
            }
	}

        void BlockPlanner::resetOneBlock(double t[])
	{
            MatrixStdDouble::mat_clear(t,BLOCK64);
	}

        void matrixZoomSet(matrix* a, matrix* detail)
        {
            if(a->level == 0){
                if(a->blockindex > 0){
		    SOGLU::BlockPlanner::appendBlockStorageAgainL2(detail, a->blockindex);
                }else{
                    std::cout<<a<<"  "<<detail<<std::endl;
                }
            }else if(a->level > 0){
                for(int i=0; i<4; i++)
                    if(a->submatrix[i] != NULL){
                        matrixZoomSet(a->submatrix[i], detail->submatrix[i]);
                    }
            }
        }
        void matrixZoomUpdate(matrix* l, matrix* bl2)
        {
            if(l->level == 1){
                for(int i=0; i<4; i++)
                    if(l->submatrix[i] != NULL && l->submatrix[i]->blockindex > 0){
                        bl2->submatrix[i] = BlockPlanner::blockstorageL2[l->submatrix[i]->blockindex];
                    }
            }else if(l->level > 1){
                for(int i=0; i<4; i++)
                    if(l->submatrix[i] != NULL){
                        bl2->submatrix[i] = memutil::newmatrix(0,bl2->blockrows2/2,bl2->level-1);
                        matrixZoomUpdate(l->submatrix[i], bl2->submatrix[i]);
                    }
            }
        }

        void BlockPlanner::copyOperatorX2(matrix* a, matrix* x1, matrix* y1, matrix* lhs1, matrix* mx2, int n)
	{
            blockstorageL2.clear();
            storageCountL2 =0;

	    int originSize = data::blockRows * data::blockRows;
	    int i,j;
	    int L2Rows = data::blockRows;
            data::blockSize = config::blockSize;
            data::blockRows = config::blockRows;
	    int scaleL2 = config::blockSizeL2 / config::blockSize;
            int levelL2 = __builtin_popcount(scaleL2 - 1);
		   
            mx2->blockrows2 = data::blockRows;
            mx2->level = __builtin_popcount(data::blockRows - 1);

	    blockstorageL2.resize(data::blockstorage.size(),NULL);
	    storageCountL2 = blockstorageL2.size();

	    checkBlock();
	    iniBlockStorage();

            double *xx =  (double*)memutil::getSmallMem(0, config::blockRows * config::blockSize * sizeof(double));
            std::memset(xx, 0, config::blockRows * config::blockSize * sizeof(double));
            matrix *mx =  BlockPlanner::iniVectorMatrix(xx, config::blockRows);
            matrix *my =  BlockPlanner::iniVectorMatrix(xx, config::blockRows);
            matrix *mlhs =  BlockPlanner::iniVectorMatrix(data::b, config::blockRows);

            matrixZoomSet(a, data::blocks);
            matrixZoomSet(x1, mx);
            matrixZoomSet(y1, my);
            matrixZoomSet(lhs1, mlhs);

	    graphL2.clear();
	    for (i = 0; i < data::graph.size(); i++)
	    {
		operation o = *(data::graph[i]);
                       operation* nop = memutil::newoperation(o.src, o.src2, o.op, o.result, o.result2, 0, o.groupNum);
                      nop->sequenceNum = o.sequenceNum;
		graphL2.push_back(nop);
	    }
	    data::graph.clear();


	    for (i = 0; i < graphL2.size(); i++)
	    {
		operation o = *(graphL2[i]);
		if (o.skip)
		    continue;
		if (o.result > 0)
		{
		    if (blockstorageL2[o.result] == NULL)
		    {
			matrix* dmp = memutil::newmatrix(0,scaleL2,levelL2);
			appendBlockStorageAgainL2(dmp, o.result);
		    }
		}

		if (o.result2 > 0)
		{
		    if (blockstorageL2[o.result2] == NULL)
		    {
			matrix* dmp = memutil::newmatrix(0,scaleL2,levelL2);
			appendBlockStorageAgainL2(dmp, o.result2);
		    }
		}

		switch (o.op)
		{
		    case blockOp::inv:
			blockMatrixInv(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::lowerInv:
			blockMatrixInvLower(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, NULL, o.sequenceNum);
			break;

		    case blockOp::lu:
			blockMatrixLU(blockstorageL2[o.src], blockstorageL2[o.result], blockstorageL2[o.result2], scaleL2, NULL, NULL, o.sequenceNum);
			break;

		    case blockOp::llt:
			blockMatrixLLT(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, NULL, o.sequenceNum);
			break;

		    case blockOp::mul:
			
                        if(blockstorageL2[o.src] == NULL || blockstorageL2[o.src2] == NULL){
                               std::cout<<o.src<<"  "<<blockstorageL2[o.src]<<"   "<<o.src2<<"  "<<blockstorageL2[o.src2]<<std::endl;
                        }
			blockMatrixMul(blockstorageL2[o.src], blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::mult:
			
			blockMatrixMulT(blockstorageL2[o.src], blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::mulneg:
			blockMatrixMulNeg(blockstorageL2[o.src], blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::sub:
			if(o.src>0){
			    if(o.src2>0){
				blockMatrixSub(blockstorageL2[o.src2], blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			    }
			    else{
				blockMatrixNeg(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			    }
			}
			else{
			    if(o.src2>0){
				//printInt(blockstorageL2[o.src2], scaleL2);
				blockMatrixCopy(blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			    }
			}
			break;

		    case blockOp::upperInv:
			blockMatrixInvUpper(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, NULL, o.sequenceNum);
			break;

		    default:
			break;
		}

	    }

            matrixZoomUpdate(x1, mx2);


            blockstorageL2.clear();
            storageCountL2 =0;
            graphL2.clear();
	}

        void BlockPlanner::copyOperatorL2(matrix* a, matrix* l, matrix* u, matrix* LL2, matrix* UL2, int n)
	{
            blockstorageL2.clear();
            storageCountL2 =0;

	    int originSize = data::blockRows * data::blockRows;
	    int i,j;
	    int L2Rows = data::blockRows;
            data::blockSize = config::blockSize;
            data::blockRows = config::blockRows;
	    int scaleL2 = config::blockSizeL2 / config::blockSize;
            int levelL2 = __builtin_popcount(scaleL2 - 1);
		   
            LL2->blockrows2 = data::blockRows;
            LL2->level = __builtin_popcount(data::blockRows - 1);
            UL2->blockrows2 = data::blockRows;
            UL2->level = __builtin_popcount(data::blockRows - 1);

	    blockstorageL2.resize(data::blockstorage.size(),NULL);
	    storageCountL2 = blockstorageL2.size();

	    checkBlock();
	    iniBlockStorage();

            matrixZoomSet(a, data::blocks);

	    graphL2.clear();
	    for (i = 0; i < data::graph.size(); i++)
	    {
		operation o = *(data::graph[i]);
                       operation* nop = memutil::newoperation(o.src, o.src2, o.op, o.result, o.result2, 0, o.groupNum);
                      nop->sequenceNum = o.sequenceNum;
		graphL2.push_back(nop);
	    }
	    data::graph.clear();


	    for (i = 0; i < graphL2.size(); i++)
	    {
		operation o = *(graphL2[i]);
		if (o.skip)
		    continue;
		if (o.result > 0)
		{
		    if (blockstorageL2[o.result] == NULL)
		    {
			matrix* dmp = memutil::newmatrix(0,scaleL2,levelL2);
			appendBlockStorageAgainL2(dmp, o.result);
		    }
		}

		if (o.result2 > 0)
		{
		    if (blockstorageL2[o.result2] == NULL)
		    {
			matrix* dmp = memutil::newmatrix(0,scaleL2,levelL2);
			appendBlockStorageAgainL2(dmp, o.result2);
		    }
		}

		switch (o.op)
		{
		    case blockOp::inv:
			blockMatrixInv(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::lowerInv:
			blockMatrixInvLower(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, NULL, o.sequenceNum);
			break;

		    case blockOp::lu:
			blockMatrixLU(blockstorageL2[o.src], blockstorageL2[o.result], blockstorageL2[o.result2], scaleL2, NULL, NULL, o.sequenceNum);
			break;

		    case blockOp::llt:
			blockMatrixLLT(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, NULL, o.sequenceNum);
			break;

		    case blockOp::mul:
			
                        if(blockstorageL2[o.src] == NULL || blockstorageL2[o.src2] == NULL){
                               std::cout<<o.src<<"  "<<blockstorageL2[o.src]<<"   "<<o.src2<<"  "<<blockstorageL2[o.src2]<<std::endl;
                        }
			blockMatrixMul(blockstorageL2[o.src], blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::mult:
			
			blockMatrixMulT(blockstorageL2[o.src], blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::mulneg:
			blockMatrixMulNeg(blockstorageL2[o.src], blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			break;

		    case blockOp::sub:
			if(o.src>0){
			    if(o.src2>0){
				blockMatrixSub(blockstorageL2[o.src2], blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			    }
			    else{
				blockMatrixNeg(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			    }
			}
			else{
			    if(o.src2>0){
				//printInt(blockstorageL2[o.src2], scaleL2);
				blockMatrixCopy(blockstorageL2[o.src2], blockstorageL2[o.result], scaleL2, o.sequenceNum);
			    }
			}
			break;

		    case blockOp::upperInv:
			blockMatrixInvUpper(blockstorageL2[o.src], blockstorageL2[o.result], scaleL2, NULL, o.sequenceNum);
			break;

		    default:
			break;
		}

	    }
            if(u!=NULL)
                matrixZoomUpdate(u, UL2);
            matrixZoomUpdate(l, LL2);

            blockstorageL2.clear();
            storageCountL2 =0;
            graphL2.clear();
	}

	void BlockPlanner::checkBlock()
	{
	    int count = 0;
	    data::blockstorage.clear();
	    data::blockstorage.push_back(NULL);
	    data::storageCount = 1;
	}

        void setupFirstColumn(matrix *b)
        {
            if(b->level == 0){
                b->blockindex = BlockPlanner::claimBlock();
                return;
            }
            int n2 = b->blockrows2 / 2;
            b->submatrix[0] = memutil::newmatrix(0,n2,b->level-1);
            b->submatrix[2] = memutil::newmatrix(0,n2,b->level-1);
            setupFirstColumn(b->submatrix[0]);
            setupFirstColumn(b->submatrix[2]);
        }
        matrix* BlockPlanner::iniVectorMatrix(int n)
        {
            int levels = __builtin_popcount(n-1);
            matrix *b =  memutil::newmatrix(0,n,levels);
            setupFirstColumn(b);
            return b;
        }

        void BlockPlanner::getFirstColumn(matrix *m, double* b, int n)
        {
            if(m->level == 0){
                double* dmp = data::blockstorage[m->blockindex];
                unsigned short *metad = (unsigned short *) (dmp + DETAILOFFSET);
                for(int i=0;i<BLOCK64;i++){
      //              if(metad[i%32+(i/32)*DETAILSKIPSHORT] & 1 )
                        b[i] = dmp[i*BLOCKCOL];
                }
                return;
            }
            int n2 = n / 2;
            getFirstColumn(m->submatrix[0], b, n2);
            getFirstColumn(m->submatrix[2], b+n2, n2);
        }
        void assignFirstColumn(matrix *m, double* b, int n)
        {
            if(m->level == 0){
                m->blockindex = BlockPlanner::claimBlock();
                double* dmp = (double *) memutil::newalignedblock(0,8);
                BlockPlanner::resetOneBlock(dmp);
                for(int i=0;i<BLOCK64;i++){
                    dmp[i*BLOCKCOL] = b[i];
                }
                uint *metay = (uint*) (dmp + METAOFFSET);
                for(int i=0;i<n/8;i++) *metay = 1;
                unsigned short *metad = (unsigned short *) (dmp + DETAILOFFSET);
                for(int i=0;i<BLOCK64;i++) 
                    metad[i%32+(i/32)*DETAILSKIPSHORT] = 1;
                BlockPlanner::appendBlockStorageAgain(dmp, m->blockindex);
                return;
            }
            int n2 = m->blockrows2 / 2;
            m->submatrix[0] = memutil::newmatrix(0,n2,m->level-1);
            m->submatrix[2] = memutil::newmatrix(0,n2,m->level-1);
            assignFirstColumn(m->submatrix[0], b, n2);
            assignFirstColumn(m->submatrix[2], b+n2*BLOCK64, n2);
        }
        matrix* BlockPlanner::iniVectorMatrix(double *b, int n)
        {
            int levels = __builtin_popcount(n-1);
            matrix *m =  memutil::newmatrix(0,n,levels);
            assignFirstColumn(m,b,n);
            return m;
        }

	void BlockPlanner::iniBlockStorage()
	{
	    int i, bi, bj;

            data::blocks = memutil::newmatrix(0, data::blockRows, __builtin_popcount(data::blockRows-1));

	    uint mask32[32];
	    for(i=0; i<32; i++) mask32[i] = 1u<<i;

            int expmask = data::blockSize-1;
            int exp = __builtin_popcount(data::blockSize-1);
            if(data::PlanL2){
	        for (i = 0; i < data::valcount; i++)
	        {
	    	    bi = data::indexi[i] >> exp;  
		    bj = data::indexj[i] >> exp;  
		    if(data::blocks[0][bi * data::blockRows + bj] == 0)
		    {
		        allocateBlock(bi, bj);
		    }
	        }
            }else{
                std::vector<cell> inlist(data::valcount);
                for (i = 0; i < data::valcount; i++){
                    inlist[i].row = data::indexi[i];
                    inlist[i].col = data::indexj[i];
                    inlist[i].val = data::vals[i];
                }
                   std::sort (inlist.begin(), inlist.end(), cellCompare);
	        for (i = 0; i < data::valcount; i++)
	        {
			bi = inlist[i].row >> exp;  
			bj = inlist[i].col >> exp;  
                        uint64_t blockindex = data::blocks[0][bi * data::blockRows + bj];
			if(blockindex == 0)
			{
			    blockindex = allocateBlock(bi, bj);
			}
			int ri = inlist[i].row & expmask; 
			int rj = inlist[i].col & expmask; 
			data::blockstorage[blockindex][ri * BLOCKCOL + rj] = inlist[i].val;
			    int mi = ri / 8;
			    int mj = rj / 8;
			    uint *m = (uint*)(data::blockstorage[blockindex] + METAOFFSET);
			    m[mi] = m[mi] | mask32[mj];
                            unsigned short *d = (unsigned short *)(data::blockstorage[blockindex] + DETAILOFFSET);
                            d += DETAILSKIPSHORT * (ri / 32);
			    d[ri%32] = d[ri%32] | mask32[mj];   
		    }
                 }
		    for (i = data::mSize; i< data::blockRows * data::blockSize; i++)
		    {
			bi = i / data::blockSize;
			int ri = i - bi * data::blockSize;
                        uint64_t blockindex = data::blocks[0][bi * data::blockRows + bi];
			if (data::blocks[0][bi * data::blockRows + bi] == 0)
			{
			    blockindex = allocateBlock(bi, bi);
			}

			if(!data::PlanL2) {
			    data::blockstorage[blockindex][ri * BLOCKCOL + ri] = 1.0;
			    int mi = ri / 8;
			    uint *m = (uint*)(data::blockstorage[blockindex] + METAOFFSET);
			    m[mi] = m[mi] | mask32[mi];
                            unsigned short *dd = (unsigned short *)(data::blockstorage[blockindex] + DETAILOFFSET);
                            dd += DETAILSKIPSHORT * (ri / 32);
                            dd[ri%32] = dd[ri%32] | mask32[mi];
			}
		    }
		if(!data::PlanL2)
                    memutil::clearmemory = false;
		}
               
        uint64_t BlockPlanner::claimBlock()
        {
            double* dmp = NULL;
            uint64_t t = data::blockstorage.size();
            data::blockstorage.push_back( dmp );
            data::storageCount = t+1;
            return t;
        }
	uint64_t BlockPlanner::allocateBlock(int bi, int bj)
	{
	    double* dmp = NULL;
	    if(!data::PlanL2)
	    {
		dmp = memutil::newalignedblock(0,8);
   //             std::memset(dmp, 0, ALLOCBLOCK);
            }

            uint64_t t = data::blockstorage.size();
            data::blockstorage.push_back( dmp );
            data::blocks->set(((uint64_t)bi) * data::blockRows + bj, t);
            data::storageCount = t+1;
            return t;
        }
        void BlockPlanner::appendBlockStorageAgain(double t[], int rtn)
        {
            data::blockstorage[rtn] = t;
        }
        void BlockPlanner::appendBlockStorageAgainL2(matrix* t, int rtn)
        {
            blockstorageL2[rtn] = t;
        }
        int BlockPlanner::appendBlockStorage(double t[])
        {
            int rtn = data::blockstorage.size();
            data::blockstorage.push_back( t );
            data::storageCount = rtn + 1;
            if(rtn < 0){
                std::cout<<" storage overflow "<<std::endl;
                exit(0);
            }
            return rtn;
        }
}

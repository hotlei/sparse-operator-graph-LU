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

#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <string>
#include <cstring>
#include <memory>
#include "operation.h"
#include "matrix.h"
#include "memutil.h"

#define ALIGN64 64

namespace SOGLU{

        std::vector<std::vector<double*>> memutil::blockStorageBuffer(MAXTHREAD,std::vector<double*>(10,NULL));
        int memutil::blockStoragePointer[MAXTHREAD]={};
        std::vector<std::vector<double*>> memutil::reclaimedStorage(MAXTHREAD,std::vector<double*>(10,NULL));

        std::vector<std::vector<void*>> memutil::memBlobs(MAXTHREAD,std::vector<void*>(10,NULL));
        std::vector<std::vector<void*>> memutil::reclaimedBlobs(MAXTHREAD,std::vector<void*>(10,NULL));

        std::vector<std::vector<void*>> memutil::memNugget(MAXTHREAD,std::vector<void*>(10,NULL));
        int memutil::memNuggetPointer[MAXTHREAD]={};

        std::vector<std::vector<void*>> memutil::matrixBuffer(MAXTHREAD,std::vector<void*>(10,NULL));
        uint memutil::matrixBufferPointer[MAXTHREAD]={};

        std::vector<std::vector<void*>> memutil::operationBuffer(MAXTHREAD,std::vector<void*>(10,NULL));
        int memutil::operationPointer[MAXTHREAD]={};

        bool memutil::clearmemory = true;
        void memutil::ini()
        {
            for(int i=0;i<MAXTHREAD;i++){
                memutil::blockStorageBuffer[i].clear();
                memutil::blockStorageBuffer[i].reserve(10000);
                memutil::blockStoragePointer[i] = -1;
                memutil::reclaimedStorage[i].clear();

                memutil::memBlobs[i].clear();
                memutil::reclaimedBlobs[i].clear();

                memutil::memNugget[i].clear();
                memutil::memNuggetPointer[i] = -1;

                memutil::matrixBuffer[i].clear();
                memutil::matrixBufferPointer[i] = -1;

                memutil::operationBuffer[i].clear();
                memutil::operationPointer[i] = -1;
            }
            for(std::vector<double*> q:memutil::blockStorageBuffer) q.resize(0);
            clearmemory = true;
        }

        double* memutil::newalignedblock(int threadId, int columns)
        {
            long zipsize = ALLOCBLOCK;
            int blockcount = MEMBLOB / zipsize;
            int que = threadId;
            int lst = reclaimedStorage[que].size() - 1;
            if(lst>=0){
                double* dmp = reclaimedStorage[que][lst];
                reclaimedStorage[que].pop_back();
                return dmp;
            }
            int last = blockStorageBuffer[que].size()-1;
            if(blockStoragePointer[que]>= blockcount-1 || last == -1 || blockStoragePointer[que] == -1){
                double *p = (double *) getMemBlob(threadId);  
                
                if(clearmemory){
        //            std::cout<<"clear 1 memblob"<<std::endl;
                    std::memset(p, 0, MEMBLOB);
                }
                blockStorageBuffer[que].push_back(p);
                last = blockStorageBuffer[que].size()-1;
                blockStoragePointer[que] = 0;
            }
            double* rtn = blockStorageBuffer[que][last] + blockStoragePointer[que] * zipsize/8; //BLOCK64 * BLOCK64;
            blockStoragePointer[que]++;
            return rtn;
        }
        void memutil::freeblock(int threadId, double* dmp)
        {
            int que = threadId;
            if(!reclaimedStorage[que].empty()){
            double* tmp = reclaimedStorage[que].back();
            if(tmp == dmp)
                  return;

//            if(std::find(reclaimedStorage[que].begin(),reclaimedStorage[que].end(),dmp)!=reclaimedStorage[que].end())
//                return;
            }
            reclaimedStorage[que].push_back(dmp);
        }
        void memutil::releaseAllBlockBuffer()
        {
            for(int i=0;i<MAXTHREAD; i++){
                for(void* p:blockStorageBuffer[i]){
                    releaseOne( p);
                }
                blockStorageBuffer[i].clear();
            }
        }

        void* memutil::getMemBlob(int threadId, size_t bytes)
        {
            int que = threadId;
            size_t len = bytes;
            if(len < MEMBLOB) len = MEMBLOB;
            int lst = reclaimedBlobs[que].size() - 1;
            if(lst>=0 && len<= MEMBLOB){
                void* dmp = reclaimedBlobs[que][lst];
                reclaimedBlobs[que].pop_back();
    //            std::cout<<"reuse mem blob from queue "<<que<<std::endl;
                return dmp;
            }

                void *p = (void *) aligned_alloc(64, len);

                memBlobs[que].push_back(p);

            return p;
        }

        void memutil::freeMemBlob(int threadId, void* vmp)
        {
            int que = threadId;
            if(!reclaimedBlobs[que].empty()){
            void* tmp = reclaimedBlobs[que].back();
            if(tmp == vmp)
                  return;
            }
            reclaimedBlobs[que].push_back(vmp);
        }

        void* memutil::getSmallMem(int threadId, size_t bytes)
        {
            int que = threadId;
            size_t len = bytes;
            if(len<=0) return NULL;

            int last = memNugget[que].size()-1;
            if(last == -1 || memNuggetPointer[que] == -1 || memNuggetPointer[que] + len > MEMBLOB){

                if(len > MEMBLOB){
                    size_t blobs =  (len + MEMBLOB - 1) / MEMBLOB;
                    len = (blobs+1) * MEMBLOB;
                }

                void *p = getMemBlob(threadId, len); //aligned_calloc_block( zipsize * 100);

                memNugget[que].push_back(p);
                last = memNugget[que].size()-1;
                memNuggetPointer[que] = 0;

                if(len > MEMBLOB){
                    size_t blobs = len / MEMBLOB;
                    for(uint64_t i=1;i<blobs;i++) memNugget[que].push_back(p+i*MEMBLOB);
     //               std::cout<<"alloc :"<<len<<" start at:"<<std::hex<<p<<std::dec<<" in "<<blobs<<" pieces"<<std::endl;
                    return p;
                }
            }
            void* rtn = memNugget[que][last] + memNuggetPointer[que]; //BLOCK64 * BLOCK64;
            memNuggetPointer[que] += len;

          //          std::cout<<"alloc :"<<len<<" start at:"<<std::hex<<rtn<<std::dec<<std::endl;
            return rtn;
        }
        void memutil::freeAllSmallMem()
        {
            int nuggets = 0; 
            for(int i=0;i<MAXTHREAD; i++){
              nuggets += memNugget[i].size();
            }

            for(int i=0;i<MAXTHREAD; i++){
              for(void* p:memNugget[i]){
                  freeMemBlob(i, p);
              }
              memNugget[i].clear();
              memNuggetPointer[i] = -1;
            }
        }

        void memutil::freeAllSmallMem(int threadId)
        {
            int nuggets = 0;
            for(int i=0;i<MAXTHREAD; i++){
              if(i==threadId)
              nuggets += memNugget[i].size();
            }

            for(int i=0;i<MAXTHREAD; i++){
              if(i==threadId){
              for(void* p:memNugget[i]){
                  freeMemBlob(i, p);
              }
              memNugget[i].clear();
              memNuggetPointer[i] = -1;
              }
            }
        }

        matrix* memutil::newmatrix(int threadId)
        {
            int que = threadId;

            int last = matrixBuffer[que].size()-1;
            if(last == -1 || matrixBufferPointer[que] == -1 || matrixBufferPointer[que] + sizeof(matrix) >= MEMBLOB){
                void *p = getMemBlob(threadId, MEMBLOB); 

                matrixBuffer[que].push_back(p);
                last = matrixBuffer[que].size()-1;
                matrixBufferPointer[que] = 0;
            }
            void* rtn = matrixBuffer[que][last] + matrixBufferPointer[que]; //BLOCK64 * BLOCK64;
            matrixBufferPointer[que] += sizeof(matrix);
            std::memset(rtn,0,sizeof(matrix));

            return (matrix*)rtn;
        }

        matrix* memutil::newmatrix(int threadId, int blockrows, int level)
        {
            int que = threadId;

            uint max = MEMBLOB - 1;
            max = ~max;
            int last = matrixBuffer[que].size()-1;
            if(((matrixBufferPointer[que] + sizeof(matrix)) & max) != 0){
                que = threadId;
            }
            if(last == -1 || matrixBufferPointer[que] == -1 || matrixBufferPointer[que] + sizeof(matrix) >= MEMBLOB){
                void *p = getMemBlob(threadId, MEMBLOB); 

                matrixBuffer[que].push_back(p);
                last = matrixBuffer[que].size()-1;
                matrixBufferPointer[que] = 0;
            }
            matrix* rtn = (matrix*)(matrixBuffer[que][last] + matrixBufferPointer[que]); 
            matrixBufferPointer[que] += sizeof(matrix);

            rtn->blockrows2 = blockrows;
            rtn->level = level;
            rtn->blockindex = 0;
            for(int i=0;i<4;i++){
                rtn->submatrix[i] = NULL;
            }
            if(level<0)
                std::cout<<"level "<<level<<std::endl;

            return rtn;
        }

        operation* memutil::newoperation(int src1, int s2, blockOp o, int d1, int d2, int threadId, int group)
        {
            int que = threadId;

            int last = operationBuffer[que].size()-1;
            if(last == -1 || operationPointer[que] == -1 || operationPointer[que] + sizeof(operation) >= MEMBLOB){

                void *p = getMemBlob(threadId, MEMBLOB); //aligned_calloc_block( zipsize * 100);

                operationBuffer[que].push_back(p);
                last = operationBuffer[que].size()-1;
                operationPointer[que] = 0;
            }
            operation* rtn = (operation*)(operationBuffer[que][last] + operationPointer[que]); //BLOCK64 * BLOCK64;
            operationPointer[que] += sizeof(operation);

            rtn->src = src1;
            rtn->src2 = s2;
            rtn->op = o;
            rtn->result = d1;
            rtn->result2 = d2;
            rtn->skip = false;
            rtn->stage = 0;
            rtn->sequenceNum = operation::seq;
            rtn->groupNum = group;
            operation::seq++;

            return rtn;
        }

        void memutil::report()
        {
            uint64_t total = 0;
            for(int i=0;i<MAXTHREAD; i++){
                total += blockStorageBuffer[0].size();
            }
            uint64_t reclaim = 0;
            for(int i=0;i<MAXTHREAD; i++){
                reclaim += reclaimedStorage[i].size();
            }
            reclaim = reclaim * 36 /  1024;

            std::cout<<"block buffer count: "<<total<<"  "<<total * 4<<" M  reclaimed "<<reclaim<<" M"<<std::endl;
            total = 0;
            for(int i=0;i<MAXTHREAD; i++){
                total += memNugget[i].size();
            }
            std::cout<<"small buffer count: "<<total<<"  "<<total * 4<<" M"<<std::endl;
            total = 0;
            for(int i=0;i<MAXTHREAD; i++){
                total += matrixBuffer[i].size();
            }
            std::cout<<"matrix buffer count: "<<total<<"  "<<total * 4<<" M"<<std::endl;
            total = 0;
            for(int i=0;i<MAXTHREAD; i++){
                total += operationBuffer[i].size();
            }
            std::cout<<"operation buffer count: "<<total<<"  "<<total * 4<<" M"<<std::endl;
        }
        
        void memutil::releaseOne(void* p)
        {
            for(int i=0;i<MAXTHREAD; i++){
                for(int j=0;j<memBlobs[i].size();j++){
                    if(p == memBlobs[i][j]){
                        free(p);
                        memBlobs[i][j] = NULL;
                        return;
                    }
                }
            }
        }

        void memutil::clear()
        {
            for(int i=0;i<MAXTHREAD; i++){
                reclaimedBlobs[i].clear();
                for(void* p:memBlobs[i]){
                    if(p != NULL)
                        free( p);
                }
                memBlobs[i].clear();
            }
        }
}

#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <string>
#include <memory>

#include "data.h"
#include "broker.h"

namespace vmatrix{
        position::position(int idx, int val)
        {
            index = idx;
            value = val;
        }

        bool positionCompare(position x, position y)
        {
            return (x.value < y.value);
        }

        edge::edge()
        {
            row = 0;
            col = 0;
        }
        edge::edge(int i, int j)
        {
            row = i;
            col = j;
        }
        void edge::sett(int i, int j)
        {
            row = i;
            col = j;
        }
        bool edgeCompare(edge* x, edge* y)
        {
            if (x->row == y->row)
                return (x->col < y->col);
            return (x->row < y->row);
        }

        int operation::seq = 0;
        operation::operation(int src1, int s2, blockOp o, int d1, int d2)
        {
            src = src1;
            src2 = s2;
            op = o;
            result = d1;
            result2 = d2;
            skip = false;
            stage = 0;
            sequenceNum = ++seq;
//            if(src1> 1348160 ){
//                std::cout<<src1 << " s2 " << s2 << " o " << o  << " d1 " << d1 <<" d2 "<< d2 << "seq " << sequenceNum<<std::endl;
//            }
        }
        void operation::sett(int src1, int s2, blockOp o, int d1, int d2)
        {
            src = src1;
            src2 = s2;
            op = o;
            result = d1;
            result2 = d2;
            skip = false;
            stage = 0;
            sequenceNum = ++seq;
        }

        operation::operation()
        {
/*            src = 0;
            src2 = 0;
            op = blockOp::noop;
            result = 0;
            result2 = 0;
            skip = false;
            stage = 0;
            sequenceNum = 0;
 */       }

    
        bool operationCompare(operation* x, operation* y)
        {
            if(x->stage == y->stage)
            {
                int xlk = x->result % NUMSTREAM;
                int ylk = y->result % NUMSTREAM;
                if(xlk == ylk){    
                  //  if(x->result == y->result){
                         if(x->op == y->op){
                             return (x->sequenceNum < y->sequenceNum);
                         }
                         return(x->op < y->op);
                  //  }
                  //  return(x->result < y->result);
                }
                return (xlk < ylk);
            }
            return (x->stage < y->stage);
        }

        int data::mSize = 0;
        int data::mSize_out = 0;
        int* data::blocks = NULL;
        int data::blockRows = 0;
        int data::blockSize = BLOCK64;
        int data::blockRows_sq = 0;
        int data::valcount = 0;
        int data::linvcount = 0;
        int data::uinvcount = 0;
        int data::mulcount = 0;
        int data::invertcount = 0;
        int data::foundzeroblock = 0;
        int data::lucount = 0;
        int data::storageCount = 0;
        bool data::UseCPU = false;
        int data::getBlockCount = 0;
        int data::blockMulCount = 0;
        int data::gpuMB = 5000;
        
        int* data::indexi = NULL;
        int* data::indexj = NULL;
        double* data::b = NULL;
        double* data::x = NULL;
        double* data::vals = NULL;

        std::vector<double*> data::blockstorage = {};
        std::vector<operation*> data::seq = {};
        std::unique_ptr<int[]> data::stage = nullptr;
        std::unique_ptr<int[]> data::laststage = nullptr;

        std::vector<operation*> data::operationBuffer = {};
        int data::operationPointer=-1;
        std::vector<edge*> data::edgeBuffer = {};
        int data::edgePointer=-1;
        std::vector<double*> data::blockStorageBuffer = {};
        int data::blockStoragePointer=-1;

        int data::blockRowsL2 = 0;
        int data::blockSizeL2 = 2048;
        int data::blockScaleL2 = 2048;
        std::vector<int*> data::blockstorageL2 = {};
        int data::storageCountL2 =0;
        std::vector<operation*> data::seqL2 = {};
        bool data::PlanL2=false;
        std::vector<operation*> data::operationBufferL2 = {};
        int data::operationPointerL2 = -1;
        operation* data::newoperationL2(int src1, int s2, blockOp o, int d1, int d2)
        {
            return NULL;
        }
     

        operation* data::newoperation(int src1, int s2, blockOp o, int d1, int d2)
        {
            int last = operationBuffer.size()-1;
            if((s2 > blockstorage.size() && s2>blockstorageL2.size()) || (src1 > blockstorage.size() && src1>blockstorageL2.size()))
         std::cout<<"  src2 "<<s2<<" op "<<o<<" src1 "<<src1<<" result "<<d1<<std::endl;
            if(operationPointer>= 10000 || last == -1 || operationPointer == -1){
                operationBuffer.push_back(new operation[10000]);
                last = operationBuffer.size()-1;
                operationPointer = 0;
            }
            operation* rtn = &operationBuffer[last][operationPointer];
            rtn->src = src1;
            rtn->src2 = s2;
            rtn->op = o;
            rtn->result = d1;
            rtn->result2 = d2;
            rtn->skip = false;
            rtn->stage = 0;
            rtn->sequenceNum = operation::seq;
            operation::seq++;
            operationPointer++;
            return rtn;
        }

        edge* data::newedge(int i, int j)
        {
            int last = edgeBuffer.size()-1;
            if(edgePointer>= 10000 || last == -1 || edgePointer == -1){
                edgeBuffer.push_back(new edge[10000]);
                last = edgeBuffer.size()-1;
                edgePointer = 0;
            }
            edge* rtn = &edgeBuffer[last][edgePointer];
            rtn->row = i;
            rtn->col = j;
            edgePointer++;
            return rtn;
        }

        double* data::newalignedblock()
        {
            int last = blockStorageBuffer.size()-1;
            if(blockStoragePointer>= 1000 || last == -1 || blockStoragePointer == -1){
                double *p = (double *) aligned_alloc (64, BLOCK64 * BLOCK64*sizeof(double) * 1000);
                blockStorageBuffer.push_back(p);
                last = blockStorageBuffer.size()-1;
                blockStoragePointer = 0;
            }
            double* rtn = blockStorageBuffer[last] + blockStoragePointer * BLOCK64 * BLOCK64;
            blockStoragePointer++;
            return rtn;
        }

        int data::checkBandwidth()
        {
            int band = 0;
            int count = 0;
            for (int i = 0; i < data::valcount; i++)
            {
                int diff = data::indexi[i] - data::indexj[i];
                if(diff == 0){
                    count++;
                }
                if (band < diff)
                {
                    band = diff;
                }
                else
                {
                    if (band < -diff)
                    {
                        band = -diff;
                    }
                }
            }
            
            
            broker::postMessage(" raw data band width: " + std::to_string(band) + "  diag non zero: " + 
                     std::to_string(count) + " out of dim: " + std::to_string(mSize));
            return band;
        }

        void data::clear()
        {
           // static int blocks[];

//        static std::vector<double*> blockstorage;
//        static double vals[];
//        static double b[];
//        static double x[];
//        static std::vector<int*> watchlist;
//        static std::vector<operation> seq;
        //std::vector<operation*> data::operationBuffer = {};

          for(operation* p:operationBuffer){
              delete[] p;
          }
          seq.clear();
          operationBuffer.clear();

          for(edge* p:edgeBuffer){
              delete[] p;
          }
          edgeBuffer.clear();
          edgePointer = -1;

          for(double* p:blockStorageBuffer){
              delete[] p;
          }
          blockStorageBuffer.clear();
          blockStoragePointer = -1;

//      std::cout<<"release blockstorage"<<std::endl;
//          for(double* dmp:blockstorage){
//             if(dmp != NULL)
//              	free(dmp);
//          }
          blockstorage.clear();

//      std::cout<<"release blockstorage"<<std::endl;
          for(int* imp:blockstorageL2){
             if(imp != NULL)
              	free(imp);
          }
          blockstorageL2.clear();
//      std::cout<<"release vals"<<std::endl;

          delete[] indexi;
          delete[] indexj;
          delete[] vals;
          delete[] b;
          delete[] x;
          delete[] blocks;

        }
        void data::clearSeq()
        {
          for(operation* p:operationBuffer){
              delete[] p;
          }   
          seq.clear();
          operationBuffer.clear();
          operationPointer = -1;
        }
}

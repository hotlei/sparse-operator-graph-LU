#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <math.h>
#include <chrono>
#include "data.h"
#include "MatrixStdDouble.h"
#include "BlockPlanner.h"
#include "broker.h"
#include "Matrixcuda.h"

namespace vmatrix
{
        ushort BlockPlanner::mask[16] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u,
                256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u};
        int BlockPlanner::countBlockEntry(int a[], int n)
        {
            int rtn=0;
            int b = 0;
            double TINY = 2.0e-10;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int t = a[i * n + j];
                    if (t > 0)
                    {
                        for (int m = 0; m < data::blockSize; m++)
                        {
                            for (int k = 0; k < data::blockSize; k++)
                            {
                                if (data::blockstorage[t][m * data::blockSize + k] > TINY || 
                                                         data::blockstorage[t][m * data::blockSize + k] < -TINY)
                                    rtn++;
                            }
                        }
                        b++;
                    }
                }
            }

            broker::postMessage("block count: " + std::to_string(b) + "     nozeros: " + 
                                       std::to_string(rtn) + "    diag: " + std::to_string(n * data::blockSize));
            return rtn;
        }

          void BlockPlanner::blockPlan(int saved[], int savedLength, int savedu[], int saveduLength)
        {
            int i, cnt = 0;
            //int maxstg = 0;
            //bool done = true;

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
   //         std::cout<<"start storage count "<<data::storageCount<<std::endl;
            for (i = 0; i < data::blockRows_sq; i++)
            {
                if (data::blocks[i] > 0)
                {
                    data::stage[data::blocks[i]] = 1;
                    data::laststage[data::blocks[i]] = 1;
                    lastuse[data::blocks[i]] = 0;
                    cnt++;
                }
            }

      //      std::cout<<"count non zero " <<cnt<< "  storage count " <<data::storageCount <<std::endl;
            if (saved != NULL && savedLength > 0)
            {
                for (i = 0; i < savedLength; i++)
                {
                    if (saved[i] > 0 && saved[i] <data::storageCount)
                    {
                        lastuse[saved[i]] = maxx;
                        data::laststage[saved[i]] = maxx;
                        maxcnt++;
                    }
                }
            }

            if (savedu != NULL && saveduLength > 0)
            {
                for (i = 0; i < saveduLength; i++)
                {
                    if (savedu[i] > 0 && savedu[i] <data::storageCount)
                    {
                        lastuse[savedu[i]] = maxx;
                        data::laststage[savedu[i]] = maxx;
                        maxcnt++;
                    }
                }
            }
    //        std::cout<<" result count: "<< maxcnt<<" result stage:"<<maxx<<std::endl;

            broker::postMessage("total storage: " + std::to_string(data::storageCount) + "    ini stage: " + std::to_string(cnt) + "    operations: " + std::to_string(data::seq.size()) + "    start planning: " );

    //        std::cout<<"seq size: "<<data::seq.size()<<std::endl;
            for (i = data::seq.size() - 1; i >= 0; i--)
            {
                operation o = *(data::seq[i]);

//                    std::cout<<"seq result "<<o.result<< " result2 " << o.result2<< " i: " << i<<std::endl;
//                    std::cout<<"seq src "<<o.src<< " res2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<o.result<<" op "<<o.op<<std::endl;
                int stg = 0;
                if (o.result > 0)
                {
                    stg = o.result;
                    if (o.result2 > 0 && o.result2 > stg)
                    {
                        stg = o.result2;
                    }
                }

                if (o.result > 0 && o.result2 > 0)
                {
                    if (lastuse[o.result] == 0 && lastuse[o.result2] == 0)
                        continue;
                }

                if (o.result > 0 && o.result2 == 0)
                {
                    if (lastuse[o.result] == 0)
                        continue;
                }

                if (stg > 0)
                {
                    if (o.src > 0){
                        if (lastuse[o.src] < stg)
                            lastuse[o.src] = stg;
                    }
                    if (o.src2 > 0)
                    {
                        if(o.src2>= data::storageCount){
       std::cout<<"o.src2 "<<o.src2 << " "<<data::storageCount<<" op "<<o.op<<" src1 "<<o.src<<" result "<<o.result<<std::endl;
                           continue;
                        }
                        if (lastuse[o.src2] < stg)
                            lastuse[o.src2] = stg;
                    }
 //                   std::cout<<" stge "<<stg<< " i: " << i<<std::endl;
                }
            }
  //          std::cout<<"count non zero " <<cnt<< "  storage count " <<maxcnt <<std::endl;

            int wastecount = 0;

            for (i = 0; i < data::seq.size(); i++)
            {
                operation o = *(data::seq[i]);
                if (lastuse[o.result] > 0) continue;
                if (o.result2 > 0 && lastuse[o.result2] > 0) continue;
                ++wastecount;
            }
//            std::cout<<" waste " <<wastecount<< std::endl;
            int bufcount = data::seq.size() - wastecount;
            operation *buf = new operation[bufcount];
            int bufcounter = 0;
            for (i = 0; i < data::seq.size(); i++)
            {
                operation o = *(data::seq[i]);
                
                if (lastuse[o.result] > 0)
                {
                    buf[bufcounter].sett(o.src, o.src2, o.op, o.result, o.result2);
                    bufcounter++;
                    continue;
                }
                if (o.result2 > 0 && lastuse[o.result2] > 0)
                {
                    buf[bufcounter].sett(o.src, o.src2, o.op, o.result, o.result2);
                    bufcounter++;
                    continue;
                }
                
            }
 //           std::cout<<" seq size " <<bufcount<< std::endl;

            data::clearSeq();
            for(i=0; i< bufcount;i++){
                data::seq.push_back(data::newoperation(buf[i].src,buf[i].src2, buf[i].op, buf[i].result, buf[i].result2));
            }
            delete[] buf;

            broker::postMessage("reduced ops to: " + std::to_string(data::seq.size()) + "      start allocate memory: " );

            int maxstg = 1;
            for (operation* oo : data::seq)
            {
                operation o = *oo;
                int stg = 0;
               // if (data::stage[o.src] > 0)
                if (o.src > 0)
                {
                    stg = data::stage[o.src] + 1;
                    if(stg < 2) stg = 2;
                  //  if (data::stage[o.src2] > 0)
                    if (o.src2 > 0)
                    {
                        if (data::stage[o.src2] + 1 > stg)
                        {
                            stg = data::stage[o.src2] + 1;
                        }
                    }
                }else{

                    if (o.src2 > 0)
                    {
                        if (data::stage[o.src2] + 1 > stg)
                        {
                            stg = data::stage[o.src2] + 1;
                        }
                    }
                    if(stg < 2) stg = 2;
                }

                if (stg > 1)
                {
                    oo->stage = stg;
                    if (o.result > 0 && data::stage[o.result] < stg)
                    {
                        data::stage[o.result] = stg;
                    }
                    if (o.result2 > 0 && data::stage[o.result2] < stg)
                    {
                        data::stage[o.result2] = stg;
                    }
                    if (stg > maxstg)
                        maxstg = stg;
                }
            }

            broker::postMessage("total stage: " + std::to_string(maxstg));

            for (operation* o : data::seq)
            {
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
       //     broker::postMessage("sorting");

            std::sort (data::seq.begin(), data::seq.end(), operationCompare);

            int memCount = 0;
            for (i = 0; i < data::blockRows_sq; i++)
            {
                if (data::blocks[i] > 0)
                {
                    memCount++;
                }
            }

//            broker::postMessage("input nonzero count: " + std::to_string(memCount) + "   result size: " + std::to_string(maxcnt));
            delete [] lastuse;
        }

          void BlockPlanner::calculate()
        {
            int count = 0;
            int clearcount = 0;
            int alloccount = 0;
         
            int invcount = 0;
            int linvcount = 0;
            int lucount = 0;
            int uinvcount = 0;
            int subcount = 0;

            int curStage = -1;
            std::vector<int> tbd; 
            tbd.reserve(10000);
            double tmpinv[4096];
            data::mulcount=0;

            for (int i = 0; i < data::seq.size(); i++)
            {
                operation o = *(data::seq[i]);
                if (o.skip)
                    continue;
     //               std::cout<<"seq src "<<o.src<< " src2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<o.result<<" op "<<o.op<<std::endl;
                count++;
                if (o.result > 0)
                {
                    if (data::blockstorage[o.result] == NULL)
                    {
                //        double* dmp = (double *) aligned_alloc (64, data::blockSize * data::blockSize*sizeof(double));
                        double* dmp = data::newalignedblock();
 
                        for(int ii=0;ii<data::blockSize * data::blockSize;ii++){
                            dmp[ii] = 0;
                        }
                        appendBlockStorageAgain(dmp, o.result);
                        alloccount++;
                    }
                }

                if (o.result2 > 0)
                {
                    if (data::blockstorage[o.result2] == NULL)
                    {
               //         double* dmp = (double *) aligned_alloc (64, data::blockSize * data::blockSize*sizeof(double));
                        double* dmp = data::newalignedblock();
 
                        for(int ii=0;ii<data::blockSize * data::blockSize;ii++){
                            dmp[ii] = 0;
                        }
                        appendBlockStorageAgain(dmp, o.result2);
                        alloccount++;
                    }
                }
        //            std::cout<<"seq src "<<data::blockstorage[o.src]<< " res2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<data::blockstorage[o.result]<<" op "<<o.op<<std::endl;

                switch (o.op)
                {
                    case blockOp::inv:
                        MatrixStdDouble::mat_inv(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        invcount++;
                        break;
                    case blockOp::lowerInv:
                        MatrixStdDouble::inv_lower(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        if(!MatrixStdDouble::inv_check_diag(data::blockstorage[o.src],data::blockstorage[o.result],data::blockSize)){
                            std::cout<<" lower out of tolerance at: "<<linvcount<<std::endl;
                        }
                        linvcount++;
                        if(linvcount==5){
                  //          MatrixStdDouble::printMatrix(data::blockstorage[o.src],data::blockSize);
                  //          MatrixStdDouble::printMatrix(data::blockstorage[o.result],data::blockSize);
                        }
                        break;
                    case blockOp::lu:
                  //      std::cout<<" lu start: "<<lucount<<std::endl;
                        MatrixStdDouble::ludcmpSimple(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result], data::blockstorage[o.result2]);
                        MatrixStdDouble::mat_mul(data::blockstorage[o.result], data::blockstorage[o.result2],
                                  data::blockSize, data::blockSize, data::blockSize, tmpinv);
                        if(!MatrixStdDouble::mat_equ(data::blockstorage[o.src],tmpinv,data::blockSize)){
                            std::cout<<" lu out of tolerance at: "<<lucount<<std::endl;
                        }
                        if(lucount == 1034)
                        {
              //              std::cout<<" lu no: "<<lucount<<std::endl;
               //             MatrixStdDouble::printMatrix(data::blockstorage[o.src],data::blockSize);
                //            MatrixStdDouble::printMatrix(data::blockstorage[o.result],data::blockSize);
                 //           MatrixStdDouble::printMatrix(data::blockstorage[o.result2],data::blockSize);
                        }
                        lucount++;
                        break;
                    case blockOp::mul:


                   //     MatrixStdDouble::blockMulOneAvx(data::blockstorage[o.src], data::blockstorage[o.src2], data::blockstorage[o.result]);
                   //     MatrixStdDouble::blockMulOneAvxBlock(data::blockstorage[o.src], data::blockstorage[o.src2], data::blockstorage[o.result], 0); 
                        blockMulOneNaive(data::blockstorage[o.src], data::blockstorage[o.src2], data::blockstorage[o.result]);
                   //      mat_mul(data::blockstorage[o.src], data::blockstorage[o.src2], data::blockSize, data::blockstorage[o.result]);
                        break;
                    case blockOp::mulneg:

                        blockMulOneNegNaive(data::blockstorage[o.src], data::blockstorage[o.src2], data::blockstorage[o.result]);
                  //      MatrixStdDouble::blockMulOneNegAvx(data::blockstorage[o.src], data::blockstorage[o.src2], data::blockstorage[o.result]);
                        break;
                    case blockOp::sub:

                        subcount++;
                        if (o.src2 > 0 && data::blockstorage[o.src2] != NULL && o.src > 0 && data::blockstorage[o.src] != NULL)
                        {
                            MatrixStdDouble::mat_sub(data::blockstorage[o.src2], data::blockstorage[o.src],
                                                         data::blockSize, data::blockstorage[o.result]);
                            break;
                        }

                        if(o.src2 > 0 && data::blockstorage[o.src2] != NULL){
                            MatrixStdDouble::mat_copy(data::blockstorage[o.src2], 
                                                         data::blockSize, data::blockstorage[o.result]);
                            break;
                        }
                       

                        if (o.src > 0 && data::blockstorage[o.src] != NULL)
                        {
                             MatrixStdDouble::mat_neg(data::blockstorage[o.src],
                                                         data::blockSize, data::blockstorage[o.result]);
                        }
                        break;
                    case blockOp::upperInv:
                        MatrixStdDouble::inv_upper(data::blockstorage[o.src], data::blockSize, data::blockstorage[o.result]);
                        if(!MatrixStdDouble::inv_check_diag(data::blockstorage[o.src],data::blockstorage[o.result],data::blockSize)){
                            std::cout<<" upper out of tolerance at: "<<uinvcount<<std::endl;
                        }
                        uinvcount++;
                        break;
                    default:
                        break;
                }

         //         if(o.src>0)
         //       if(data::stage[o.src] == 1)
         //           std::cout<<"stage no. "<< curStage<< "  "<<o.stage<< "  count "<<tbd.size()<<" laststage "<<o.src<< ": "<<data::stage[o.src]<< ": "<<data::laststage[o.src]<<std::endl;
                if(o.stage != curStage)
                {
                    std::sort (tbd.begin(), tbd.end());
                    int ddx = -1;
                    for (int idx = 0;idx <tbd.size(); idx++)
                    {
             //   std::cout<<" debuging here "<<data::blockstorage[tbd[idx]]<<std::endl;
                        if(tbd[idx] == ddx)
                             continue;
                        ddx = tbd[idx];
                        if(data::blockstorage[tbd[idx]] != NULL){
                          free( data::blockstorage[tbd[idx]] );
                          data::blockstorage[tbd[idx]] = NULL;
                          clearcount++;
                        }
                    }
                    tbd.clear();
                    curStage = o.stage;
                }
                if (o.src > 0)
                {
                    if(data::laststage[o.src] == o.stage)
                    {
                        tbd.push_back(o.src);
                    }
                }
                if (o.src2 > 0)
                {
                    if (data::laststage[o.src2] == o.stage)
                    {
                        tbd.push_back(o.src2);
                    }
                }
            }

    //        broker::postMessage("   total ops: " + std::to_string(count));
            std::cout<<" clear count " << clearcount<<" alloc " << alloccount<<std::endl;
            std::cout<<"inv "<<invcount<<" lower inv "<<linvcount<<" upper inv "<<uinvcount<<" lu "<<lucount<<" sub "<<subcount<<" mul "<<data::mulcount<<std::endl;
        }

          void BlockPlanner::sovle(int* bl, int* bu, double* b, int n)
        {
            double* y = new double[n * data::blockSize];
            double* x = new double[n * data::blockSize];
            int i,j;

            for(i=0;i<n* data::blockSize;i++){
                y[i] = 0;
                x[i] = 0;
            }
            for (i = 0; i < n; i++)
            {
                for(j=0; j< i; j++)
                {
                    if (bl[i * n + j] > 0)
                    {
                        cy(data::blockstorage[bl[i * n + j]], b, y, data::blockSize, i * data::blockSize, j * data::blockSize);
                    }
                }
                solveLower(data::blockstorage[bl[i * n + i]], b, y, data::blockSize, i * data::blockSize);
            }

            for(i=n-1; i>=0; i--)
            {
                for (j = i + 1; j < n; j++)
                {
                    if (bu[i * n + j] > 0)
                    {
                        bx(data::blockstorage[bu[i * n + j]], y, x, data::blockSize, i * data::blockSize, j * data::blockSize);
                    }
                }
                solveUpper(data::blockstorage[bu[i * n + i]], y, x, data::blockSize, i * data::blockSize);
            }

            int nancount = 0;
            
            if(data::x != NULL)
                delete[] data::x;
            data::x = new double[data::mSize];
            for (i = 0; i < data::mSize; i++)
            {
                data::x[i] = x[i];
                if (std::isnan(x[i]))
                {
                    nancount++;
                }
            }

            delete[] x;
            delete[] y;
            if(nancount>0)
                broker::postMessage("found NaN:  " + std::to_string(nancount));
        }

          void BlockPlanner::assignB()
        {
            configL2();
            if(data::b != NULL)
                delete[] data::b;
            data::b = new double[data::blockRows * data::blockSize];
            for (int i = 0; i < data::blockRows * data::blockSize; i++)
            {
                data::b[i] = 0;
            }
            for (int i = 0; i < data::valcount; i++)
            {
                if(data::indexi[i]<0){
                    std::cout<<"indexi "<<data::indexi[i]<<" i "<<i<<std::endl;
                }
                data::b[data::indexi[i]] += data::indexj[i] * data::vals[i];
            }
            for (int i = data::mSize; i < data::blockRows * data::blockSize; i++)
                data::b[i] = 1;
        }

          void BlockPlanner::solveUpper(double bu[], double y[], double x[], int n, int offset)
        {
            for(int i = n - 1; i >= 0; i--)
            {
                for(int j=i+1; j<n; j++)
                {
                    y[i + offset] -= bu[i * n + j] * x[j + offset];
                }
                x[i + offset] = y[i + offset] / bu[i * n + i];
            }
        }
          void BlockPlanner::bx(double bu[], double y[], double x[], int n, int ioffset, int joffset)
        {
//#pragma omp parallel for
            for(int m=0;m<8;m++){
                double* yy = y+ioffset;
                double* xx = x + joffset;
            for (int p = 0; p < 8; p++)
            {
                int i = m*8+p;
                double* bb=bu + i* BLOCK64;
                double tmp = 0;
//#pragma omp parallel for simd schedule(static, 16)
//#pragma omp simd reduction(+:tmp)
                for (int j = 0; j < BLOCK64; j++)
                {
                    tmp += bb[j] * xx[j];
                }

                yy[i] -= tmp;
            }
            }
        }

          void BlockPlanner::solveLower(double bl[], double b[], double y[], int n, int offset)
        {
            for(int i=0; i< n; i++)
            {
                for(int j=0; j < i; j++)
                {
                    b[i + offset] -= bl[i * n + j] * y[j + offset];
                }
                y[i + offset] = b[i + offset] / bl[i * n + i];
            }
        }

          void BlockPlanner::cy(double bl[], double b[], double y[], int n, int ioffset, int joffset)
        {
//#pragma omp parallel for
            for(int m=0;m<8;m++){
                double* bb = b+ioffset;
                double* yy = y + joffset;
            for (int p = 0; p < 8; p++)
            {
                int i=m*8+p;
                double* bll = bl+ i * BLOCK64;
                double tmp = 0;
//#pragma omp parallel for simd schedule(static, 16)
//#pragma omp simd reduction(+:tmp)
                for (int j = 0; j < BLOCK64; j++)
                {
                    tmp += bll[j] * yy[j];
                }

                bb[i] -= tmp;
            }
            }
        }

        
          int BlockPlanner::countStrorageEntry(int n, int stage)
        {
            int rtn = 0;
            double TINY = 2.0e-20f;

            for (int t = 1; t < n; t++)
            {               
                        for (int m = 0; m < data::blockSize; m++)
                        {
                            for (int k = 0; k < data::blockSize; k++)
                            {
                                if (data::blockstorage[t][m * data::blockSize + k] > TINY || data::blockstorage[t][m * data::blockSize + k] < -TINY)
                                    rtn++;
                         //       if (double.IsNaN(data::blockstorage[t][m * data::blockSize + k]))
                            }
                        }        
            }

            broker::postMessage("storage entry count: " + std::to_string(rtn) + " stage: " + std::to_string(stage));
            return rtn;
        }

          int BlockPlanner::checkMatrixEntry(int a[], int n)
	{
            int rtn = 0;
            for(int i=0;i<n;i++){
                int line = 0;
                for(int j=0;j<n;j++){
   if(i>=16&&i<=23) std::cout<<"     \b"<<a[i*n+j];
                    if(a[i*n+j]>0){
                        line++;
                    }
                }
                std::cout<<" line "<<i<<": "<<line<<std::endl;
                rtn += line;
            }
            std::cout<<"total non empty "<<rtn<<std::endl;
            return rtn;
        }
          double* BlockPlanner::checkDiagEntry(int a[], int n)
        {
            int rtn = 0;
            int b = 0;
            int nancount=0;
            int infcount=0;
            double TINY = 1.0e-20f;
            double* dia = new double[n * data::blockSize];
            double sum = 0;
            double prod = 1;
            int    prodinx = 0;
            for(int i=0;i<n * data::blockSize;i++){
                dia[i] = 0;
            }

            for (int i = 0; i < n; i++)
            {
                int j = i;
                {
                    int t = a[i * n + j];
                    if (t > 0)
                    {
                        for (int m = 0; m < data::blockSize; m++)
                        {
                            int k = m;
                            {
                                dia[i * data::blockSize + m] = data::blockstorage[t][m * data::blockSize + k];
                                if(std::isinf(data::blockstorage[t][m * data::blockSize + k])){
                //           std::cout<<i * data::blockSize + m<<": "<<  data::blockstorage[t][m * data::blockSize + k]<< std::endl;
                                    infcount++;
                                }
                                if(std::isnan(data::blockstorage[t][m * data::blockSize + k])){
                 //          std::cout<<i * data::blockSize + m<<": "<<  data::blockstorage[t][m * data::blockSize + k]<< std::endl;
                                    nancount++;
                                }
                                prod *= data::blockstorage[t][m * data::blockSize + k];
                                if(prod > 1e10 || prod < -1e10){
                                    prod *= 1e-10;
                                    prodinx += 10;
                                }
                                if(prod > -1e-3 && prod < 1e-3){
                                    prod *= 1e10;
                                    prodinx -= 10;
                                }
                                if (data::blockstorage[t][m * data::blockSize + k] > TINY || data::blockstorage[t][m * data::blockSize + k] < -TINY)
                                {
                                    rtn++;
                                    sum += data::blockstorage[t][m * data::blockSize + k];
      //            if(m==0 || m == data::blockSize/2)
       //              std::cout<<i * data::blockSize + m<<": "<<  data::blockstorage[t][m * data::blockSize + k]<< std::endl;
                                }
                            }
                        }
                    }
                }
            }

     //       broker::postMessage("diag count: " + std::to_string(rtn) + " sum " + std::to_string(sum) + " prod " + std::to_string(prod)
      //         +" e " + std::to_string(prodinx) +" inf " + std::to_string(infcount) + " nan " + std::to_string(nancount));
            return dia;
        }

          void BlockPlanner::blockLUOne(int a[], int l[], int u[])
        {
            double* tmpl = NULL;
            double* tmpu = NULL;
            if (data::UseCPU)
            {
                tmpl = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmpl);
                tmpu = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmpu);
                MatrixStdDouble::ludcmpSimple(data::blockstorage[a[0]], data::blockSize, tmpl, tmpu);
            }
            l[0] = appendBlockStorage(tmpl);
            u[0] = appendBlockStorage(tmpu);

            data::seq.push_back(data::newoperation(a[0], 0, blockOp::lu, l[0], u[0]));
            if(a[0] == 0)
            {
                a[0] = 0;
            }
            data::lucount++;
        }
          void BlockPlanner::blockLU(int a[], int l[], int u[], int n)
        {
            if(n == 1)
            {
                blockLUOne(a, l, u);
                return;
            }
            int hn = n / 2;
 //               broker::postMessage("half size : " + std::to_string(hn) + "  op count: " + std::to_string(data::seq.size()));
            int* ha = getBlockList(a,n, 0, 0, hn);
            int* hinv = new int[hn * hn];
            //blockInv(ha, hinv, hn);

            
            int* l1 = new int[hn * hn];
            int* u1 = new int[hn * hn];
            int* l2 = new int[hn * hn];
            int* u2 = new int[hn * hn];
            for(int i=0;i<hn*hn;i++){
                l1[i] = 0;
                u1[i] = 0;
                l2[i] = 0;
                u2[i] = 0;
                hinv[i] = 0;
            }

            blockLU(ha, l1, u1, hn);
            blockInvLower(l1, l2, hn);
            blockInvUpper(u1, u2, hn);
         //   if(n>=1024)
      //          broker::postMessage("size : " + std::to_string(n) + " mul start: " + "  op count: " + std::to_string(data::seq.size()));
            if(hn==1){
      //          broker::postMessage("size : " + std::to_string(n) + " block inv1 " );
      //          blockInv(ha,hinv,hn);
      //xx          blockMul(u2, l2, hinv, hn);  //??
            }else{
     //xx           blockMul(u2, l2, hinv, hn);
            }
         //   if (n >= 1024)
      //          broker::postMessage("size : " + std::to_string(n) + " mul stop: "  + "  op count: " + std::to_string(data::seq.size()));

    //            broker::postMessage("update : " + std::to_string(hn) + " watch stop: "  + "  op count: " + std::to_string(data::seq.size()));
            putBlockList(l,n, 0, 0, hn, l1);
     //           broker::postMessage("update : " + std::to_string(hn) + " watch stop: "  + "  op count: " + std::to_string(data::seq.size()));
            putBlockList(u,n, 0, 0, hn, u1);
            int* b = getBlockList(a,n, 0, hn, hn);
       //         std::cout << "b : " << b << std::endl;

            int* a1b = new int[hn * hn];
            for(int i=0;i<hn*hn;i++){
                a1b[i] = 0;
            }
        //        std::cout << "a1b : " << a1b << std::endl;
//xx            blockMul(hinv, b, a1b, hn);
         //       std::cout << "a1b : " << a1b << std::endl;
            int* aua1b = new int[hn * hn];
            for(int i=0;i<hn*hn;i++){
                aua1b[i] = 0;
            }
   //             std::cout << "aua1b : " << aua1b << std::endl;
            if(hn==1)
   //             blockMul(u1, a1b, aua1b, hn);
                blockMul(l2, b, aua1b, hn);  
            else
                blockMul(l2, b, aua1b, hn);  
            putBlockList(u,n, 0, hn, hn, aua1b);
 //               broker::postMessage("size : " + std::to_string(n) + " watch stop: "  + "  op count: " + std::to_string(data::seq.size()));

            int* c = getBlockList(a, n,hn, 0, hn);
            int* ca1 = new int[hn * hn];
            int* ca1al = new int[hn * hn];
            for(int i=0;i<hn*hn;i++){
                ca1[i] = 0;
                ca1al[i] = 0;
            }
   //xx         blockMul(c, hinv, ca1, hn);
            if(hn == 1)
  //              blockMul(ca1, l1, ca1al, hn);
                blockMul(c, u2, ca1al, hn); 
            else
                blockMul(c, u2, ca1al, hn); 
            putBlockList(l,n, hn, 0, hn, ca1al);
  //              broker::postMessage("size : " + std::to_string(n) + " watch stop  2: "  + "  op count: " + std::to_string(data::seq.size()));

            int* ca1b = new int[hn * hn];
            int* dsub = new int[hn * hn];
            for(int i=0;i<hn*hn;i++){
                ca1b[i] = 0;
                dsub[i] = 0;
            }
            blockMul(ca1al, aua1b, ca1b, hn); // (ca1, b, ca1b, hn)
            int* d = getBlockList(a,n, hn, hn, hn);
            blockSub(d, ca1b, dsub, hn);
            resetBlocks(l1, hn);
            resetBlocks(u1, hn);
   //             broker::postMessage("size : " + std::to_string(n) + " watch stop 3  "  + "  op count: " + std::to_string(data::seq.size()));
            blockLU(dsub, l1, u1, hn);
            putBlockList(l, n, hn, hn, hn, l1);
            putBlockList(u, n, hn, hn, hn, u1);

            delete []ha;
            delete []l1;
            delete []u1;
            delete []l2;
            delete []u2;
            delete []hinv;
            delete []b;
            delete []a1b;
            delete []aua1b;
            delete []c;
            delete []ca1;
            delete []ca1al;
            delete []ca1b;
            delete []d;
            delete []dsub;
        }
          void BlockPlanner::blockInvOne(int a[], int y[])
        {
            double* tmp = NULL;
            if (data::UseCPU)
            {
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
                MatrixStdDouble::mat_inv(data::blockstorage[a[0]], data::blockSize, tmp);
            }
       //         broker::postMessage("push : block inv1 " );
            y[0] = BlockPlanner::appendBlockStorage(tmp);
            data::seq.push_back(data::newoperation(a[0], 0, blockOp::inv, y[0], 0));
            data::invertcount++;
        }
          void BlockPlanner::blockInv(int a[], int y[], int n)
        {
            if (n == 1)
            {
                blockInvOne(a, y);
                return;
            }
            int* l = new int[n * n];
            int* u = new int[n * n];
            int* l1 = new int[n * n];
            int* u1 = new int[n * n];
            for(int i=0; i<n*n; i++){
                l[i] = 0;
                u[i] = 0;
                l1[i] = 0;
                u1[i] = 0;
            }
            blockLU(a, l, u, n);
            blockInvLower(l, l1, n);
            blockInvUpper(u, u1, n);
            blockMul(u1, l1, y, n);

            delete []l;
            delete []u;
            delete []l1;
            delete []u1;
        }
          void BlockPlanner::blockInvLowerOne(int l[], int y[])
        {
            double* tmp = NULL;
            if (data::UseCPU)
            {
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
                MatrixStdDouble::inv_lower(data::blockstorage[l[0]], data::blockSize, tmp);
            }
            y[0] = BlockPlanner::appendBlockStorage(tmp);
            //if (data::seq.Count > data::seq.Capacity- (data::seq.Capacity>>2))
            //{
            //    data::seq.Capacity = data::seq.Capacity + (data::seq.Capacity >> 1);
            //}
            data::seq.push_back(data::newoperation(l[0], 0, blockOp::lowerInv, y[0], 0));
            data::linvcount++;
        }
          void BlockPlanner::blockInvLower(int l[], int y[], int n)
        {
            
            if (n == 1)
            {
                blockInvLowerOne(l, y);
                return;
            }
            int n2 = n / 2;
            int nn4 = n2 * n2;
            int *a = getBlockList(l,n, 0, 0, n2);
            int *c = getBlockList(l,n, n2, 0, n2);
            int *d = getBlockList(l,n, n2, n2, n2);

            int *d1 = new int[nn4];
            for (int i=0; i< nn4; i++)
               d1[i] = 0;
            blockInvLower(d, d1, n2);
            putBlockList(y, n, n2, n2, n2, d1);

            int *d1c = d;
            for(int i=0;i<nn4; i++)
               d1c[i] = 0;
            blockMul(d1, c, d1c, n2);

            int *a1 = d1;
            for(int i=0;i<nn4; i++)
               a1[i] = 0;
            blockInvLower(a, a1, n2);
            putBlockList(y, n, 0, 0, n2, a1);

            int *c21 = c;
            for(int i=0;i<nn4; i++)
               c21[i] = 0;
            blockMulNeg(d1c, a1, c21, n2);
            putBlockList(y,n, n2, 0, n2, c21);

            delete [] d1c;
            delete [] c21;
            delete [] a;
            delete []a1;
        }

          void BlockPlanner::blockInvUpperOne(int u[], int y[])
        {
            double *tmp = NULL;
            if (data::UseCPU)
            {
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
                MatrixStdDouble::inv_upper(data::blockstorage[u[0]], data::blockSize, tmp);
            }
            y[0] = BlockPlanner::appendBlockStorage(tmp);
            data::seq.push_back(data::newoperation(u[0], 0, blockOp::upperInv, y[0], 0));
            data::uinvcount++;
        }
          void BlockPlanner::blockInvUpper(int u[], int y[], int n)
        {

            if (n == 1)
            {
                blockInvUpperOne(u, y);
                return;
            }
            int n2 = n / 2;
            int nn4 = n2 * n2;

            int *a = getBlockList(u,n, 0, 0, n / 2);
            int *b = getBlockList(u,n, 0, n / 2, n / 2);
            int *d = getBlockList(u,n, n / 2, n / 2, n / 2);

            int *a1 = new int[n / 2 * n / 2];
            for (int i=0; i< n*n/4; i++)
               a1[i] = 0;
            blockInvUpper(a, a1, n / 2);
            putBlockList(y, n, 0, 0, n / 2, a1);

            int *a1b = a;
            for(int i=0;i<nn4; i++)
               a1b[i] = 0;
            blockMul(a1, b, a1b, n / 2);

            int *d1 = a1;
            for(int i=0;i<nn4; i++)
               d1[i] = 0;
            blockInvUpper(d, d1, n / 2);
            putBlockList(y, n, n / 2, n / 2, n / 2, d1);

            int *a1bd1 = b;      
            for(int i=0;i<nn4; i++)
               a1bd1[i] = 0;
            blockMulNeg(a1b, d1, a1bd1, n / 2);
            putBlockList(y,n, 0, n / 2, n / 2, a1bd1);

            delete [] a1bd1;
            delete [] d1;
            delete [] a1b;
            delete [] d;

        }

          void BlockPlanner::blockScanNAN(int a[], int n)
        {
            int rtn = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (a[i * n + j] == 0) continue;


                    for (int m = 0; m < data::blockSize; m++)
                        for (int k = 0; k < data::blockSize; k++)
                        {
                            //if(double.IsNaN(data::blockstorage[a[i * n + j]][m * data::blockSize + k]))
                            // rtn++;
                        }

                }
            }
            broker::postMessage("number of NAN: " + std::to_string(rtn) + " matrix block rows: " + std::to_string(n));
        }

          void BlockPlanner::printInt(int a[], int n)
          {
               int line = 8;
  /*             for(int i=0;i<n*n;i++){
                  std::cout<<a[i]<<"  \b";
                  if((i%line)==line-1)
                     std::cout<<std::endl;
               }
  /*    */    }

          void BlockPlanner::blockSub(int a[], int b[], int d[], int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (b[i * n + j] == 0 && a[i * n + j] == 0) continue;

                    double* tmp = NULL;
                    d[i * n + j] = appendBlockStorage(tmp);

                    if (data::UseCPU)
                    {
                        tmp = new double[data::blockSize * data::blockSize];   // ????
                        resetOneBlock(tmp);   // ????
                        appendBlockStorageAgain(tmp, d[i*n+j]);  // ????
                        if (a[i * n + j] > 0)
                        {
                            for (int m = 0; m < data::blockSize; m++)
                                for (int k = 0; k < data::blockSize; k++)
                                {
                                    data::blockstorage[d[i * n + j]][m * data::blockSize + k] = data::blockstorage[a[i * n + j]][m * data::blockSize + k];
                                }
                        }

                        if (b[i * n + j] > 0)
                        {
                            for (int m = 0; m < data::blockSize; m++)
                                for (int k = 0; k < data::blockSize; k++)
                                {
                                    data::blockstorage[d[i * n + j]][m * data::blockSize + k] -= data::blockstorage[b[i * n + j]][m * data::blockSize + k];
                                }
                        }
                    }

                    data::seq.push_back(data::newoperation(b[i * n + j], a[i * n + j], blockOp::sub, d[i * n + j], 0));
                }
            }
        }
          void BlockPlanner::blockCopy(int a[], int d[], int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (a[i * n + j] == 0) continue;

                    double* tmp = NULL;
                    d[i * n + j] = appendBlockStorage(tmp);

                    if (data::UseCPU)
                    {
                        tmp = new double[data::blockSize * data::blockSize];   // ????
                        resetOneBlock(tmp);   // ????
                        appendBlockStorageAgain(tmp, d[i*n+j]);  // ????
                        if (a[i * n + j] > 0)
                        {
                            for (int m = 0; m < data::blockSize; m++)
                                for (int k = 0; k < data::blockSize; k++)
                                {
                                    data::blockstorage[d[i * n + j]][m * data::blockSize + k] = data::blockstorage[a[i * n + j]][m * data::blockSize + k];
                                }
                        }
                    }

                    data::seq.push_back(data::newoperation(0, a[i * n + j], blockOp::sub, d[i * n + j], 0));
                }
            }
        }
          void BlockPlanner::blockNeg(int b[], int d[], int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (b[i * n + j] == 0 ) continue;

                    double* tmp = NULL;
                    d[i * n + j] = appendBlockStorage(tmp);

                    if (data::UseCPU)
                    {
                        tmp = new double[data::blockSize * data::blockSize];   // ????
                        resetOneBlock(tmp);   // ????
                        appendBlockStorageAgain(tmp, d[i*n+j]);  // ????

                        if (b[i * n + j] > 0)
                        {
                            for (int m = 0; m < data::blockSize; m++)
                                for (int k = 0; k < data::blockSize; k++)
                                {
                                    data::blockstorage[d[i * n + j]][m * data::blockSize + k] = -data::blockstorage[b[i * n + j]][m * data::blockSize + k];
                                }
                        }
                    }

                    data::seq.push_back(data::newoperation(b[i * n + j], 0, blockOp::sub, d[i * n + j], 0));
                }
            }
        }
        bool BlockPlanner::isZeroBlock(double tmp[])
        {
            double TINY = 2.0e-14;

            int n = data::blockSize;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {

                    if (tmp[i * n + j] > TINY || tmp[i * n + j] < -TINY)
                    {
                        return false;
                    }
                }
            }
            data::foundzeroblock++;
            return true;
        }

          void BlockPlanner::blockMulSparseNeg(int aa[], int bb[], int c[], int n)
        {
            //int[][] a = new int[n][];
            //int[] astart = new int[n];
            //int[] alen = new int[n];
            //int** a = (int**) malloc(n * sizeof(int *));
            //int* astart = (int*) malloc(n * sizeof(int));
            //int* alen = (int*) malloc(n * sizeof(int));
            int **a = new int*[n];
            int *astart = new int[n];
            int *alen = new int[n];

            for(int p=0;p<n;p++){
                a[p] = NULL;
                astart[p] = 0;
                alen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int pn = p * n;
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (aa[pn + q] > 0)
                    {
                        end = q;
                        break;
                    }
                }
                for (int q = 0; q < n; q++)
                {
                    if (aa[pn + q] > 0)
                    {
                        astart[p] = q;
                        break;
                    }
                }
                alen[p] = end - astart[p] + 1;
                a[p] = (int *) malloc ( alen[p] * sizeof(int));
                for (int q = 0; q < alen[p]; q++)
                {
                    a[p][q] = aa[pn + astart[p] + q];
                }
            }

            //int** b = (int**) malloc(n * sizeof(int *));
            //int* bstart = (int*) malloc(n * sizeof(int));
            //int* blen = (int*) malloc(n * sizeof(int));
            int **b = new int*[n];
            int *bstart = new int[n];
            int *blen = new int[n];
            for(int p=0;p<n;p++){
                b[p] = NULL;
                bstart[p] = 0;
                blen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (bb[q * n + p] > 0)
                    {
                        end = q;
                        break;
                    }
                }
                for (int q = 0; q < n; q++)
                {
                    if (bb[q * n + p] > 0)
                    {
                        bstart[p] = q;
                        break;
                    }
                }
                blen[p] = end - bstart[p] + 1;
                b[p] = (int *) malloc(blen[p] * sizeof(int));
                for (int q = 0; q < blen[p]; q++)
                {
                    b[p][q] = bb[(q + bstart[p]) * n + p];
                }
            }

            double* tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                int ni = astart[i];
                for (int j = 0; j < n; j++)
                {
                    flag = true;
                    int jn = bstart[j];
                    int ks = ni > jn ? ni : jn;
                    int ke = ni + alen[i] < jn + blen[j] ? ni + alen[i] : jn + blen[j];
                    for (int k = ks; k < ke; k++)
                    {
                        if (a[i][k - ni] == 0 || b[j][k - jn] == 0)
                            continue;
                        flag = false;
                    }

                    if (flag)
                          continue;
            //            c[n * i + j] = 0;
                    else
                    {
                        if(c[n * i + j] == 0)
                        c[n * i + j] = appendBlockStorage(tmp);

                        for (int k = ks; k < ke; k++)
                        {
                            int kn = k * n;
                            if
                                (a[i][-ni + k] == 0 || b[j][-jn + k] == 0)
                                //       (a[ni + k] == 0 || b[kn + j] == 0) 
                                continue;
                            if (data::UseCPU)
                            {
                                blockMulOneNeg(data::blockstorage[a[i][-ni + k]], data::blockstorage[b[j][-jn + k]], tmp);
                                //       blockMulOne(data::blockstorage[a[ni + k]], data::blockstorage[b[kn + j]], tmp);  
                            }
                            data::seq.push_back(data::newoperation(a[i][-ni + k], b[j][-jn + k], blockOp::mulneg, c[n * i + j], 0));
                            //   data::seq.Add(new operation(a[ni + k], b[kn + j], blockOp.mul, c[ni + j], 0)); 
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }
            for( int kk =0; kk< n; kk++){
                if(a[kk] != NULL){
                    delete [] a[kk];
                }
            }
            delete [] a;
            delete [] astart;
            delete [] alen;
            for( int kk =0; kk< n; kk++){
                if(b[kk] != NULL){
                    delete [] b[kk];
                }
            }
            delete [] b;
            delete [] bstart;
            delete [] blen;
            if(tmp != NULL)
                delete [] tmp;
        }

          void BlockPlanner::blockMulBit(int aa[], int bb[], int c[], int n)
        {
                     std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
            //int** a = (int**) malloc(n * sizeof(int *));
            //int* astart = (int*) malloc(n * sizeof(int));
            //int* alen = (int*) malloc(n * sizeof(int));
            ushort** a = new ushort*[n];
       //     std::cout<<"** a " <<a<<std::endl;
            int* astart = new int[n];
            int* alen = new int[n];
       //     std::cout<<"alen " <<alen<<std::endl;
            for(int p=0;p<n;p++){
                a[p] = NULL;
                astart[p] = 0;
                alen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int pn = p * n;
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (aa[pn + q] > 0)
                    {
                        end = q/16;
                        end = end * 16 + 15;
                        break;
                    }
                }
                        astart[p] = 0;
                for (int q = 0; q < n; q++)
                {
                    if (aa[pn + q] > 0)
                    {
                        astart[p] = q/16;
                        astart[p] = astart[p] * 16;
                        break;
                    }
                }
                alen[p] = (end - astart[p]) / 16 + 1;
       //     std::cout<<" alen[p] " <<alen[p]<<std::endl;
                a[p] = new ushort[alen[p]];
                for (int q = 0; q < alen[p]; q++)
                {
                    uint bits = 0;
                    //for(int j = 0; j < 16; j++)
                    //{
                    //    if(aa[pn + astart[p] + q * 16 + j] > 0)
                    //    {
                    //        bits += 1;
                    //    }
                    //    bits = bits << 1;
                    //}
                    //a[p][q] = (ushort)bits;
                    //  a[p][q] = aa[pn + astart[p] + q];

                    bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (aa[pn + astart[p] + q * 16 + j] > 0)
                        {
                            bits = bits | mask[j];
                        }
                    }
                    a[p][q] = (ushort)bits;
                }
            }

            ushort** b = new ushort*[n];
            int* bstart = new int[n];
            int* blen = new int[n];
            for(int p=0;p<n;p++){
                b[p] = NULL;
                bstart[p] = 0;
                blen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (bb[q * n + p] > 0)
                    {
                        end = q / 16;
                        end = end * 16 + 15;
                        break;
                    }
                }
                        bstart[p] = 0;
                for (int q = 0; q < n; q++)
                {
                    if (bb[q * n + p] > 0)
                    {
                        bstart[p] = q / 16;
                        bstart[p] = bstart[p] * 16;
                        break;
                    }
                }
                blen[p] = (end - bstart[p]) / 16 + 1;
       //     std::cout<<" blen[p] " <<blen[p]<<std::endl;
                b[p] = new ushort[blen[p]];
                for (int q = 0; q < blen[p]; q++)
                {
                    uint bits = 0;
                    //for(int j=0; j<16; j++)
                    //{
                    //    if(bb[(q*16 + j + bstart[p]) * n + p] > 0)
                    //    {
                    //        bits += 1;
                    //    }
                    //    bits = bits << 1;
                    //}
                    //b[p][q] = (ushort)bits;
                    //         b[p][q] = bb[(q + bstart[p]) * n + p];
                    bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (bb[(q * 16 + j + bstart[p]) * n + p] > 0)
                        {
                            bits = bits | mask[j];
                        }
                    }
                    b[p][q] = (ushort)bits;
                }
            }

            double* tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                int ni = astart[i];
                for (int j = 0; j < n; j++)
                {
                    flag = true;
                    int jn = bstart[j];
                    int ks = ni > jn ? ni : jn;
                    int ke = ni + alen[i] * 16 < jn + blen[j] * 16 ? ni + alen[i] * 16 : jn + blen[j] * 16;
                    int ks16 = ks / 16;
                    int ke16 = ke / 16;
                    int ni16 = ni / 16;
                    int jn16 = jn / 16;

                    for (int k = ks16; k < ke16; k++)
                    {
                        if ((a[i][k - ni16] & b[j][k - jn16]) == 0)
                            continue;
                        flag = false;
                    }

                    if (flag)
                        c[n * i + j] = 0;
                    else
                    {
                        c[n * i + j] = appendBlockStorage(tmp);

                        for (int k = ks16; k < ke16; k++)
                        {
                            int bits = a[i][k - ni16] & b[j][k - jn16];

            ///                for (int m = 15; m >= 0; m--)
                            for (int m = 0; m < 16; m++)
                            {
            ///                    int kn = (k *16 + m) * n;

                                if ((bits & mask[m]) == 0)
                                {
                                    //       (a[ni + k] == 0 || b[kn + j] == 0) 
                                    continue;
                                }
                                if (data::UseCPU)
                                {
                                    blockMulOne(data::blockstorage[aa[i * n + k * 16 + m]], data::blockstorage[bb[j + (k * 16 + m) * n]], tmp);
                                    //       blockMulOne(data::blockstorage[a[ni + k]], data::blockstorage[b[kn + j]], tmp);  
                                }
                                data::seq.push_back(data::newoperation(aa[i *n + k * 16 + m], bb[j  + (k * 16 + m)*n], blockOp::mul, c[n * i + j], 0));
                                //   data::seq.Add(new operation(a[ni + k], b[kn + j], blockOp.mul, c[ni + j], 0)); 
          ///                      if (aa[i * n + k * 16 + m] == 0 || bb[j + (k * 16 + m) * n] == 0)
           ///                         continue;
                            }
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }

                  std::chrono::duration<double> time_lu = 
                  std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
                  if(n>200)
                  std::cout << "time for bit LU: "<<time_lu.count() << "  size:  "<< n <<" "<<data::UseCPU<<std::endl;
            for( int kk =0; kk< n; kk++){
                if(a[kk] != NULL){
                    delete [] a[kk];
                }
            }
            delete [] a;
            delete [] astart;
            delete [] alen;
            for( int kk =0; kk< n; kk++){
                if(b[kk] != NULL){
                    delete [] b[kk];
                }
            }
            delete [] b;
            delete [] bstart;
            delete [] blen;
            if(tmp != NULL)
                delete [] tmp;
        }

          void BlockPlanner::blockMulBit2(int aa[], int bb[], int c[], int n)
        {
            //int** a = (int**) malloc(n * sizeof(int *));
            //int* astart = (int*) malloc(n * sizeof(int));
            //int* alen = (int*) malloc(n * sizeof(int));
            ushort** a = new ushort*[n];
            int* astart = new int[n];
            int* alen = new int[n];
            for(int p=0; p<n;p++){
                a[p] = NULL;
                astart[p] = 0;
                alen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int pn = p * n;
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (aa[pn + q] > 0)
                    {
                        end = q / 16;
                        end = end * 16 + 15;
                        break;
                    }
                }
                for (int q = 0; q < n; q++)
                {
                    if (aa[pn + q] > 0)
                    {
                        astart[p] = q / 16;
                        astart[p] = astart[p] * 16;
                        break;
                    }
                }
                alen[p] = (end - astart[p]) / 16 + 1;
                a[p] = new ushort[alen[p]];
                for (int q = 0; q < alen[p]; q++)
                {
                    uint bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (aa[pn + astart[p] + q * 16 + j] > 0)
                        {
                            bits += 1;
                        }
                        bits = bits << 1;
                    }
                    a[p][q] = (ushort)bits;
                    //  a[p][q] = aa[pn + astart[p] + q];
                }
            }

            //int** a = (int**) malloc(n * sizeof(int *));
            //int* astart = (int*) malloc(n * sizeof(int));
            //int* alen = (int*) malloc(n * sizeof(int));
            ushort** b = new ushort*[n];
            int* bstart = new int[n];
            int* blen = new int[n];
            for(int p=0; p<n; p++){
                b[p] = NULL;
                bstart[p] = 0;
                blen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (bb[q * n + p] > 0)
                    {
                        end = q / 16;
                        end = end * 16 + 15;
                        break;
                    }
                }
                for (int q = 0; q < n; q++)
                {
                    if (bb[q * n + p] > 0)
                    {
                        bstart[p] = q / 16;
                        bstart[p] = bstart[p] * 16;
                        break;
                    }
                }
                blen[p] = (end - bstart[p]) / 16 + 1;
                b[p] = new ushort[blen[p]];
                for (int q = 0; q < blen[p]; q++)
                {
                    uint bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (bb[(q * 16 + j + bstart[p]) * n + p] > 0)
                        {
                            bits += 1;
                        }
                        bits = bits << 1;
                    }
                    b[p][q] = (ushort)bits;
                    //         b[p][q] = bb[(q + bstart[p]) * n + p];
                }
            }

            double* tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                int ni = astart[i];
                for (int j = 0; j < n; j++)
                {
                    flag = true;
                    int jn = bstart[j];
                    int ks = ni > jn ? ni : jn;
                    int ke = ni + alen[i] * 16 < jn + blen[j] * 16 ? ni + alen[i] * 16 : jn + blen[j] * 16;
                    int ks16 = ks / 16;
                    int ke16 = ke / 16;
                    int ni16 = ni / 16;
                    int jn16 = jn / 16;

                    for (int k = ks16; k < ke16; k++)
                    {
                        if ((a[i][k - ni16] & b[j][k - jn16]) == 0)
                            continue;
                        flag = false;
                    }

                    if (flag)
                        c[n * i + j] = 0;
                    else
                    {
                        c[n * i + j] = appendBlockStorage(tmp);

                        for (int k = ks16; k < ke16; k++)
                        {
                            int bits = a[i][k - ni16] & b[j][k - jn16];

                            for (int m = 15; m >= 0; m--)
                            {
                                int kn = (k * 16 + m) * n;

                                if ((bits & 1) == 0)
                                {
                                    //       (a[ni + k] == 0 || b[kn + j] == 0) 
                                    bits = bits >> 1;
                                    continue;
                                }
                                bits = bits >> 1;
                                if (data::UseCPU)
                                {
                                    blockMulOne(data::blockstorage[aa[i * n + k]], data::blockstorage[bb[j * n + k]], tmp);
                                    //       blockMulOne(data::blockstorage[a[ni + k]], data::blockstorage[b[kn + j]], tmp);  
                                }
                                data::seq.push_back(data::newoperation(aa[i * n + k * 16 + m], bb[j + (k * 16 + m) * n], blockOp::mul, c[n * i + j], 0));
                                //   data::seq.Add(new operation(a[ni + k], b[kn + j], blockOp.mul, c[ni + j], 0)); 
                                if (aa[i * n + k * 16 + m] == 0 || bb[j + (k * 16 + m) * n] == 0)
                                    continue;
                            }
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }
            for( int kk =0; kk< n; kk++){
                if(a[kk] != NULL){
                    delete [] a[kk];
                }
            }
            delete [] a;
            delete [] astart;
            delete [] alen;
            for( int kk =0; kk< n; kk++){
                if(b[kk] != NULL){
                    delete [] b[kk];
                }
            }
            delete [] b;
            delete [] bstart;
            delete [] blen;
            if(tmp != NULL)
                delete [] tmp;
        }

          void BlockPlanner::blockMulBitNeg(int aa[], int bb[], int c[], int n)
        {
          //  int** a = (int**) malloc(n * sizeof(int *));
          //  int* astart = (int*) malloc(n * sizeof(int));
          //  int* alen = (int*) malloc(n * sizeof(int));
            ushort** a = new ushort*[n];
            int* astart = new int[n];
            int* alen = new int[n];
            for(int p=0;p<n;p++){
                a[p] = NULL;
                astart[p] = 0;
                alen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int pn = p * n;
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (aa[pn + q] > 0)
                    {
                        end = q / 16;
                        end = end * 16 + 15;
                        break;
                    }
                }
                        astart[p] = 0;
                for (int q = 0; q < n; q++)
                {
                    if (aa[pn + q] > 0)
                    {
                        astart[p] = q / 16;
                        astart[p] = astart[p] * 16;
                        break;
                    }
                }
                alen[p] = (end - astart[p]) / 16 + 1;
                a[p] = new ushort[alen[p]];
                for (int q = 0; q < alen[p]; q++)
                {
                    uint bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (aa[pn + astart[p] + q * 16 + j] > 0)
                        {
                            bits += 1;
                        }
                        bits = bits << 1;
                    }
                    a[p][q] = (ushort)bits;
                    //  a[p][q] = aa[pn + astart[p] + q];

                    bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (aa[pn + astart[p] + q * 16 + j] > 0)
                        {
                            bits = bits | mask[j];
                        }
                    }
                    a[p][q] = (ushort)bits;
                }
            }

            //int** a = (int**) malloc(n * sizeof(int *));
            //int* astart = (int*) malloc(n * sizeof(int));
            //int* alen = (int*) malloc(n * sizeof(int));
            ushort** b = new ushort*[n];
            int* bstart = new int[n];
            int* blen = new int[n];

            for(int p=0;p<n;p++){
                b[p] = NULL;
                bstart[p] = 0;
                blen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (bb[q * n + p] > 0)
                    {
                        end = q / 16;
                        end = end * 16 + 15;
                        break;
                    }
                }
                        bstart[p] = 0;
                for (int q = 0; q < n; q++)
                {
                    if (bb[q * n + p] > 0)
                    {
                        bstart[p] = q / 16;
                        bstart[p] = bstart[p] * 16;
                        break;
                    }
                }
                blen[p] = (end - bstart[p]) / 16 + 1;
                b[p] = new ushort[blen[p]];
                for (int q = 0; q < blen[p]; q++)
                {
                    uint bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (bb[(q * 16 + j + bstart[p]) * n + p] > 0)
                        {
                            bits += 1;
                        }
                        bits = bits << 1;
                    }
                    b[p][q] = (ushort)bits;
                    //         b[p][q] = bb[(q + bstart[p]) * n + p];
                    bits = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        if (bb[(q * 16 + j + bstart[p]) * n + p] > 0)
                        {
                            bits = bits | mask[j];
                        }
                    }
                    b[p][q] = (ushort)bits;
                }
            }

            double *tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                int ni = astart[i];
                for (int j = 0; j < n; j++)
                {
                    flag = true;
                    int jn = bstart[j];
                    int ks = ni > jn ? ni : jn;
                    int ke = ni + alen[i] * 16 < jn + blen[j] * 16 ? ni + alen[i] * 16 : jn + blen[j] * 16;
                    int ks16 = ks / 16;
                    int ke16 = ke / 16;
                    int ni16 = ni / 16;
                    int jn16 = jn / 16;

                    for (int k = ks16; k < ke16; k++)
                    {
                        if ((a[i][k - ni16] & b[j][k - jn16]) == 0)
                            continue;
                        flag = false;
                    }

                    if (flag)
                        c[n * i + j] = 0;
                    else
                    {
                        c[n * i + j] = appendBlockStorage(tmp);

                        for (int k = ks16; k < ke16; k++)
                        {
                            int bits = a[i][k - ni16] & b[j][k - jn16];

                            for (int m = 15; m >= 0; m--)
                            {
                                int kn = (k * 16 + m) * n;

                                if ((bits & mask[m]) == 0)
                                {
                                    //       (a[ni + k] == 0 || b[kn + j] == 0) 
                                    continue;
                                }
                                if (data::UseCPU)
                                {
                                    blockMulOneNeg(data::blockstorage[aa[i * n + k * 16 + m]], data::blockstorage[bb[j + (k * 16 + m) * n]], tmp);
                                    //       blockMulOne(data::blockstorage[a[ni + k]], data::blockstorage[b[kn + j]], tmp);  
                                }
                                data::seq.push_back(data::newoperation(aa[i * n + k * 16 + m], bb[j + (k * 16 + m) * n], blockOp::mulneg, c[n * i + j], 0));
                                //   data::seq.Add(new operation(a[ni + k], b[kn + j], blockOp.mul, c[ni + j], 0)); 
                                if (aa[i * n + k * 16 + m] == 0 || bb[j + (k * 16 + m) * n] == 0)
                                    continue;
                            }
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }
            for( int kk =0; kk< n; kk++){
                if(a[kk] != NULL){
                    delete [] a[kk];
                }
            }
            delete [] a;
            delete [] astart;
            delete [] alen;
            for( int kk =0; kk< n; kk++){
                if(b[kk] != NULL){
                    delete [] b[kk];
                }
            }
            delete [] b;
            delete [] bstart;
            delete [] blen;
            if(tmp != NULL)
                delete [] tmp;
        }

          void BlockPlanner::blockMulSparse(int aa[], int bb[], int c[], int n)
        {
                   std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
            if(n>200)
            std::cout<<"size: "<<n<<std::endl;

            int **a = new int*[n];
            int *astart = new int[n];
            int *alen = new int[n];

            for (int p=0; p<n; p++){
               a[p] = NULL;
               astart[p] = 0;
               alen[p] = n;
            }
            for (int p = 0; p < n; p++)
            {
                int pn = p * n;
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (aa[pn + q] > 0)
                    {
                        end = q;
                        break;
                    }
                }
                for (int q = 0; q < n; q++)
                {
                    if (aa[pn + q] > 0)
                    {
                        astart[p] = q;
                        break;
                    }
                }
                alen[p] = end - astart[p] + 1;
                a[p] = new int[alen[p]];
                for (int q = 0; q < alen[p]; q++)
                {
                    a[p][q] = aa[pn + astart[p] + q];
                }
            }

            int **b = new int*[n];
            int *bstart = new int[n];
            int *blen = new int[n];
            for(int p = 0;p<n;p++){
                b[p] = NULL;
                bstart[p] = 0;
                blen[p] = n;
            }

            for (int p = 0; p < n; p++)
            {
                int end = 0;
                for (int q = n - 1; q >= 0; q--)
                {
                    if (bb[q * n + p] > 0)
                    {
                        end = q;
                        break;
                    }
                }
                for (int q = 0; q < n; q++)
                {
                    if (bb[q * n + p] > 0)
                    {
                        bstart[p] = q;
                        break;
                    }
                }
                blen[p] = end - bstart[p] + 1;
                b[p] = new int[blen[p]];
                for (int q = 0; q < blen[p]; q++)
                {
                    b[p][q] = bb[(q + bstart[p]) * n + p];
                }
            }

            double *tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                int ni = astart[i];
                    flag = true;
                    int jn = bstart[j];
                    int ks = ni > jn ? ni : jn;
                    int ke = ni + alen[i] < jn + blen[j] ? ni + alen[i] : jn + blen[j];
                    for (int k = ks; k < ke; k++)
                    {
                        if (a[i][k - ni] == 0 || b[j][k-jn] == 0)
                            continue;
                        flag = false;
                    }

                    if (flag)
                        continue;
            //            c[n*i + j] = 0;
                    else
                    {
                         if(c[n*i + j] == 0)
                        c[n*i + j] = appendBlockStorage(tmp);

                        for (int k = ks; k < ke; k++)
                        {
                            int kn = k * n;
                            if
                                (a[i][-ni + k] == 0 || b[j][-jn + k] == 0)
                                //       (a[ni + k] == 0 || b[kn + j] == 0) 
                                continue;
                            if (data::UseCPU)
                            {
                                blockMulOne(data::blockstorage[a[i][-ni + k]], data::blockstorage[b[j][-jn + k]], tmp);
                                //       blockMulOne(data::blockstorage[a[ni + k]], data::blockstorage[b[kn + j]], tmp);  
                            }
                            data::seq.push_back(data::newoperation(a[i][-ni + k], b[j][-jn + k], blockOp::mul, c[n*i + j], 0));
                            //   data::seq.Add(new operation(a[ni + k], b[kn + j], blockOp.mul, c[ni + j], 0)); 
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }
                  std::chrono::duration<double> time_lu = 
                  std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            long vecSize = data::seq.size()/1024;
            
                  if(n>200)
                  std::cout << "time for sparse LU: "<<time_lu.count() << "  size:  "<< n 
                            << " storage " << vecSize * sizeof(operation) / 1024<<"M "<<sizeof(operation)<<std::endl;
            //int **a = new *int[n];
            //int *astart = new int[n];
            //int *alen = new int[n];
            for( int kk =0; kk< n; kk++){
                if(a[kk] != NULL){
                    delete [] a[kk];
                }
            }
            delete [] a;
            delete [] astart;
            delete [] alen;
            for( int kk =0; kk< n; kk++){
                if(b[kk] != NULL){
                    delete [] b[kk];
                }
            }
            delete [] b;
            delete [] bstart;
            delete [] blen;
            if(tmp != NULL)
                delete [] tmp;
        }

          void BlockPlanner::blockMul(int a[], int b[], int c[], int n)
        {
            if (n > 50)
            {
                if (data::UseCPU){
                    blockMulSparse(a, b, c, n);
                }
                else{
                    blockMulSparse(a, b, c, n);
             //       blockMulBit(a, b, c, n);
                }
                return;
            }
            data::blockMulCount++;
            
            double *tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                int ni = i * n;
                for (int j = 0; j < n; j++)
                { 
                    flag = true;
                    int jn = j * n;
                    for (int k = 0; k < n; k++) 
                    { 
                        if (a[ni + k] == 0 || b[k * n + j] == 0) 
                                continue;
                        flag = false;
                    }
                    if (flag)
                          continue;
       //                 c[ni + j] = 0;
                    else
                    {
                        if(c[ni + j] == 0)
                            c[ni + j] = appendBlockStorage(tmp);

                        for (int k = 0; k < n; k++)
                        {
                            int kn = k * n;
                            if (a[ni + k] == 0 || b[kn + j] == 0) 
                                continue;
                            if (data::UseCPU)
                            {
                                blockMulOne(data::blockstorage[a[ni + k]], data::blockstorage[b[kn + j]], tmp);  
                            }
                            data::seq.push_back(data::newoperation(a[ni + k], b[kn + j], blockOp::mul, c[ni + j], 0)); 
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }
        }

          void BlockPlanner::blockMulDiag(int a[], int b[], int c[], int n)
        {
            double *tmp = NULL;
            if (data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }
            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    flag = true;
                    for (int k = 0; k < n; k++)
                    {
                        if (a[i * n + k] == 0 || b[k * n + j] == 0)
                            continue;
                        flag = false;
                    }
                    if (flag || i != j)
                        c[i * n + j] = 0;
                    else
                    {
                        c[i * n + j] = appendBlockStorage(tmp);

                        for (int k = 0; k < n; k++)
                        {
                            if (a[i * n + k] == 0 || b[k * n + j] == 0)
                                continue;
                            if (data::UseCPU)
                                blockMulOne(data::blockstorage[a[i * n + k]], data::blockstorage[b[k * n + j]], tmp);
                            data::seq.push_back(data::newoperation(a[i * n + k], b[k * n + j], blockOp::mul, c[i * n + j], 0));
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                              resetOneBlock(tmp);
                        }
                    }
                }
            }
        }

          void BlockPlanner::blockMulNeg(int a[], int e[], int c[], int n)
        {
            if (n > 50)
            {
                if (data::UseCPU)
                    blockMulSparseNeg(a, e, c, n);
                else
                    blockMulBitNeg(a, e, c, n);
                return;
            }

            data::blockMulCount++;
            int *b = new int[n*n];
            for (int p = 0; p < n; p++)
            {
                int pn = p * n;
                for (int q = 0; q < n; q++)
                {
                    int qn = q * n;
                    b[pn + q] = e[qn + p];
                }
            }

            double *tmp = NULL;

            if(data::UseCPU){
                tmp = new double[data::blockSize * data::blockSize];
                resetOneBlock(tmp);
            }

            bool flag = true;

            for (int i = 0; i < n; i++)
            {
                int ni = i * n;
                for (int j = 0; j < n; j++)
                {
                    flag = true;
                    int jn = j * n;
                    for (int k = 0; k < n; k++)
                    {
                        if (a[ni + k] == 0 || b[jn + k] == 0)//b[k * n + j] == 0)
                            continue;
                        flag = false;
                    }
                    if (flag)
                        continue;
          //              c[ni + j] = 0;
                    else
                    {
                        if(c[ni + j] == 0)
                        c[ni + j] = appendBlockStorage(tmp);

                        for (int k = 0; k < n; k++)
                        {
                            int kn = k * n;
                            if (a[ni + k] == 0 || b[jn + k] == 0) //b[kn + j] == 0)
                                continue;
                            if (data::UseCPU)
                                blockMulOneNeg(data::blockstorage[a[ni + k]], data::blockstorage[b[jn + k]], tmp); //
                            data::seq.push_back(data::newoperation(a[ni + k], b[jn + k], blockOp::mulneg, c[ni + j], 0)); //
                        }

                        if (data::UseCPU){
                            tmp = new double[data::blockSize * data::blockSize];
                            resetOneBlock(tmp);
                        }
                    }
                }
            }
            delete [] b;
        //    delete [] tmp;
        }

          void BlockPlanner::resetOneBlock(double t[])
        {
            std::memset(t, 0, sizeof(double)*data::blockSize * data::blockSize);
         //   for (int i = 0; i < data::blockSize * data::blockSize; i++)
          //      t[i] = 0;
        }
          void BlockPlanner::resetBlocks(int t[], int n)
        {
            std::memset(t, 0, sizeof(int)*n*n);
        //    for (int i = 0; i < n * n; i++)
        //        t[i] = 0;
        }

          void BlockPlanner::blockMulOneTrans(double a[], double b[], double c[])
        {
            //Parallel.For(0, 4, m =>
                  for (int m = 0; m < 4; m++)
            {
                int n = BLOCK64;
                int n4 = BLOCK16;

                for (int p = 0; p < n4; p++)
                {
                    int i = m * n4 + p;
                    int inn = i * n;
                    for (int j = 0; j < n; j++)
                    {
                        double sum[8];
                        int jn = j * n;
                        for (int k=0; k<8; k++)
                            sum[k] = 0;
                        
                        for(int w=0; w< BLOCK64; w=w+8){
                            int jnw = jn + w;
                            int innw = inn + w;
                            for(int k=0;k<8;k++){
                                sum[k] += a[innw + k] * b[jnw+k]; 
                        //    sum += a[i * n + k] * b[j * n + k];
                            }
                        }
                        double summ = 0;
                        for(int k = 0; k<8; k++){
                            summ += sum[k];
                        } 
                        //for (int k = 0; k < n; k++)
                        //{
                        //    sum += a[i * n + k] * b[k * n + j];
                        //}
                        c[i * n + j] += summ;
                    }
                }
            }
           // );
            data::mulcount++;
        }

          void BlockPlanner::blockMulOneNegTrans(double a[], double b[], double c[])
        {
            //Parallel.For(0, 4, m =>
                  for (int m = 0; m < 4; m++)
            {
                int n = BLOCK64;
                int n4 = BLOCK16;

                for (int p = 0; p < n4; p++)
                {
                    int i = m * n4 + p;
                    int inn = i * n;
                    for (int j = 0; j < n; j++)
                    {
                        double sum[8];
                        int jn = j * n;
                        for (int k=0; k<8; k++)
                            sum[k] = 0;
                        
                        for(int w=0; w< BLOCK64; w=w+8){
                            int jnw = jn + w;
                            int innw = inn + w;
                            for(int k=0;k<8;k++){
                                sum[k] += a[innw + k] * b[jnw+k]; 
                        //    sum += a[i * n + k] * b[j * n + k];
                            }
                        }
                        double summ = 0;
                        for(int k = 0; k<8; k++){
                            summ += sum[k];
                        } 
                        //for (int k = 0; k < n; k++)
                        //{
                        //    sum += a[i * n + k] * b[k * n + j];
                        //}
                        c[i * n + j] -= summ;
                    }
                }
            }
           // );
            data::mulcount++;
        }

          void BlockPlanner::blockMulOne(double a[], double b[], double c[])
        {
            //Parallel.For(0, 4, m =>
                  for (int m = 0; m < 4; m++)
            {
                int n = BLOCK64;
                int n4 = BLOCK16;
                int kn[8];

                for(int p = 0;p<8;p++){
                    kn[p] = p * n;
                }
                for (int p = 0; p < n4; p++)
                {
                    int i = m * n4 + p;
                    int inn = i * n;
                    for (int j = 0; j < n; j++)
                    {
                        double sum[8];
                        for (int k=0; k<8; k++)
                            sum[k] = 0;
                        
                        for(int w=0; w< BLOCK64; w=w+8){
                            int wn = w * n + j;
                            int innw = inn + w;
                            for(int k=0;k<8;k++){
                                sum[k] += a[innw + k] * b[wn+kn[k]]; 
                            }
                        }
                        double summ = 0;
                        for(int k = 0; k<8; k++){
                            summ += sum[k];
                        } 
                        //for (int k = 0; k < n; k++)
                        //{
                        //    sum += a[i * n + k] * b[k * n + j];
                        //}
                        c[i * n + j] += summ;
                    }
                }
            }
           // );
            data::mulcount++;
        }


          void BlockPlanner::blockMulOneNeg(double a[], double b[], double c[])
        {
            //Parallel.For(0, 4, m =>
                  for (int m = 0; m < 4; m++)
            {
                int n = BLOCK64;
                int n4 = BLOCK16;
                int kn[8];

                for(int p = 0;p<8;p++){
                    kn[p] = p * n;
                }
                for (int p = 0; p < n4; p++)
                {
                    int i = m * n4 + p;
                    int inn = i * n;
                    for (int j = 0; j < n; j++)
                    {
                        double sum[8];
                        for (int k=0; k<8; k++)
                            sum[k] = 0;
                        
                        for(int w=0; w< BLOCK64; w=w+8){
                            int wn = w * n + j;
                            int innw = inn + w;
                            for(int k=0;k<8;k++){
                                sum[k] += a[innw + k] * b[wn+kn[k]]; 
                            }
                        }
                        double summ = 0;
                        for(int k = 0; k<8; k++){
                            summ += sum[k];
                        } 
                        //for (int k = 0; k < n; k++)
                        //{
                        //    sum += a[i * n + k] * b[k * n + j];
                        //}
                        c[i * n + j] -= summ;
                    }
                }
            }
           // );
            data::mulcount++;
        }


          void BlockPlanner::blockMulOneNaive(double a[], double b[], double c[])
        {
            //Parallel.For(0, 4, m =>
                  for (int m = 0; m < 4; m++)
            {
                int n = BLOCK64;
                int n4 = BLOCK16;
                for (int p = 0; p < n4; p++)
                {
                    int i = m * n4 + p;
/**/
                        for (int k = 0; k < n; k++)
                    {
                        long double aink = a[i*n+k];
                        int kn = k*n;
                        int inn = i*n;
                        double* cinn = c+inn;
                        double* bkn = b+kn;
          //     #pragma omp simd
                    for (int j = 0; j < BLOCK64; j++)
                        {
                            cinn[j] += aink * bkn[j];
                        }
                    }
/*
                    for (int j = 0; j < n; j++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++)
                        {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] += sum;
                    }
 */               }
            }
           // );
            data::mulcount++;
        }

          void BlockPlanner::blockMulOneNegNaive(double a[], double b[], double c[])
        {

            //Parallel.For(0, 4, m =>
                  for (int m = 0; m < 4; m++)
            {
                int n = BLOCK64;
                int n4 = BLOCK16;
                for (int p = 0; p < n4; p++)
                {
                    int i = m * n4 + p;
/*i*/
                        for (int k = 0; k < n; k++)
                    {
                        double* cin = c + i * n;
                        double aink = a[i * n + k];
                        double* bkn = b + k* n;
         // #pragma omp simd
                    for (int j = 0; j < BLOCK64; j++)
                        {
                            cin[j] -= aink * bkn[j];
                        }
                  //      c[i * n + j] -= sum;
                    }
/*
                    for (int j = 0; j < n; j++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++)
                        {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] -= sum;
                    }
       */         }
            }
            //);

            //int i, j, k;

            //int n = data::blockSize;
            //for (i = 0; i < n; i++)
            //{
            //    for (j = 0; j < n; j++)
            //    {
            //        double sum = 0.0;
            //        for (k = 0; k < n; k++)
            //        {
            //            sum += a[i * n + k] * b[k * n + j];
            //        }
            //        c[i * n + j] -= sum;
            //    }
            //}
            data::mulcount++;
        }
          void BlockPlanner::putBlockList(int tgt[], int tgtn, int rowStart, int colStart, int n, int val[])
        {
            int i, j;
            for (i = 0; i < n; i++)
            {
                int ni = i * n;
                int tgtoff = (i + rowStart) * tgtn + colStart;
                for (j = 0; j < n; j++)
                {
                     tgt[tgtoff + j] = val[i * n + j];
                }
            }
        }
          int* BlockPlanner::getBlockList(int src[],int srcn, int rowStart, int colStart, int n)
        {
            if(rowStart+n>srcn || colStart+n>srcn){
               std::cout<<"srcn: "<<srcn<<" rowstart: "<<rowStart<<" colstart "<<colStart<<" h n: "<<n<<std::endl;
            }
            int *rtn = new int[n * n];
            int i, j;
            for(i=0;i<n*n;i++)
                rtn[i] = 0;
            for (i = 0; i < n; i++)
            {
                int ni = i * n;
                int srcoff = (i + rowStart) * srcn + colStart;
                for (j = 0; j < n; j++)
                {
                    rtn[ni + j] = src[srcoff + j];
                }
            }
            data::getBlockCount++;
            return rtn;
        }

      // :blockLU(data::blocks, bl, bu, n);
          void BlockPlanner::copySeqL2(int a[], int l[], int u[], int **bl2, int **bu2, int n)
        {
            int maxA = 0;
            int originSize = data::blockRows * data::blockRows;
            int i,j;
            for(i=0;i<data::blockRows;i++)
                for(j=0;j<data::blockRows;j++){
                    if(a[i*data::blockRows+j]>maxA){
                        maxA = a[i*data::blockRows+j];
                    }
                }
   //         checkMatrixEntry(a,data::blockRows);
            std::cout<<"block Rows "<<data::blockRows<<" storage range "<<maxA<<" origin size "<<originSize<<std::endl;
            int* indexA = (int*)malloc((maxA+1)*sizeof(int));
            int* jndexA = (int*)malloc((maxA+1)*sizeof(int));
            for(i=0;i<=maxA;i++){
                indexA[i] = -1;
                jndexA[i] = -1;
            }
            for(i=0;i<data::blockRows;i++)
                for(j=0;j<data::blockRows;j++){
                    if(a[i*data::blockRows+j]>0){
                       indexA[a[i*data::blockRows+j]] = i;
                       jndexA[a[i*data::blockRows+j]] = j;
                    }
                }
            int maxLU = 0;
            for(i=0;i<data::blockRows;i++)
                for(j=0;j<data::blockRows;j++){
                    if(l[i*data::blockRows+j]>maxLU){
                        maxLU = l[i*data::blockRows+j];
                    }
                    if(u[i*data::blockRows+j]>maxLU){
                        maxLU = u[i*data::blockRows+j];
                    }
                }
            int* indexL = (int*)malloc((maxLU+1)*sizeof(int));
            int* jndexL = (int*)malloc((maxLU+1)*sizeof(int));
            int* indexU = (int*)malloc((maxLU+1)*sizeof(int));
            int* jndexU = (int*)malloc((maxLU+1)*sizeof(int));
            for(i=0;i<maxLU+1;i++){
                indexL[i] = -1;
                jndexL[i] = -1;
                indexU[i] = -1;
                jndexU[i] = -1;
            }
            for(i=0;i<data::blockRows;i++)
                for(j=0;j<data::blockRows;j++){
                    if(l[i*data::blockRows+j]<=0)
                        continue;
                    indexL[l[i*data::blockRows+j]] = i;
                    jndexL[l[i*data::blockRows+j]] = j;
                }
            for(i=0;i<data::blockRows;i++)
                for(j=0;j<data::blockRows;j++){
                    if(u[i*data::blockRows+j]<=0)
                        continue;
                    indexU[u[i*data::blockRows+j]] = i;
                    jndexU[u[i*data::blockRows+j]] = j;
                }

            int L2Rows = data::blockRows;
            data::blockSize = BLOCK64;
            configL2();
            int scaleblock = data::blockScaleL2 * data::blockScaleL2;
            int scaleL2 = data::blockScaleL2;
            int *aa = (int *) aligned_alloc (64, L2Rows * L2Rows *sizeof(int));
            for(i=0;i<L2Rows * L2Rows;i++)
                 aa[i] = a[i];
           
            data::blockstorageL2.resize(data::blockstorage.size(),NULL);
            data::storageCountL2 = data::blockstorageL2.size();
            std::cout<<"L2 storage size: "<<data::storageCountL2<<" block size "<<scaleblock<<std::endl;
            for(i=0;i<L2Rows * L2Rows;i++)
                     if(a[i]>0){
                         int* dmp = (int *) aligned_alloc (64, scaleblock*sizeof(int));

                        for(int ii=0;ii<scaleblock;ii++){
                            dmp[ii] = 0;
                        }
                        appendBlockStorageAgainL2(dmp, a[i]);
                     }


            //configL2();
            checkBlock();
            iniBlockStorage();

            std::cout<<"L2 Rows "<<L2Rows<<" scale "<< scaleL2<<" rows "<<data::blockRows<<std::endl;
            for(i=0;i<L2Rows * L2Rows;i++)
                     if(aa[i]>0){
                         int *t = getBlockList(data::blocks,data::blockRows,indexA[aa[i]]*scaleL2, jndexA[aa[i]]*scaleL2,scaleL2);
                         for(j=0;j<scaleblock;j++)
                             data::blockstorageL2[aa[i]][j] = t[j];
                         delete[] t;
                     }
            free(aa);
            delete[] a;

            *bl2 = (int*)malloc(data::blockRows * data::blockRows *sizeof(int));
            *bu2 = (int*)malloc(data::blockRows * data::blockRows *sizeof(int));
            int *LL2 = *bl2;
            int *UL2 = *bu2;
            for(i=0;i<data::blockRows * data::blockRows;i++){
                LL2[i] = 0;
                UL2[i] = 0;
            }

            data::seqL2.clear();
            for (i = 0; i < data::seq.size(); i++)
            {
                operation o = *(data::seq[i]);
         //       if(i<200)
         //           std::cout<<"seq1 src "<<o.src<< " src2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<o.result<<" op "<<o.op<<std::endl;
                data::seqL2.push_back(data::newoperation(o.src, o.src2, o.op, o.result, o.result2));
            }
            data::seq.clear();


            for (i = 0; i < data::seqL2.size(); i++)
            {
                operation o = *(data::seqL2[i]);
                if (o.skip)
                    continue;
          //      if(i<200)
          //          std::cout<<"seq2 src "<<o.src<< " src2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<o.result<<" op "<<o.op<<std::endl;
                //count++;
                if (o.result > 0)
                {
                    if (data::blockstorageL2[o.result] == NULL)
                    {
                        int* dmp = (int *) aligned_alloc (64, scaleblock*sizeof(int));
 
                        for(int ii=0;ii<scaleblock;ii++){
                            dmp[ii] = 0;
                        }
                        appendBlockStorageAgainL2(dmp, o.result);
      //                  alloccount++;
                    }
                }

                if (o.result2 > 0)
                {
                    if (data::blockstorageL2[o.result2] == NULL)
                    {
                        int* dmp = (int *) aligned_alloc (64, scaleblock*sizeof(int));

                        for(int ii=0;ii<scaleblock;ii++){
                            dmp[ii] = 0;
                        }
                        appendBlockStorageAgainL2(dmp, o.result2);
        //                alloccount++;
                    }
                }
        //            std::cout<<"seq src "<<data::blockstorage[o.src]<< " res2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<data::blockstorage[o.result]<<" op "<<o.op<<std::endl;

                switch (o.op)
                {
                    case blockOp::inv:
                        blockInv(data::blockstorageL2[o.src], data::blockstorageL2[o.result], data::blockScaleL2);
                     //   invcount++;
                        break;

                    case blockOp::lowerInv:
                        blockInvLower(data::blockstorageL2[o.src], data::blockstorageL2[o.result], data::blockScaleL2);
                    //    linvcount++;
                        break;

                    case blockOp::lu:
                        blockLU(data::blockstorageL2[o.src], data::blockstorageL2[o.result], data::blockstorageL2[o.result2], data::blockScaleL2);
                    //    lucount++;
                        break;

                    case blockOp::mul:
                        
                        blockMul(data::blockstorageL2[o.src], data::blockstorageL2[o.src2], data::blockstorageL2[o.result], data::blockScaleL2);
                        break;

                    case blockOp::mulneg:
                        blockMulNeg(data::blockstorageL2[o.src], data::blockstorageL2[o.src2], data::blockstorageL2[o.result], data::blockScaleL2);
                        break;

                    case blockOp::sub:
                        if(o.src>0){
                            if(o.src2>0){
                                blockSub(data::blockstorageL2[o.src2], data::blockstorageL2[o.src], data::blockstorageL2[o.result], data::blockScaleL2);
                            }
                            else{
                                blockNeg(data::blockstorageL2[o.src], data::blockstorageL2[o.result], data::blockScaleL2);
                            }
                        }
                        else{
                            if(o.src2>0){
           //                     std::cout<<data::UseCPU<<" index "<<o.src2<<std::endl;
                                printInt(data::blockstorageL2[o.src2], data::blockScaleL2);
                                blockCopy(data::blockstorageL2[o.src2], data::blockstorageL2[o.result], data::blockScaleL2);
                            }
                        }
                        break;

                    case blockOp::upperInv:
                        blockInvUpper(data::blockstorageL2[o.src], data::blockstorageL2[o.result], data::blockScaleL2);
                        break;

                    default:
                        break;
                }

            /*    if(o.result>0 && o.result<=maxLU && indexL[o.result]>=0){
                      putBlockList(LL2,data::blockRows,indexL[o.result]*scaleL2, jndexL[o.result]*scaleL2,scaleL2, data::blockstorageL2[o.result]);
                 if(indexL[o.result]*scaleL2==16)
                    std::cout<<"seq2 src "<<o.src<< " src2 " << o.src2<< " i: " << i<<" seq "<<o.sequenceNum<<" result "<<o.result<<" op "<<o.op<<" result j "<<jndexL[o.result]<<std::endl;
                }
                if(o.result>0 && o.result<=maxLU && indexU[o.result]>=0){
                      putBlockList(UL2,data::blockRows,indexU[o.result]*scaleL2, jndexU[o.result]*scaleL2,scaleL2, data::blockstorageL2[o.result]);
                }
                if(o.result2>0 && o.result2<=maxLU && indexL[o.result2]>=0){
                      putBlockList(LL2,data::blockRows,indexL[o.result2]*scaleL2, jndexL[o.result2]*scaleL2,scaleL2, data::blockstorageL2[o.result2]);
                }
                if(o.result2>0 && o.result2<=maxLU && indexU[o.result2]>=0){
                      putBlockList(UL2,data::blockRows,indexU[o.result2]*scaleL2, jndexU[o.result2]*scaleL2,scaleL2, data::blockstorageL2[o.result2]);
                }
*/
            }
            for(i=0;i<=maxLU;i++){
                if(indexL[i]>=0){
                    putBlockList(LL2,data::blockRows,indexL[i]*scaleL2, jndexL[i]*scaleL2,scaleL2, data::blockstorageL2[i]);
                    if(indexL[i]*scaleL2==16){
   //                 std::cout<<"before scale i "<<indexL[i]<< " j " << jndexL[i]<<std::endl;
                         printInt(data::blockstorageL2[i],scaleL2);
                    }
                }
                if(indexU[i]>=0){
                      putBlockList(UL2,data::blockRows,indexU[i]*scaleL2, jndexU[i]*scaleL2,scaleL2, data::blockstorageL2[i]);
                }
            }

            free(indexA);
            free(jndexA);
            free(indexL);
            free(jndexL);
            free(indexU);
            free(jndexU);
//            broker::postMessage("   total ops: " + std::to_string(data::seq.size()));
        }

          void BlockPlanner::checkBlock()
        {
            data::blocks = new int[data::blockRows * data::blockRows];
            data::blockRows_sq = data::blockRows * data::blockRows;

//            broker::postMessage("block rows: " + std::to_string(data::blockRows) + "  total blocks: " + std::to_string(data::blockRows_sq));

            int i;
            for(i=0; i<data::blockRows_sq; i++)
            {
                data::blocks[i] = 0;
            }

            int bi, bj;
            for (i = 0; i < data::valcount; i++)
            {
                bi = data::indexi[i] / data::blockSize;
                bj = data::indexj[i] / data::blockSize;
                data::blocks[bi * data::blockRows + bj] = 1;
                if(data::indexi[i] >= data::mSize || data::indexj[i]  >= data::mSize )
   std::cout<<"out of range "<<i<<": "<<data::indexi[i]<<" " <<data::indexj[i]<<" "<<data::vals[i]<<std::endl; 
            }

            int count = 0;
            for (i = 0; i < data::blockRows_sq; i++)
            {
                count += data::blocks[i];
            }

            broker::postMessage("valid blocks: " + std::to_string(count));
            data::blockstorage.clear();
            data::blockstorage.push_back(NULL);
            data::storageCount = 1;
        }

          void BlockPlanner::pushUp()
        {
            int i, j;
            int *dir = new int[data::mSize];
            int *first = new int[data::mSize];
            for (i = 0; i < data::mSize; i++)
            {
                dir[i] = i;
                first[i] = data::mSize;
            }


            for (i = 0; i < data::mSize; i++)
            {
                if(first[data::indexi[i]] > data::indexj[i])
                {
                    first[data::indexi[i]] = data::indexj[i];
                }
            }

            std::vector<position> list;
            list.reserve(data::mSize);
            
            for (i = 0; i < data::mSize; i++)
            {
                list.push_back(position(i, first[i]));
            }

            std::sort (list.begin(), list.end(), positionCompare);

            for (i = 0; i < list.size(); i++)
            {
                if(list[i].index < data::mSize)
                {
                    dir[list[i].index] = i;
                }
            }

            for (i = 0; i < data::mSize; i++)
            {
                data::indexi[i] = dir[data::indexi[i]];
            }

            double* tmp = new double[data::mSize];
            for (i = 0; i < data::mSize; i++)
            {
                tmp[i] = data::b[i];
            }
            for (i = 0; i < data::mSize; i++)
            {
                data::b[i] = data::b[dir[i]];
            }

            delete [] tmp;
            delete [] dir;
            delete [] first;
        }

          void BlockPlanner::pushDown()
        {
            int i, j;
            int *dir = new int[data::mSize];
            int *first = new int[data::mSize];
            for (i = 0; i < data::mSize; i++)
            {
                dir[i] = i;
                first[i] = 0;
            }


            for (i = 0; i < data::mSize; i++)
            {
                if (first[data::indexi[i]] < data::indexj[i])
                {
                    first[data::indexi[i]] = data::indexj[i];
                }
            }

            std::vector<position> list;
            list.reserve(data::mSize);
            
            for (i = 0; i < data::mSize; i++)
            {
                list.push_back(position(i, first[i]));
            }

            std::sort (list.begin(), list.end(), positionCompare);

            for (i = 0; i < list.size(); i++)
            {
                if (list[i].index < data::mSize)
                {
                    dir[list[i].index] = i;
                }
            }

            for (i = 0; i < data::mSize; i++)
            {
                data::indexi[i] = dir[data::indexi[i]];
            }

            double *tmp = new double[data::mSize];
            for (i = 0; i < data::mSize; i++)
            {
                tmp[i] = data::b[i];
            }
            for (i = 0; i < data::mSize; i++)
            {
                data::b[i] = data::b[dir[i]];
            }
            delete [] tmp;
            delete [] dir;
            delete [] first;
        }

          void BlockPlanner::iniBlockStorage()
        {
            int i, bi, bj;


            for (i = 0; i < data::blockRows_sq; i++)
            {
                data::blocks[i] = 0;
            }
 //           std::cout<<"count "<<data::storageCount<<" size " << data::blockstorage.size()<<" blockRows_sq "<<data::blockRows_sq<<std::endl;

            for (i = 0; i < data::valcount; i++)
            {
                bi = data::indexi[i] / data::blockSize;
                bj = data::indexj[i] / data::blockSize;
                if(data::blocks[bi * data::blockRows + bj] == 0)
                {
                    allocateBlock(bi, bj);
                }
                int ri = data::indexi[i] - bi * data::blockSize;
                int rj = data::indexj[i] - bj * data::blockSize;
                if(ri * data::blockSize + rj > data::blockSize * data::blockSize){
                     std::cout<<" ri " <<ri<< " rj "<<rj<<std::endl;
                }
//  std::cout<<" planL2 "<<data::blockstorage[data::blocks[bi * data::blockRows + bj]]<<std::endl;
                if(!data::PlanL2)
                data::blockstorage[data::blocks[bi * data::blockRows + bj]][ri * data::blockSize + rj] = data::vals[i];
            }
 //           broker::postMessage("valid blocks: " + std::to_string(data::storageCount));
            for (i = data::mSize; i< data::blockRows * data::blockSize; i++)
            {
                bi = i / data::blockSize;
                int ri = i - bi * data::blockSize;
                if (data::blocks[bi * data::blockRows + bi] == 0)
                {
                    allocateBlock(bi, bi);
                }

                if(!data::PlanL2)
                data::blockstorage[data::blocks[bi * data::blockRows + bi]][ri * data::blockSize + ri] = 1;
            }
            
   //         std::cout<<"block start "<<data::blocks[0] << "  stop  " << data::blocks[data::blockRows * data::blockRows -1] <<std::endl;
    //        broker::postMessage("valid blocks: " + std::to_string(data::storageCount));
//            printInt(data::blocks,data::blockRows);
        }
          void BlockPlanner::allocateBlock(int bi, int bj)
        {
            double* dmp = NULL;
            if(!data::PlanL2)
            {
             //   dmp = (double *) aligned_alloc (64, data::blockSize * data::blockSize*sizeof(double));
                dmp = data::newalignedblock();
 
                for(int i=0;i<data::blockSize * data::blockSize;i++){
                    dmp[i] = 0;
                }
            }

            int t = data::blockstorage.size();
  //        std::cout<<t<<" allocate "<<bi * data::blockRows + bj<<" to "<<static_cast<void*>(dmp)<<" count "<<data::storageCount<<std::endl;
            data::blockstorage.push_back( dmp );
            data::blocks[bi * data::blockRows + bj] = t;
            data::storageCount = t+1;
        }
          int BlockPlanner::appendBlockStorageReal(double t[])
        {
            int rtn = data::blockstorage.size();
            data::blockstorage.push_back( t );
            data::storageCount = rtn + 1;
            return rtn;
        }
          void BlockPlanner::appendBlockStorageAgain(double t[], int rtn)
        {
            data::blockstorage[rtn] = t;
        }
          void BlockPlanner::appendBlockStorageAgainL2(int t[], int rtn)
        {
            data::blockstorageL2[rtn] = t;
        }
          int BlockPlanner::appendBlockStorage(double t[])
        {
            int rtn = data::blockstorage.size();
            data::blockstorage.push_back( t );
            data::storageCount = rtn + 1;
            return rtn;
        }
          void BlockPlanner::configL2()
        {
            data::blockRows = (data::mSize_out + data::blockSize - 1) / data::blockSize;
            data::blockRows = data::mSize_out / data::blockSize;
//         std::cout<<" rows: "<<data::blockRows<<" msize "<<data::mSize_out<<" blocksize "<<data::blockSize<<std::endl;
            data::blockRows = roundup(data::blockRows);
 //        std::cout<<" rows: "<<data::blockRows<<std::endl;
            data::blockRowsL2 = data::blockRows * data::blockSize / data::blockSizeL2;
  //       std::cout<<" rowsL2: "<<data::blockRowsL2<<std::endl;
            data::blockScaleL2 = data::blockSizeL2 / data::blockSize;
   //      std::cout<<" scale: "<<data::blockScaleL2<<std::endl;
        }
          int BlockPlanner::roundup(int count)
        {
            int rtn = 1;
            int counter = 0;
            data::blockSizeL2 = data::blockSize/2;
            while (count > 0)
            {
                count = count / 2;
                rtn = rtn * 2;
                if(counter % 2 == 1){
                   data::blockSizeL2 = data::blockSizeL2 * 2;
                }
//         std::cout<<" size2: "<<data::blockSizeL2<<std::endl;
                counter++;
            }
            return rtn;
        }
          int BlockPlanner::stageCount(int blockRows)
        {
            int rtn = 0;
            while (blockRows > 1)
            {
                blockRows = blockRows / 2;
                rtn = rtn + 1;
            }
            return rtn;
        }

}

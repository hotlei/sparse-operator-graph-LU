
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#include "broker.h"
#include "data.h"
#include "BlockPlanner.h"
#include "BlockReOrder.h"
#include "cudaPlan.h"

namespace vmatrix {
int readMTX(std::string fname){
  long    count=0;
  int    row, col;
  double val;
  std::string line;
  int    rows, cols, counts;

  char  cstr[1024];
  char * ps;
  char * pe;
  std::ifstream myfile (fname.c_str());
  if (myfile.is_open())
  {
    while ( std::getline (myfile,line) ){
         if(line.length()<=3) continue;
         if(line.find('%') != std::string::npos) continue;

         strcpy (cstr, line.c_str());
         
         rows = (int)strtol(cstr,&ps,10);
         cols = (int)strtol(ps, &pe,10);
         counts = (int)strtol(pe, NULL,10);
  if(rows != cols){
        std::cout<<" not square matrix "<<std::endl; 
        if( rows > cols)
            rows = cols;
  }
         
         data::mSize = rows;
         data::mSize_out = rows;
         data::valcount= counts;
         data::indexi = new int[counts];
         data::indexj = new int[counts];
         data::vals = new double[counts];
         for(int i =0; i<counts; i++){
             data::indexi[i] = -1;
             data::indexj[i] = -1;
             data::vals[i] = 0;
         }
         std::cout<<rows  << " " << cols << " " << counts;
         break;
    }
         
    while ( std::getline (myfile,line) )
    {
         if(line.length()<=3) continue;
         if(line.find('%') != std::string::npos) continue;
         if(line.length()>=1000) continue;

         strcpy (cstr, line.c_str());

         row = (int)strtol(cstr,&ps,10);
         col = (int)strtol(ps, &pe,10);
         val = strtod(pe, NULL);

         if(val == 0)
             continue;
         if(row>rows || col>rows)
             continue;

         data::indexi[count] = row-1;
         data::indexj[count] = col-1;
         data::vals[count] = val;
if(row>rows || col>cols){
    std::cout<<" "<<count<<": "<<line<<std::endl;
}

         count++;
    }
    myfile.close();
    data::valcount= count;
    std::cout <<" "<< row << " " << col << " " << val ;

    std::cout <<" "<< count <<'\n';
  }

  else std::cout << "Can not open file"<<'\n'; 
  return count;
}

int iniData(){
    ushort u = 1u;
    for( int i=0; i<16; i++){
        BlockPlanner::mask[i] = u;
        u  *= 2;
    }
    data::blockSize = BLOCK64;
    return 0;
}



        ///expected:
        ///start read: 10/22/2018 8:43:11 PM
//90449       90449     2455670
//total lines: 2455670    done read: 10/22/2018 8:43:14 PM
//start: 00:00:02.9665631
//block rows: 2048  total blocks: 4194304
//valid blocks: 4194304
//valid blocks: 6283
//block count: 6282     nozeros: 2008220    diag: 131072
//entry count: 2008220
//start lu: 00:00:03.2532142
//size : 1024 mul start: 00:00:03.4140203
//size : 1024 mul stop: 00:00:03.4225845
//size : 2048 mul start: 00:00:03.7550260
//size : 2048 mul stop: 00:00:03.8781632
//size : 1024 mul start: 00:00:04.1114705
//size : 1024 mul stop: 00:00:04.1191219
//Done dry run: 00:00:04.1736847
//get block count: 63490
//block mul count: 48768
//blocks: 2048
//op seqs: 9441328
//storage: 2069986
//utilize: 0.493523120880127
//capacity: 8388608
//unit LU count: 2048
//unit lower invert count: 11264
//unit upper invert count: 11264
//unit multiple count: 0
//unit invert count: 0
//found zero block count: 0
//total storage: 2069986    ini stage: 6282    operations: 9441328    start planning: 00:00:04.2007094
//reduced ops to: -9375246      start allocate memory: 00:00:04.4025693
//new storage count: 41371   result size: 8330
//Done planing: 00:00:05.0576281
//   total ops: 33041
//All Done: 00:00:07.6419941
//start validate... 10/22/2018 8:43:19 PM
//max diff:  3.56376403942704E-08   finished: 00:00:07.7457878


        int *finx=NULL;
        int *fjnx=NULL;
        double *fb=NULL;

        void Start_Calculate()
        {
            std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
            int count = data::valcount;
//            count = readMTX("d:\\dev\\ecl32.mtx");// e20r2000.mtx :4241 : 3min  // e30r4000.mtx :9661 // nos5.mtx 468 nos7 729  // cavity11 :2597 // bcsstm10 :1086//sherman2 :1080);
                                                              //orsreg_1 :2205 //orsirr_1 :1030 //e05r0500 :236 //bcsstk18 :11948
                                                              //s3dkq4m2.mtx :90449 //ecl32 :51993 // cage10 :11397 // cage9

            // cage10 ops: 238323 1:36
//            broker::postMessage("calculation start: " );
            BlockPlanner::appendBlockStorage(NULL);
  //          std::cout<<"count "<<data::storageCount<<" size " << data::blockstorage.size()<<std::endl;
            BlockReOrder::sortInBlock();


            BlockPlanner::configL2();
            BlockPlanner::checkBlock();
            BlockPlanner::iniBlockStorage();
            double* dg = BlockPlanner::checkDiagEntry(data::blocks, data::blockRows);
            delete[] dg;

            data::UseCPU = false;
            int n = data::blockRows;
            int *bl = (int*) std::malloc(n*n*sizeof(int));
            int *bu = (int*) std::malloc(n*n*sizeof(int));
            int *by = (int*) std::malloc(n*n*sizeof(int));
            for(int i=0;i<n*n;i++){
                bl[i] = 0;
                bu[i] = 0;
                by[i] = 0;
            }

 //           broker::postMessage("start lu: " );
            BlockPlanner::blockLU(data::blocks, bl, bu, n);
  //          broker::postMessage("Done dry run: " );
            std::chrono::duration<double> time_dry = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout << "time now: "<<time_dry.count() <<'\n';

   //         broker::postMessage("get block count: " + std::to_string(data::getBlockCount));

            broker::postMessage("blocks: " + std::to_string(data::blockRows) + 
            " op seqs: " + std::to_string(data::seq.size()) +
            " storage: " + std::to_string(data::storageCount) +
            " utilize: " + std::to_string(1.0* data::storageCount/data::blockRows/data::blockRows));

    //        broker::postMessage("found zero block count: " + std::to_string(data::foundzeroblock));

     //       BlockPlanner::blockPlan(bl,n*n, bu, n*n);

            broker::postMessage("Done planing: " );

            std::chrono::duration<double> time_plan = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout << "planning: "<<time_plan.count() <<'\n';
            std::chrono::steady_clock::time_point calcstart = std::chrono::steady_clock::now();

            data::UseCPU = true;
            BlockPlanner::calculate();

      //      broker::postMessage("All Done: " );

            std::chrono::duration<double> time_calc = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-calcstart);
     //       std::cout << "calculated :"<<time_calc.count() <<'\n';

       //     broker::postMessage("start validate ... " );

            BlockPlanner::sovle(bl, bu, data::b, data::blockRows);


            if (BlockReOrder::reverseOrder != NULL)
            {
                double* tmp = data::x;
                data::x = new double[data::mSize];
                for (int i = 0; i < data::mSize; i++) data::x[i] = 0;
                for (int i = 0; i < data::mSize; i++)
                {
                    int idx = i ;

                    data::x[i] = tmp[BlockReOrder::reverseOrder[idx]];
                }
                delete[] tmp;
            }

                double max = 0;
                for (int i = 0; i < data::mSize_out; i++)
                {
                    if (std::isnan(data::x[i]))
                    {
                        max = 1000000;
                        continue;
                    }
                    int ii = i;
                    if (data::x[i] - ii > max){
                        max = data::x[i] - ii;
                        std::cout << " i: "<<i<<" error: "<<max<<std::endl;
                    }
                    if (ii - data::x[i] > max){
                        max = ii - data::x[i];
                        std::cout << " i: "<<i<<" error: "<<max<<std::endl;
                    }
                }

                broker::postMessage("max diff:  " + std::to_string(max) + "   finished: " );
            
            free(bl);
            free(bu);
            free(by);
            delete[] finx;
            delete[] fjnx;
            delete[] fb;
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
  //          std::cout << "totaltime: "<< time_span.count() <<" ";
        }

        void saveInx()
        {
            if(finx != NULL)
                delete[] finx;
            if(fjnx != NULL)
                delete[] fjnx;
            if(fb != NULL)
                delete[] fb;
            finx = new int[data::valcount];
            fjnx = new int[data::valcount];
            fb = new double[data::valcount];
            for(int i = 0; i < data::valcount; i++)
            {
                finx[i] = data::indexi[i];
                fjnx[i] = data::indexj[i];
            }
            for (int i = 0; i < data::blockRows * data::blockSize; i++)
            {
                fb[i] = data::b[i];
            }
        }
        void recoverInx()
        {
            for (int i = 0; i < data::valcount; i++)
            {
                data::indexi[i] = finx[i];
                data::indexj[i] = fjnx[i];
            }
            for (int i = 0; i < data::blockRows * data::blockSize; i++)
            {
                data::b[i] = fb[i];
            }
        }
        void reOrder()
        {
        //    int count = readMTX("d:\\dev\\ecl32.mtx");// e20r2000.mtx :4241 : 3min  // e30r4000.mtx :9661 // nos5.mtx 468 nos7 729  // cavity11 :2597 // bcsstm10 :1086//sherman2 :1080);
                                                                  //orsreg_1 :2205 //orsirr_1 :1030 //e05r0500 :236 //bcsstk18 :11948
                                                                  //s3dkq4m2.mtx :90449 //ecl32 :51993 // cage10 :11397 // cage9


            //BlockReOrder.filltoBlocks();
            //BlockPlanner.assignB();
            //ReOrder.checkBandwidth();
            //ReOrder.updateEdges();
            //ReOrder.findStartNode();
            //ReOrder.checkBandwidth();

            
            //block
        //    BlockPlanner::assignB();
            int orig_bandwidth = data::checkBandwidth();

            saveInx();
            data::blockSize = BLOCK64;

            int minBandwidth=data::mSize,  minStart=0;  // blockSize 8, bandwidth 4148, 1:48
                                                       
            BlockReOrder::getBlockIndexRaw();
            BlockReOrder::checkBandwidth();
            BlockReOrder::updateEdges(0);

            int startNode = BlockReOrder::findMinConnection();
// std::cout<<" step 1 "<<startNode<<std::endl;

            startNode = BlockReOrder::findSideNode(startNode);

            recoverInx();

            BlockReOrder::getBlockIndexRaw();
// std::cout<<" reOrder step2 : "<<startNode<<std::endl;

            BlockReOrder::checkBandwidth();
            BlockReOrder::updateEdges(0);
            BlockReOrder::assignLevel(startNode);
std::cout<<" reOrder with start Node: "<<startNode<<std::endl;
            BlockReOrder::reOrder();

            data::blockSize = BLOCK64;

            int new_band = data::checkBandwidth();
            if(new_band > orig_bandwidth){
                recoverInx();
                std::cout<<"keep orignal order"<<std::endl;
                for(int i=0;i<data::mSize;i++){
                    BlockReOrder::reverseOrder[i] = i;
                    BlockReOrder::newOrder[i] = i;
                }
            }else{
                std::cout<<"updated node order"<<std::endl;
            }
        }

        void Start_CalculateL2()
        {
            std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
            int count = data::valcount;

            // cage10 ops: 238323 1:36
     //       broker::postMessage("calculation start: " );
            BlockPlanner::appendBlockStorage(NULL);
      //      std::cout<<"count "<<data::storageCount<<" size " << data::blockstorage.size()<<std::endl;
            BlockReOrder::sortInBlock();
            data::PlanL2 = true;

            BlockPlanner::configL2();
            BlockPlanner::checkBlock();
            BlockPlanner::iniBlockStorage();
          //  double* dgb = BlockPlanner::checkDiagEntry(data::blocks, data::blockRows);
          //  delete[] dgb;

            data::UseCPU = false;
            int n = data::blockRows;
            int *bl = (int*) std::malloc(n*n*sizeof(int));
            int *bu = (int*) std::malloc(n*n*sizeof(int));
            int *by = (int*) std::malloc(n*n*sizeof(int));
            for(int i=0;i<n*n;i++){
                bl[i] = 0;
                bu[i] = 0;
                by[i] = 0;
            }

            
   //         broker::postMessage("start lu: " );
            BlockPlanner::blockLU(data::blocks, bl, bu, n);
    //        broker::postMessage("Done dry run: " );
            std::chrono::duration<double> time_dry = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
   //         std::cout << "time now: "<<time_dry.count() <<'\n';

    //        broker::postMessage("get block count: " + std::to_string(data::getBlockCount));

            broker::postMessage("blocks: " + std::to_string(data::blockRows) + 
            " blockSize: " + std::to_string(data::blockSize) +
            " inputSize: " + std::to_string(data::mSize) + 
            " extend: " + std::to_string(data::blockRows * data::blockSize) +
            " op seqs: " + std::to_string(data::seq.size()) +
            " storage: " + std::to_string(data::storageCount) +
            " utilize: " + std::to_string(1.0* data::storageCount/data::blockRows/data::blockRows));

     //       broker::postMessage("found zero block count: " + std::to_string(data::foundzeroblock));

            BlockPlanner::blockPlan(bl,n*n, bu, n*n);

            data::PlanL2 = false;
            data::blockSize = BLOCK64;
         //   BlockPlanner::checkBlock();
         //   BlockPlanner::iniBlockStorage();
            int *bl2;
            int *bu2;
            BlockPlanner::copySeqL2(data::blocks, bl, bu, &bl2, &bu2,n);
            n = data::blockRows;

    //        BlockPlanner::checkMatrixEntry(bl2,n);
     //       BlockPlanner::checkMatrixEntry(bu2,n);
            broker::postMessage("blocks: " + std::to_string(data::blockRows) + 
            " blockSize: " + std::to_string(data::blockSize) +
            " inputSize: " + std::to_string(data::mSize) + 
            " extend: " + std::to_string(data::blockRows * data::blockSize) +
            " op seqs: " + std::to_string(data::seq.size()) +
            " storage: " + std::to_string(data::storageCount) +
            " utilize: " + std::to_string(1.0* data::storageCount/data::blockRows/data::blockRows));

             BlockPlanner::blockPlan(bl2,n*n, bu2, n*n);
//std::cout << "fb:"<<fb<<std::endl;
//std::cout << "fb:"<<fb<<std::endl;
            double* dgb = BlockPlanner::checkDiagEntry(data::blocks, data::blockRows);
            delete[] dgb;
            std::chrono::duration<double> time_plan = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout << "plan time: "<<time_plan.count() <<'\n';
            std::chrono::steady_clock::time_point calcstart = std::chrono::steady_clock::now();

            copyCuda(data::blocks,data::blockRows,data::blockSize,bl2,bu2);

            data::UseCPU = true;
         //   BlockPlanner::calculate();

   //         broker::postMessage("All Done: " );

            std::chrono::duration<double> time_calc = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-calcstart);
    //        std::cout << "calculated :"<<time_calc.count() <<'\n';

    //        broker::postMessage("start validate ... " );

            data::clearSeq();
            double* dg = BlockPlanner::checkDiagEntry(bu2, data::blockRows);
            delete[] dg;

            std::chrono::steady_clock::time_point solvestart = std::chrono::steady_clock::now();
            BlockPlanner::sovle(bl2, bu2, data::b, data::blockRows);
            std::chrono::duration<double> time_solve =
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-solvestart);
            std::cout << "solve triangled :"<<time_solve.count() <<'\n';

            // broker::postMessage("solved");

            if (BlockReOrder::reverseOrder != NULL)
            {
                double* tmp = data::x;
                data::x = new double[data::mSize];
                for (int i = 0; i < data::mSize; i++) data::x[i] = 0;
                for (int i = 0; i < data::mSize; i++)
                {
                    int idx = i ;

                    data::x[i] = tmp[BlockReOrder::reverseOrder[idx]];
                }
                delete[] tmp;
            }

                double max = 0;
                for (int i = 0; i < data::mSize_out; i++)
                {
                    if (std::isnan(data::x[i]))
                    {
                        max = 1000000;
                        continue;
                    }
                    int ii = i;
                    if (data::x[i] - ii > max){
                        max = data::x[i] - ii;
  //            std::cout<<i<<" "<<data::x[i]<<"   "<<max<<std::endl;
                    }
                    if (ii - data::x[i] > max){
                        max = ii - data::x[i];
   //           std::cout<<i<<" "<<data::x[i]<<"   "<<max<<std::endl;
                    }
                    if(i%100==0){
//              std::cout<<i<<" "<<data::x[i]<<std::endl;
                    }
                }

                broker::postMessage("max diff:  " + std::to_string(max) + "   finished: " );
            
            free(bl);
            free(bu);
            free(by);
            free(bl2);
//std::cout << "free bu2"<<std::endl;
            free(bu2);
//std::cout << "free finx"<<std::endl;
            delete[] finx;
//std::cout << "free fjnx"<<std::endl;
            delete[] fjnx;
//std::cout << "free fb:"<<fb<<std::endl;
            delete[] fb;
//std::cout << "free finx"<<std::endl;
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
   //         std::cout << "totaltime: "<< time_span.count() <<" ";
        }

}

int main (int argc, char* argv[]) {

            std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
    int devID = 0;
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);
    if (error != cudaSuccess)
        printf("cudaGetDevice  %s (code %d)\n", cudaGetErrorString(error), error);

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error == cudaSuccess){
        float mb = (float)deviceProp.totalGlobalMem/1048576.0f;
        vmatrix::data::gpuMB = (int)(mb * 0.9f);
        printf("GPU Device %d: \"%s\" with compute capability %d.%d mem:%.0f MB %lu\n\n", 
                                devID, deviceProp.name, deviceProp.major, 
                                     deviceProp.minor, mb, sizeof(long double));
    }
    cudaFree(0);
  //  vmatrix::data::gpuMB = 800;


  vmatrix::iniData();
  std::string fname="/home/data/cage10.mtx";
  if(argc>1){
     fname.replace(fname.find("cage10"),6,argv[1]);
  }
  if(vmatrix::readMTX(fname)==0)
       return 0;
  vmatrix::data::checkBandwidth();
  vmatrix::BlockPlanner::assignB();
  vmatrix::reOrder();
            std::chrono::steady_clock::time_point timereorder = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_plan = std::chrono::duration_cast<std::chrono::duration<double>>(timereorder-timestart);
            std::cout << "re Order time: "<<time_plan.count() <<'\n';
  vmatrix::data::blockSize = vmatrix::data::blockSizeL2;
  vmatrix::data::blockRows = vmatrix::data::blockRowsL2;
  vmatrix::Start_CalculateL2();
  vmatrix::data::clear();
  vmatrix::BlockReOrder::clear();
            std::chrono::duration<double> time_calc = 
                     std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timereorder);
            std::chrono::duration<double> time_total = 
                     std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout <<"totaltime: "<<time_total.count()<< " reordertime: "<<time_plan.count()<<" calculationtime: "<<time_calc.count()<< " ";
  std::cout<<vmatrix::data::mSize<<" "<<vmatrix::data::mSize_out<<" "<<vmatrix::data::valcount<<" ";
  std::cout<<fname<<std::endl;

  return 0;
}

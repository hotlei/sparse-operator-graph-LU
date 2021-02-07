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

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>

#include "operation.h"
#include "matrix.h"
#include "data.h"
#include "memutil.h"
#include "BlockPlanner.h"
#include "GPSOrder.h"
#include "config.h"

namespace SOGLU {

        int iniData(){
            ushort u = 1u;
            for( int i=0; i<16; i++){
                BlockPlanner::mask[i] = u;
                u *= 2;
            }
            data::blockSize = BLOCK64;
            memutil::ini();
            data::blockstorage.clear();
            data::blockstorage.reserve(10000);
            return 0;
        }

        void decompose_solveLU()
        {
            if(data::symmetric) std::cout << "symmetric" <<std::endl;
            std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
            int count = data::valcount;

            BlockPlanner::appendBlockStorage(NULL);
            data::PlanL2 = true;

            data::blockSize = config::blockSizeL2;
            data::blockRows = config::blockRowsL2;

            BlockPlanner::checkBlock();
            BlockPlanner::iniBlockStorage();

            int n = data::blockRows;
            int levels = __builtin_popcount(data::blockRows-1);
            matrix *bl =  memutil::newmatrix(0,n,levels);
            matrix *bu =  memutil::newmatrix(0,n,levels);
            
            if(data::symmetric)
                BlockPlanner::blockMatrixLLT(data::blocks, bl, n, NULL);
            else
                BlockPlanner::blockMatrixLU(data::blocks, bl, bu, n, NULL, NULL);

            std::cout<<"blocks: "<<data::blockRows<<" blockSize: "<<data::blockSize<<" inputSize: "<<data::mSize  
                     <<" extend: "<<data::blockRows * data::blockSize<<" op count: "<<data::graph.size()<<
                       " storage: "<<data::storageCount<<std::endl;

            if(data::symmetric)
                BlockPlanner::blockPlan(bl, NULL);
            else
                BlockPlanner::blockPlan(bl, bu);

            data::PlanL2 = false;
            data::blockSize = BLOCK64;
            int levelall = __builtin_popcount(data::blockRows-1);
            matrix *bl2 = memutil::newmatrix(0,data::blockRows,levelall);
            matrix *bu2 = memutil::newmatrix(0,data::blockRows,levelall);
            if(data::symmetric)
                BlockPlanner::copyOperatorL2(data::blocks, bl, NULL, bl2, bu2,n);
            else
                BlockPlanner::copyOperatorL2(data::blocks, bl, bu, bl2, bu2,n);
            memutil::freeAllSmallMem(0);
            n = data::blockRows;

            std::cout<<"blocks: "<<data::blockRows<<" blockSize: "<<data::blockSize<<" inputSize: "<<data::mSize  
                     <<" extend: "<<data::blockRows * data::blockSize<<" op count: "<<data::graph.size()<<
                       " storage: "<<data::storageCount<<std::endl;

            BlockPlanner::blockPlan(bl2, bu2);
            std::chrono::duration<double> time_plan = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout << "plan time: "<<time_plan.count() <<'\n';
            std::chrono::steady_clock::time_point calcstart = std::chrono::steady_clock::now();

            BlockPlanner::calculate();

            std::chrono::duration<double> time_calc = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-calcstart);
            std::cout << "kernel time: "<<time_calc.count() <<'\n';

            data::clearOperations();

            std::chrono::steady_clock::time_point solvestart = std::chrono::steady_clock::now();
            BlockPlanner::solve(bl2, bu2, data::b, data::blockRows*data::blockSize);
            std::chrono::duration<double> time_solve =
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-solvestart);
            std::cout << "solve triangled :"<<time_solve.count() <<'\n';
        }

        double* decompose_solveX()
        {
            if(data::symmetric) std::cout << "symmetric" <<std::endl;
            std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();
            int count = data::valcount;

            BlockPlanner::appendBlockStorage(NULL);
            data::PlanL2 = true;

            data::blockSize = config::blockSizeL2;
            data::blockRows = config::blockRowsL2;

            BlockPlanner::checkBlock();
            BlockPlanner::iniBlockStorage();

            int n = data::blockRows;
            int levels = __builtin_popcount(data::blockRows-1);
            matrix *bl =  memutil::newmatrix(0,n,levels);
            matrix *bu =  memutil::newmatrix(0,n,levels);

            matrix *mx =  BlockPlanner::iniVectorMatrix(n);
            matrix *my =  BlockPlanner::iniVectorMatrix(n);
            matrix *mlhs =  BlockPlanner::iniVectorMatrix(n);
            
            if(data::symmetric)
                BlockPlanner::blockMatrixLLT(data::blocks, bl, n, NULL);
            else
                BlockPlanner::blockMatrixLU(data::blocks, bl, bu, n, NULL, NULL,0, mx, my, mlhs);

            std::cout<<"blocks: "<<data::blockRows<<" blockSize: "<<data::blockSize<<" inputSize: "<<data::mSize  
                     <<" extend: "<<data::blockRows * data::blockSize<<" op count: "<<data::graph.size()<<
                       " storage: "<<data::storageCount<<std::endl;

            if(data::symmetric)
                BlockPlanner::blockPlan(bl, NULL);
            else
                BlockPlanner::blockPlan(mx, NULL, mlhs);

            data::PlanL2 = false;
            data::blockSize = BLOCK64;
            int levelall = __builtin_popcount(data::blockRows-1);
            matrix *x2 = memutil::newmatrix(0,data::blockRows,levelall);

     //       if(!data::symmetric)
                BlockPlanner::copyOperatorX2(data::blocks, mx, my, mlhs, x2, n);
            memutil::freeAllSmallMem(0);
            n = data::blockRows;

            std::cout<<"blocks: "<<data::blockRows<<" blockSize: "<<data::blockSize<<" inputSize: "<<data::mSize  
                     <<" extend: "<<data::blockRows * data::blockSize<<" op count: "<<data::graph.size()<<
                       " storage: "<<data::storageCount<<std::endl;

            BlockPlanner::blockPlan(x2, NULL);
            std::chrono::duration<double> time_plan = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout << "plan time: "<<time_plan.count() <<'\n';
            std::chrono::steady_clock::time_point calcstart = std::chrono::steady_clock::now();

            BlockPlanner::calculate();

            std::chrono::duration<double> time_calc = 
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-calcstart);
            std::cout << "kernel time: "<<time_calc.count() <<'\n';

            data::clearOperations();

            std::chrono::steady_clock::time_point solvestart = std::chrono::steady_clock::now();

            double *xx =  (double*)memutil::getSmallMem(0, config::blockRows * config::blockSize * sizeof(double));
            std::memset(xx, 0, config::blockRows * config::blockSize * sizeof(double));
            BlockPlanner::getFirstColumn(x2, xx, config::blockRows * config::blockSize);
            std::chrono::duration<double> time_solve =
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-solvestart);
            std::cout << "solve triangled :"<<time_solve.count() <<'\n';
            return xx;
        }

        double* solveLU(int dim, int valcount, bool symmetric, int* index_i, int* index_j, double* vals, double* b) 
        {
            std::chrono::steady_clock::time_point timestart = std::chrono::steady_clock::now();

            data::mSize = dim;
            data::mSize_out = dim;
            data::valcount = valcount;
            data::symmetric = symmetric;

            if(symmetric) data::valcount = valcount * 2 - dim;

            data::indexi = memutil::getSmallMem(1,sizeof(int)*data::valcount);
            data::indexj = memutil::getSmallMem(1,sizeof(int)*data::valcount);
            data::vals = memutil::getSmallMem(1,sizeof(double)*data::valcount);
    
            int ii = 0;
            for(int i=0; i<valcount; i++){
                data::indexi[ii] = index_i[i];
                data::indexj[ii] = index_j[i];
                data::vals[ii] = vals[i];
                ii++;
                if(symmetric && index_i[i] != index_j[i]){
                    data::indexi[ii] = index_j[i];
                    data::indexj[ii] = index_i[i];
                    data::vals[ii] = vals[i];
                    ii++;
                }
            }
            data::valcount = ii;

            config::set(dim);
            data::blockRows = config::blockRows;
            data::blockSize = config::blockSize;

            data::b = memutil::getSmallMem(1,data::blockRows * data::blockSize *sizeof(double));
            for(int i=0; i<dim; i++){
                data::b[i] = b[i];
            }
            for(int i=dim; i<data::blockRows * data::blockSize; i++){
                data::b[i] = 1;
            }

            GOrder::Reorder();
            GOrder::sortInBlock();
            std::chrono::steady_clock::time_point timereorder = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_plan = std::chrono::duration_cast<std::chrono::duration<double>>(timereorder-timestart);
            std::cout << "re Order time: "<<time_plan.count() <<'\n';
            GOrder::clearmost();

            double* x = decompose_solveX();

            double* nr =  GOrder::reOrderResult(x);

            data::clear();
            GOrder::clear();

            std::chrono::duration<double> time_calc = 
                     std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timereorder);
            std::chrono::duration<double> time_total = 
                     std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-timestart);
            std::cout <<"totaltime: "<<time_total.count()<< " reordertime: "<<time_plan.count()<<" calculationtime: "<<time_calc.count()<<std::endl;

            return nr;
        }
}

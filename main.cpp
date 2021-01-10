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
#include "mtx.h"
#include "solver.h"

int main (int argc, char* argv[]) {

    SOGLU::iniData();

    if(argc<=1 || std::string(argv[1]).find(".mtx")==std::string::npos){
        std::cout<<"usage: ./solve filename.mtx"<<std::endl;
        return 0;
    }
    std::string fname = argv[1];
    
    std::string filebase = fname.substr(0,fname.find(".mtx"));
    if(SOGLU::mtx::readMTX(fname)==0)
        return 0;

    SOGLU::mtx::readArray(filebase + "_b.mtx", SOGLU::mtx::mSize);

    double* x = SOGLU::solveLU(
        SOGLU::mtx::mSize, 
        SOGLU::mtx::valcount, 
        SOGLU::mtx::symmetric, 
        SOGLU::mtx::indexi, 
        SOGLU::mtx::indexj, 
        SOGLU::mtx::vals, 
        SOGLU::mtx::b
        );

    SOGLU::mtx::checkResult(x);
    SOGLU::mtx::writeArray(filebase + "_x.mtx", SOGLU::data::b, SOGLU::data::mSize_out);
    SOGLU::mtx::sampleArray(x, SOGLU::mtx::mSize);

    SOGLU::memutil::clear();
    return 0;
}

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
#include "memutil.h"
#include "data.h"
#include "mtx.h"

namespace SOGLU {

     int mtx::mSize=0;
     bool mtx::symmetric=false;

     int* mtx::indexi=NULL;
     int* mtx::indexj=NULL;
     double* mtx::vals=NULL;
     int mtx::valcount=0;
     double* mtx::b=NULL;

int mtx::readMTX(std::string fname){
    long    count=0;
    int    row, col;
    double val;
    std::string line;
    int    rows, cols, counts;

    char  cstr[1024];
    char * ps;
    char * pe;
    int    sym = 0;

    std::ifstream myfile (fname.c_str());
    if (myfile.is_open())
    {
        std::getline(myfile,line);
        if(line.find("symmetric") != std::string::npos) {sym = 1; symmetric = true;}
        myfile.seekg(0,myfile.beg);
        while ( std::getline (myfile,line) ){
            if(line.length()<=3) continue;
            if(line.find('%') != std::string::npos) continue;

            strcpy (cstr, line.c_str());
         
            rows = (int)strtol(cstr,&ps,10);
            cols = (int)strtol(ps, &pe,10);
            counts = (int)strtol(pe, NULL,10);
            if(rows != cols){
                std::cout<<" not square matrix "<<std::endl; 
                if(rows > cols)
                    rows = cols;
            }
         
            mSize = rows;
            valcount= counts ;
            indexi = memutil::getSmallMem(1,sizeof(int)*counts);
            indexj = memutil::getSmallMem(1,sizeof(int)*counts);
            vals = memutil::getSmallMem(1,sizeof(double)*counts);
            for(int i =0; i<valcount; i++){
                indexi[i] = -1;
                indexj[i] = -1;
                vals[i] = 0;
            }
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

            indexi[count] = row-1;
            indexj[count] = col-1;
            vals[count] = val;
            count++;
        }
        myfile.close();
        valcount= count;
    }
    else std::cout << "Can not open file"<<'\n'; 
    return count;
}

void mtx::sampleArray(double *a, int dim){
        if(dim>3){
            std::cout<<a[0]<<" "<<a[1]<<" "<<a[2]<<" ";
        }
        std::cout<<"...";
        if(dim>3){
            std::cout<<" "<<a[dim-3]<<" "<<a[dim-2]<<" "<<a[dim-1]<<" ";
        }
        std::cout <<'\n';
}

void mtx::writeArray(std::string fname, double *a, int dim){
    std::ofstream myfile (fname.c_str());
    if (myfile.is_open())
    {
        myfile<<"%%MatrixMarket matrix array real general"<<std::endl;
        myfile<<dim<<" 1"<<std::endl;
        for(int i=0;i<dim;i++){
            myfile<<a[i]<<std::endl;
        }
        myfile.close();
    }else {
        std::cout << "Can not open file"<<'\n';
    }
}

int mtx::readArray(std::string fname, int dim){
    long    count=0;
    double val;
    std::string line;

    char  cstr[1024];
    std::ifstream myfile (fname.c_str());
    b = memutil::getSmallMem(1,sizeof(double)*dim);
    if (myfile.is_open())
    {
        while ( std::getline (myfile,line) ){
            if(line.find('%') != std::string::npos) continue;
            break;
        }
         
        std::cout<<line<<"\n";
        while ( std::getline (myfile,line) ){
            if(count >= dim) break;
            if(line.find('%') != std::string::npos) continue;
            if(line.length()>=1000) continue;

            strcpy (cstr, line.c_str());

            val = strtod(cstr, NULL);

            b[count] = val;

            count++;
        }
        myfile.close();
    } 
    
    for(int i=count; i<dim; i++){
        b[i] = 1;
    }
   
    return count;
}
    double mtx::checkResult( double* nx)
        {

            if(b == NULL || nx == NULL)
                return 1;

            double* bb = new double[mSize];
            for (int i = 0; i < mSize; i++) {
                bb[i] = 0;
            }
            if(symmetric){
                for (int i = 0; i < valcount; i++) {
                    bb[indexi[i]] += vals[i] * nx[indexj[i]];
                    if(indexi[i] != indexj[i])
                        bb[indexj[i]] += vals[i] * nx[indexi[i]];
                }
            }
            else{
                for (int i = 0; i < valcount; i++) {
                    bb[indexi[i]] += vals[i] * nx[indexj[i]];
                }
            }
            double max = 0;
            for (int i = 0; i < mSize; i++){
                double t = fabs(b[i] - bb[i]);
                if(t > max || std::isnan(t) || std::isinf(t)) {
                    max = t;
                }
            }
            std::cout<<"max rhs error:"<<max<<std::endl;
            delete[] bb;
            return max;
        }

}

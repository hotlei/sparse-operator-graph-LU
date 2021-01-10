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

#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <sys/mman.h>
#include "operation.h"
#include "matrix.h"
#include "data.h"
#include "MatrixStdDouble.h"
#define TINY  1.0e-20
#define TINY1  1.0e-18
#define TINY2  1.0e-9
#define STEPCOL 4
#define STEPROW 4

namespace SOGLU
{
        uint MatrixStdDouble::mask16[16] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u,
                256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u};

        void resetMetaField(double * y){
            uint *metay = (uint*) (y + METAOFFSET);
            unsigned short *metad = (unsigned short *) (y + DETAILOFFSET);

            std::memset(metay, 0, METASIZE);
            for(int j = 0; j < 4; j++){
                for(int i = 0; i<CACHE64/sizeof(unsigned short); i++) metad[i] = 0;
                metad += DETAILSKIPSHORT;
            }
        }
        void copyMetaField(double * src, double* dst){
            unsigned short *metas = (unsigned short *) (src + DETAILOFFSET);
            unsigned short *metad = (unsigned short *) (dst + DETAILOFFSET);

            for(int j = 0; j < 4; j++){
                std::memcpy(metad,metas,64);
                metas += DETAILSKIPSHORT;
                metad += DETAILSKIPSHORT;
            }
            metas = (unsigned short *) (src + METAOFFSET);
            metad = (unsigned short *) (dst + METAOFFSET);
            std::memcpy(metad,metas,METASIZE);
        }
        void updateMeta(double* y, int n){

          //uint16_t *metab = (uint16_t*)(b + DETAILOFFSET + (j/32)*DETAILSKIPSHORT/4);
            uint *metay = (uint*) (y + METAOFFSET);
            unsigned short *metad = (unsigned short *) (y + DETAILOFFSET);
            int m, p, s, t, i, j, nonzero;

            resetMetaField(y);

            int n8 = BLOCK64 / 8;
            for(m = 0; m<n8; m++)
            for(p = 0; p<n8; p++){
                nonzero = 0;
                metad = (unsigned short *) (y + DETAILOFFSET);
                metad = metad + DETAILSKIPSHORT * (m/4);
                for(s=0; s<8; s++){
                    int line = 0;
                    i = m * 8 + s;
                    for(t=0; t<8; t++){
                        j = p * 8 + t;
                        if(y[i * BLOCKCOL + j] > TINY || y[i * BLOCKCOL + j] < -TINY){
                            line = 1;
                            break;
                        }
                    }
                    if(line > 0) {
              //          metad[(m%4) * 8 + s] = metad[(m%4) * 8 + s] | MatrixStdDouble::mask16[p];   
                        metad[i%32] = metad[i%32] | MatrixStdDouble::mask16[p];   
                        nonzero = 1;
                    }
                }
                if(nonzero > 0)
                    metay[m] = metay[m] | MatrixStdDouble::mask16[p];
            }
        }
         
        void MatrixStdDouble::mat_clear(double* b, int n)
        {
            std::memset(b, 0, ALLOCBLOCK);
        }

        void MatrixStdDouble::mat_clean(double* y, int n)
        {
            uint *metay = (uint*) (y + METAOFFSET);
            int m, p, s, t, i, j;

            int n8 = BLOCK64 / 8;
            for(m = 0; m<n8; m++)
            for(p = 0; p<n8; p++){
                if((metay[m] & MatrixStdDouble::mask16[p]) == 0){
                    for(s=0;s<8;s++){
                        i = m * 8 + s;
                        for(t=0; t<8; t++){
                            j = p * 8 + t;
                            y[i * BLOCKCOL + j] = 0;
                        }
                    }
                    continue;
                }
                unsigned short *metad = (unsigned short *) (y + DETAILOFFSET);
                metad = metad + (m/4) * DETAILSKIPSHORT;
                for(s=0; s<8; s++){
                    i = m * 8 + s;
                    if((metad[i%32] & MatrixStdDouble::mask16[p]) == 0)
                    for(t=0; t<8; t++){
                        j = p * 8 + t;
                        y[i * BLOCKCOL + j] = 0;
                    }
                }
            }
        }
        void MatrixStdDouble::mat_clean_lower(double* b, int n)
        {
            mat_clean(b,n);
            int m, p, s, t, i, j;
            int n8 = n / 8;
            for(p = 0; p<n8; p++){
                m = p;
                for(s=0;s<8;s++){
                    i = m * 8 + s;
                    for(t=s+1; t<8; t++){
                        j = p * 8 + t;
                        b[i * BLOCKCOL + j] = 0;
                    }
                }
            }
        }
        void MatrixStdDouble::mat_clean_upper(double* b, int n)
        {
            mat_clean(b,n);
            int m, p, s, t, i, j;
            int n8 = n / 8;
            for(p = 0; p<n8; p++){
                m = p;
                for(s=0;s<8;s++){
                    i = m * 8 + s;
                    for(t=0; t<s; t++){
                        j = p * 8 + t;
                        b[i * BLOCKCOL + j] = 0;
                    }   
                }
            }
        }

        void MatrixStdDouble::blockMulOneAvxBlockNeg(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {
            for(int i=0;i<BLOCK64;i++){
                    for(int k=0;k<BLOCK64;k++){
                for(int j=0;j<BLOCK64;j++){
                        c[i*BLOCKCOL+j] -= a[i*BLOCKCOL+k] * b[k*BLOCKCOL+j];
                    }
                }
            }
            updateMeta(c, BLOCK64);
        }

//
        void MatrixStdDouble::mat_mult(double *a, double *c, unsigned long msk, int coreIdx) //64
        {      // 2.8 sec
            for(int i=0;i<BLOCK64;i++){
                for(int j=0;j<BLOCK64;j++){
                    double sum = 0;
                    for(int k=0;k<BLOCK64;k++){
                        sum += a[i*BLOCKCOL+k] * a[j*BLOCKCOL+k];
                    }
                    c[i*BLOCKCOL+j] += sum;
                }
            }
            updateMeta(c, BLOCK64);
        }
        void MatrixStdDouble::mat_mult(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {      // 2.8 sec
            for(int i=0;i<BLOCK64;i++){
                for(int j=0;j<BLOCK64;j++){
                    double sum = 0;
                    for(int k=0;k<BLOCK64;k++){
                        sum += a[i*BLOCKCOL+k] * b[j*BLOCKCOL+k];
                    }
                    c[i*BLOCKCOL+j] += sum;
                }
            }
            updateMeta(c, BLOCK64);
        }
/* */

        void MatrixStdDouble::blockMulOneAvxBlock(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {                                                            //    HOT hot hotspot
            for(int i=0;i<BLOCK64;i++){
                    for(int k=0;k<BLOCK64;k++){
                for(int j=0;j<BLOCK64;j++){
                        c[i*BLOCKCOL+j] += a[i*BLOCKCOL+k] * b[k*BLOCKCOL+j];
                    }
                }
            }
            updateMeta(c, BLOCK64);
        }

        void MatrixStdDouble::printMatrix(double* a, int n)
        {
            int count = 0;
            for(int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double t = a[i * BLOCKCOL + j];

             //       if(i==j && i%4==0)
                    if ((!(t < TINY2 && t > -TINY2)) || i==j){
                        std::cout<<i<<" "<<j<<" "<<t<<std::endl;
                        count++;
                    }
                }
            }
            std::cout<<n<<" "<<n<<" non zero "<<count<<std::endl;
        }

        void expandNoneZero(double *ldouble, double *udouble, int n)
        {
            uint *l = (uint *)(ldouble + METAOFFSET);
            uint *u = (uint *)(udouble + METAOFFSET);
            uint16_t *metal =  (uint16_t *)(ldouble + DETAILOFFSET);
            uint16_t *metau =  (uint16_t *)(udouble + DETAILOFFSET);
            uint16_t ui = *metal;
            uint masku = METAMASK;
            uint maskl = 1u;
            for(int m=0; m<BLOCK64/8; m++){
                uint ml = 0;
                uint mu = 0;
                for(int k=0;k<8;k++){
                    int i = m * 8 + k;
                    uint detail = (metal[DETAILSKIPSHORT * (i/32) + i%32] & METAMASK);
                    uint dl = detail & maskl;
                    dl = dl | dl >> 8;
                    dl = dl | dl >> 4;
                    dl = dl | dl >> 2;
                    dl = dl | dl >> 1;
                    dl = dl & maskl;

                    uint du = detail & masku;
                    ui = ui | du;
                    du = ui;
                   
                    metal[DETAILSKIPSHORT * (i/32) + i%32] = dl;
                    metau[DETAILSKIPSHORT * (i/32) + i%32] = du;
                    ml = ml | dl;
                    mu = mu | du;
                }
                masku = (masku << 1) & METAMASK;
                maskl = (maskl << 1) + 1;
                l[m] = ml;
                u[m] = mu;
            }
        }
        void expandNoneZero(double *ldouble, int n)
        {
            uint *l = (uint *)(ldouble + METAOFFSET);
            uint16_t *metal =  (uint16_t *)(ldouble + DETAILOFFSET);
            uint16_t ui = *metal;
            uint masku = METAMASK;
            uint maskl = 1u;
            for(int m=0; m<BLOCK64/8; m++){
                uint ml = 0;
                uint mu = 0;
                for(int k=0;k<8;k++){
                    int i = m * 8 + k;
                    uint detail = (metal[DETAILSKIPSHORT * (i/32) + i%32] & METAMASK);
                    uint dl = detail & maskl;
                    dl = dl | dl >> 8;
                    dl = dl | dl >> 4;
                    dl = dl | dl >> 2;
                    dl = dl | dl >> 1;
                    dl = dl & maskl;

                    uint du = detail & masku;
                    ui = ui | du;
                    du = ui;
                   
                    metal[DETAILSKIPSHORT * (i/32) + i%32] = dl;
                    ml = ml | dl;
                    mu = mu | du;
                }
                masku = (masku << 1) & METAMASK;
                maskl = (maskl << 1) + 1;
                l[m] = ml;
            }
        }

        void MatrixStdDouble::lltdcmpSimple(double* a, int n, double* l)
 //       void lltdcmpSimple_nr(double* a, int n, double* l)
        { // cout
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }
            std::memset(l,0,ALLOCBLOCK);

            for (int j=0; j<BLOCK64; j++){
                double sum = 0;
                for(int k=0;k<j;k++)
                    sum += l[j * BLOCKCOL + k] * l[j * BLOCKCOL + k];
                double p = a[j * BLOCKCOL + j] - sum;
                if(p<TINY) p = TINY;
                p = sqrt(p);
                l[j * BLOCKCOL + j] = p;
                p = 1/ p;

                for(int i=j+1; i<BLOCK64; i++){
                    double sun = 0;
                    for(int k=0;k<j;k++)
                        sun += l[i * BLOCKCOL + k] * l[j * BLOCKCOL + k];
                    l[i * BLOCKCOL + j] = p * (a[i * BLOCKCOL + j] - sun);
                }
            }

            updateMeta(l, BLOCK64);
        }

        void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* u)
 //       void ludcmpSimple_nr(double* a, int n, double* l, double* u)
        {
            long double sum = 0;
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }
            std::memset(u,0,ALLOCBLOCK);
            std::memset(l,0,ALLOCBLOCK);
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += l[i * BLOCKCOL + k] * u[k * BLOCKCOL + j];
                    u[i * BLOCKCOL + j] = a[i * BLOCKCOL + j] - sum;
                }
                if (u[i * BLOCKCOL + i] < TINY2 && u[i * BLOCKCOL + i] > -TINY2)
                {
       //             std::cout<<" i " <<i<<" j "<<i<<" val "<<u[i*n+i]<<std::endl;
                    if (u[i * BLOCKCOL + i] < 0)
                        u[i * BLOCKCOL + i] = -TINY2;
                    else
                        u[i * BLOCKCOL + i] = TINY2;
                }
                long double u1 = u[i * BLOCKCOL + i];
                u1 = 1 / u1;
                for (int j = i + 1; j < n; j++)
                {
                    sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += l[j * BLOCKCOL + k] * u[k * BLOCKCOL + i];
                    l[j * BLOCKCOL + i] = u1 * (a[j * BLOCKCOL + i] - sum);
                }
                l[i * BLOCKCOL + i] = 1;
            }

            updateMeta(l, n);
            updateMeta(u, n);
        }

        void MatrixStdDouble::inv_lower(double* l, int n, double* y)
        {
            std::memset(y, 0, BLOCK64 * BLOCKCOL * sizeof(double));
            for(int i=0; i<BLOCK64; i++){
                double q = 1.0/l[i * BLOCKCOL + i];
                y[i * BLOCKCOL + i] = q;
                for(int j=0; j<i; j++){
                    double sum = 0;
                    for(int k=j;k<i; k++){
                        sum += l[i*BLOCKCOL+k] * y[k*BLOCKCOL+j];
                    }
                    y[i*BLOCKCOL+j] = - sum * q;
                }
            }
            updateMeta(y,BLOCK64);
        }

        void MatrixStdDouble::inv_upper(double* u, int n, double* y)
        {
            int i, j;
            std::memset(y, 0, BLOCK64 * BLOCKCOL * sizeof(double));
            for (i = 0; i < n; i++)
                y[i * BLOCKCOL + i] = 1.0;
            for (j = n-1; j >=0; j--)
            {
                for (i = j-1; i >=0  ; i--)
                {
                    double scale = u[j * BLOCKCOL + j];
                    scale = u[i * BLOCKCOL + j] / scale;
                    for (int k = j; k < n; k++)
                        y[i * BLOCKCOL + k] -= y[j * BLOCKCOL + k] * scale;
                }

                long double rate = u[j * BLOCKCOL + j];
                rate = 1.0 / rate;
                for (i = j; i < n; i++)
                    y[j * BLOCKCOL + i] = y[j * BLOCKCOL + i] * rate;
            }
            updateMeta(y, n);
        }

        void MatrixStdDouble::mat_inv(double* a, int n, double* y)
        {
        }

        bool MatrixStdDouble::inv_check_diag(double* a, double* b, int n)
        {
            int i, j, k;
            double sum;
            double min, max;
            min = 1;
            max = 1;
            double tor = 1e-3;
            double* y = new double[n];
            for(i=0;i<n;i++) y[i] = 0;
            for (i = 0; i < n; i++)
            {
                j = i;
                {
                    sum = 0.0f;
                    for (k = 0; k < n; k++)
                    {
                        sum += a[i * BLOCKCOL + k] * b[k * BLOCKCOL + j];
                    }
                    y[i] = sum;
                    if(std::isnan(sum)){
                        max = 10;
                        continue;
                    }
                    if (min > sum)
                        min = sum;
                    if (max < sum)
                        max = sum;
                }
            }
            if( max > 1+tor || max < 1-tor || min > 1+ tor || min < 1-tor){
                std::cout<<"diag min:" + std::to_string(min) + "  diag max:" + std::to_string(max)<<std::endl;
                delete[] y;
                return false;
            }

            max = 0;
            min = 0;
            for(i=0;i<n;i++) y[i] = 0;
            for (i = 2; i < n; i++)
            {
                j = i/2;
                {
                    sum = 0.0f;
                    for (k = 0; k < n; k++)
                    {
                        sum += a[i * BLOCKCOL + k] * b[k * BLOCKCOL + j];
                    }
                    y[i] = sum;
                    if(std::isnan(sum)){
                        max = 10;
                        continue;
                    }
                    if (min > sum)
                        min = sum;
                    if (max < sum)
                        max = sum;
                }
            }
                delete[] y;
            if( max > 0+tor || max < 0-tor || min > 0+ tor || min < 0-tor){
                std::cout<<"middle min:" + std::to_string(min) + "  middle max:" + std::to_string(max)<<std::endl;
                return false;
            }

            return true;
        }

        void MatrixStdDouble::mat_sub(double* a, double* b, int n, double* r, int coreIdx)
        {
            for(int i=0;i<BLOCK64;i++)
                for(int j=0;j<BLOCK64;j++){
                    r[i*BLOCKCOL+j] = a[i*BLOCKCOL+j] - b[i*BLOCKCOL+j];
                }
            updateMeta(r, BLOCK64);
        }

        void MatrixStdDouble::mat_copy(double* a, int n, double* r, int coreIdx)
        {
            copyMetaField(a,r);
            for(int i=0;i<BLOCK64;i++)
                for(int j=0;j<BLOCK64;j++){
                    r[i*BLOCKCOL+j] = a[i*BLOCKCOL+j];
                }
        }

        void MatrixStdDouble::mat_neg(double* b, int n, double* r, int coreIdx)
        {
            copyMetaField(b,r);
            for(int i=0;i<BLOCK64;i++)
                for(int j=0;j<BLOCK64;j++){
                    r[i*BLOCKCOL+j] =  - b[i*BLOCKCOL+j];
                }
        }
}


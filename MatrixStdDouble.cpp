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
#include <immintrin.h>
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
        alignas(64) const double  MatrixStdDouble::blanckdata[512] = 
                                        { 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                        };


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
        void combineMetaField(double * src, double * src2, double* dst){
            unsigned short *metas = (unsigned short *) (src + DETAILOFFSET);
            unsigned short *metat = (unsigned short *) (src2 + DETAILOFFSET);
            unsigned short *metad = (unsigned short *) (dst + DETAILOFFSET);

            for(int j = 0; j < 4; j++){
                __m512i mrs = _mm512_load_epi32((void *)metas);
                __m512i mrt = _mm512_load_epi32((void *)metat);
                __m512i mrd = _mm512_or_epi32(mrs,mrt);
                _mm512_store_epi32((void *)metad, mrd);
                metas += DETAILSKIPSHORT;
                metat += DETAILSKIPSHORT;
                metad += DETAILSKIPSHORT;
            }
            metas = (unsigned short *) (src + METAOFFSET);
            metat = (unsigned short *) (src2 + METAOFFSET);
            metad = (unsigned short *) (dst + METAOFFSET);
            __m512i mrs = _mm512_load_epi32((void *)metas);
            __m512i mrt = _mm512_load_epi32((void *)metat);
            __m512i mrd = _mm512_or_epi32(mrs,mrt);
            _mm512_store_epi32((void *)metad, mrd);
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
/*
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
/* */
//
        void MatrixStdDouble::blockMulOneAvxBlockNeg(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {
            const int n = BLOCK64;
            uint calc;
            uint czero;

            _mm_prefetch(c+METAOFFSET, _MM_HINT_T1);
            _mm_prefetch(c+DETAILOFFSET, _MM_HINT_T1);
            uint *metaa = (uint*) (a + METAOFFSET);
            uint *metab = (uint*) (b + METAOFFSET);
            uint *metac = (uint*) (c + METAOFFSET);
            _mm_prefetch(metaa, _MM_HINT_T1);
            _mm_prefetch(metab, _MM_HINT_T1);
            int skipped = 0;

            int seq;
                double* cinn;
                double* aaik;
                double* bbkp;
//  //
              for(int m=0;m<BLOCK64/8;m++){
                     uint16_t *metacd = (uint16_t *)(c + DETAILOFFSET);
                     metacd += DETAILSKIPSHORT * (m/4); 
                                   uint16_t metacm0 = metacd[(m%4)*8+0];
                                   uint16_t metacm1 = metacd[(m%4)*8+1];
                                   uint16_t metacm2 = metacd[(m%4)*8+2];
                                   uint16_t metacm3 = metacd[(m%4)*8+3];
                                   uint16_t metacm4 = metacd[(m%4)*8+4];
                                   uint16_t metacm5 = metacd[(m%4)*8+5];
                                   uint16_t metacm6 = metacd[(m%4)*8+6];
                                   uint16_t metacm7 = metacd[(m%4)*8+7];
                    if((metaa[m] & METAMASK)>0)
                    _mm_prefetch(a+DETAILOFFSET+DETAILSKIPSHORT*(m/4)/4,_MM_HINT_T1); 
                for(int p=0;p<BLOCK64/8;p++){

                    int nonzero = 0;
                    int i = m * 8 ;
                        int inn = i*BLOCKCOL + p * 8;
                        cinn = c+inn;
                    uint mask16p = 1<<p;

                    calc = 0;
                    czero = 0;
                    for(int k=0; k<BLOCK64/8; k++){
                        calc = calc | 1<<k;
                        if((metaa[m] & 1<<k) == 0 || (metab[k] & 1<<p) == 0)
                            calc = calc & (METAMASK - (1<<k));
                    }
                    for(int k=BLOCK64/8-1; k>=0; k--){
                        if((calc & 1<<k) == 0) continue;
                        nonzero = 1;
                    }
                    if(nonzero == 0) {continue;}
                     uint16_t *metaad = (uint16_t *)(a + DETAILOFFSET);
                     metaad += DETAILSKIPSHORT * (i/32); 
                           uint16_t metaai0 = metaad[i%32+0];
                           uint16_t metaai1 = metaad[i%32+1];
                           uint16_t metaai2 = metaad[i%32+2];
                           uint16_t metaai3 = metaad[i%32+3];
                           uint16_t metaai4 = metaad[i%32+4];
                           uint16_t metaai5 = metaad[i%32+5];
                           uint16_t metaai6 = metaad[i%32+6];
                           uint16_t metaai7 = metaad[i%32+7];

                           __m512d r16; 
                           __m512d r17; 
                           __m512d r18; 
                           __m512d r19; 
                           __m512d r20; 
                           __m512d r21; 
                           __m512d r22; 
                           __m512d r23; 
                           r16 = _mm512_xor_pd(r16,r16); 
                           r17 = _mm512_xor_pd(r17,r17); 
                           r18 = _mm512_xor_pd(r18,r18); 
                           r19 = _mm512_xor_pd(r19,r19); 
                           r20 = _mm512_xor_pd(r20,r20); 
                           r21 = _mm512_xor_pd(r21,r21); 
                           r22 = _mm512_xor_pd(r22,r22); 
                           r23 = _mm512_xor_pd(r23,r23); 

                           if(__builtin_expect((metacm0 & mask16p)>0,0)){ czero = czero | 1u;
                           r16 = _mm512_load_pd ((void const*) (cinn));}
                           if(__builtin_expect((metacm1 & mask16p)>0,0)){ czero = czero | 2u;
                           r17 = _mm512_load_pd ((void const*) (cinn+ROWSKIP64));}
                           if(__builtin_expect((metacm2 & mask16p)>0,0)){ czero = czero | 4u;
                           r18 = _mm512_load_pd ((void const*) (cinn+ROWSKIP128));}
                           if(__builtin_expect((metacm3 & mask16p)>0,0)){ czero = czero | 8u;
                           r19 = _mm512_load_pd ((void const*) (cinn+ROWSKIP192));}
                           if(__builtin_expect((metacm4 & mask16p)>0,0)){ czero = czero | 16u;
                           r20 = _mm512_load_pd ((void const*) (cinn+ROWSKIP256));}
                           if(__builtin_expect((metacm5 & mask16p)>0,0)){ czero = czero | 32u;
                           r21 = _mm512_load_pd ((void const*) (cinn+ROWSKIP320));}
                           if(__builtin_expect((metacm6 & mask16p)>0,0)){ czero = czero | 64u;
                           r22 = _mm512_load_pd ((void const*) (cinn+ROWSKIP384));}
                           if(__builtin_expect((metacm7 & mask16p)>0,0)){ czero = czero | 128u;
                           r23 = _mm512_load_pd ((void const*) (cinn+ROWSKIP448));}

                        for (int k = 0; k < BLOCK64/8; k++)
                    {
                        if((calc & 1<<k) == 0) continue;
                        _mm_prefetch(b+DETAILOFFSET+DETAILSKIPSHORT*(k/4)/4,_MM_HINT_T1);
                        int k8 = k*8;
                        int aik = i*BLOCKCOL + k8;
                        uint mask16k = 1<<k;

                        int bkp = k8 * BLOCKCOL + p * 8;
                         aaik = a+aik;
                         bbkp = b+bkp;

                           __m512d r00;
                           __m512d r01;
                           __m512d r02;
                           __m512d r03;
                           __m512d r04;
                           __m512d r05;
                           __m512d r06;
                           __m512d r07;
                           __m512d r24;
                           __m512d r25;
                           __m512d r26;
                           __m512d r27;
                           __m512d r28;
                           __m512d r29;
                           __m512d r30;
                           __m512d r31;
                        void * bbkp0 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp64 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp128 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp192 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp256 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp320 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp384 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp448 = (void *)(MatrixStdDouble::blanckdata);
                               _mm_prefetch(MatrixStdDouble::blanckdata,_MM_HINT_T1);

                     uint16_t *metabd = (uint16_t *)(b + DETAILOFFSET);
                     metabd += DETAILSKIPSHORT * (k/4); 
                           uint metabk0 = metabd[(k%4)*8+0];
                           uint metabk1 = metabd[(k%4)*8+1];
                           uint metabk2 = metabd[(k%4)*8+2];
                           uint metabk3 = metabd[(k%4)*8+3];
                           uint metabk4 = metabd[(k%4)*8+4];
                           uint metabk5 = metabd[(k%4)*8+5];
                           uint metabk6 = metabd[(k%4)*8+6];
                           uint metabk7 = metabd[(k%4)*8+7];

                           __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                           {
                           uint16_t *metabk8 = (uint16_t *)(metabd + (k%4)*8);
                           __m512i bbkp512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(bbkp));
                           __m512i meta512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)metabk8));
                           __m512i mask512 = _mm512_set1_epi64(mask16p);
                           __m512i offset512 = _mm512_slli_epi64(_mm512_set_epi64(ROWSKIP448,ROWSKIP384,ROWSKIP320,ROWSKIP256,ROWSKIP192,ROWSKIP128,ROWSKIP64,0),3);
                                   bbkp512 = _mm512_add_epi64(bbkp512,offset512);
                                   mask512 = _mm512_srai_epi64(_mm512_and_epi64(meta512,mask512),p); 
                                   mask512 = _mm512_sub_epi64(mask512,_mm512_set1_epi64(1));
                           __m512i addr512 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,bbkp512));        

                           __m256i lo4 =_mm512_extracti64x4_epi64(addr512,0);
                           __m256i hi4 =_mm512_extracti64x4_epi64(addr512,1);
               
                           bbkp0 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 0));
                               __builtin_prefetch(bbkp0,0,3);
                           bbkp64 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 1));
                               __builtin_prefetch(bbkp64,0,3);
                           bbkp128 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 2));
                               __builtin_prefetch(bbkp128,0,3);
                           bbkp192 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 3));
                               __builtin_prefetch(bbkp192,0,3);
                           bbkp256 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 0));
                               __builtin_prefetch(bbkp256,0,3);
                           bbkp320 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 1));
                               __builtin_prefetch(bbkp320,0,3);
                           bbkp384 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 2));
                               __builtin_prefetch(bbkp384,0,3);
                           bbkp448 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 3));
                               __builtin_prefetch(bbkp448,0,3);
                           }

                        double *aaik0 = MatrixStdDouble::blanckdata;
                        double *aaik64 = MatrixStdDouble::blanckdata;
                        double *aaik128 = MatrixStdDouble::blanckdata;
                        double *aaik192 = MatrixStdDouble::blanckdata;
                        double *aaik256 = MatrixStdDouble::blanckdata;
                        double *aaik320 = MatrixStdDouble::blanckdata;
                        double *aaik384 = MatrixStdDouble::blanckdata;
                        double *aaik448 = MatrixStdDouble::blanckdata;
                       
                           { 
                           __m512i aaik512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(aaik));
                           __m512i metaa512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metaad+i%32)));
                           __m512i maskk512 = _mm512_set1_epi64(mask16k);
                           __m512i offsetk512 = _mm512_slli_epi64(_mm512_set_epi64(ROWSKIP448,ROWSKIP384,ROWSKIP320,ROWSKIP256,ROWSKIP192,ROWSKIP128,ROWSKIP64,0),3);
                           aaik512 = _mm512_add_epi64(aaik512,offsetk512);
                           maskk512 = _mm512_srai_epi64(_mm512_and_epi64(metaa512,maskk512),k);
                           maskk512 = _mm512_sub_epi64(maskk512,_mm512_set1_epi64(1));
                           __m512i addrk512 = _mm512_or_epi64(_mm512_and_epi64(maskk512,blanck512),_mm512_andnot_epi64(maskk512,aaik512));

                           __m256i lok4 =_mm512_extracti64x4_epi64(addrk512,0);
                           __m256i hik4 =_mm512_extracti64x4_epi64(addrk512,1);
                           aaik0 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 0));
                           aaik64 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 1));
                               __builtin_prefetch(aaik64,0,3);
                           aaik128 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 2));
                               __builtin_prefetch(aaik128,0,3);
                           aaik192 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 3));
                               __builtin_prefetch(aaik192,0,3);
                           aaik256 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 0));
                               __builtin_prefetch(aaik256,0,3);
                           aaik320 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 1));
                               __builtin_prefetch(aaik320,0,3);
                           aaik384 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 2));
                               __builtin_prefetch(aaik384,0,3);
                           aaik448 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 3));
                               __builtin_prefetch(aaik448,0,3);
                           }
                      
                           {
                           r00 = _mm512_set1_pd(aaik0[0]);
                           r01 = _mm512_set1_pd(aaik64[0]);
                           r02 = _mm512_set1_pd(aaik128[0]);
                           r03 = _mm512_set1_pd(aaik192[0]);
                           r04 = _mm512_set1_pd(aaik256[0]);
                           r05 = _mm512_set1_pd(aaik320[0]);
                           r06 = _mm512_set1_pd(aaik384[0]);
                           r07 = _mm512_set1_pd(aaik448[0]);

                           __m512d r08 = _mm512_load_pd (bbkp0);
                           r16 = _mm512_fnmadd_pd (r00, r08, r16);
                           r17 = _mm512_fnmadd_pd (r01, r08, r17);
                           r18 = _mm512_fnmadd_pd (r02, r08, r18);
                           r19 = _mm512_fnmadd_pd (r03, r08, r19);
                           r20 = _mm512_fnmadd_pd (r04, r08, r20);
                           r21 = _mm512_fnmadd_pd (r05, r08, r21);
                           r22 = _mm512_fnmadd_pd (r06, r08, r22);
                           r23 = _mm512_fnmadd_pd (r07, r08, r23);
                           }
                           {
                           r24 = _mm512_set1_pd(aaik0[1]);
                           r25 = _mm512_set1_pd(aaik64[1]);
                           r26 = _mm512_set1_pd(aaik128[1]);
                           r27 = _mm512_set1_pd(aaik192[1]);
                           r28 = _mm512_set1_pd(aaik256[1]);
                           r29 = _mm512_set1_pd(aaik320[1]);
                           r30 = _mm512_set1_pd(aaik384[1]);
                           r31 = _mm512_set1_pd(aaik448[1]);
                           __m512d r09 = _mm512_load_pd ( bbkp64);
                           r16 = _mm512_fnmadd_pd (r24, r09, r16);
                           r17 = _mm512_fnmadd_pd (r25, r09, r17);
                           r18 = _mm512_fnmadd_pd (r26, r09, r18);
                           r19 = _mm512_fnmadd_pd (r27, r09, r19);
                           r20 = _mm512_fnmadd_pd (r28, r09, r20);
                           r21 = _mm512_fnmadd_pd (r29, r09, r21);
                           r22 = _mm512_fnmadd_pd (r30, r09, r22);
                           r23 = _mm512_fnmadd_pd (r31, r09, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[2]);
                            r01 = _mm512_set1_pd(aaik64[2]);
                            r02 = _mm512_set1_pd(aaik128[2]);
                            r03 = _mm512_set1_pd(aaik192[2]);
                            r04 = _mm512_set1_pd(aaik256[2]);
                            r05 = _mm512_set1_pd(aaik320[2]);
                            r06 = _mm512_set1_pd(aaik384[2]);
                            r07 = _mm512_set1_pd(aaik448[2]);
                           __m512d r10 = _mm512_load_pd (bbkp128);
                           r16 = _mm512_fnmadd_pd (r00, r10, r16);
                           r17 = _mm512_fnmadd_pd (r01, r10, r17);
                           r18 = _mm512_fnmadd_pd (r02, r10, r18);
                           r19 = _mm512_fnmadd_pd (r03, r10, r19);
                           r20 = _mm512_fnmadd_pd (r04, r10, r20);
                           r21 = _mm512_fnmadd_pd (r05, r10, r21);
                           r22 = _mm512_fnmadd_pd (r06, r10, r22);
                           r23 = _mm512_fnmadd_pd (r07, r10, r23);
                           }
                           {
                            r24 = _mm512_set1_pd(aaik0[3]);
                            r25 = _mm512_set1_pd(aaik64[3]);
                            r26 = _mm512_set1_pd(aaik128[3]);
                            r27 = _mm512_set1_pd(aaik192[3]);
                            r28 = _mm512_set1_pd(aaik256[3]);
                            r29 = _mm512_set1_pd(aaik320[3]);
                            r30 = _mm512_set1_pd(aaik384[3]);
                            r31 = _mm512_set1_pd(aaik448[3]);
                           __m512d r11 = _mm512_load_pd (bbkp192);
                           r16 = _mm512_fnmadd_pd (r24, r11, r16);
                           r17 = _mm512_fnmadd_pd (r25, r11, r17);
                           r18 = _mm512_fnmadd_pd (r26, r11, r18);
                           r19 = _mm512_fnmadd_pd (r27, r11, r19);
                           r20 = _mm512_fnmadd_pd (r28, r11, r20);
                           r21 = _mm512_fnmadd_pd (r29, r11, r21);
                           r22 = _mm512_fnmadd_pd (r30, r11, r22);
                           r23 = _mm512_fnmadd_pd (r31, r11, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[4]);
                            r01 = _mm512_set1_pd(aaik64[4]);
                            r02 = _mm512_set1_pd(aaik128[4]);
                            r03 = _mm512_set1_pd(aaik192[4]);
                            r04 = _mm512_set1_pd(aaik256[4]);
                            r05 = _mm512_set1_pd(aaik320[4]);
                            r06 = _mm512_set1_pd(aaik384[4]);
                            r07 = _mm512_set1_pd(aaik448[4]);
                           __m512d r12 = _mm512_load_pd (bbkp256);
                           r16 = _mm512_fnmadd_pd (r00, r12, r16);
                           r17 = _mm512_fnmadd_pd (r01, r12, r17);
                           r18 = _mm512_fnmadd_pd (r02, r12, r18);
                           r19 = _mm512_fnmadd_pd (r03, r12, r19);
                           r20 = _mm512_fnmadd_pd (r04, r12, r20);
                           r21 = _mm512_fnmadd_pd (r05, r12, r21);
                           r22 = _mm512_fnmadd_pd (r06, r12, r22);
                           r23 = _mm512_fnmadd_pd (r07, r12, r23);
                           }

                           {
                            r24 = _mm512_set1_pd(aaik0[5]);
                            r25 = _mm512_set1_pd(aaik64[5]);
                            r26 = _mm512_set1_pd(aaik128[5]);
                            r27 = _mm512_set1_pd(aaik192[5]);
                            r28 = _mm512_set1_pd(aaik256[5]);
                            r29 = _mm512_set1_pd(aaik320[5]);
                            r30 = _mm512_set1_pd(aaik384[5]);
                            r31 = _mm512_set1_pd(aaik448[5]);
                           __m512d r13 = _mm512_load_pd (bbkp320);
                           r16 = _mm512_fnmadd_pd (r24, r13, r16);
                           r17 = _mm512_fnmadd_pd (r25, r13, r17);
                           r18 = _mm512_fnmadd_pd (r26, r13, r18);
                           r19 = _mm512_fnmadd_pd (r27, r13, r19);
                           r20 = _mm512_fnmadd_pd (r28, r13, r20);
                           r21 = _mm512_fnmadd_pd (r29, r13, r21);
                           r22 = _mm512_fnmadd_pd (r30, r13, r22);
                           r23 = _mm512_fnmadd_pd (r31, r13, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[6]);
                            r01 = _mm512_set1_pd(aaik64[6]);
                            r02 = _mm512_set1_pd(aaik128[6]);
                            r03 = _mm512_set1_pd(aaik192[6]);
                            r04 = _mm512_set1_pd(aaik256[6]);
                            r05 = _mm512_set1_pd(aaik320[6]);
                            r06 = _mm512_set1_pd(aaik384[6]);
                            r07 = _mm512_set1_pd(aaik448[6]);
                           __m512d r14 = _mm512_load_pd (bbkp384);
                           r16 = _mm512_fnmadd_pd (r00, r14, r16);
                           r17 = _mm512_fnmadd_pd (r01, r14, r17);
                           r18 = _mm512_fnmadd_pd (r02, r14, r18);
                           r19 = _mm512_fnmadd_pd (r03, r14, r19);
                           r20 = _mm512_fnmadd_pd (r04, r14, r20);
                           r21 = _mm512_fnmadd_pd (r05, r14, r21);
                           r22 = _mm512_fnmadd_pd (r06, r14, r22);
                           r23 = _mm512_fnmadd_pd (r07, r14, r23);
                           }

                           {
                            r24 = _mm512_set1_pd(aaik0[7]);
                            r25 = _mm512_set1_pd(aaik64[7]);
                            r26 = _mm512_set1_pd(aaik128[7]);
                            r27 = _mm512_set1_pd(aaik192[7]);
                            r28 = _mm512_set1_pd(aaik256[7]);
                            r29 = _mm512_set1_pd(aaik320[7]);
                            r30 = _mm512_set1_pd(aaik384[7]);
                            r31 = _mm512_set1_pd(aaik448[7]);
                           __m512d r15 = _mm512_load_pd (bbkp448);
                           r16 = _mm512_fnmadd_pd (r24, r15, r16);
                           r17 = _mm512_fnmadd_pd (r25, r15, r17);
                           r18 = _mm512_fnmadd_pd (r26, r15, r18);
                           r19 = _mm512_fnmadd_pd (r27, r15, r19);
                           r20 = _mm512_fnmadd_pd (r28, r15, r20);
                           r21 = _mm512_fnmadd_pd (r29, r15, r21);
                           r22 = _mm512_fnmadd_pd (r30, r15, r22);
                           r23 = _mm512_fnmadd_pd (r31, r15, r23);
                           }

                    }
                           int fnd = 0;
                           int fn1 = 0;
                           int fn2 = 0;
                           int fn3 = 0;
                           int fn4 = 0;
                           int fn5 = 0;
                           int fn6 = 0;
                           int fn7 = 0;
                           __m512i rzero = _mm512_set1_epi32(0x60000000u);
//
                           __m512d r00 = _mm512_set1_pd(TINY1);
                           __m512d r01 = _mm512_set1_pd(-TINY1);
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r16, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r16, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm0 = metacm0 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 1u) > 0 || (metacm0 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn), r16);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r17, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r17, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm1 = metacm1 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 2u) > 0 || (metacm1 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP64), r17);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r18, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r18, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm2 = metacm2 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 4u) > 0 || (metacm2 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP128), r18);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r19, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r19, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm3 = metacm3 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 8u) > 0 || (metacm3 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP192), r19);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r20, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r20, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm4 = metacm4 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 16u) > 0 || (metacm4 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP256), r20);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r21, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r21, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm5 = metacm5 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 32u) > 0 || (metacm5 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP320), r21);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r22, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r22, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm6 = metacm6 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 64u) > 0 || (metacm6 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP384), r22);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r23, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r23, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm7 = metacm7 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 128u) > 0 || (metacm7 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP448), r23);}
                           if(fnd > 0)
                           metac[m] = metac[m] | mask16p;
                      // // 
                }
                                    metacd[(m%4)*8+0] = metacm0;
                                    metacd[(m%4)*8+1] = metacm1;
                                    metacd[(m%4)*8+2] = metacm2;
                                    metacd[(m%4)*8+3] = metacm3;
                                    metacd[(m%4)*8+4] = metacm4;
                                    metacd[(m%4)*8+5] = metacm5;
                                    metacd[(m%4)*8+6] = metacm6;
                                    metacd[(m%4)*8+7] = metacm7;
            }
        }
/* */
/*
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
/* */
//
        void MatrixStdDouble::mat_mult(double *a, double *c, unsigned long msk, int coreIdx) //64
        {
            for(int i=0;i<BLOCK64;i+=STEPROW){
                for(int j=0;j<BLOCK64;j+=2){ // STEPCOL){
                    double sum00 = 0;
                    double sum10 = 0;
                    double sum20 = 0;
                    double sum30 = 0;
                    double sum01 = 0;
                    double sum11 = 0;
                    double sum21 = 0;
                    double sum31 = 0;
                    double cum00 = 0;
                    double cum10 = 0;
                    double cum20 = 0;
                    double cum30 = 0;
                    double cum01 = 0;
                    double cum11 = 0;
                    double cum21 = 0;
                    double cum31 = 0;
                    uint16_t *metac = (uint16_t*)(c + DETAILOFFSET + (i/32)*DETAILSKIPSHORT/4);
                    uint16_t maskj = 1 << (j/8);
                    metac = metac + (i % 32);
                    if(metac[0] & maskj) { cum00 = c[i*BLOCKCOL+j]; cum01 = c[i*BLOCKCOL+j+1]; }
                    if(metac[1] & maskj) { cum10 = c[i*BLOCKCOL+BLOCKCOL+j]; cum11 = c[i*BLOCKCOL+BLOCKCOL+j+1]; }
                    if(metac[2] & maskj) { cum20 = c[i*BLOCKCOL+BLOCKCOL*2+j]; cum21 = c[i*BLOCKCOL+BLOCKCOL*2+j+1]; }
                    if(metac[3] & maskj) { cum30 = c[i*BLOCKCOL+BLOCKCOL*3+j]; cum31 = c[i*BLOCKCOL+BLOCKCOL*3+j+1]; }
                    for(int k=0;k<BLOCK64;k++){
                        sum00 += a[i*BLOCKCOL+k] * a[j*BLOCKCOL+k];
                        sum10 += a[i*BLOCKCOL+BLOCKCOL+k] * a[j*BLOCKCOL+k];
                        sum20 += a[i*BLOCKCOL+BLOCKCOL*2+k] * a[j*BLOCKCOL+k];
                        sum30 += a[i*BLOCKCOL+BLOCKCOL*3+k] * a[j*BLOCKCOL+k];
                        sum01 += a[i*BLOCKCOL+k] * a[j*BLOCKCOL+BLOCKCOL+k];
                        sum11 += a[i*BLOCKCOL+BLOCKCOL+k] * a[j*BLOCKCOL+BLOCKCOL+k];
                        sum21 += a[i*BLOCKCOL+BLOCKCOL*2+k] * a[j*BLOCKCOL+BLOCKCOL+k];
                        sum31 += a[i*BLOCKCOL+BLOCKCOL*3+k] * a[j*BLOCKCOL+BLOCKCOL+k];
                    }

                    c[i*BLOCKCOL+j] = sum00 + cum00;
                    c[i*BLOCKCOL+BLOCKCOL+j] = sum10 + cum10;
                    c[i*BLOCKCOL+BLOCKCOL*2+j] = sum20 + cum20;
                    c[i*BLOCKCOL+BLOCKCOL*3+j] = sum30 + cum30;
                    c[i*BLOCKCOL+j+1] = sum01 + cum01;
                    c[i*BLOCKCOL+BLOCKCOL+j+1] = sum11 + cum11;
                    c[i*BLOCKCOL+BLOCKCOL*2+j+1] = sum21 + cum21;
                    c[i*BLOCKCOL+BLOCKCOL*3+j+1] = sum31 + cum31;
                }
            }
            updateMeta(c, BLOCK64);
        }
/* */
/*
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
//
        void MatrixStdDouble::mat_mult(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {
            for(int k=0; k<BLOCK64; k+=8){
                for(int j=0; j<BLOCK64; j+=8){
                    uint16_t *metab = (uint16_t*)(b + DETAILOFFSET + (j/32)*DETAILSKIPSHORT/4);
                    __m512d rbkp11, rbkp12, rbkp13, rbkp14, rbkp15, rbkp16, rbkp17, rbkp10; 
                    uint16_t bmask = 0;
                    for(int m=0;m<8;m++){
                        bmask = bmask | metab[m+(j%32)];
                    }
                    if((bmask & (1<<(k/8))) == 0) continue; 
                    rbkp10 = _mm512_xor_pd(rbkp10,rbkp10);
                    rbkp11 = _mm512_xor_pd(rbkp11,rbkp11);
                    rbkp12 = _mm512_xor_pd(rbkp12,rbkp12);
                    rbkp13 = _mm512_xor_pd(rbkp13,rbkp13);
                    rbkp14 = _mm512_xor_pd(rbkp14,rbkp14);
                    rbkp15 = _mm512_xor_pd(rbkp15,rbkp15);
                    rbkp16 = _mm512_xor_pd(rbkp16,rbkp16);
                    rbkp17 = _mm512_xor_pd(rbkp17,rbkp17);
                    {
                        __m512i addb0;
                        {
                            __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                            __m512i offset512 = _mm512_set_epi64(ROWSKIP448*8,ROWSKIP384*8,ROWSKIP320*8,ROWSKIP256*8,ROWSKIP192*8,ROWSKIP128*8,ROWSKIP64*8,0);
                            __m512i sel512 = _mm512_set1_epi64(1u<<(k/8));
                            __m512i one512 = _mm512_set1_epi64(1);
                            __m512i bbjk512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(b+j*BLOCKCOL+k));
                            bbjk512 = _mm512_add_epi64(bbjk512,offset512);
                            __m512i metab512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metab+j%32)));
                            __m512i mask512 = _mm512_sub_epi64(_mm512_and_epi64(metab512,sel512),one512);
                            mask512 = _mm512_srai_epi64(mask512,63);
                            addb0 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,bbjk512));
                        }
                        rbkp10 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,0), 0)));
                        rbkp11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,0), 1)));
                        rbkp12 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,0), 2)));
                        rbkp13 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,0), 3)));
                        rbkp14 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,1), 0)));
                        rbkp15 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,1), 1)));
                        rbkp16 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,1), 2)));
                        rbkp17 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_extracti64x4_epi64(addb0,1), 3)));

                            _mm_prefetch(c + j, _MM_HINT_ET1 );
                            _mm_prefetch(c + BLOCKCOL + j, _MM_HINT_ET1 );
                            _mm_prefetch(c + 2 * BLOCKCOL + j, _MM_HINT_ET1 );
                            _mm_prefetch(c + 3 * BLOCKCOL + j, _MM_HINT_ET1 );

                        __m512i d1 = _mm512_set_epi64(11,10,9,8,3,2,1,0);
                        __m512i d2 = _mm512_set_epi64(15,14,13,12,7,6,5,4);

                        __m512d r21 = _mm512_permutex2var_pd(rbkp10, d1, rbkp14);
                        __m512d r25 = _mm512_permutex2var_pd(rbkp10, d2, rbkp14);
                        __m512d r22 = _mm512_permutex2var_pd(rbkp11, d1, rbkp15);
                        __m512d r26 = _mm512_permutex2var_pd(rbkp11, d2, rbkp15);
                        __m512d r23 = _mm512_permutex2var_pd(rbkp12, d1, rbkp16);
                        __m512d r27 = _mm512_permutex2var_pd(rbkp12, d2, rbkp16);
                        __m512d r24 = _mm512_permutex2var_pd(rbkp13, d1, rbkp17);
                        __m512d r28 = _mm512_permutex2var_pd(rbkp13, d2, rbkp17);

                        __m512i d3 = _mm512_set_epi64(13,12,5,4,9,8,1,0);
                        __m512i d4 = _mm512_set_epi64(15,14,7,6,11,10,3,2);

                        __m512d r31 = _mm512_permutex2var_pd(r21, d3, r23);
                        __m512d r33 = _mm512_permutex2var_pd(r21, d4, r23);
                        __m512d r32 = _mm512_permutex2var_pd(r22, d3, r24);
                        __m512d r34 = _mm512_permutex2var_pd(r22, d4, r24);
                        __m512d r35 = _mm512_permutex2var_pd(r25, d3, r27);
                        __m512d r37 = _mm512_permutex2var_pd(r25, d4, r27);
                        __m512d r36 = _mm512_permutex2var_pd(r26, d3, r28);
                        __m512d r38 = _mm512_permutex2var_pd(r26, d4, r28);

                        d1 = _mm512_set_epi64(14,6,12,4,10,2,8,0);
                        d2 = _mm512_set_epi64(15,7,13,5,11,3,9,1);

                        rbkp10 = _mm512_permutex2var_pd(r31, d1, r32);
                        rbkp11 = _mm512_permutex2var_pd(r31, d2, r32);
                        rbkp12 = _mm512_permutex2var_pd(r33, d1, r34);
                        rbkp13 = _mm512_permutex2var_pd(r33, d2, r34);
                        rbkp14 = _mm512_permutex2var_pd(r35, d1, r36);
                        rbkp15 = _mm512_permutex2var_pd(r35, d2, r36);
                        rbkp16 = _mm512_permutex2var_pd(r37, d1, r38);
                        rbkp17 = _mm512_permutex2var_pd(r37, d2, r38);

                    }
                    {
                        __m512i addb0;
                        {
                            __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                            __m512i offset512 = _mm512_set_epi64(ROWSKIP448*8,ROWSKIP384*8,ROWSKIP320*8,ROWSKIP256*8,ROWSKIP192*8,ROWSKIP128*8,ROWSKIP64*8,0);
                            __m512i sel512 = _mm512_set1_epi64(1u<<(k/8));
                            __m512i one512 = _mm512_set1_epi64(1);
                            __m512i bbjk512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(b+((j+8)%64)*BLOCKCOL+k));
                            bbjk512 = _mm512_add_epi64(bbjk512,offset512);
                            __m512i metab512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metab+((j+8)%64)%32)));
                            __m512i mask512 = _mm512_sub_epi64(_mm512_and_epi64(metab512,sel512),one512);
                            mask512 = _mm512_srai_epi64(mask512,63);
                            addb0 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,bbjk512));
                        }
                        __m256i bddnext = _mm512_extracti64x4_epi64(addb0,0);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(bddnext,0)), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(bddnext,1)), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(bddnext,2)), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(bddnext,3)), _MM_HINT_T1);
                        __m256i eddnext = _mm512_extracti64x4_epi64(addb0,1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(eddnext,0)), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(eddnext,1)), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(eddnext,2)), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(eddnext,3)), _MM_HINT_T1);
                   }
                
                __m256i addnext;
                        {
                            uint16_t *metaa = (uint16_t*)(a + DETAILOFFSET);
                            __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                            __m512i offset512 = _mm512_set_epi64(ROWSKIP448*8,ROWSKIP384*8,ROWSKIP320*8,ROWSKIP256*8,ROWSKIP192*8,ROWSKIP128*8,ROWSKIP64*8,0);
                            __m512i sel512 = _mm512_set1_epi64(1u<<(k/8));
                            __m512i one512 = _mm512_set1_epi64(1);
                            __m512i aaik512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(a+k));
                            aaik512 = _mm512_add_epi64(aaik512,offset512);
                            __m512i metaa512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metaa)));
                            __m512i mask512 = _mm512_sub_epi64(_mm512_and_epi64(metaa512,sel512),one512);
                            mask512 = _mm512_srai_epi64(mask512,63);
                            __m512i adda512 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,aaik512));
                            addnext = _mm512_extracti64x4_epi64(adda512,0);
                        }  

                int ii = 0;
                __m256i adda0=addnext;
                for(int i=0; i<BLOCK64; i=ii,adda0 = addnext){  // i+=4){
                    __m512d rc0, rc1, rc2, rc3;
                    rc0 = _mm512_xor_pd(rc0,rc0);
                    rc1 = _mm512_xor_pd(rc1,rc1);
                    rc2 = _mm512_xor_pd(rc2,rc2);
                    rc3 = _mm512_xor_pd(rc3,rc3);
                    uint16_t *metac = (uint16_t*)(c + DETAILOFFSET + (i/32)*DETAILSKIPSHORT/4);
                    uint16_t maskj = 1 << (j/8);
                    metac = metac + (i % 32);
                    if(metac[0] & maskj)
                    rc0 = _mm512_load_pd( c + i * BLOCKCOL + j ); 
                    if(metac[1] & maskj)
                    rc1 = _mm512_load_pd( c + (i+1) * BLOCKCOL + j ); 
                    if(metac[2] & maskj)
                    rc2 = _mm512_load_pd( c + (i+2) * BLOCKCOL + j ); 
                    if(metac[3] & maskj)
                    rc3 = _mm512_load_pd( c + (i+3) * BLOCKCOL + j ); 

                    metac[0] = metac[0] | maskj;
                    metac[1] = metac[1] | maskj;
                    metac[2] = metac[2] | maskj;
                    metac[3] = metac[3] | maskj;
                    uint16_t amask = 0;
                        {
                        do{
                            uint16_t *metaa = (uint16_t*)(a + DETAILOFFSET + ((ii+4)/32)*DETAILSKIPSHORT/4);
                            _mm_prefetch(metaa + BLOCKCOL *4, _MM_HINT_T1); 
                            __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                            __m512i offset512 = _mm512_set_epi64(ROWSKIP448*8,ROWSKIP384*8,ROWSKIP320*8,ROWSKIP256*8,ROWSKIP192*8,ROWSKIP128*8,ROWSKIP64*8,0);
                            __m512i sel512 = _mm512_set1_epi64(1u<<(k/8));
                            __m512i one512 = _mm512_set1_epi64(1);
                            __m512i aaik512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(a+((ii+4)%64)*BLOCKCOL+k));
                            aaik512 = _mm512_add_epi64(aaik512,offset512);
                            __m512i metaa512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metaa+(ii+4)%32)));
                            __m512i mask512 = _mm512_sub_epi64(_mm512_and_epi64(metaa512,sel512),one512);
                            mask512 = _mm512_srai_epi64(mask512,63);
                            __m512i adda512 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,aaik512));
                            addnext = _mm512_extracti64x4_epi64(adda512,0);
                            amask = 0;
                            for(int m=0;m<4;m++){
                                amask = amask | metaa[m+((ii+4)%32)];
                            }
                            amask = amask & (1u<<(k/8));
                            ii = ii + 4;
                        }while(amask==0 && ii<BLOCK64);
               //             _mm_prefetch(c + ((ii+4)&63) * BLOCKCOL + j, _MM_HINT_ET1 );
               //             _mm_prefetch(c + ((ii+5)&63) * BLOCKCOL + j, _MM_HINT_ET1 );
               //             _mm_prefetch(c + ((ii+6)&63) * BLOCKCOL + j, _MM_HINT_ET1 );
               //             _mm_prefetch(c + ((ii+7)&63) * BLOCKCOL + j, _MM_HINT_ET1 );
                        }

                    __m512d aaik20 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(adda0, 0))); //_mm512_extracti64x4_epi64(adda0,0), 0)));
                    __m512d aaik21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(adda0, 1))); //_mm512_extracti64x4_epi64(adda0,0), 1)));
                    __m512d aaik22 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(adda0, 2))); //_mm512_extracti64x4_epi64(adda0,0), 2)));
                    __m512d aaik23 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(adda0, 3))); //_mm512_extracti64x4_epi64(adda0,0), 3)));
                        {
                         //   __m256i add256 = _mm512_extracti64x4_epi64(addnext,0);
                            _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(addnext,0)), _MM_HINT_T1); // _mm512_extracti64x4_epi64(addnext,0), 0)), _MM_HINT_T1);
                            _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(addnext,1)), _MM_HINT_T1); // _mm512_extracti64x4_epi64(addnext,0), 1)), _MM_HINT_T1);
                            _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(addnext,2)), _MM_HINT_T1); // _mm512_extracti64x4_epi64(addnext,0), 2)), _MM_HINT_T1);
                            _mm_prefetch(reinterpret_cast<void *>(_mm256_extract_epi64(addnext,3)), _MM_HINT_T1); // _mm512_extracti64x4_epi64(addnext,0), 3)), _MM_HINT_T1);
                        }
                    {
                        __m128d bottom10 = _mm512_extractf64x2_pd(aaik20,0);
                        __m128d bottom11 = _mm512_extractf64x2_pd(aaik21,0);
                        __m128d bottom12 = _mm512_extractf64x2_pd(aaik22,0);
                        __m128d bottom13 = _mm512_extractf64x2_pd(aaik23,0);

                        __m512d aaik10 = _mm512_broadcastsd_pd(bottom10);
                        __m512d aaik11 = _mm512_broadcastsd_pd(bottom11);
                        __m512d aaik12 = _mm512_broadcastsd_pd(bottom12);
                        __m512d aaik13 = _mm512_broadcastsd_pd(bottom13);

                        rc0 = _mm512_fmadd_pd (aaik10, rbkp10, rc0);
                        rc1 = _mm512_fmadd_pd (aaik11, rbkp10, rc1);
                        rc2 = _mm512_fmadd_pd (aaik12, rbkp10, rc2);
                        rc3 = _mm512_fmadd_pd (aaik13, rbkp10, rc3);

                        bottom10 = _mm_permute_pd(bottom10,1);
                        bottom11 = _mm_permute_pd(bottom11,1);
                        bottom12 = _mm_permute_pd(bottom12,1);
                        bottom13 = _mm_permute_pd(bottom13,1);

                        aaik10 = _mm512_broadcastsd_pd(bottom10);
                        aaik11 = _mm512_broadcastsd_pd(bottom11);
                        aaik12 = _mm512_broadcastsd_pd(bottom12);
                        aaik13 = _mm512_broadcastsd_pd(bottom13);

                        rc0 = _mm512_fmadd_pd (aaik10, rbkp11, rc0);
                        rc1 = _mm512_fmadd_pd (aaik11, rbkp11, rc1);
                        rc2 = _mm512_fmadd_pd (aaik12, rbkp11, rc2);
                        rc3 = _mm512_fmadd_pd (aaik13, rbkp11, rc3);
                    }
                    {
                        __m128d bottom16 = _mm512_extractf64x2_pd(aaik20,1);
                        __m128d bottom17 = _mm512_extractf64x2_pd(aaik21,1);
                        __m128d bottom14 = _mm512_extractf64x2_pd(aaik22,1);
                        __m128d bottom15 = _mm512_extractf64x2_pd(aaik23,1);
                        __m512d aaik16 = _mm512_broadcastsd_pd(bottom16);
                        __m512d aaik17 = _mm512_broadcastsd_pd(bottom17);
                        __m512d aaik14 = _mm512_broadcastsd_pd(bottom14);
                        __m512d aaik15 = _mm512_broadcastsd_pd(bottom15);

                        rc0 = _mm512_fmadd_pd (aaik16, rbkp12, rc0);
                        rc1 = _mm512_fmadd_pd (aaik17, rbkp12, rc1);
                        rc2 = _mm512_fmadd_pd (aaik14, rbkp12, rc2);
                        rc3 = _mm512_fmadd_pd (aaik15, rbkp12, rc3);
                        bottom16 = _mm_permute_pd(bottom16,1);
                        bottom17 = _mm_permute_pd(bottom17,1);
                        bottom14 = _mm_permute_pd(bottom14,1);
                        bottom15 = _mm_permute_pd(bottom15,1);
                        aaik16 = _mm512_broadcastsd_pd(bottom16);
                        aaik17 = _mm512_broadcastsd_pd(bottom17);
                        aaik14 = _mm512_broadcastsd_pd(bottom14);
                        aaik15 = _mm512_broadcastsd_pd(bottom15);
                        rc0 = _mm512_fmadd_pd (aaik16, rbkp13, rc0);
                        rc1 = _mm512_fmadd_pd (aaik17, rbkp13, rc1);
                        rc2 = _mm512_fmadd_pd (aaik14, rbkp13, rc2);
                        rc3 = _mm512_fmadd_pd (aaik15, rbkp13, rc3);
                    }
                    {
                        __m128d bottom18 = _mm512_extractf64x2_pd(aaik20,2);
                        __m128d bottom19 = _mm512_extractf64x2_pd(aaik21,2);
                        __m128d bottom20 = _mm512_extractf64x2_pd(aaik22,2);
                        __m128d bottom21 = _mm512_extractf64x2_pd(aaik23,2);
                        __m512d aaik18 = _mm512_broadcastsd_pd(bottom18);
                        __m512d aaik19 = _mm512_broadcastsd_pd(bottom19);
                        __m512d aaik20 = _mm512_broadcastsd_pd(bottom20);
                        __m512d aaik21 = _mm512_broadcastsd_pd(bottom21);
                        rc0 = _mm512_fmadd_pd (aaik18, rbkp14, rc0);
                        rc1 = _mm512_fmadd_pd (aaik19, rbkp14, rc1);
                        rc2 = _mm512_fmadd_pd (aaik20, rbkp14, rc2);
                        rc3 = _mm512_fmadd_pd (aaik21, rbkp14, rc3);
                        bottom18 = _mm_permute_pd(bottom18,1);
                        bottom19 = _mm_permute_pd(bottom19,1);
                        bottom20 = _mm_permute_pd(bottom20,1);
                        bottom21 = _mm_permute_pd(bottom21,1);
                        aaik18 = _mm512_broadcastsd_pd(bottom18);
                        aaik19 = _mm512_broadcastsd_pd(bottom19);
                        aaik20 = _mm512_broadcastsd_pd(bottom20);
                        aaik21 = _mm512_broadcastsd_pd(bottom21);
                        rc0 = _mm512_fmadd_pd (aaik18, rbkp15, rc0);
                        rc1 = _mm512_fmadd_pd (aaik19, rbkp15, rc1);
                        rc2 = _mm512_fmadd_pd (aaik20, rbkp15, rc2);
                        rc3 = _mm512_fmadd_pd (aaik21, rbkp15, rc3);
                    }
                    {  
                        __m128d bottom40 = _mm512_extractf64x2_pd(aaik20,3);
                        __m128d bottom41 = _mm512_extractf64x2_pd(aaik21,3);
                        __m128d bottom30 = _mm512_extractf64x2_pd(aaik22,3);
                        __m128d bottom31 = _mm512_extractf64x2_pd(aaik23,3);
                        __m512d aaik40 = _mm512_broadcastsd_pd(bottom40);
                        __m512d aaik41 = _mm512_broadcastsd_pd(bottom41);
                        __m512d aaik30 = _mm512_broadcastsd_pd(bottom30);
                        __m512d aaik31 = _mm512_broadcastsd_pd(bottom31);
                        rc0 = _mm512_fmadd_pd (aaik40, rbkp16, rc0);
                        rc1 = _mm512_fmadd_pd (aaik41, rbkp16, rc1);
                        rc2 = _mm512_fmadd_pd (aaik30, rbkp16, rc2);
                        rc3 = _mm512_fmadd_pd (aaik31, rbkp16, rc3);
                        bottom40 = _mm_permute_pd(bottom40,1);
                        bottom41 = _mm_permute_pd(bottom41,1);
                        aaik40 = _mm512_broadcastsd_pd(bottom40);
                        aaik41 = _mm512_broadcastsd_pd(bottom41);
                        rc0 = _mm512_fmadd_pd (aaik40, rbkp17, rc0);
                        rc1 = _mm512_fmadd_pd (aaik41, rbkp17, rc1);

                        bottom30 = _mm_permute_pd(bottom30,1);
                        bottom31 = _mm_permute_pd(bottom31,1);
                        aaik30 = _mm512_broadcastsd_pd(bottom30);
                        aaik31 = _mm512_broadcastsd_pd(bottom31);
                        rc2 = _mm512_fmadd_pd (aaik30, rbkp17, rc2);
                        rc3 = _mm512_fmadd_pd (aaik31, rbkp17, rc3);
                    }  //
                    _mm512_store_pd((void*)( c + i * BLOCKCOL + j ),rc0); 
                    _mm512_store_pd((void*)( c + (i+1) * BLOCKCOL + j ),rc1); 
                    _mm512_store_pd((void*)( c + (i+2) * BLOCKCOL + j ),rc2); 
                    _mm512_store_pd((void*)( c + (i+3) * BLOCKCOL + j ),rc3); 
                }
                }
            }
            updateMeta(c, BLOCK64);
        }
/* */
//
        void MatrixStdDouble::mat_mult4(double *__restrict ab[], double *__restrict bb[], double *__restrict c, int blocks, unsigned long msk, int coreIdx) //64
    //    void mat_mult_n(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {          // 3.28 sec
            for(int i=0;i<BLOCK64;i+=STEPROW){
                for(int j=0;j<BLOCK64;j+=STEPCOL){
                    int iBLOCKCOL = i*BLOCKCOL;
                    int jBLOCKCOL = j*BLOCKCOL;
                    int BLOCKCOL2 = BLOCKCOL*2;
                    int BLOCKCOL3 = BLOCKCOL*3;
                    
                    __m512i adda0, adda1, adda2, adda3, addb0, addb1, addb2, addb3;
                    
                    //
                    _mm_prefetch(c+i*BLOCKCOL+j,_MM_HINT_ET1); 
                    _mm_prefetch(c+i*BLOCKCOL+BLOCKCOL+j,_MM_HINT_ET1); 
                    _mm_prefetch(c+i*BLOCKCOL+BLOCKCOL*2+j,_MM_HINT_ET1); 
                    _mm_prefetch(c+i*BLOCKCOL+BLOCKCOL*3+j,_MM_HINT_ET1); 
                    // //

                    __m512d s00, s10, s20, s30, s01, s11, s21, s31;
                    __m512d s02, s12, s22, s32, s03, s13, s23, s33;
                    s00 = _mm512_xor_pd(s00,s00);
                    s10 = _mm512_xor_pd(s10,s10);
                    s20 = _mm512_xor_pd(s20,s20);
                    s30 = _mm512_xor_pd(s30,s30);
                    s01 = _mm512_xor_pd(s01,s01);
                    s11 = _mm512_xor_pd(s11,s11);
                    s21 = _mm512_xor_pd(s21,s21);
                    s31 = _mm512_xor_pd(s31,s31);

                    s02 = _mm512_xor_pd(s02,s02);
                    s12 = _mm512_xor_pd(s12,s12);
                    s22 = _mm512_xor_pd(s22,s22);
                    s32 = _mm512_xor_pd(s32,s32);
                    s03 = _mm512_xor_pd(s03,s03);
                    s13 = _mm512_xor_pd(s13,s13);
                    s23 = _mm512_xor_pd(s23,s23);
                    s33 = _mm512_xor_pd(s33,s33);
                    double cum00 = 0;
                    double cum10 = 0;
                    double cum20 = 0;
                    double cum30 = 0;
                    double cum01 = 0;
                    double cum11 = 0;
                    double cum21 = 0;
                    double cum31 = 0;
                    double cum02 = 0;
                    double cum12 = 0;
                    double cum22 = 0;
                    double cum32 = 0;
                    double cum03 = 0;
                    double cum13 = 0;
                    double cum23 = 0;
                    double cum33 = 0;
                    uint16_t *metac = (uint16_t*)(c + DETAILOFFSET + (i/32)*DETAILSKIPSHORT/4);
                    uint16_t maskj = 1 << (j/8);
                    metac = metac + (i % 32);
                    if(metac[0] & maskj) { 
                                            cum00 = c[i*BLOCKCOL+j]; cum01 = c[i*BLOCKCOL+j+1]; 
                                            cum02 = c[i*BLOCKCOL+j+2]; cum03 = c[i*BLOCKCOL+j+3]; }
                    if(metac[1] & maskj) { 
                                            cum10 = c[i*BLOCKCOL+BLOCKCOL+j]; cum11 = c[i*BLOCKCOL+BLOCKCOL+j+1]; 
                                            cum12 = c[i*BLOCKCOL+BLOCKCOL+j+2]; cum13 = c[i*BLOCKCOL+BLOCKCOL+j+3]; }
                    if(metac[2] & maskj) { 
                                            cum20 = c[i*BLOCKCOL+BLOCKCOL*2+j]; cum21 = c[i*BLOCKCOL+BLOCKCOL*2+j+1]; 
                                            cum22 = c[i*BLOCKCOL+BLOCKCOL*2+j+2]; cum23 = c[i*BLOCKCOL+BLOCKCOL*2+j+3]; }
                    if(metac[3] & maskj) { 
                                            cum30 = c[i*BLOCKCOL+BLOCKCOL*3+j]; cum31 = c[i*BLOCKCOL+BLOCKCOL*3+j+1]; 
                                            cum32 = c[i*BLOCKCOL+BLOCKCOL*3+j+2]; cum33 = c[i*BLOCKCOL+BLOCKCOL*3+j+3]; }
                for(int blockidx=0; blockidx<blocks; blockidx++){
                        double* a = ab[blockidx];
                        double* b = bb[blockidx];
                        uint16_t *metaa = (uint16_t*)(a + DETAILOFFSET + (i/32)*DETAILSKIPSHORT/4);
                        uint16_t *metab = (uint16_t*)(b + DETAILOFFSET + (j/32)*DETAILSKIPSHORT/4);
                    uint16_t* metaai32 = metaa+i%32;
                    uint16_t* metabj32 = metab+j%32;

                        uint16_t abit = (*(metaai32)) | (*(metaai32+1)) | (*(metaai32+2)) | (*(metaai32+3));
                        uint16_t bbit = (*(metabj32)) | (*(metabj32+1)) | (*(metabj32+2)) | (*(metabj32+3));

                        if((abit & bbit & METAMASK) == 0) {
                            continue;
                        }
                        {
                        __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                        __m512i offset512 = _mm512_set_epi64(448,384,320,256,192,128,64,0);
                        __m512i sel512 = _mm512_set_epi64(128,64,32,16,8,4,2,1);
                        __m512i one512 = _mm512_set1_epi64(1);
 
                        {
                            __m512i aaik5120 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(a+iBLOCKCOL));
                            __m512i aaik5121 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(a+iBLOCKCOL+BLOCKCOL));
                            aaik5120 = _mm512_add_epi64(aaik5120,offset512);
                            aaik5121 = _mm512_add_epi64(aaik5121,offset512);
                            __m512i mask5120 = _mm512_set1_epi64(*(metaai32));
                            __m512i metaa5121 = _mm512_set1_epi64(*(metaai32+1));
                            mask5120 = _mm512_sub_epi64(_mm512_and_epi64(mask5120,sel512),one512);
                            __m512i mask5121 = _mm512_sub_epi64(_mm512_and_epi64(metaa5121,sel512),one512);
                            mask5120 = _mm512_srai_epi64(mask5120,63);
                            mask5121 = _mm512_srai_epi64(mask5121,63);
                            __m512i aaik5122 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(a+iBLOCKCOL+BLOCKCOL*2));
                            __m512i aaik5123 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(a+iBLOCKCOL+BLOCKCOL*3));

                            adda0 = _mm512_or_epi64(_mm512_and_epi64(mask5120,blanck512),_mm512_andnot_epi64(mask5120,aaik5120));
                            adda1 = _mm512_or_epi64(_mm512_and_epi64(mask5121,blanck512),_mm512_andnot_epi64(mask5121,aaik5121));
                            aaik5122 = _mm512_add_epi64(aaik5122,offset512);
                            aaik5123 = _mm512_add_epi64(aaik5123,offset512);
                            __m512i metaa5122 = _mm512_set1_epi64(*(metaai32+2));
                            __m512i metaa5123 = _mm512_set1_epi64(*(metaai32+3));
                            __m512i mask5122 = _mm512_sub_epi64(_mm512_and_epi64(metaa5122,sel512),one512);
                            __m512i mask5123 = _mm512_sub_epi64(_mm512_and_epi64(metaa5123,sel512),one512);
                            mask5122 = _mm512_srai_epi64(mask5122,63);
                            mask5123 = _mm512_srai_epi64(mask5123,63);
                            __m512i bbjk5120 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(b+jBLOCKCOL));
                            __m512i bbjk5121 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(b+jBLOCKCOL+BLOCKCOL));
                            adda2 = _mm512_or_epi64(_mm512_and_epi64(mask5122,blanck512),_mm512_andnot_epi64(mask5122,aaik5122));
                            adda3 = _mm512_or_epi64(_mm512_and_epi64(mask5123,blanck512),_mm512_andnot_epi64(mask5123,aaik5123));
                            bbjk5120 = _mm512_add_epi64(bbjk5120,offset512);
                            bbjk5121 = _mm512_add_epi64(bbjk5121,offset512);
                            __m512i metab5120 = _mm512_set1_epi64(*(metabj32));
                            __m512i metab5121 = _mm512_set1_epi64(*(metabj32+1));
                            __m512i mbsk5120 = _mm512_sub_epi64(_mm512_and_epi64(metab5120,sel512),one512);
                            __m512i mbsk5121 = _mm512_sub_epi64(_mm512_and_epi64(metab5121,sel512),one512);
                            mbsk5120 = _mm512_srai_epi64(mbsk5120,63);
                            mbsk5121 = _mm512_srai_epi64(mbsk5121,63);
                            __m512i bbjk5122 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(b+jBLOCKCOL+BLOCKCOL2));
                            addb0 = _mm512_or_epi64(_mm512_and_epi64(mbsk5120,blanck512),_mm512_andnot_epi64(mbsk5120,bbjk5120));
                            addb1 = _mm512_or_epi64(_mm512_and_epi64(mbsk5121,blanck512),_mm512_andnot_epi64(mbsk5121,bbjk5121));
                            __m512i bbjk5123 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(b+jBLOCKCOL+BLOCKCOL3));

                            bbjk5122 = _mm512_add_epi64(bbjk5122,offset512);
                            __m512i metab5122 = _mm512_set1_epi64(*(metabj32+2));
                            __m512i mbsk5122 = _mm512_sub_epi64(_mm512_and_epi64(metab5122,sel512),one512);
                            mbsk5122 = _mm512_srai_epi64(mbsk5122,63);
                            addb2 = _mm512_or_epi64(_mm512_and_epi64(mbsk5122,blanck512),_mm512_andnot_epi64(mbsk5122,bbjk5122));
                            bbjk5123 = _mm512_add_epi64(bbjk5123,offset512);
                            __m512i metab5123 = _mm512_set1_epi64(*(metabj32+3));
                            __m512i mbsk5123 = _mm512_sub_epi64(_mm512_and_epi64(metab5123,sel512),one512);
                            mbsk5123 = _mm512_srai_epi64(mbsk5123,63);
                            addb3 = _mm512_or_epi64(_mm512_and_epi64(mbsk5123,blanck512),_mm512_andnot_epi64(mbsk5123,bbjk5123));
                        }
   
                        }


                            
                    {
                        {
                            __m512d bx0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0),0))); 
                            __m512d bx1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1),0)));
                            __m512d ax0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0),0)));
                            __m512d ax1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1),0)));
                            double *ai4 = a+((i+4)&63)*BLOCKCOL;
                            double *ai5 = a+((i+5)&63)*BLOCKCOL;
                            double *ai6 = a+((i+6)&63)*BLOCKCOL;
                            double *ai7 = a+((i+7)&63)*BLOCKCOL;
                            __m512d ax2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2),0)));
                            __m512d ax3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3),0)));
                            __m512d bx2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2),0)));
                            __m512d bx3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3),0)));
                            double *bj4 = b+((j+4)&63)*BLOCKCOL;
                            double *bj5 = b+((j+5)&63)*BLOCKCOL;
                            double *bj6 = b+((j+6)&63)*BLOCKCOL;
                            double *bj7 = b+((j+7)&63)*BLOCKCOL;
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax0, bx0, s00);
                            s10 = _mm512_fmadd_pd (ax1, bx0, s10);
                            s01 = _mm512_fmadd_pd (ax0, bx1, s01);
                            s11 = _mm512_fmadd_pd (ax1, bx1, s11);
                            s20 = _mm512_fmadd_pd (ax2, bx0, s20);
                            s30 = _mm512_fmadd_pd (ax3, bx0, s30);
                            s21 = _mm512_fmadd_pd (ax2, bx1, s21);
                            s31 = _mm512_fmadd_pd (ax3, bx1, s31);

                            s02 = _mm512_fmadd_pd (ax0, bx2, s02);
                            s12 = _mm512_fmadd_pd (ax1, bx2, s12);
                            s03 = _mm512_fmadd_pd (ax0, bx3, s03);
                            s13 = _mm512_fmadd_pd (ax1, bx3, s13);
                            s22 = _mm512_fmadd_pd (ax2, bx2, s22);
                            s32 = _mm512_fmadd_pd (ax3, bx2, s32);
                            s23 = _mm512_fmadd_pd (ax2, bx3, s23);
                            s33 = _mm512_fmadd_pd (ax3, bx3, s33);
                            __m512d bx01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0),1))); 
                            __m512d bx11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1),1))); 
                            __m512d ax01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0),1))); 
                            __m512d ax11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1),1))); 
                            __m512d ax21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2),1))); 
                            __m512d ax31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3),1))); 
                            __m512d bx21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2),1))); 
                            __m512d bx31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3),1))); 
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax01, bx01, s00);
                            s10 = _mm512_fmadd_pd (ax11, bx01, s10);
                            s01 = _mm512_fmadd_pd (ax01, bx11, s01);
                            s11 = _mm512_fmadd_pd (ax11, bx11, s11);
                            s20 = _mm512_fmadd_pd (ax21, bx01, s20);
                            s30 = _mm512_fmadd_pd (ax31, bx01, s30);
                            s21 = _mm512_fmadd_pd (ax21, bx11, s21);
                            s31 = _mm512_fmadd_pd (ax31, bx11, s31);

                            s02 = _mm512_fmadd_pd (ax01, bx21, s02);
                            s12 = _mm512_fmadd_pd (ax11, bx21, s12);
                            s03 = _mm512_fmadd_pd (ax01, bx31, s03);
                            s13 = _mm512_fmadd_pd (ax11, bx31, s13);
                            s22 = _mm512_fmadd_pd (ax21, bx21, s22);
                            s32 = _mm512_fmadd_pd (ax31, bx21, s32);
                            s23 = _mm512_fmadd_pd (ax21, bx31, s23);
                            s33 = _mm512_fmadd_pd (ax31, bx31, s33);
                            bx0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0),2))); 
                            bx1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1),2))); 
                            ax0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0),2))); 
                            ax1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1),2))); 
                            ax2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2),2))); 
                            ax3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3),2))); 
                            bx2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2),2))); 
                            bx3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3),2))); 
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax0, bx0, s00);
                            s10 = _mm512_fmadd_pd (ax1, bx0, s10);
                            s01 = _mm512_fmadd_pd (ax0, bx1, s01);
                            s11 = _mm512_fmadd_pd (ax1, bx1, s11);
                            s20 = _mm512_fmadd_pd (ax2, bx0, s20);
                            s30 = _mm512_fmadd_pd (ax3, bx0, s30);
                            s21 = _mm512_fmadd_pd (ax2, bx1, s21);
                            s31 = _mm512_fmadd_pd (ax3, bx1, s31);

                            s02 = _mm512_fmadd_pd (ax0, bx2, s02);
                            s12 = _mm512_fmadd_pd (ax1, bx2, s12);
                            s03 = _mm512_fmadd_pd (ax0, bx3, s03);
                            s13 = _mm512_fmadd_pd (ax1, bx3, s13);
                            s22 = _mm512_fmadd_pd (ax2, bx2, s22);
                            s32 = _mm512_fmadd_pd (ax3, bx2, s32);
                            s23 = _mm512_fmadd_pd (ax2, bx3, s23);
                            s33 = _mm512_fmadd_pd (ax3, bx3, s33);
                            bx01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0),3))); 
                            bx11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1),3))); 
                            ax01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0),3))); 
                            ax11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1),3))); 
                            ax21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2),3))); 
                            ax31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3),3))); 
                            bx21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2),3))); 
                            bx31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3),3))); 
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax01, bx01, s00);
                            s10 = _mm512_fmadd_pd (ax11, bx01, s10);
                            s01 = _mm512_fmadd_pd (ax01, bx11, s01);
                            s11 = _mm512_fmadd_pd (ax11, bx11, s11);
                            s20 = _mm512_fmadd_pd (ax21, bx01, s20);
                            s30 = _mm512_fmadd_pd (ax31, bx01, s30);
                            s21 = _mm512_fmadd_pd (ax21, bx11, s21);
                            s31 = _mm512_fmadd_pd (ax31, bx11, s31);

                        __m512i dp = _mm512_set_epi64(3,2,1,0,7,6,5,4);

                            s02 = _mm512_fmadd_pd (ax01, bx21, s02);
                            s12 = _mm512_fmadd_pd (ax11, bx21, s12);
                            s03 = _mm512_fmadd_pd (ax01, bx31, s03);
                            s13 = _mm512_fmadd_pd (ax11, bx31, s13);
                            s22 = _mm512_fmadd_pd (ax21, bx21, s22);
                            s32 = _mm512_fmadd_pd (ax31, bx21, s32);
                            s23 = _mm512_fmadd_pd (ax21, bx31, s23);
                            s33 = _mm512_fmadd_pd (ax31, bx31, s33);


                        addb0 = _mm512_permutexvar_epi64(dp, addb0);
                        addb1 = _mm512_permutexvar_epi64(dp, addb1);
                        adda0 = _mm512_permutexvar_epi64(dp, adda0);
                        adda1 = _mm512_permutexvar_epi64(dp, adda1);
                        adda2 = _mm512_permutexvar_epi64(dp, adda2);
                        adda3 = _mm512_permutexvar_epi64(dp, adda3);
                        addb2 = _mm512_permutexvar_epi64(dp, addb2);
                        addb3 = _mm512_permutexvar_epi64(dp, addb3);

                            bx0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0), 0)));
                            bx1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1), 0)));
                            ax0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0), 0)));
                            ax1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1), 0)));
                            ax2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2), 0)));
                            ax3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3), 0)));
                            bx2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2), 0)));
                            bx3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3), 0)));
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax0, bx0, s00);
                            s10 = _mm512_fmadd_pd (ax1, bx0, s10);
                            s01 = _mm512_fmadd_pd (ax0, bx1, s01);
                            s11 = _mm512_fmadd_pd (ax1, bx1, s11);
                            s20 = _mm512_fmadd_pd (ax2, bx0, s20);
                            s30 = _mm512_fmadd_pd (ax3, bx0, s30);
                            s21 = _mm512_fmadd_pd (ax2, bx1, s21);
                            s31 = _mm512_fmadd_pd (ax3, bx1, s31);

                            s02 = _mm512_fmadd_pd (ax0, bx2, s02);
                            s12 = _mm512_fmadd_pd (ax1, bx2, s12);
                            s03 = _mm512_fmadd_pd (ax0, bx3, s03);
                            s13 = _mm512_fmadd_pd (ax1, bx3, s13);
                            s22 = _mm512_fmadd_pd (ax2, bx2, s22);
                            s32 = _mm512_fmadd_pd (ax3, bx2, s32);
                            s23 = _mm512_fmadd_pd (ax2, bx3, s23);
                            s33 = _mm512_fmadd_pd (ax3, bx3, s33);
                            bx01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0), 1)));
                            bx11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1), 1)));
                            ax01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0), 1)));
                            ax11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1), 1)));
                            ax21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2), 1)));
                            ax31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3), 1)));
                            bx21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2), 1)));
                            bx31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3), 1)));
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax01, bx01, s00);
                            s10 = _mm512_fmadd_pd (ax11, bx01, s10);
                            s01 = _mm512_fmadd_pd (ax01, bx11, s01);
                            s11 = _mm512_fmadd_pd (ax11, bx11, s11);
                            s20 = _mm512_fmadd_pd (ax21, bx01, s20);
                            s30 = _mm512_fmadd_pd (ax31, bx01, s30);
                            s21 = _mm512_fmadd_pd (ax21, bx11, s21);
                            s31 = _mm512_fmadd_pd (ax31, bx11, s31);

                            s02 = _mm512_fmadd_pd (ax01, bx21, s02);
                            s12 = _mm512_fmadd_pd (ax11, bx21, s12);
                            s03 = _mm512_fmadd_pd (ax01, bx31, s03);
                            s13 = _mm512_fmadd_pd (ax11, bx31, s13);
                            s22 = _mm512_fmadd_pd (ax21, bx21, s22);
                            s32 = _mm512_fmadd_pd (ax31, bx21, s32);
                            s23 = _mm512_fmadd_pd (ax21, bx31, s23);
                            s33 = _mm512_fmadd_pd (ax31, bx31, s33);
                            bx0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0), 2)));
                            bx1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1), 2)));
                            ax0 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0), 2)));
                            ax1 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1), 2)));
                            ax2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2), 2)));
                            ax3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3), 2)));
                            bx2 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2), 2)));
                            bx3 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3), 2)));
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax0, bx0, s00);
                            s10 = _mm512_fmadd_pd (ax1, bx0, s10);
                            s01 = _mm512_fmadd_pd (ax0, bx1, s01);
                            s11 = _mm512_fmadd_pd (ax1, bx1, s11);
                            s20 = _mm512_fmadd_pd (ax2, bx0, s20);
                            s30 = _mm512_fmadd_pd (ax3, bx0, s30);
                            s21 = _mm512_fmadd_pd (ax2, bx1, s21);
                            s31 = _mm512_fmadd_pd (ax3, bx1, s31);

                            s02 = _mm512_fmadd_pd (ax0, bx2, s02);
                            s12 = _mm512_fmadd_pd (ax1, bx2, s12);
                            s03 = _mm512_fmadd_pd (ax0, bx3, s03);
                            s13 = _mm512_fmadd_pd (ax1, bx3, s13);
                            s22 = _mm512_fmadd_pd (ax2, bx2, s22);
                            s32 = _mm512_fmadd_pd (ax3, bx2, s32);
                            s23 = _mm512_fmadd_pd (ax2, bx3, s23);
                            s33 = _mm512_fmadd_pd (ax3, bx3, s33);
                            bx01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb0), 3)));
                            bx11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb1), 3)));
                            ax01 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda0), 3)));
                            ax11 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda1), 3)));
                            ax21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda2), 3)));
                            ax31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(adda3), 3)));
                            bx21 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb2), 3)));
                            bx31 = _mm512_load_pd(reinterpret_cast<void *>(_mm256_extract_epi64(_mm512_castsi512_si256(addb3), 3))); 
                            _mm_prefetch(ai4,_MM_HINT_T1); ai4 += 8;
                            _mm_prefetch(ai5,_MM_HINT_T1); ai5 += 8;
                            _mm_prefetch(ai6,_MM_HINT_T1); ai6 += 8;
                            _mm_prefetch(ai7,_MM_HINT_T1); ai7 += 8;
                            _mm_prefetch(bj4,_MM_HINT_T1); bj4 += 8;
                            _mm_prefetch(bj5,_MM_HINT_T1); bj5 += 8;
                            _mm_prefetch(bj6,_MM_HINT_T1); bj6 += 8;
                            _mm_prefetch(bj7,_MM_HINT_T1); bj7 += 8;
                            s00 = _mm512_fmadd_pd (ax01, bx01, s00);
                            s10 = _mm512_fmadd_pd (ax11, bx01, s10);
                            s01 = _mm512_fmadd_pd (ax01, bx11, s01);
                            s11 = _mm512_fmadd_pd (ax11, bx11, s11);
                            s20 = _mm512_fmadd_pd (ax21, bx01, s20);
                            s30 = _mm512_fmadd_pd (ax31, bx01, s30);
                            s21 = _mm512_fmadd_pd (ax21, bx11, s21);
                            s31 = _mm512_fmadd_pd (ax31, bx11, s31);

                            s02 = _mm512_fmadd_pd (ax01, bx21, s02);
                            s12 = _mm512_fmadd_pd (ax11, bx21, s12);
                            s03 = _mm512_fmadd_pd (ax01, bx31, s03);
                            s13 = _mm512_fmadd_pd (ax11, bx31, s13);
                            s22 = _mm512_fmadd_pd (ax21, bx21, s22);
                            s32 = _mm512_fmadd_pd (ax31, bx21, s32);
                            s23 = _mm512_fmadd_pd (ax21, bx31, s23);
                            s33 = _mm512_fmadd_pd (ax31, bx31, s33);
                        }
                   }
                }

                    c[iBLOCKCOL+j] = _mm512_reduce_add_pd(s00) + cum00;
                    c[iBLOCKCOL+BLOCKCOL+j] = _mm512_reduce_add_pd(s10) + cum10;
                    c[iBLOCKCOL+BLOCKCOL2+j] = _mm512_reduce_add_pd(s20) + cum20;
                    c[iBLOCKCOL+BLOCKCOL3+j] = _mm512_reduce_add_pd(s30) + cum30;
                    c[iBLOCKCOL+j+1] = _mm512_reduce_add_pd(s01) + cum01;
                    c[iBLOCKCOL+BLOCKCOL+j+1] = _mm512_reduce_add_pd(s11) + cum11;
                    c[iBLOCKCOL+BLOCKCOL2+j+1] = _mm512_reduce_add_pd(s21) + cum21;
                    c[iBLOCKCOL+BLOCKCOL3+j+1] = _mm512_reduce_add_pd(s31) + cum31;

                    c[iBLOCKCOL+j+2] = _mm512_reduce_add_pd(s02) + cum02;
                    c[iBLOCKCOL+BLOCKCOL+j+2] = _mm512_reduce_add_pd(s12) + cum12;
                    c[iBLOCKCOL+BLOCKCOL2+j+2] = _mm512_reduce_add_pd(s22) + cum22;
                    c[iBLOCKCOL+BLOCKCOL3+j+2] = _mm512_reduce_add_pd(s32) + cum32;
                    c[iBLOCKCOL+j+3] = _mm512_reduce_add_pd(s03) + cum03;
                    c[iBLOCKCOL+BLOCKCOL+j+3] = _mm512_reduce_add_pd(s13) + cum13;
                    c[iBLOCKCOL+BLOCKCOL2+j+3] = _mm512_reduce_add_pd(s23) + cum23;
                    c[iBLOCKCOL+BLOCKCOL3+j+3] = _mm512_reduce_add_pd(s33) + cum33;
                }
            }
            updateMeta(c, BLOCK64);
        }
/* */
//
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
/* */

        void MatrixStdDouble::blockMulOneAvxBlock4(double* ab[], double* bb[], double c[], int blocks, unsigned long msk, int coreIdx) //64
        {
            const int n = BLOCK64;
            uint calc;
            uint czero;

            _mm_prefetch(c+METAOFFSET, _MM_HINT_T1);
            _mm_prefetch(c+DETAILOFFSET, _MM_HINT_T1);
            uint *metac = (uint*) (c + METAOFFSET);
            int skipped = 0;

            int seq;
                double* cinn;

              for(int m=0;m<BLOCK64/8;m++){
                     uint16_t *metacd = (uint16_t *)(c + DETAILOFFSET);
                     metacd += DETAILSKIPSHORT * (m/4); 
                                   uint16_t metacm0 = metacd[(m%4)*8+0];
                                   uint16_t metacm1 = metacd[(m%4)*8+1];
                                   uint16_t metacm2 = metacd[(m%4)*8+2];
                                   uint16_t metacm3 = metacd[(m%4)*8+3];
                                   uint16_t metacm4 = metacd[(m%4)*8+4];
                                   uint16_t metacm5 = metacd[(m%4)*8+5];
                                   uint16_t metacm6 = metacd[(m%4)*8+6];
                                   uint16_t metacm7 = metacd[(m%4)*8+7];
                for(int p=0;p<BLOCK64/8;p++){

                    int nonzero = 0;
                    int i = m * 8 ;
                        int inn = i*BLOCKCOL + p * 8;
                        cinn = c+inn;
                    uint mask16p = 1<<p;

                     for(int sidx =0; sidx < blocks; sidx++){
                        double* a = ab[sidx];
                        double* b = bb[sidx];
            uint *metaa = (uint*) (a + METAOFFSET);
            uint *metab = (uint*) (b + METAOFFSET);
            _mm_prefetch(metaa, _MM_HINT_T1);
            _mm_prefetch(metab, _MM_HINT_T1);
                    if((metaa[m] & METAMASK)>0)
                    _mm_prefetch(a+DETAILOFFSET+DETAILSKIPSHORT*(m/4)/4,_MM_HINT_T1); 
                double* aaik;
                double* bbkp;

                    calc = 0;
                    czero = 0;
                    for(int k=0; k<BLOCK64/8; k++){
                        calc = calc | 1<<k;
                        if((metaa[m] & 1<<k) == 0 || (metab[k] & 1<<p) == 0)
                            calc = calc & (METAMASK - (1<<k));
                    }
                    for(int k=BLOCK64/8-1; k>=0; k--){
                        if((calc & 1<<k) == 0) continue;
                        nonzero = 1;
                    }
                    if(nonzero == 0) {continue;}
                     uint16_t *metaad = (uint16_t *)(a + DETAILOFFSET);
                     metaad += DETAILSKIPSHORT * (i/32); 
                           uint16_t metaai0 = metaad[i%32+0];
                           uint16_t metaai1 = metaad[i%32+1];
                           uint16_t metaai2 = metaad[i%32+2];
                           uint16_t metaai3 = metaad[i%32+3];
                           uint16_t metaai4 = metaad[i%32+4];
                           uint16_t metaai5 = metaad[i%32+5];
                           uint16_t metaai6 = metaad[i%32+6];
                           uint16_t metaai7 = metaad[i%32+7];

                           __m512d r16; 
                           __m512d r17; 
                           __m512d r18; 
                           __m512d r19; 
                           __m512d r20; 
                           __m512d r21; 
                           __m512d r22; 
                           __m512d r23; 
                           r16 = _mm512_xor_pd(r16,r16); 
                           r17 = _mm512_xor_pd(r17,r17); 
                           r18 = _mm512_xor_pd(r18,r18); 
                           r19 = _mm512_xor_pd(r19,r19); 
                           r20 = _mm512_xor_pd(r20,r20); 
                           r21 = _mm512_xor_pd(r21,r21); 
                           r22 = _mm512_xor_pd(r22,r22); 
                           r23 = _mm512_xor_pd(r23,r23); 

                           if(__builtin_expect((metacm0 & mask16p)>0,0)){ czero = czero | 1u;
                           r16 = _mm512_load_pd ((void const*) (cinn));}
                           if(__builtin_expect((metacm1 & mask16p)>0,0)){ czero = czero | 2u;
                           r17 = _mm512_load_pd ((void const*) (cinn+ROWSKIP64));}
                           if(__builtin_expect((metacm2 & mask16p)>0,0)){ czero = czero | 4u;
                           r18 = _mm512_load_pd ((void const*) (cinn+ROWSKIP128));}
                           if(__builtin_expect((metacm3 & mask16p)>0,0)){ czero = czero | 8u;
                           r19 = _mm512_load_pd ((void const*) (cinn+ROWSKIP192));}
                           if(__builtin_expect((metacm4 & mask16p)>0,0)){ czero = czero | 16u;
                           r20 = _mm512_load_pd ((void const*) (cinn+ROWSKIP256));}
                           if(__builtin_expect((metacm5 & mask16p)>0,0)){ czero = czero | 32u;
                           r21 = _mm512_load_pd ((void const*) (cinn+ROWSKIP320));}
                           if(__builtin_expect((metacm6 & mask16p)>0,0)){ czero = czero | 64u;
                           r22 = _mm512_load_pd ((void const*) (cinn+ROWSKIP384));}
                           if(__builtin_expect((metacm7 & mask16p)>0,0)){ czero = czero | 128u;
                           r23 = _mm512_load_pd ((void const*) (cinn+ROWSKIP448));}

                        for (int k = 0; k < BLOCK64/8; k++)
                    {
                        if((calc & 1<<k) == 0) continue;
                        _mm_prefetch(b+DETAILOFFSET+DETAILSKIPSHORT*(k/4)/4,_MM_HINT_T1);
                        int k8 = k*8;
                        int aik = i*BLOCKCOL + k8;
                        uint mask16k = 1<<k;

                        int bkp = k8 * BLOCKCOL + p * 8;
                         aaik = a+aik;
                         bbkp = b+bkp;

                           __m512d r00;
                           __m512d r01;
                           __m512d r02;
                           __m512d r03;
                           __m512d r04;
                           __m512d r05;
                           __m512d r06;
                           __m512d r07;
                           __m512d r24;
                           __m512d r25;
                           __m512d r26;
                           __m512d r27;
                           __m512d r28;
                           __m512d r29;
                           __m512d r30;
                           __m512d r31;
                        void * bbkp0 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp64 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp128 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp192 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp256 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp320 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp384 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp448 = (void *)(MatrixStdDouble::blanckdata);
                               _mm_prefetch(MatrixStdDouble::blanckdata,_MM_HINT_T1);

                     uint16_t *metabd = (uint16_t *)(b + DETAILOFFSET);
                     metabd += DETAILSKIPSHORT * (k/4); 
                           uint metabk0 = metabd[(k%4)*8+0];
                           uint metabk1 = metabd[(k%4)*8+1];
                           uint metabk2 = metabd[(k%4)*8+2];
                           uint metabk3 = metabd[(k%4)*8+3];
                           uint metabk4 = metabd[(k%4)*8+4];
                           uint metabk5 = metabd[(k%4)*8+5];
                           uint metabk6 = metabd[(k%4)*8+6];
                           uint metabk7 = metabd[(k%4)*8+7];

                           __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                           {
                           uint16_t *metabk8 = (uint16_t *)(metabd + (k%4)*8);
                           __m512i bbkp512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(bbkp));
                           __m512i meta512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)metabk8));
                           __m512i mask512 = _mm512_set1_epi64(mask16p);
                           __m512i offset512 = _mm512_slli_epi64(_mm512_set_epi64(ROWSKIP448,ROWSKIP384,ROWSKIP320,ROWSKIP256,ROWSKIP192,ROWSKIP128,ROWSKIP64,0),3);
                                   bbkp512 = _mm512_add_epi64(bbkp512,offset512);
                                   mask512 = _mm512_srai_epi64(_mm512_and_epi64(meta512,mask512),p); 
                                   mask512 = _mm512_sub_epi64(mask512,_mm512_set1_epi64(1));
                           __m512i addr512 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,bbkp512));        

                           __m256i lo4 =_mm512_extracti64x4_epi64(addr512,0);
                           __m256i hi4 =_mm512_extracti64x4_epi64(addr512,1);
               
                           bbkp0 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 0));
                               __builtin_prefetch(bbkp0,0,3);
                           bbkp64 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 1));
                               __builtin_prefetch(bbkp64,0,3);
                           bbkp128 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 2));
                               __builtin_prefetch(bbkp128,0,3);
                           bbkp192 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 3));
                               __builtin_prefetch(bbkp192,0,3);
                           bbkp256 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 0));
                               __builtin_prefetch(bbkp256,0,3);
                           bbkp320 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 1));
                               __builtin_prefetch(bbkp320,0,3);
                           bbkp384 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 2));
                               __builtin_prefetch(bbkp384,0,3);
                           bbkp448 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 3));
                               __builtin_prefetch(bbkp448,0,3);
                           }

                        double *aaik0 = MatrixStdDouble::blanckdata;
                        double *aaik64 = MatrixStdDouble::blanckdata;
                        double *aaik128 = MatrixStdDouble::blanckdata;
                        double *aaik192 = MatrixStdDouble::blanckdata;
                        double *aaik256 = MatrixStdDouble::blanckdata;
                        double *aaik320 = MatrixStdDouble::blanckdata;
                        double *aaik384 = MatrixStdDouble::blanckdata;
                        double *aaik448 = MatrixStdDouble::blanckdata;
                       
                           { 
                           __m512i aaik512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(aaik));
                           __m512i metaa512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metaad+i%32)));
                           __m512i maskk512 = _mm512_set1_epi64(mask16k);
                           __m512i offsetk512 = _mm512_slli_epi64(_mm512_set_epi64(ROWSKIP448,ROWSKIP384,ROWSKIP320,ROWSKIP256,ROWSKIP192,ROWSKIP128,ROWSKIP64,0),3);
                           aaik512 = _mm512_add_epi64(aaik512,offsetk512);
                           maskk512 = _mm512_srai_epi64(_mm512_and_epi64(metaa512,maskk512),k);
                           maskk512 = _mm512_sub_epi64(maskk512,_mm512_set1_epi64(1));
                           __m512i addrk512 = _mm512_or_epi64(_mm512_and_epi64(maskk512,blanck512),_mm512_andnot_epi64(maskk512,aaik512));

                           __m256i lok4 =_mm512_extracti64x4_epi64(addrk512,0);
                           __m256i hik4 =_mm512_extracti64x4_epi64(addrk512,1);
                           aaik0 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 0));
                           aaik64 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 1));
                               __builtin_prefetch(aaik64,0,3);
                           aaik128 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 2));
                               __builtin_prefetch(aaik128,0,3);
                           aaik192 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 3));
                               __builtin_prefetch(aaik192,0,3);
                           aaik256 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 0));
                               __builtin_prefetch(aaik256,0,3);
                           aaik320 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 1));
                               __builtin_prefetch(aaik320,0,3);
                           aaik384 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 2));
                               __builtin_prefetch(aaik384,0,3);
                           aaik448 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 3));
                               __builtin_prefetch(aaik448,0,3);
                           }
                      
                           {
                           r00 = _mm512_set1_pd(aaik0[0]);
                           r01 = _mm512_set1_pd(aaik64[0]);
                           r02 = _mm512_set1_pd(aaik128[0]);
                           r03 = _mm512_set1_pd(aaik192[0]);
                           r04 = _mm512_set1_pd(aaik256[0]);
                           r05 = _mm512_set1_pd(aaik320[0]);
                           r06 = _mm512_set1_pd(aaik384[0]);
                           r07 = _mm512_set1_pd(aaik448[0]);

                           __m512d r08 = _mm512_load_pd (bbkp0);
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r01, r08, r17);
                           r18 = _mm512_fmadd_pd (r02, r08, r18);
                           r19 = _mm512_fmadd_pd (r03, r08, r19);
                           r20 = _mm512_fmadd_pd (r04, r08, r20);
                           r21 = _mm512_fmadd_pd (r05, r08, r21);
                           r22 = _mm512_fmadd_pd (r06, r08, r22);
                           r23 = _mm512_fmadd_pd (r07, r08, r23);
                           }
                           {
                           r24 = _mm512_set1_pd(aaik0[1]);
                           r25 = _mm512_set1_pd(aaik64[1]);
                           r26 = _mm512_set1_pd(aaik128[1]);
                           r27 = _mm512_set1_pd(aaik192[1]);
                           r28 = _mm512_set1_pd(aaik256[1]);
                           r29 = _mm512_set1_pd(aaik320[1]);
                           r30 = _mm512_set1_pd(aaik384[1]);
                           r31 = _mm512_set1_pd(aaik448[1]);
                           __m512d r09 = _mm512_load_pd ( bbkp64);
                           r16 = _mm512_fmadd_pd (r24, r09, r16);
                           r17 = _mm512_fmadd_pd (r25, r09, r17);
                           r18 = _mm512_fmadd_pd (r26, r09, r18);
                           r19 = _mm512_fmadd_pd (r27, r09, r19);
                           r20 = _mm512_fmadd_pd (r28, r09, r20);
                           r21 = _mm512_fmadd_pd (r29, r09, r21);
                           r22 = _mm512_fmadd_pd (r30, r09, r22);
                           r23 = _mm512_fmadd_pd (r31, r09, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[2]);
                            r01 = _mm512_set1_pd(aaik64[2]);
                            r02 = _mm512_set1_pd(aaik128[2]);
                            r03 = _mm512_set1_pd(aaik192[2]);
                            r04 = _mm512_set1_pd(aaik256[2]);
                            r05 = _mm512_set1_pd(aaik320[2]);
                            r06 = _mm512_set1_pd(aaik384[2]);
                            r07 = _mm512_set1_pd(aaik448[2]);
                           __m512d r10 = _mm512_load_pd (bbkp128);
                           r16 = _mm512_fmadd_pd (r00, r10, r16);
                           r17 = _mm512_fmadd_pd (r01, r10, r17);
                           r18 = _mm512_fmadd_pd (r02, r10, r18);
                           r19 = _mm512_fmadd_pd (r03, r10, r19);
                           r20 = _mm512_fmadd_pd (r04, r10, r20);
                           r21 = _mm512_fmadd_pd (r05, r10, r21);
                           r22 = _mm512_fmadd_pd (r06, r10, r22);
                           r23 = _mm512_fmadd_pd (r07, r10, r23);
                           }
                           {
                            r24 = _mm512_set1_pd(aaik0[3]);
                            r25 = _mm512_set1_pd(aaik64[3]);
                            r26 = _mm512_set1_pd(aaik128[3]);
                            r27 = _mm512_set1_pd(aaik192[3]);
                            r28 = _mm512_set1_pd(aaik256[3]);
                            r29 = _mm512_set1_pd(aaik320[3]);
                            r30 = _mm512_set1_pd(aaik384[3]);
                            r31 = _mm512_set1_pd(aaik448[3]);
                           __m512d r11 = _mm512_load_pd (bbkp192);
                           r16 = _mm512_fmadd_pd (r24, r11, r16);
                           r17 = _mm512_fmadd_pd (r25, r11, r17);
                           r18 = _mm512_fmadd_pd (r26, r11, r18);
                           r19 = _mm512_fmadd_pd (r27, r11, r19);
                           r20 = _mm512_fmadd_pd (r28, r11, r20);
                           r21 = _mm512_fmadd_pd (r29, r11, r21);
                           r22 = _mm512_fmadd_pd (r30, r11, r22);
                           r23 = _mm512_fmadd_pd (r31, r11, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[4]);
                            r01 = _mm512_set1_pd(aaik64[4]);
                            r02 = _mm512_set1_pd(aaik128[4]);
                            r03 = _mm512_set1_pd(aaik192[4]);
                            r04 = _mm512_set1_pd(aaik256[4]);
                            r05 = _mm512_set1_pd(aaik320[4]);
                            r06 = _mm512_set1_pd(aaik384[4]);
                            r07 = _mm512_set1_pd(aaik448[4]);
                           __m512d r12 = _mm512_load_pd (bbkp256);
                           r16 = _mm512_fmadd_pd (r00, r12, r16);
                           r17 = _mm512_fmadd_pd (r01, r12, r17);
                           r18 = _mm512_fmadd_pd (r02, r12, r18);
                           r19 = _mm512_fmadd_pd (r03, r12, r19);
                           r20 = _mm512_fmadd_pd (r04, r12, r20);
                           r21 = _mm512_fmadd_pd (r05, r12, r21);
                           r22 = _mm512_fmadd_pd (r06, r12, r22);
                           r23 = _mm512_fmadd_pd (r07, r12, r23);
                           }

                           {
                            r24 = _mm512_set1_pd(aaik0[5]);
                            r25 = _mm512_set1_pd(aaik64[5]);
                            r26 = _mm512_set1_pd(aaik128[5]);
                            r27 = _mm512_set1_pd(aaik192[5]);
                            r28 = _mm512_set1_pd(aaik256[5]);
                            r29 = _mm512_set1_pd(aaik320[5]);
                            r30 = _mm512_set1_pd(aaik384[5]);
                            r31 = _mm512_set1_pd(aaik448[5]);
                           __m512d r13 = _mm512_load_pd (bbkp320);
                           r16 = _mm512_fmadd_pd (r24, r13, r16);
                           r17 = _mm512_fmadd_pd (r25, r13, r17);
                           r18 = _mm512_fmadd_pd (r26, r13, r18);
                           r19 = _mm512_fmadd_pd (r27, r13, r19);
                           r20 = _mm512_fmadd_pd (r28, r13, r20);
                           r21 = _mm512_fmadd_pd (r29, r13, r21);
                           r22 = _mm512_fmadd_pd (r30, r13, r22);
                           r23 = _mm512_fmadd_pd (r31, r13, r23);
                           }

                           {
                            r00 = _mm512_set1_pd(aaik0[6]);
                            r01 = _mm512_set1_pd(aaik64[6]);
                            r02 = _mm512_set1_pd(aaik128[6]);
                            r03 = _mm512_set1_pd(aaik192[6]);
                            r04 = _mm512_set1_pd(aaik256[6]);
                            r05 = _mm512_set1_pd(aaik320[6]);
                            r06 = _mm512_set1_pd(aaik384[6]);
                            r07 = _mm512_set1_pd(aaik448[6]);
                           __m512d r14 = _mm512_load_pd (bbkp384);
                           r16 = _mm512_fmadd_pd (r00, r14, r16);
                           r17 = _mm512_fmadd_pd (r01, r14, r17);
                           r18 = _mm512_fmadd_pd (r02, r14, r18);
                           r19 = _mm512_fmadd_pd (r03, r14, r19);
                           r20 = _mm512_fmadd_pd (r04, r14, r20);
                           r21 = _mm512_fmadd_pd (r05, r14, r21);
                           r22 = _mm512_fmadd_pd (r06, r14, r22);
                           r23 = _mm512_fmadd_pd (r07, r14, r23);
                           }

                           {
                            r24 = _mm512_set1_pd(aaik0[7]);
                            r25 = _mm512_set1_pd(aaik64[7]);
                            r26 = _mm512_set1_pd(aaik128[7]);
                            r27 = _mm512_set1_pd(aaik192[7]);
                            r28 = _mm512_set1_pd(aaik256[7]);
                            r29 = _mm512_set1_pd(aaik320[7]);
                            r30 = _mm512_set1_pd(aaik384[7]);
                            r31 = _mm512_set1_pd(aaik448[7]);
                           __m512d r15 = _mm512_load_pd (bbkp448);
                           r16 = _mm512_fmadd_pd (r24, r15, r16);
                           r17 = _mm512_fmadd_pd (r25, r15, r17);
                           r18 = _mm512_fmadd_pd (r26, r15, r18);
                           r19 = _mm512_fmadd_pd (r27, r15, r19);
                           r20 = _mm512_fmadd_pd (r28, r15, r20);
                           r21 = _mm512_fmadd_pd (r29, r15, r21);
                           r22 = _mm512_fmadd_pd (r30, r15, r22);
                           r23 = _mm512_fmadd_pd (r31, r15, r23);
                           }

                    }
                           int fnd = 0;
                           int fn1 = 0;
                           int fn2 = 0;
                           int fn3 = 0;
                           int fn4 = 0;
                           int fn5 = 0;
                           int fn6 = 0;
                           int fn7 = 0;
                           __m512i rzero = _mm512_set1_epi32(0x60000000u);
//*
                           __m512d r00 = _mm512_set1_pd(TINY1);
                           __m512d r01 = _mm512_set1_pd(-TINY1);
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r16, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r16, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm0 = metacm0 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 1u) > 0 || (metacm0 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn), r16);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r17, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r17, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm1 = metacm1 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 2u) > 0 || (metacm1 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP64), r17);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r18, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r18, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm2 = metacm2 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 4u) > 0 || (metacm2 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP128), r18);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r19, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r19, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm3 = metacm3 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 8u) > 0 || (metacm3 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP192), r19);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r20, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r20, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm4 = metacm4 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 16u) > 0 || (metacm4 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP256), r20);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r21, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r21, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm5 = metacm5 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 32u) > 0 || (metacm5 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP320), r21);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r22, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r22, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm6 = metacm6 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 64u) > 0 || (metacm6 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP384), r22);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r23, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r23, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm7 = metacm7 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 128u) > 0 || (metacm7 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP448), r23);}
                           if(fnd > 0)
                           metac[m] = metac[m] | mask16p;
                      /* */ 
                }
                }
                                    metacd[(m%4)*8+0] = metacm0;
                                    metacd[(m%4)*8+1] = metacm1;
                                    metacd[(m%4)*8+2] = metacm2;
                                    metacd[(m%4)*8+3] = metacm3;
                                    metacd[(m%4)*8+4] = metacm4;
                                    metacd[(m%4)*8+5] = metacm5;
                                    metacd[(m%4)*8+6] = metacm6;
                                    metacd[(m%4)*8+7] = metacm7;
            }
        }
/*
        void MatrixStdDouble::blockMulOneAvxBlock(double *__restrict a, double *__restrict b, double *__restrict c, unsigned long msk, int coreIdx) //64
        {
            const int n = BLOCK64;
            uint calc;
            uint czero;

            _mm_prefetch(c+METAOFFSET, _MM_HINT_T1);
            _mm_prefetch(c+DETAILOFFSET, _MM_HINT_T1);
            uint *metaa = (uint*) (a + METAOFFSET);
            uint *metab = (uint*) (b + METAOFFSET);
            uint *metac = (uint*) (c + METAOFFSET);
            _mm_prefetch(metaa, _MM_HINT_T1);
            _mm_prefetch(metab, _MM_HINT_T1);
            int skipped = 0;

            int seq;
                double* cinn;
                double* aaik;
                double* bbkp;
//  //
              for(int m=0;m<BLOCK64/8;m++){
                     uint16_t *metacd = (uint16_t *)(c + DETAILOFFSET);
                     metacd += DETAILSKIPSHORT * (m/4); 
                                   uint16_t metacm0 = metacd[(m%4)*8+0];
                                   uint16_t metacm1 = metacd[(m%4)*8+1];
                                   uint16_t metacm2 = metacd[(m%4)*8+2];
                                   uint16_t metacm3 = metacd[(m%4)*8+3];
                                   uint16_t metacm4 = metacd[(m%4)*8+4];
                                   uint16_t metacm5 = metacd[(m%4)*8+5];
                                   uint16_t metacm6 = metacd[(m%4)*8+6];
                                   uint16_t metacm7 = metacd[(m%4)*8+7];
                    if((metaa[m] & METAMASK)>0)
                    _mm_prefetch(a+DETAILOFFSET+DETAILSKIPSHORT*(m/4)/4,_MM_HINT_T1); 
                for(int p=0;p<BLOCK64/8;p++){

                    int nonzero = 0;
                    int i = m * 8 ;
                        int inn = i*BLOCKCOL + p * 8;
                        cinn = c+inn;
                    uint mask16p = 1<<p;

                    calc = 0;
                    czero = 0;
                    for(int k=0; k<BLOCK64/8; k++){
                        calc = calc | 1<<k;
                        if((metaa[m] & 1<<k) == 0 || (metab[k] & 1<<p) == 0)
                            calc = calc & (METAMASK - (1<<k));
                    }
                    for(int k=BLOCK64/8-1; k>=0; k--){
                        if((calc & 1<<k) == 0) continue;
                        nonzero = 1;
                    }
                    if(nonzero == 0) {continue;}
                     uint16_t *metaad = (uint16_t *)(a + DETAILOFFSET);
                     metaad += DETAILSKIPSHORT * (i/32); 
                           uint16_t metaai0 = metaad[i%32+0];
                           uint16_t metaai1 = metaad[i%32+1];
                           uint16_t metaai2 = metaad[i%32+2];
                           uint16_t metaai3 = metaad[i%32+3];
                           uint16_t metaai4 = metaad[i%32+4];
                           uint16_t metaai5 = metaad[i%32+5];
                           uint16_t metaai6 = metaad[i%32+6];
                           uint16_t metaai7 = metaad[i%32+7];

                           __m512d r16; 
                           __m512d r17; 
                           __m512d r18; 
                           __m512d r19; 
                           __m512d r20; 
                           __m512d r21; 
                           __m512d r22; 
                           __m512d r23; 
                           r16 = _mm512_xor_pd(r16,r16); 
                           r17 = _mm512_xor_pd(r17,r17); 
                           r18 = _mm512_xor_pd(r18,r18); 
                           r19 = _mm512_xor_pd(r19,r19); 
                           r20 = _mm512_xor_pd(r20,r20); 
                           r21 = _mm512_xor_pd(r21,r21); 
                           r22 = _mm512_xor_pd(r22,r22); 
                           r23 = _mm512_xor_pd(r23,r23); 

                           if(__builtin_expect((metacm0 & mask16p)>0,0)){ czero = czero | 1u;
                           r16 = _mm512_load_pd ((void const*) (cinn));}
                           if(__builtin_expect((metacm1 & mask16p)>0,0)){ czero = czero | 2u;
                           r17 = _mm512_load_pd ((void const*) (cinn+ROWSKIP64));}
                           if(__builtin_expect((metacm2 & mask16p)>0,0)){ czero = czero | 4u;
                           r18 = _mm512_load_pd ((void const*) (cinn+ROWSKIP128));}
                           if(__builtin_expect((metacm3 & mask16p)>0,0)){ czero = czero | 8u;
                           r19 = _mm512_load_pd ((void const*) (cinn+ROWSKIP192));}
                           if(__builtin_expect((metacm4 & mask16p)>0,0)){ czero = czero | 16u;
                           r20 = _mm512_load_pd ((void const*) (cinn+ROWSKIP256));}
                           if(__builtin_expect((metacm5 & mask16p)>0,0)){ czero = czero | 32u;
                           r21 = _mm512_load_pd ((void const*) (cinn+ROWSKIP320));}
                           if(__builtin_expect((metacm6 & mask16p)>0,0)){ czero = czero | 64u;
                           r22 = _mm512_load_pd ((void const*) (cinn+ROWSKIP384));}
                           if(__builtin_expect((metacm7 & mask16p)>0,0)){ czero = czero | 128u;
                           r23 = _mm512_load_pd ((void const*) (cinn+ROWSKIP448));}

                        for (int k = 0; k < BLOCK64/8; k++)
                    {
                        if((calc & 1<<k) == 0) continue;
                        _mm_prefetch(b+DETAILOFFSET+DETAILSKIPSHORT*(k/4)/4,_MM_HINT_T1);
                        int k8 = k*8;
                        int aik = i*BLOCKCOL + k8;
                        uint mask16k = 1<<k;

                        int bkp = k8 * BLOCKCOL + p * 8;
                         aaik = a+aik;
                         bbkp = b+bkp;

                           __m512d r00;
                           __m512d r01;
                           __m512d r02;
                           __m512d r03;
                           __m512d r04;
                           __m512d r05;
                           __m512d r06;
                           __m512d r07;
                           __m512d r24;
                           __m512d r25;
                           __m512d r26;
                           __m512d r27;
                           __m512d r28;
                           __m512d r29;
                           __m512d r30;
                           __m512d r31;
                        void * bbkp0 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp64 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp128 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp192 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp256 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp320 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp384 = (void *)(MatrixStdDouble::blanckdata);
                        void * bbkp448 = (void *)(MatrixStdDouble::blanckdata);
                               _mm_prefetch(MatrixStdDouble::blanckdata,_MM_HINT_T1);

                     uint16_t *metabd = (uint16_t *)(b + DETAILOFFSET);
                     metabd += DETAILSKIPSHORT * (k/4); 
                           uint metabk0 = metabd[(k%4)*8+0];
                           uint metabk1 = metabd[(k%4)*8+1];
                           uint metabk2 = metabd[(k%4)*8+2];
                           uint metabk3 = metabd[(k%4)*8+3];
                           uint metabk4 = metabd[(k%4)*8+4];
                           uint metabk5 = metabd[(k%4)*8+5];
                           uint metabk6 = metabd[(k%4)*8+6];
                           uint metabk7 = metabd[(k%4)*8+7];

                           __m512i blanck512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(MatrixStdDouble::blanckdata+coreIdx*32));
                           {
                           uint16_t *metabk8 = (uint16_t *)(metabd + (k%4)*8);
                           __m512i bbkp512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(bbkp));
                           __m512i meta512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)metabk8));
                           __m512i mask512 = _mm512_set1_epi64(mask16p);
                           __m512i offset512 = _mm512_slli_epi64(_mm512_set_epi64(ROWSKIP448,ROWSKIP384,ROWSKIP320,ROWSKIP256,ROWSKIP192,ROWSKIP128,ROWSKIP64,0),3);
                                   bbkp512 = _mm512_add_epi64(bbkp512,offset512);
                                   mask512 = _mm512_srai_epi64(_mm512_and_epi64(meta512,mask512),p); 
                                   mask512 = _mm512_sub_epi64(mask512,_mm512_set1_epi64(1));
                           __m512i addr512 = _mm512_or_epi64(_mm512_and_epi64(mask512,blanck512),_mm512_andnot_epi64(mask512,bbkp512));        

                           __m256i lo4 =_mm512_extracti64x4_epi64(addr512,0);
                           __m256i hi4 =_mm512_extracti64x4_epi64(addr512,1);
               
                           bbkp0 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 0));
                               __builtin_prefetch(bbkp0,0,3);
                           bbkp64 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 1));
                               __builtin_prefetch(bbkp64,0,3);
                           bbkp128 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 2));
                               __builtin_prefetch(bbkp128,0,3);
                           bbkp192 = reinterpret_cast<void *>(_mm256_extract_epi64(lo4, 3));
                               __builtin_prefetch(bbkp192,0,3);
                           bbkp256 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 0));
                               __builtin_prefetch(bbkp256,0,3);
                           bbkp320 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 1));
                               __builtin_prefetch(bbkp320,0,3);
                           bbkp384 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 2));
                               __builtin_prefetch(bbkp384,0,3);
                           bbkp448 = reinterpret_cast<void *>(_mm256_extract_epi64(hi4, 3));
                               __builtin_prefetch(bbkp448,0,3);
                           }

                        double *aaik0 = MatrixStdDouble::blanckdata;
                        double *aaik64 = MatrixStdDouble::blanckdata;
                        double *aaik128 = MatrixStdDouble::blanckdata;
                        double *aaik192 = MatrixStdDouble::blanckdata;
                        double *aaik256 = MatrixStdDouble::blanckdata;
                        double *aaik320 = MatrixStdDouble::blanckdata;
                        double *aaik384 = MatrixStdDouble::blanckdata;
                        double *aaik448 = MatrixStdDouble::blanckdata;
                       
                           { 
                           __m512i aaik512 = _mm512_set1_epi64(reinterpret_cast<uintptr_t>(aaik));
                           __m512i metaa512 = _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i*)(metaad+i%32)));
                           __m512i maskk512 = _mm512_set1_epi64(mask16k);
                           __m512i offsetk512 = _mm512_slli_epi64(_mm512_set_epi64(ROWSKIP448,ROWSKIP384,ROWSKIP320,ROWSKIP256,ROWSKIP192,ROWSKIP128,ROWSKIP64,0),3);
                           aaik512 = _mm512_add_epi64(aaik512,offsetk512);
                           maskk512 = _mm512_srai_epi64(_mm512_and_epi64(metaa512,maskk512),k);
                           maskk512 = _mm512_sub_epi64(maskk512,_mm512_set1_epi64(1));
                           __m512i addrk512 = _mm512_or_epi64(_mm512_and_epi64(maskk512,blanck512),_mm512_andnot_epi64(maskk512,aaik512));

                           __m256i lok4 =_mm512_extracti64x4_epi64(addrk512,0);
                           __m256i hik4 =_mm512_extracti64x4_epi64(addrk512,1);
                           aaik0 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 0));
                           aaik64 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 1));
                               __builtin_prefetch(aaik64,0,3);
                           aaik128 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 2));
                               __builtin_prefetch(aaik128,0,3);
                           aaik192 = reinterpret_cast<double *>(_mm256_extract_epi64(lok4, 3));
                               __builtin_prefetch(aaik192,0,3);
                           aaik256 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 0));
                               __builtin_prefetch(aaik256,0,3);
                           aaik320 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 1));
                               __builtin_prefetch(aaik320,0,3);
                           aaik384 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 2));
                               __builtin_prefetch(aaik384,0,3);
                           aaik448 = reinterpret_cast<double *>(_mm256_extract_epi64(hik4, 3));
                               __builtin_prefetch(aaik448,0,3);
                           }
                      
                           {
                           r00 = _mm512_set1_pd(aaik0[0]);
                           r01 = _mm512_set1_pd(aaik64[0]);
                           r02 = _mm512_set1_pd(aaik128[0]);
                           r03 = _mm512_set1_pd(aaik192[0]);
                           r04 = _mm512_set1_pd(aaik256[0]);
                           r05 = _mm512_set1_pd(aaik320[0]);
                           r06 = _mm512_set1_pd(aaik384[0]);
                           r07 = _mm512_set1_pd(aaik448[0]);

                           __m512d r08 = _mm512_load_pd (bbkp0);
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r01, r08, r17);
                           r18 = _mm512_fmadd_pd (r02, r08, r18);
                           r19 = _mm512_fmadd_pd (r03, r08, r19);
                           r20 = _mm512_fmadd_pd (r04, r08, r20);
                           r21 = _mm512_fmadd_pd (r05, r08, r21);
                           r22 = _mm512_fmadd_pd (r06, r08, r22);
                           r23 = _mm512_fmadd_pd (r07, r08, r23);
                           }
                           {
                           r24 = _mm512_set1_pd(aaik0[1]);
                           r25 = _mm512_set1_pd(aaik64[1]);
                           r26 = _mm512_set1_pd(aaik128[1]);
                           r27 = _mm512_set1_pd(aaik192[1]);
                           r28 = _mm512_set1_pd(aaik256[1]);
                           r29 = _mm512_set1_pd(aaik320[1]);
                           r30 = _mm512_set1_pd(aaik384[1]);
                           r31 = _mm512_set1_pd(aaik448[1]);
                           __m512d r09 = _mm512_load_pd ( bbkp64);
                           r16 = _mm512_fmadd_pd (r24, r09, r16);
                           r17 = _mm512_fmadd_pd (r25, r09, r17);
                           r18 = _mm512_fmadd_pd (r26, r09, r18);
                           r19 = _mm512_fmadd_pd (r27, r09, r19);
                           r20 = _mm512_fmadd_pd (r28, r09, r20);
                           r21 = _mm512_fmadd_pd (r29, r09, r21);
                           r22 = _mm512_fmadd_pd (r30, r09, r22);
                           r23 = _mm512_fmadd_pd (r31, r09, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[2]);
                            r01 = _mm512_set1_pd(aaik64[2]);
                            r02 = _mm512_set1_pd(aaik128[2]);
                            r03 = _mm512_set1_pd(aaik192[2]);
                            r04 = _mm512_set1_pd(aaik256[2]);
                            r05 = _mm512_set1_pd(aaik320[2]);
                            r06 = _mm512_set1_pd(aaik384[2]);
                            r07 = _mm512_set1_pd(aaik448[2]);
                           __m512d r10 = _mm512_load_pd (bbkp128);
                           r16 = _mm512_fmadd_pd (r00, r10, r16);
                           r17 = _mm512_fmadd_pd (r01, r10, r17);
                           r18 = _mm512_fmadd_pd (r02, r10, r18);
                           r19 = _mm512_fmadd_pd (r03, r10, r19);
                           r20 = _mm512_fmadd_pd (r04, r10, r20);
                           r21 = _mm512_fmadd_pd (r05, r10, r21);
                           r22 = _mm512_fmadd_pd (r06, r10, r22);
                           r23 = _mm512_fmadd_pd (r07, r10, r23);
                           }
                           {
                            r24 = _mm512_set1_pd(aaik0[3]);
                            r25 = _mm512_set1_pd(aaik64[3]);
                            r26 = _mm512_set1_pd(aaik128[3]);
                            r27 = _mm512_set1_pd(aaik192[3]);
                            r28 = _mm512_set1_pd(aaik256[3]);
                            r29 = _mm512_set1_pd(aaik320[3]);
                            r30 = _mm512_set1_pd(aaik384[3]);
                            r31 = _mm512_set1_pd(aaik448[3]);
                           __m512d r11 = _mm512_load_pd (bbkp192);
                           r16 = _mm512_fmadd_pd (r24, r11, r16);
                           r17 = _mm512_fmadd_pd (r25, r11, r17);
                           r18 = _mm512_fmadd_pd (r26, r11, r18);
                           r19 = _mm512_fmadd_pd (r27, r11, r19);
                           r20 = _mm512_fmadd_pd (r28, r11, r20);
                           r21 = _mm512_fmadd_pd (r29, r11, r21);
                           r22 = _mm512_fmadd_pd (r30, r11, r22);
                           r23 = _mm512_fmadd_pd (r31, r11, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[4]);
                            r01 = _mm512_set1_pd(aaik64[4]);
                            r02 = _mm512_set1_pd(aaik128[4]);
                            r03 = _mm512_set1_pd(aaik192[4]);
                            r04 = _mm512_set1_pd(aaik256[4]);
                            r05 = _mm512_set1_pd(aaik320[4]);
                            r06 = _mm512_set1_pd(aaik384[4]);
                            r07 = _mm512_set1_pd(aaik448[4]);
                           __m512d r12 = _mm512_load_pd (bbkp256);
                           r16 = _mm512_fmadd_pd (r00, r12, r16);
                           r17 = _mm512_fmadd_pd (r01, r12, r17);
                           r18 = _mm512_fmadd_pd (r02, r12, r18);
                           r19 = _mm512_fmadd_pd (r03, r12, r19);
                           r20 = _mm512_fmadd_pd (r04, r12, r20);
                           r21 = _mm512_fmadd_pd (r05, r12, r21);
                           r22 = _mm512_fmadd_pd (r06, r12, r22);
                           r23 = _mm512_fmadd_pd (r07, r12, r23);
                           }

                           {
                            r24 = _mm512_set1_pd(aaik0[5]);
                            r25 = _mm512_set1_pd(aaik64[5]);
                            r26 = _mm512_set1_pd(aaik128[5]);
                            r27 = _mm512_set1_pd(aaik192[5]);
                            r28 = _mm512_set1_pd(aaik256[5]);
                            r29 = _mm512_set1_pd(aaik320[5]);
                            r30 = _mm512_set1_pd(aaik384[5]);
                            r31 = _mm512_set1_pd(aaik448[5]);
                           __m512d r13 = _mm512_load_pd (bbkp320);
                           r16 = _mm512_fmadd_pd (r24, r13, r16);
                           r17 = _mm512_fmadd_pd (r25, r13, r17);
                           r18 = _mm512_fmadd_pd (r26, r13, r18);
                           r19 = _mm512_fmadd_pd (r27, r13, r19);
                           r20 = _mm512_fmadd_pd (r28, r13, r20);
                           r21 = _mm512_fmadd_pd (r29, r13, r21);
                           r22 = _mm512_fmadd_pd (r30, r13, r22);
                           r23 = _mm512_fmadd_pd (r31, r13, r23);
                           }
                           {
                            r00 = _mm512_set1_pd(aaik0[6]);
                            r01 = _mm512_set1_pd(aaik64[6]);
                            r02 = _mm512_set1_pd(aaik128[6]);
                            r03 = _mm512_set1_pd(aaik192[6]);
                            r04 = _mm512_set1_pd(aaik256[6]);
                            r05 = _mm512_set1_pd(aaik320[6]);
                            r06 = _mm512_set1_pd(aaik384[6]);
                            r07 = _mm512_set1_pd(aaik448[6]);
                           __m512d r14 = _mm512_load_pd (bbkp384);
                           r16 = _mm512_fmadd_pd (r00, r14, r16);
                           r17 = _mm512_fmadd_pd (r01, r14, r17);
                           r18 = _mm512_fmadd_pd (r02, r14, r18);
                           r19 = _mm512_fmadd_pd (r03, r14, r19);
                           r20 = _mm512_fmadd_pd (r04, r14, r20);
                           r21 = _mm512_fmadd_pd (r05, r14, r21);
                           r22 = _mm512_fmadd_pd (r06, r14, r22);
                           r23 = _mm512_fmadd_pd (r07, r14, r23);
                           }

                           {
                            r24 = _mm512_set1_pd(aaik0[7]);
                            r25 = _mm512_set1_pd(aaik64[7]);
                            r26 = _mm512_set1_pd(aaik128[7]);
                            r27 = _mm512_set1_pd(aaik192[7]);
                            r28 = _mm512_set1_pd(aaik256[7]);
                            r29 = _mm512_set1_pd(aaik320[7]);
                            r30 = _mm512_set1_pd(aaik384[7]);
                            r31 = _mm512_set1_pd(aaik448[7]);
                           __m512d r15 = _mm512_load_pd (bbkp448);
                           r16 = _mm512_fmadd_pd (r24, r15, r16);
                           r17 = _mm512_fmadd_pd (r25, r15, r17);
                           r18 = _mm512_fmadd_pd (r26, r15, r18);
                           r19 = _mm512_fmadd_pd (r27, r15, r19);
                           r20 = _mm512_fmadd_pd (r28, r15, r20);
                           r21 = _mm512_fmadd_pd (r29, r15, r21);
                           r22 = _mm512_fmadd_pd (r30, r15, r22);
                           r23 = _mm512_fmadd_pd (r31, r15, r23);
                           }

                    }
                           int fnd = 0;
                           int fn1 = 0;
                           int fn2 = 0;
                           int fn3 = 0;
                           int fn4 = 0;
                           int fn5 = 0;
                           int fn6 = 0;
                           int fn7 = 0;
                           __m512i rzero = _mm512_set1_epi32(0x60000000u);
//
                           __m512d r00 = _mm512_set1_pd(TINY1);
                           __m512d r01 = _mm512_set1_pd(-TINY1);
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r16, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r16, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm0 = metacm0 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 1u) > 0 || (metacm0 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn), r16);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r17, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r17, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm1 = metacm1 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 2u) > 0 || (metacm1 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP64), r17);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r18, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r18, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm2 = metacm2 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 4u) > 0 || (metacm2 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP128), r18);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r19, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r19, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm3 = metacm3 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 8u) > 0 || (metacm3 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP192), r19);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r20, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r20, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm4 = metacm4 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 16u) > 0 || (metacm4 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP256), r20);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r21, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r21, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm5 = metacm5 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 32u) > 0 || (metacm5 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP320), r21);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r22, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r22, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm6 = metacm6 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 64u) > 0 || (metacm6 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP384), r22);}
                               if(_cvtmask8_u32(_mm512_cmp_pd_mask(r23, r00, _CMP_NLE_UQ))>0 ||
                                   _cvtmask8_u32(_mm512_cmp_pd_mask(r23, r01, _CMP_NGE_UQ))>0 )
                               {   
                                   metacm7 = metacm7 | mask16p;
                                   fnd = 1;
                               }
                           if(__builtin_expect((czero & 128u) > 0 || (metacm7 & mask16p) > 0,0)){
                           _mm512_store_pd ((void*) (cinn+ROWSKIP448), r23);}
                           if(fnd > 0)
                           metac[m] = metac[m] | mask16p;
                      // // 
                }
                                    metacd[(m%4)*8+0] = metacm0;
                                    metacd[(m%4)*8+1] = metacm1;
                                    metacd[(m%4)*8+2] = metacm2;
                                    metacd[(m%4)*8+3] = metacm3;
                                    metacd[(m%4)*8+4] = metacm4;
                                    metacd[(m%4)*8+5] = metacm5;
                                    metacd[(m%4)*8+6] = metacm6;
                                    metacd[(m%4)*8+7] = metacm7;
            }
        }
/* */

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
            for(int m=0; m<BLOCK64/8; m++){
                uint ml = 0;
                uint mu = 0;
                uint maskl = (2u<<m) - 1u;
                uint masku = 256u - (1u<<m);
                for(int k=0;k<8;k++){
                    int i = m * 8 + k;
                    uint detail = (metal[DETAILSKIPSHORT * (i/32) + i%32] & METAMASK);
                    uint dl = detail & maskl;
                    dl = dl | dl << 8;
                    dl = dl | dl << 4;
                    dl = dl | dl << 2;
                    dl = dl | dl << 1;
                    dl = dl & maskl;

                    uint du = detail & masku;
                    ui = ui | du;
                    du = ui & masku;
                   
                    metal[DETAILSKIPSHORT * (i/32) + i%32] = dl;
                    metau[DETAILSKIPSHORT * (i/32) + i%32] = du;
                    ml = ml | dl;
                    mu = mu | du;
                }
            //    masku = (masku << 1) & METAMASK;
            //    maskl = (maskl << 1) + 1;
                l[m] = ml;
                u[m] = mu;
            }
        }
        void expandNoneZero(double *ldouble, int n)
        {
            uint *l = (uint *)(ldouble + METAOFFSET);
            uint16_t *metal =  (uint16_t *)(ldouble + DETAILOFFSET);
            uint16_t ui = *metal;
            for(int m=0; m<BLOCK64/8; m++){
                uint ml = 0;
                uint mu = 0;
                uint maskl = (2u<<m) - 1u;
                uint masku = 256u - (1u<<m);
                for(int k=0;k<8;k++){
                    int i = m * 8 + k;
                    uint detail = (metal[DETAILSKIPSHORT * (i/32) + i%32] & METAMASK);
                    uint dl = detail & maskl;
                    dl = dl | dl << 8;
                    dl = dl | dl << 4;
                    dl = dl | dl << 2;
                    dl = dl | dl << 1;
                    dl = dl & maskl;

                    uint du = detail & masku;
                    ui = ui | du;
                    du = ui & masku;
                   
                    metal[DETAILSKIPSHORT * (i/32) + i%32] = dl;
                    ml = ml | dl;
                    mu = mu | du;
                }
        //        masku = (masku << 1) & METAMASK;
        //        maskl = (maskl << 1) + 1;
                l[m] = ml;
            }
        }
/*
        void MatrixStdDouble::lltdcmpSimple(double* a, int n, double* l)
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
/* */
//
        void MatrixStdDouble::lltdcmpSimple(double* a, int n, double* l)
        { // cout
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }
   //         std::memset(l,0,ALLOCBLOCK);

            uint16_t *metal = (uint16_t*) (l + DETAILOFFSET);
            copyMetaField(a,l);
            expandNoneZero(l, n);

            for (int j=0; j<BLOCK64; j++){
                double sum = 0;
                        for (int r = 0; r <= j ; r+=8){
                            __m512d lx = _mm512_load_pd(l+j*BLOCKCOL+r);
                            __m512d mx = _mm512_mul_pd(lx,lx);
                            sum += _mm512_reduce_add_pd(mx); 
                        }
                double p = a[j * BLOCKCOL + j] - sum;
                if(p<TINY) p = TINY;
                p = sqrt(p);
                l[j * BLOCKCOL + j] = p;
                p = 1/ p;

                uint mski = 1u << (j / 8);
                for(int i=j+1; i<BLOCK64; i++){
                    if((metal[DETAILSKIPSHORT*(i/32)+i%32] & mski) == 0) continue;
                    double sun = 0;
                        for (int r = 0; r <= j; r+=8){
                            __m512d lx = _mm512_load_pd(l+i*BLOCKCOL+r);
                            __m512d ux = _mm512_load_pd(l+j*BLOCKCOL+r);
                            __m512d mx = _mm512_mul_pd(lx,ux);
                            sun += _mm512_reduce_add_pd(mx);
                        }
                    l[i * BLOCKCOL + j] = p * (a[i * BLOCKCOL + j] - sun);
                }
            }

 //           updateMeta(l, BLOCK64);
        }
/* */
//
        void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* u)
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
/* */
 //       void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* t)
        void ludcmpSimple_x(double* a, int n, double* l, double* t)
        {
            long double sum = 0;
            alignas(64) double u[BLOCK64 * BLOCKCOL];
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }
            std::memset(u,0,ALLOCBLOCK);
            std::memset(l,0,ALLOCBLOCK);
            std::memset(t,0,ALLOCBLOCK);

            uint16_t *metal = (uint16_t*) (l + DETAILOFFSET);
            uint16_t *metat = (uint16_t*) (t + DETAILOFFSET);
            copyMetaField(a,l);
            expandNoneZero(l, t, n);
            for (int i = 0; i < n; i++)
            {
                for(int p = i / 8; p<n/8; p++){
                    if(metat[DETAILSKIPSHORT*(i/32)+i%32] & (1u << p))
                    for(int q = 0; q < 8; q++)
              //      for(int j = i; j < n; j++) {
                    {
                        int j = p *8 + q;
                        if(j<i) continue;
                        sum = 0;
                   //     for (int r = 0; r <= i/8; r++){
                        for (int r = i/8; r >= 0; r--){
                            __m512d lx = _mm512_load_pd(l+i*BLOCKCOL+r*8);
                            __m512d ux = _mm512_load_pd(u+j*BLOCKCOL+r*8);
                            __m512d mx = _mm512_mul_pd(lx,ux);
                            sum += _mm512_reduce_add_pd(mx); 
                        }
                        long double ut = a[i * BLOCKCOL + j] - sum;
                        if(j>=i){
                            u[j * BLOCKCOL + i] = ut;
                            t[i * BLOCKCOL + j] = ut;
                        }
                    }
                }
                if (u[i * BLOCKCOL + i] < TINY2 && u[i * BLOCKCOL + i] > -TINY2)
                {
                    if (u[i * BLOCKCOL + i] < 0){
                        u[i * BLOCKCOL + i] = -TINY2;
                        t[i * BLOCKCOL + i] = -TINY2;
                    }
                    else{
                        u[i * BLOCKCOL + i] = TINY2;
                        t[i * BLOCKCOL + i] = TINY2;
                    }
                }
                long double u1 = u[i * BLOCKCOL + i];
                u1 = 1 / u1;
                uint mski = 1u << (i / 8);
                for (int j = i + 1; j < n; j++)
                {
                    if((metal[DETAILSKIPSHORT*(j/32)+j%32] & mski) == 0) continue;
                    sum = 0;
                        for (int r = 0; r <= i/8; r++){
                            __m512d lx = _mm512_load_pd(l+j*BLOCKCOL+r*8);
                            __m512d ux = _mm512_load_pd(u+i*BLOCKCOL+r*8);
                            __m512d mx = _mm512_mul_pd(lx,ux);
                            sum += _mm512_reduce_add_pd(mx);
                        }
               //     for (int k = 0; k < i; k++)
               //         sum += l[j * BLOCKCOL + k] * t[k * BLOCKCOL + i];
                    l[j * BLOCKCOL + i] = u1 * (a[j * BLOCKCOL + i] - sum);
                }
                l[i * BLOCKCOL + i] = 1;
            }

  //          updateMeta(l, n);
  //          updateMeta(t, n);
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
/*
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
/* */

        void MatrixStdDouble::inv_upper(double* u, int n, double* y)
//        void inv_upper_x(double* u, int n, double* y)
        {
            int i, j;
         //   for (i = 0; i < n*n; i++) y[i] = 0;
            std::memset(y, 0, BLOCK64 * BLOCKCOL * sizeof(double));
            for (i = 0; i < n; i++)
                y[i * BLOCKCOL + i] = 1.0;
            for (j = n-1; j >=0; j--)
            {
  /*              if (u[j * n + j] < TINY && u[j * n + j] > -TINY)
                {
                    if (u[j * n + j] < 0)
                        u[j * n + j] = -TINY;
                    else
                        u[j * n + j] = TINY;
                }
*/
                for (i = j-1; i >=0  ; i--)
                {
                    double scale = u[j * BLOCKCOL + j];
                    scale = u[i * BLOCKCOL + j] / scale;
                    for(int r=j/8; r<8; r++){
                        __m512d yjx = _mm512_load_pd(y+j*BLOCKCOL+r*8);
                        __m512d scalex = _mm512_set1_pd(scale);
                        __m512d yix = _mm512_load_pd(y+i*BLOCKCOL+r*8);
                        yix = _mm512_fnmadd_pd(scalex, yjx, yix);
                        _mm512_store_pd ((void*) (y+i*BLOCKCOL+r*8), yix);
                    }
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
/*
        void MatrixStdDouble::mat_sub(double* a, double* b, int n, double* r, int coreIdx)
        {
            for(int i=0;i<BLOCK64;i++)
                for(int j=0;j<BLOCK64;j++){
                    r[i*BLOCKCOL+j] = a[i*BLOCKCOL+j] - b[i*BLOCKCOL+j];
                }
            updateMeta(r, BLOCK64);
        }
/* */
        void MatrixStdDouble::mat_sub(double* a, double* b, int n, double* r, int coreIdx)
        {
            combineMetaField(a,b,r);
            __builtin_prefetch(a + METAOFFSET);
            __builtin_prefetch(b + METAOFFSET);
            uint *metaa = (uint*) (a + METAOFFSET);
            uint *metab = (uint*) (b + METAOFFSET);
            uint *metar = (uint*) (r + METAOFFSET);
            uint16_t *metaad = (uint16_t*) (a + DETAILOFFSET);
            uint16_t *metabd = (uint16_t*) (b + DETAILOFFSET);

               __m512i ma0 = _mm512_load_epi32((void *)metaa);
            __m512i mb0 = _mm512_load_epi32((void *)metab);
            __m512i mr0 = _mm512_or_epi32(ma0, mb0);

                __m512d zero = _mm512_set1_pd(0);
                __m512d r00 = _mm512_set1_pd(TINY1);
                __m512d r01 = _mm512_set1_pd(-TINY1);

            int n8 = BLOCK64/8;
            for(int m=0; m<n8; m++){
                __mmask16 mth = _cvtu32_mask16(1<<m);
                for(int p=0; p<n8; p++){
                    uint mask16p = 1<<p;
                    __m512i mask512p = _mm512_set1_epi32(mask16p);
                    __mmask16 load16 = _mm512_test_epi32_mask(mr0,mask512p);
                    if(__builtin_expect(_kand_mask16(mth, load16) == 0, 1)){
                        int rmp = m*8*n+p*8;

                        for(int s=0; s<8; s++){
                            double* cc= r + ( m * 8 + s) * BLOCKCOL + p * 8;
                            _mm512_store_pd(cc,zero);
                        }
                    }else{
                        for(int s=0; s<8; s++){
                            int idx = ((m*8+s)/32) * DETAILSKIPSHORT + (m*8+s)%32;
                            int mn = ( m * 8 + s) * BLOCKCOL + p * 8;
                            if((metaad[idx] & mask16p) > 0){
                                __builtin_prefetch(a+mn);
                            }
                            if((metabd[idx] & mask16p) > 0){
                                __builtin_prefetch(b+mn);
                            }
                        }
                        for(int s=0; s<8; s++){
                            int mn = ( m * 8 + s) * BLOCKCOL + p * 8;
                __m512d av;
                __m512d bv;
                            av = _mm512_xor_pd(av,av);
                            bv = _mm512_xor_pd(bv,bv);
                            int idx = ((m*8+s)/32) * DETAILSKIPSHORT + (m*8+s)%32;
                            if((metaad[idx] & mask16p) > 0){
                                av = _mm512_load_pd(a+mn);
                            }
                            if((metabd[idx] & mask16p) > 0){
                                bv = _mm512_load_pd(b+mn);
                            }
                            __m512d cv = _mm512_sub_pd(av,bv);
                            _mm512_store_pd(r+mn,cv);
                        }
                    }
                }
            }
        }

/*
        void MatrixStdDouble::mat_copy(double* a, int n, double* r, int coreIdx)
        {
            copyMetaField(a,r);
            for(int i=0;i<BLOCK64;i++)
                for(int j=0;j<BLOCK64;j++){
                    r[i*BLOCKCOL+j] = a[i*BLOCKCOL+j];
                }
        }
/* */
        void MatrixStdDouble::mat_copy(double* a, int n, double* r, int coreIdx)
        {
        //    std::memcpy(r, a, ALLOCBLOCK); //sizeof(double)*n*n);
        //    updateMeta(r, n);
            int *metaa = (int*) (a + METAOFFSET);
            uint16_t *metaad = (uint16_t*) (a + DETAILOFFSET);
            int *metar = (int*) (r + METAOFFSET);
            copyMetaField(a, r);
            int n8 = n/8;
            for(int m=0; m<n8; m++){
                for(int p=0; p<n8; p++){
                    if((metaa[m] & MatrixStdDouble::mask16[p]) == 0){
                        for(int s=0; s<8; s++){
                            double* cc= r + ( m * 8 + s) * BLOCKCOL + p * 8;
                            for(int t=0; t<8; t++){
                                cc[t] = 0;
                            }
                        }
                    }else{
                        for(int s=0; s<8; s++){
                            int mn = ( m * 8 + s) * BLOCKCOL + p * 8;
                            int idx = ((m*8+s)/32) * DETAILSKIPSHORT + (m*8+s)%32;
                            double* aa = MatrixStdDouble::blanckdata;
                            if((metaad[idx] & MatrixStdDouble::mask16[p]) > 0){
                                aa = a + mn;
                            }
                            double* cc = r + mn;
                            for(int t=0; t<8; t++){
                                cc[t] = aa[t];
                            }
                        }
                    }
                }
            } 
        }

/*
        void MatrixStdDouble::mat_neg(double* b, int n, double* r, int coreIdx)
        {
            copyMetaField(b,r);
            for(int i=0;i<BLOCK64;i++)
                for(int j=0;j<BLOCK64;j++){
                    r[i*BLOCKCOL+j] =  - b[i*BLOCKCOL+j];
                }
        }
/* */
        void MatrixStdDouble::mat_neg(double* b, int n, double* r, int coreIdx)
        {
            __builtin_prefetch(b + METAOFFSET);
            uint *metab = (uint*) (b + METAOFFSET);
            uint16_t *metabd = (uint16_t*) (b + DETAILOFFSET);
            uint *metar = (uint*) (r + METAOFFSET);

            copyMetaField(b, r);
            __m512i mr0 = _mm512_load_epi32((void *)metab);

                __m512d zero = _mm512_set1_pd(0);
                __m512d av;
                __m512d bv;
                __m512d r00 = _mm512_set1_pd(TINY1);
                __m512d r01 = _mm512_set1_pd(-TINY1);

            int n8 = n/8;
            for(int m=0; m<n8; m++){
                __mmask16 mth = _cvtu32_mask16(1<<m);
                for(int p=0; p<n8; p++){
                    uint mask16p = 1<<p;
                    __m512i mask512p = _mm512_set1_epi32(mask16p);
                    __mmask16 load16 = _mm512_test_epi32_mask(mr0,mask512p);
                    if(__builtin_expect(_kand_mask16(mth, load16) == 0, 1)){

                        for(int s=0; s<8; s++){
                            double* cc= r + ( m * 8 + s) * BLOCKCOL + p * 8;
                            _mm512_store_pd(cc,zero);
                        }
                    }else{
                        for(int s=0; s<8; s++){
                            int mn = ( m * 8 + s) * BLOCKCOL + p * 8;
                            int idx = ((m*8+s)/32) * DETAILSKIPSHORT + (m*8+s)%32;
                            if((metabd[idx] & mask16p) > 0){
                                __builtin_prefetch(b+mn);
                            }
                        }
                        for(int s=0; s<8; s++){
                            int mn = ( m * 8 + s) * BLOCKCOL + p * 8;
                            int idx = ((m*8+s)/32) * DETAILSKIPSHORT + (m*8+s)%32;
                            bv = _mm512_xor_pd(bv,bv);
                            if((metabd[idx] & mask16p) > 0){
                                bv = _mm512_load_pd(b+mn);
                            __m512d cv = _mm512_sub_pd(zero,bv);
                      //      if(_cvtmask8_u32(_mm512_cmp_pd_mask(cv, r00, _CMP_NLE_UQ))>0 ||
                      //             _cvtmask8_u32(_mm512_cmp_pd_mask(cv, r01, _CMP_NGE_UQ))>0 )
                            _mm512_store_pd(r+mn,cv);
                            }
                        }
                    }
                }
            }
        }

        void mat_clear_x(double* b, int n)
        {
            __builtin_prefetch(b + METAOFFSET);
            uint *metab = (uint*) (b + METAOFFSET);
            unsigned short *metad = (unsigned short *) (b + DETAILOFFSET);
            __m512i mr1; 
            __m512i mr2; 
            __m512i mr3; 
            __m512i mr4; 
                    mr1 = _mm512_xor_epi32(mr1,mr1);
                    mr2 = _mm512_xor_epi32(mr2,mr2);
                    mr3 = _mm512_xor_epi32(mr3,mr3);
                    mr4 = _mm512_xor_epi32(mr4,mr4);

            __m512i mr0 = _mm512_load_epi32((void *)metab);

            __m512i lower8 = _mm512_set1_epi32(0xff);
            __mmask16 lowertest = _mm512_test_epi32_mask(mr0,lower8);
            if(_kand_mask16(lowertest,_cvtu32_mask16(0x3))){
                mr1 = _mm512_load_epi32((void *)(metad));
                }
            if(_kand_mask16(lowertest,_cvtu32_mask16(0xc))){
                mr2 = _mm512_load_epi32((void *)(metad+DETAILSKIPSHORT));
                }
            if(_kand_mask16(lowertest,_cvtu32_mask16(0x30))){
                mr3 = _mm512_load_epi32((void *)(metad+DETAILSKIPSHORT*2));
            }
            if(_kand_mask16(lowertest,_cvtu32_mask16(0xc0))){
                mr4 = _mm512_load_epi32((void *)(metad+DETAILSKIPSHORT*3));
            }

            int n8 = n/8;
            for(int m=0; m<n8; m++){
                __mmask16 mth = _cvtu32_mask16(1<<m);
                for(int p=0; p<n8; p++){
                    uint mask16p = 1<<p;
                    __m512i mask512p = _mm512_set1_epi32(mask16p);
                    __mmask16 load16 = _mm512_test_epi32_mask(mr0,mask512p);
                    int doff = (m/4) * DETAILSKIPSHORT;
                    if(__builtin_expect(_kand_mask16(mth, load16), 0)){
                        for(int s=0; s<4; s++){
                            int mn = ( m * 8 + s) * SUBCOL9 * 8 + p * 8;
                            if((metad[doff+(m%4)*8+s] & mask16p) > 0){
                                __builtin_prefetch(b+mn,1,1);
                            }
                        }
                        for(int s=0; s<4; s++){
                            int mn = ( m * 8 + s) * SUBCOL9 * 8 + p * 8;
                            __m512d bv;
                            bv = _mm512_xor_pd(bv,bv);
                            if((metad[doff+(m%4)*8+s] & mask16p) > 0){
                                _mm512_store_pd(b+mn,bv);
                            }
                        }
                        for(int s=4; s<8; s++){
                            int mn = ( m * 8 + s) * SUBCOL9 * 8 + p * 8;
                            if((metad[doff+(m%4)*8+s] & mask16p) > 0){
                                __builtin_prefetch(b+mn,1,1);
                            }
                        }
                        for(int s=4; s<8; s++){
                            int mn = ( m * 8 + s) * SUBCOL9 * 8 + p * 8;
                            __m512d bv;
                            bv = _mm512_xor_pd(bv,bv);
                            if((metad[doff+(m%4)*8+s] & mask16p) > 0){
                                _mm512_store_pd(b+mn,bv);
                            }
                        }
                    }
                }
            }
            std::memset(metab, 0, METASIZE);
            for(int j = 0; j < 4; j++){
                for(int i = 0; i<CACHE64/sizeof(unsigned short); i++) metad[i] = 0;
                metad += DETAILSKIPSHORT;
            }
        }
}

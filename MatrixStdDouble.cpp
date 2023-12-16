#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
//#include <immintrin.h>
#include "data.h"
#include "broker.h"
#include "MatrixStdDouble.h"
//#include "mkl.h"
#define TINY  1.0e-20
#define TINY2  1.0e-9

namespace vmatrix
{

        void MatrixStdDouble::blockMulOneBlas(double a[], double b[], double c[])
        {
            double alpha = 1.0; double beta = 1.0;
 //           cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 64, 64, 64, alpha, a, 64, b, 64, beta, c, 64);
            data::mulcount++;
        }

        void MatrixStdDouble::blockMulOneAvxBlockTrans(double a[], double b[], double c[], unsigned long msk1, unsigned long msk2) //64
        {
            const int n = 64;
            //Parallel.For(0, 4, m =>
            int seq;
                double* cinn;
                double* aaik;
                double* bbkp;
/*
     //       #pragma omp parallel for num_threads(12) private(cinn,aaik,bbkp)
            #pragma omp parallel for private(cinn,aaik,bbkp)
                 for(seq = 0; seq<64; seq++)
            {
                int m = seq / 8;
                int p = seq % 8;

                unsigned long am = msk1 >> m*8;
                unsigned long bm = msk2 >> p*8;
                unsigned long m255 = 255;
                am = am & m255;
                bm = bm & m255;
                unsigned long m7 = am & bm;
                if(m7 != 0)
                {
                    int i = m * 8 ;
                        int inn = i*n + p * 8;
                        cinn = c+inn;

                           __m512d r16 = _mm512_load_pd ((void const*) (cinn));
                           __m512d r17 = _mm512_load_pd ((void const*) (cinn+64));
                           __m512d r18 = _mm512_load_pd ((void const*) (cinn+128));
                           __m512d r19 = _mm512_load_pd ((void const*) (cinn+192));
                           __m512d r20 = _mm512_load_pd ((void const*) (cinn+256));
                           __m512d r21 = _mm512_load_pd ((void const*) (cinn+320));
                           __m512d r22 = _mm512_load_pd ((void const*) (cinn+384));
                           __m512d r23 = _mm512_load_pd ((void const*) (cinn+448));

                        for (int k = 0; k < 8; k++)
                    {
                        if((m7&1)==0){
                            m7 = m7 >>1;
                            continue;
                        }
                        m7 = m7>>1;
                        int k8 = k*8;
                        int aik = k8*n + i;

                        int bkp = k8 * n + p * 8;
                         aaik = a+aik;
                         bbkp = b+bkp;

                           __m512d r08 = _mm512_load_pd ((void const*) (bbkp));
                           __m512d r00 = _mm512_set1_pd(aaik[0]);
                           __m512d r01 = _mm512_set1_pd(aaik[1]);
                           __m512d r02 = _mm512_set1_pd(aaik[2]);
                           __m512d r03 = _mm512_set1_pd(aaik[3]);
                           __m512d r04 = _mm512_set1_pd(aaik[4]);
                           __m512d r05 = _mm512_set1_pd(aaik[5]);
                           __m512d r06 = _mm512_set1_pd(aaik[6]);
                           __m512d r07 = _mm512_set1_pd(aaik[7]);

                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r01, r08, r17);
                           r18 = _mm512_fmadd_pd (r02, r08, r18);
                           r19 = _mm512_fmadd_pd (r03, r08, r19);
                           r20 = _mm512_fmadd_pd (r04, r08, r20);
                           r21 = _mm512_fmadd_pd (r05, r08, r21);
                           r22 = _mm512_fmadd_pd (r06, r08, r22);
                           r23 = _mm512_fmadd_pd (r07, r08, r23);

                           __m512d r09 = _mm512_load_pd ((void const*) (bbkp+64));
                           __m512d r24 = _mm512_set1_pd(aaik[64]);
                           __m512d r25 = _mm512_set1_pd(aaik[65]);
                           __m512d r26 = _mm512_set1_pd(aaik[66]);
                           __m512d r27 = _mm512_set1_pd(aaik[67]);
                           __m512d r28 = _mm512_set1_pd(aaik[68]);
                           __m512d r29 = _mm512_set1_pd(aaik[69]);
                           __m512d r30 = _mm512_set1_pd(aaik[70]);
                           __m512d r31 = _mm512_set1_pd(aaik[71]);
                           r16 = _mm512_fmadd_pd (r24, r09, r16);
                           r17 = _mm512_fmadd_pd (r25, r09, r17);
                           r18 = _mm512_fmadd_pd (r26, r09, r18);
                           r19 = _mm512_fmadd_pd (r27, r09, r19);
                           r20 = _mm512_fmadd_pd (r28, r09, r20);
                           r21 = _mm512_fmadd_pd (r29, r09, r21);
                           r22 = _mm512_fmadd_pd (r30, r09, r22);
                           r23 = _mm512_fmadd_pd (r31, r09, r23);

                           __m512d r10 = _mm512_load_pd ((void const*) (bbkp+128));
                            r00 = _mm512_set1_pd(aaik[128]);
                            r01 = _mm512_set1_pd(aaik[129]);
                            r02 = _mm512_set1_pd(aaik[130]);
                            r03 = _mm512_set1_pd(aaik[131]);
                            r04 = _mm512_set1_pd(aaik[132]);
                            r05 = _mm512_set1_pd(aaik[133]);
                            r06 = _mm512_set1_pd(aaik[134]);
                            r07 = _mm512_set1_pd(aaik[135]);
                           r16 = _mm512_fmadd_pd (r00, r10, r16);
                           r17 = _mm512_fmadd_pd (r01, r10, r17);
                           r18 = _mm512_fmadd_pd (r02, r10, r18);
                           r19 = _mm512_fmadd_pd (r03, r10, r19);
                           r20 = _mm512_fmadd_pd (r04, r10, r20);
                           r21 = _mm512_fmadd_pd (r05, r10, r21);
                           r22 = _mm512_fmadd_pd (r06, r10, r22);
                           r23 = _mm512_fmadd_pd (r07, r10, r23);

                           __m512d r11 = _mm512_load_pd ((void const*) (bbkp+192));
                            r24 = _mm512_set1_pd(aaik[192]);
                            r25 = _mm512_set1_pd(aaik[193]);
                            r26 = _mm512_set1_pd(aaik[194]);
                            r27 = _mm512_set1_pd(aaik[195]);
                            r28 = _mm512_set1_pd(aaik[196]);
                            r29 = _mm512_set1_pd(aaik[197]);
                            r30 = _mm512_set1_pd(aaik[198]);
                            r31 = _mm512_set1_pd(aaik[199]);
                           r16 = _mm512_fmadd_pd (r24, r11, r16);
                           r17 = _mm512_fmadd_pd (r25, r11, r17);
                           r18 = _mm512_fmadd_pd (r26, r11, r18);
                           r19 = _mm512_fmadd_pd (r27, r11, r19);
                           r20 = _mm512_fmadd_pd (r28, r11, r20);
                           r21 = _mm512_fmadd_pd (r29, r11, r21);
                           r22 = _mm512_fmadd_pd (r30, r11, r22);
                           r23 = _mm512_fmadd_pd (r31, r11, r23);

                           __m512d r12 = _mm512_load_pd ((void const*) (bbkp+256));
                            r00 = _mm512_set1_pd(aaik[256]);
                            r01 = _mm512_set1_pd(aaik[257]);
                            r02 = _mm512_set1_pd(aaik[258]);
                            r03 = _mm512_set1_pd(aaik[259]);
                            r04 = _mm512_set1_pd(aaik[260]);
                            r05 = _mm512_set1_pd(aaik[261]);
                            r06 = _mm512_set1_pd(aaik[262]);
                            r07 = _mm512_set1_pd(aaik[263]);
                           r16 = _mm512_fmadd_pd (r00, r12, r16);
                           r17 = _mm512_fmadd_pd (r01, r12, r17);
                           r18 = _mm512_fmadd_pd (r02, r12, r18);
                           r19 = _mm512_fmadd_pd (r03, r12, r19);
                           r20 = _mm512_fmadd_pd (r04, r12, r20);
                           r21 = _mm512_fmadd_pd (r05, r12, r21);
                           r22 = _mm512_fmadd_pd (r06, r12, r22);
                           r23 = _mm512_fmadd_pd (r07, r12, r23);

                           __m512d r13 = _mm512_load_pd ((void const*) (bbkp+320));
                            r24 = _mm512_set1_pd(aaik[320]);
                            r25 = _mm512_set1_pd(aaik[321]);
                            r26 = _mm512_set1_pd(aaik[322]);
                            r27 = _mm512_set1_pd(aaik[323]);
                            r28 = _mm512_set1_pd(aaik[324]);
                            r29 = _mm512_set1_pd(aaik[325]);
                            r30 = _mm512_set1_pd(aaik[326]);
                            r31 = _mm512_set1_pd(aaik[327]);
                           r16 = _mm512_fmadd_pd (r24, r13, r16);
                           r17 = _mm512_fmadd_pd (r25, r13, r17);
                           r18 = _mm512_fmadd_pd (r26, r13, r18);
                           r19 = _mm512_fmadd_pd (r27, r13, r19);
                           r20 = _mm512_fmadd_pd (r28, r13, r20);
                           r21 = _mm512_fmadd_pd (r29, r13, r21);
                           r22 = _mm512_fmadd_pd (r30, r13, r22);
                           r23 = _mm512_fmadd_pd (r31, r13, r23);

                           __m512d r14 = _mm512_load_pd ((void const*) (bbkp+384));
                            r00 = _mm512_set1_pd(aaik[384]);
                            r01 = _mm512_set1_pd(aaik[385]);
                            r02 = _mm512_set1_pd(aaik[386]);
                            r03 = _mm512_set1_pd(aaik[387]);
                            r04 = _mm512_set1_pd(aaik[388]);
                            r05 = _mm512_set1_pd(aaik[389]);
                            r06 = _mm512_set1_pd(aaik[390]);
                            r07 = _mm512_set1_pd(aaik[391]);
                           r16 = _mm512_fmadd_pd (r00, r14, r16);
                           r17 = _mm512_fmadd_pd (r01, r14, r17);
                           r18 = _mm512_fmadd_pd (r02, r14, r18);
                           r19 = _mm512_fmadd_pd (r03, r14, r19);
                           r20 = _mm512_fmadd_pd (r04, r14, r20);
                           r21 = _mm512_fmadd_pd (r05, r14, r21);
                           r22 = _mm512_fmadd_pd (r06, r14, r22);
                           r23 = _mm512_fmadd_pd (r07, r14, r23);

                           __m512d r15 = _mm512_load_pd ((void const*) (bbkp+448));
                            r24 = _mm512_set1_pd(aaik[448]);
                            r25 = _mm512_set1_pd(aaik[449]);
                            r26 = _mm512_set1_pd(aaik[450]);
                            r27 = _mm512_set1_pd(aaik[451]);
                            r28 = _mm512_set1_pd(aaik[452]);
                            r29 = _mm512_set1_pd(aaik[453]);
                            r30 = _mm512_set1_pd(aaik[454]);
                            r31 = _mm512_set1_pd(aaik[455]);
                           r16 = _mm512_fmadd_pd (r24, r15, r16);
                           r17 = _mm512_fmadd_pd (r25, r15, r17);
                           r18 = _mm512_fmadd_pd (r26, r15, r18);
                           r19 = _mm512_fmadd_pd (r27, r15, r19);
                           r20 = _mm512_fmadd_pd (r28, r15, r20);
                           r21 = _mm512_fmadd_pd (r29, r15, r21);
                           r22 = _mm512_fmadd_pd (r30, r15, r22);
                           r23 = _mm512_fmadd_pd (r31, r15, r23);

                    }
                           _mm512_store_pd ((void*) (cinn), r16);
                           _mm512_store_pd ((void*) (cinn+64), r17);
                           _mm512_store_pd ((void*) (cinn+128), r18);
                           _mm512_store_pd ((void*) (cinn+192), r19);
                           _mm512_store_pd ((void*) (cinn+256), r20);
                           _mm512_store_pd ((void*) (cinn+320), r21);
                           _mm512_store_pd ((void*) (cinn+384), r22);
                           _mm512_store_pd ((void*) (cinn+448), r23);
                }
            }
           // );
 */           data::mulcount++;
        }


        void MatrixStdDouble::blockMulOneAvxBlock(double a[], double b[], double c[], unsigned long msk) //64
        {
            const int n = 64;
            //Parallel.For(0, 4, m =>
            int seq;
                double* cinn;
                double* aaik;
                double* bbkp;
/*
     //       #pragma omp parallel for num_threads(12) private(cinn,aaik,bbkp)
            #pragma omp parallel for private(cinn,aaik,bbkp)
                 for(seq = 0; seq<64; seq++)
            {
                int m = seq / 8;
                int p = seq % 8;

                {
                    int i = m * 8 ;
                        int inn = i*n + p * 8;
                        cinn = c+inn;

                           __m512d r16 = _mm512_load_pd ((void const*) (cinn));
                           __m512d r17 = _mm512_load_pd ((void const*) (cinn+64));
                           __m512d r18 = _mm512_load_pd ((void const*) (cinn+128));
                           __m512d r19 = _mm512_load_pd ((void const*) (cinn+192));
                           __m512d r20 = _mm512_load_pd ((void const*) (cinn+256));
                           __m512d r21 = _mm512_load_pd ((void const*) (cinn+320));
                           __m512d r22 = _mm512_load_pd ((void const*) (cinn+384));
                           __m512d r23 = _mm512_load_pd ((void const*) (cinn+448));

                        for (int k = 0; k < 8; k++)
                    {
                        int k8 = k*8;
                        int aik = i*n + k8;

                        int bkp = k8 * n + p * 8;
                         aaik = a+aik;
                         bbkp = b+bkp;

                           __m512d r00 = _mm512_set1_pd(aaik[0]);
                           __m512d r01 = _mm512_set1_pd(aaik[64]);
                           __m512d r02 = _mm512_set1_pd(aaik[128]);
                           __m512d r03 = _mm512_set1_pd(aaik[192]);
                           __m512d r04 = _mm512_set1_pd(aaik[256]);
                           __m512d r05 = _mm512_set1_pd(aaik[320]);
                           __m512d r06 = _mm512_set1_pd(aaik[384]);
                           __m512d r07 = _mm512_set1_pd(aaik[448]);

                           __m512d r08 = _mm512_load_pd ((void const*) (bbkp));
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r01, r08, r17);
                           r18 = _mm512_fmadd_pd (r02, r08, r18);
                           r19 = _mm512_fmadd_pd (r03, r08, r19);
                           r20 = _mm512_fmadd_pd (r04, r08, r20);
                           r21 = _mm512_fmadd_pd (r05, r08, r21);
                           r22 = _mm512_fmadd_pd (r06, r08, r22);
                           r23 = _mm512_fmadd_pd (r07, r08, r23);

                           __m512d r24 = _mm512_set1_pd(aaik[1]);
                           __m512d r25 = _mm512_set1_pd(aaik[65]);
                           __m512d r26 = _mm512_set1_pd(aaik[129]);
                           __m512d r27 = _mm512_set1_pd(aaik[193]);
                           __m512d r28 = _mm512_set1_pd(aaik[257]);
                           __m512d r29 = _mm512_set1_pd(aaik[321]);
                           __m512d r30 = _mm512_set1_pd(aaik[385]);
                           __m512d r31 = _mm512_set1_pd(aaik[449]);
                           __m512d r09 = _mm512_load_pd ((void const*) (bbkp+64));
                           r16 = _mm512_fmadd_pd (r24, r09, r16);
                           r17 = _mm512_fmadd_pd (r25, r09, r17);
                           r18 = _mm512_fmadd_pd (r26, r09, r18);
                           r19 = _mm512_fmadd_pd (r27, r09, r19);
                           r20 = _mm512_fmadd_pd (r28, r09, r20);
                           r21 = _mm512_fmadd_pd (r29, r09, r21);
                           r22 = _mm512_fmadd_pd (r30, r09, r22);
                           r23 = _mm512_fmadd_pd (r31, r09, r23);

                            r00 = _mm512_set1_pd(aaik[2]);
                            r01 = _mm512_set1_pd(aaik[66]);
                            r02 = _mm512_set1_pd(aaik[130]);
                            r03 = _mm512_set1_pd(aaik[194]);
                            r04 = _mm512_set1_pd(aaik[258]);
                            r05 = _mm512_set1_pd(aaik[322]);
                            r06 = _mm512_set1_pd(aaik[386]);
                            r07 = _mm512_set1_pd(aaik[450]);
                           __m512d r10 = _mm512_load_pd ((void const*) (bbkp+128));
                           r16 = _mm512_fmadd_pd (r00, r10, r16);
                           r17 = _mm512_fmadd_pd (r01, r10, r17);
                           r18 = _mm512_fmadd_pd (r02, r10, r18);
                           r19 = _mm512_fmadd_pd (r03, r10, r19);
                           r20 = _mm512_fmadd_pd (r04, r10, r20);
                           r21 = _mm512_fmadd_pd (r05, r10, r21);
                           r22 = _mm512_fmadd_pd (r06, r10, r22);
                           r23 = _mm512_fmadd_pd (r07, r10, r23);

                            r24 = _mm512_set1_pd(aaik[3]);
                            r25 = _mm512_set1_pd(aaik[67]);
                            r26 = _mm512_set1_pd(aaik[131]);
                            r27 = _mm512_set1_pd(aaik[195]);
                            r28 = _mm512_set1_pd(aaik[259]);
                            r29 = _mm512_set1_pd(aaik[323]);
                            r30 = _mm512_set1_pd(aaik[387]);
                            r31 = _mm512_set1_pd(aaik[451]);
                           __m512d r11 = _mm512_load_pd ((void const*) (bbkp+192));
                           r16 = _mm512_fmadd_pd (r24, r11, r16);
                           r17 = _mm512_fmadd_pd (r25, r11, r17);
                           r18 = _mm512_fmadd_pd (r26, r11, r18);
                           r19 = _mm512_fmadd_pd (r27, r11, r19);
                           r20 = _mm512_fmadd_pd (r28, r11, r20);
                           r21 = _mm512_fmadd_pd (r29, r11, r21);
                           r22 = _mm512_fmadd_pd (r30, r11, r22);
                           r23 = _mm512_fmadd_pd (r31, r11, r23);

                            r00 = _mm512_set1_pd(aaik[4]);
                            r01 = _mm512_set1_pd(aaik[68]);
                            r02 = _mm512_set1_pd(aaik[132]);
                            r03 = _mm512_set1_pd(aaik[196]);
                            r04 = _mm512_set1_pd(aaik[260]);
                            r05 = _mm512_set1_pd(aaik[324]);
                            r06 = _mm512_set1_pd(aaik[388]);
                            r07 = _mm512_set1_pd(aaik[452]);
                           __m512d r12 = _mm512_load_pd ((void const*) (bbkp+256));
                           r16 = _mm512_fmadd_pd (r00, r12, r16);
                           r17 = _mm512_fmadd_pd (r01, r12, r17);
                           r18 = _mm512_fmadd_pd (r02, r12, r18);
                           r19 = _mm512_fmadd_pd (r03, r12, r19);
                           r20 = _mm512_fmadd_pd (r04, r12, r20);
                           r21 = _mm512_fmadd_pd (r05, r12, r21);
                           r22 = _mm512_fmadd_pd (r06, r12, r22);
                           r23 = _mm512_fmadd_pd (r07, r12, r23);

                            r24 = _mm512_set1_pd(aaik[5]);
                            r25 = _mm512_set1_pd(aaik[69]);
                            r26 = _mm512_set1_pd(aaik[133]);
                            r27 = _mm512_set1_pd(aaik[197]);
                            r28 = _mm512_set1_pd(aaik[261]);
                            r29 = _mm512_set1_pd(aaik[325]);
                            r30 = _mm512_set1_pd(aaik[389]);
                            r31 = _mm512_set1_pd(aaik[453]);
                           __m512d r13 = _mm512_load_pd ((void const*) (bbkp+320));
                           r16 = _mm512_fmadd_pd (r24, r13, r16);
                           r17 = _mm512_fmadd_pd (r25, r13, r17);
                           r18 = _mm512_fmadd_pd (r26, r13, r18);
                           r19 = _mm512_fmadd_pd (r27, r13, r19);
                           r20 = _mm512_fmadd_pd (r28, r13, r20);
                           r21 = _mm512_fmadd_pd (r29, r13, r21);
                           r22 = _mm512_fmadd_pd (r30, r13, r22);
                           r23 = _mm512_fmadd_pd (r31, r13, r23);

                            r00 = _mm512_set1_pd(aaik[6]);
                            r01 = _mm512_set1_pd(aaik[70]);
                            r02 = _mm512_set1_pd(aaik[134]);
                            r03 = _mm512_set1_pd(aaik[198]);
                            r04 = _mm512_set1_pd(aaik[262]);
                            r05 = _mm512_set1_pd(aaik[326]);
                            r06 = _mm512_set1_pd(aaik[390]);
                            r07 = _mm512_set1_pd(aaik[454]);
                           __m512d r14 = _mm512_load_pd ((void const*) (bbkp+384));
                           r16 = _mm512_fmadd_pd (r00, r14, r16);
                           r17 = _mm512_fmadd_pd (r01, r14, r17);
                           r18 = _mm512_fmadd_pd (r02, r14, r18);
                           r19 = _mm512_fmadd_pd (r03, r14, r19);
                           r20 = _mm512_fmadd_pd (r04, r14, r20);
                           r21 = _mm512_fmadd_pd (r05, r14, r21);
                           r22 = _mm512_fmadd_pd (r06, r14, r22);
                           r23 = _mm512_fmadd_pd (r07, r14, r23);

                            r24 = _mm512_set1_pd(aaik[7]);
                            r25 = _mm512_set1_pd(aaik[71]);
                            r26 = _mm512_set1_pd(aaik[135]);
                            r27 = _mm512_set1_pd(aaik[199]);
                            r28 = _mm512_set1_pd(aaik[263]);
                            r29 = _mm512_set1_pd(aaik[327]);
                            r30 = _mm512_set1_pd(aaik[391]);
                            r31 = _mm512_set1_pd(aaik[455]);
                           __m512d r15 = _mm512_load_pd ((void const*) (bbkp+448));
                           r16 = _mm512_fmadd_pd (r24, r15, r16);
                           r17 = _mm512_fmadd_pd (r25, r15, r17);
                           r18 = _mm512_fmadd_pd (r26, r15, r18);
                           r19 = _mm512_fmadd_pd (r27, r15, r19);
                           r20 = _mm512_fmadd_pd (r28, r15, r20);
                           r21 = _mm512_fmadd_pd (r29, r15, r21);
                           r22 = _mm512_fmadd_pd (r30, r15, r22);
                           r23 = _mm512_fmadd_pd (r31, r15, r23);

                    }
                           _mm512_store_pd ((void*) (cinn), r16);
                           _mm512_store_pd ((void*) (cinn+64), r17);
                           _mm512_store_pd ((void*) (cinn+128), r18);
                           _mm512_store_pd ((void*) (cinn+192), r19);
                           _mm512_store_pd ((void*) (cinn+256), r20);
                           _mm512_store_pd ((void*) (cinn+320), r21);
                           _mm512_store_pd ((void*) (cinn+384), r22);
                           _mm512_store_pd ((void*) (cinn+448), r23);
                }
            }
           // );
 */           data::mulcount++;
        }

       void MatrixStdDouble::blockMulOneAvx(double a[], double b[], double c[]) //64
        {
            const int n = 64;
            //Parallel.For(0, 4, m =>
            int m;
  /*          #pragma omp parallel for
                  for (m = 0; m < 16; m++)
            {
                for (int p = 0; p < 4; p++)
                {
                    int i = m * 4 + p;
                        int inn = i*n;

                           __m512d r16 = _mm512_load_pd ((void const*) (c+inn));
                           __m512d r17 = _mm512_load_pd ((void const*) (c+inn+8));
                           __m512d r18 = _mm512_load_pd ((void const*) (c+inn+16));
                           __m512d r19 = _mm512_load_pd ((void const*) (c+inn+24));
                           __m512d r20 = _mm512_load_pd ((void const*) (c+inn+32));
                           __m512d r21 = _mm512_load_pd ((void const*) (c+inn+40));
                           __m512d r22 = _mm512_load_pd ((void const*) (c+inn+48));
                           __m512d r23 = _mm512_load_pd ((void const*) (c+inn+56));

                        for (int k = 0; k < n; k+=2)
                    {
                        double aink = a[i*n+k];
                        double aink1 = a[i*n+k+1];
                        int kn = k*n;
                    //for (int j = 0; j < n; j += 8)
                        {
                           __m512d r00 = _mm512_set1_pd(aink);
                           __m512d r01 = _mm512_set1_pd(aink1);

                           __m512d r08 = _mm512_load_pd ((void const*) (b+kn));
                           __m512d r09 = _mm512_load_pd ((void const*) (b+kn+8));
                           __m512d r10 = _mm512_load_pd ((void const*) (b+kn+16));
                           __m512d r11 = _mm512_load_pd ((void const*) (b+kn+24));
                           __m512d r12 = _mm512_load_pd ((void const*) (b+kn+32));
                           __m512d r13 = _mm512_load_pd ((void const*) (b+kn+40));
                           __m512d r14 = _mm512_load_pd ((void const*) (b+kn+48));
                           __m512d r15 = _mm512_load_pd ((void const*) (b+kn+56));

                           __m512d r24 = _mm512_load_pd ((void const*) (b+kn + 64));
                           __m512d r25 = _mm512_load_pd ((void const*) (b+kn+72));
                           __m512d r26 = _mm512_load_pd ((void const*) (b+kn+80));
                           __m512d r27 = _mm512_load_pd ((void const*) (b+kn+88));
                           __m512d r28 = _mm512_load_pd ((void const*) (b+kn+96));
                           __m512d r29 = _mm512_load_pd ((void const*) (b+kn+104));
                           __m512d r30 = _mm512_load_pd ((void const*) (b+kn+112));
                           __m512d r31 = _mm512_load_pd ((void const*) (b+kn+120));
                      
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r00, r09, r17);
                           r18 = _mm512_fmadd_pd (r00, r10, r18);
                           r19 = _mm512_fmadd_pd (r00, r11, r19);
                           r20 = _mm512_fmadd_pd (r00, r12, r20);
                           r21 = _mm512_fmadd_pd (r00, r13, r21);
                           r22 = _mm512_fmadd_pd (r00, r14, r22);
                           r23 = _mm512_fmadd_pd (r00, r15, r23);

                           r16 = _mm512_fmadd_pd (r01, r24, r16);
                           r17 = _mm512_fmadd_pd (r01, r25, r17);
                           r18 = _mm512_fmadd_pd (r01, r26, r18);
                           r19 = _mm512_fmadd_pd (r01, r27, r19);
                           r20 = _mm512_fmadd_pd (r01, r28, r20);
                           r21 = _mm512_fmadd_pd (r01, r29, r21);
                           r22 = _mm512_fmadd_pd (r01, r30, r22);
                           r23 = _mm512_fmadd_pd (r01, r31, r23);
                     //       c[inn + j] += aink * b[kn + j];
                        }
                    }
                           _mm512_store_pd ((void*) (c+inn), r16);
                           _mm512_store_pd ((void*) (c+inn+8), r17);
                           _mm512_store_pd ((void*) (c+inn+16), r18);
                           _mm512_store_pd ((void*) (c+inn+24), r19);
                           _mm512_store_pd ((void*) (c+inn+32), r20);
                           _mm512_store_pd ((void*) (c+inn+40), r21);
                           _mm512_store_pd ((void*) (c+inn+48), r22);
                           _mm512_store_pd ((void*) (c+inn+56), r23);
                }
            }
           // );
   */         data::mulcount++;
        }

        void MatrixStdDouble::blockMulOneNegAvx(double a[], double b[], double c[]) //64
        {
            const int n = 64;
            //Parallel.For(0, 4, m =>
            int m;
    /*        #pragma omp parallel for
                  for (m = 0; m < 4; m++)
            {
                for (int p = 0; p < 16; p++)
                {
                    int i = m * BLOCK16 + p;
                        int inn = i*n;

                           __m512d r16 = _mm512_load_pd ((void const*) (c+inn));
                           __m512d r17 = _mm512_load_pd ((void const*) (c+inn+8));
                           __m512d r18 = _mm512_load_pd ((void const*) (c+inn+16));
                           __m512d r19 = _mm512_load_pd ((void const*) (c+inn+24));
                           __m512d r20 = _mm512_load_pd ((void const*) (c+inn+32));
                           __m512d r21 = _mm512_load_pd ((void const*) (c+inn+40));
                           __m512d r22 = _mm512_load_pd ((void const*) (c+inn+48));
                           __m512d r23 = _mm512_load_pd ((void const*) (c+inn+56));

                        for (int k = 0; k < n; k+=2)
                    {
                        double aink = -a[i*n+k];
                        double aink1 = -a[i*n+k+1];
                        int kn = k*n;
                    //for (int j = 0; j < n; j += 8)
                        {
                           __m512d r00 = _mm512_set1_pd(aink);
                           __m512d r01 = _mm512_set1_pd(aink1);

                           __m512d r08 = _mm512_load_pd ((void const*) (b+kn));
                           __m512d r09 = _mm512_load_pd ((void const*) (b+kn+8));
                           __m512d r10 = _mm512_load_pd ((void const*) (b+kn+16));
                           __m512d r11 = _mm512_load_pd ((void const*) (b+kn+24));
                           __m512d r12 = _mm512_load_pd ((void const*) (b+kn+32));
                           __m512d r13 = _mm512_load_pd ((void const*) (b+kn+40));
                           __m512d r14 = _mm512_load_pd ((void const*) (b+kn+48));
                           __m512d r15 = _mm512_load_pd ((void const*) (b+kn+56));

                           __m512d r24 = _mm512_load_pd ((void const*) (b+kn + 64));
                           __m512d r25 = _mm512_load_pd ((void const*) (b+kn+72));
                           __m512d r26 = _mm512_load_pd ((void const*) (b+kn+80));
                           __m512d r27 = _mm512_load_pd ((void const*) (b+kn+88));
                           __m512d r28 = _mm512_load_pd ((void const*) (b+kn+96));
                           __m512d r29 = _mm512_load_pd ((void const*) (b+kn+104));
                           __m512d r30 = _mm512_load_pd ((void const*) (b+kn+112));
                           __m512d r31 = _mm512_load_pd ((void const*) (b+kn+120));
                      
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r00, r09, r17);
                           r18 = _mm512_fmadd_pd (r00, r10, r18);
                           r19 = _mm512_fmadd_pd (r00, r11, r19);
                           r20 = _mm512_fmadd_pd (r00, r12, r20);
                           r21 = _mm512_fmadd_pd (r00, r13, r21);
                           r22 = _mm512_fmadd_pd (r00, r14, r22);
                           r23 = _mm512_fmadd_pd (r00, r15, r23);

                           r16 = _mm512_fmadd_pd (r01, r24, r16);
                           r17 = _mm512_fmadd_pd (r01, r25, r17);
                           r18 = _mm512_fmadd_pd (r01, r26, r18);
                           r19 = _mm512_fmadd_pd (r01, r27, r19);
                           r20 = _mm512_fmadd_pd (r01, r28, r20);
                           r21 = _mm512_fmadd_pd (r01, r29, r21);
                           r22 = _mm512_fmadd_pd (r01, r30, r22);
                           r23 = _mm512_fmadd_pd (r01, r31, r23);
                     //       c[inn + j] += aink * b[kn + j];
                        }
                    }
                           _mm512_store_pd ((void*) (c+inn), r16);
                           _mm512_store_pd ((void*) (c+inn+8), r17);
                           _mm512_store_pd ((void*) (c+inn+16), r18);
                           _mm512_store_pd ((void*) (c+inn+24), r19);
                           _mm512_store_pd ((void*) (c+inn+32), r20);
                           _mm512_store_pd ((void*) (c+inn+40), r21);
                           _mm512_store_pd ((void*) (c+inn+48), r22);
                           _mm512_store_pd ((void*) (c+inn+56), r23);
                }
            }
           // );
     */       data::mulcount++;
        }


        void blockMulOneAvxBlock32(double a[], double b[], double c[], unsigned long msk)
        {
            const int n = 32;
            //Parallel.For(0, 4, m =>
            int seq;

      /*      #pragma omp parallel for
                 for(seq = 0; seq<16; seq++)
            {
                int m = seq / 4;
                int p = seq % 4;

                {
                    int i = m * 8 ;
                        int inn = i*n + p * 8;

                           __m512d r16 = _mm512_load_pd ((void const*) (c+inn));
                           __m512d r17 = _mm512_load_pd ((void const*) (c+inn+32));
                           __m512d r18 = _mm512_load_pd ((void const*) (c+inn+64));
                           __m512d r19 = _mm512_load_pd ((void const*) (c+inn+96));
                           __m512d r20 = _mm512_load_pd ((void const*) (c+inn+128));
                           __m512d r21 = _mm512_load_pd ((void const*) (c+inn+160));
                           __m512d r22 = _mm512_load_pd ((void const*) (c+inn+192));
                           __m512d r23 = _mm512_load_pd ((void const*) (c+inn+224));

                        for (int k = 0; k < 8; k++)
                    {
                        int k8 = k*4;
                        int aik = i*n + k8;

                        int bkp = k8 * n + p * 8;

                           __m512d r00 = _mm512_set1_pd(a[aik]);
                           __m512d r01 = _mm512_set1_pd(a[aik+32]);
                           __m512d r02 = _mm512_set1_pd(a[aik+64]);
                           __m512d r03 = _mm512_set1_pd(a[aik+96]);
                           __m512d r04 = _mm512_set1_pd(a[aik+128]);
                           __m512d r05 = _mm512_set1_pd(a[aik+160]);
                           __m512d r06 = _mm512_set1_pd(a[aik+192]);
                           __m512d r07 = _mm512_set1_pd(a[aik+224]);

                           __m512d r08 = _mm512_load_pd ((void const*) (b+bkp));
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r01, r08, r17);
                           r18 = _mm512_fmadd_pd (r02, r08, r18);
                           r19 = _mm512_fmadd_pd (r03, r08, r19);
                           r20 = _mm512_fmadd_pd (r04, r08, r20);
                           r21 = _mm512_fmadd_pd (r05, r08, r21);
                           r22 = _mm512_fmadd_pd (r06, r08, r22);
                           r23 = _mm512_fmadd_pd (r07, r08, r23);

                           __m512d r24 = _mm512_set1_pd(a[aik+1]);
                           __m512d r25 = _mm512_set1_pd(a[aik+33]);
                           __m512d r26 = _mm512_set1_pd(a[aik+65]);
                           __m512d r27 = _mm512_set1_pd(a[aik+97]);
                           __m512d r28 = _mm512_set1_pd(a[aik+129]);
                           __m512d r29 = _mm512_set1_pd(a[aik+161]);
                           __m512d r30 = _mm512_set1_pd(a[aik+193]);
                           __m512d r31 = _mm512_set1_pd(a[aik+225]);
                           __m512d r09 = _mm512_load_pd ((void const*) (b+bkp+32));
                           r16 = _mm512_fmadd_pd (r24, r09, r16);
                           r17 = _mm512_fmadd_pd (r25, r09, r17);
                           r18 = _mm512_fmadd_pd (r26, r09, r18);
                           r19 = _mm512_fmadd_pd (r27, r09, r19);
                           r20 = _mm512_fmadd_pd (r28, r09, r20);
                           r21 = _mm512_fmadd_pd (r29, r09, r21);
                           r22 = _mm512_fmadd_pd (r30, r09, r22);
                           r23 = _mm512_fmadd_pd (r31, r09, r23);

                            r00 = _mm512_set1_pd(a[aik+2]);
                            r01 = _mm512_set1_pd(a[aik+34]);
                            r02 = _mm512_set1_pd(a[aik+66]);
                            r03 = _mm512_set1_pd(a[aik+98]);
                            r04 = _mm512_set1_pd(a[aik+130]);
                            r05 = _mm512_set1_pd(a[aik+162]);
                            r06 = _mm512_set1_pd(a[aik+194]);
                            r07 = _mm512_set1_pd(a[aik+226]);
                           __m512d r10 = _mm512_load_pd ((void const*) (b+bkp+64));
                           r16 = _mm512_fmadd_pd (r00, r10, r16);
                           r17 = _mm512_fmadd_pd (r01, r10, r17);
                           r18 = _mm512_fmadd_pd (r02, r10, r18);
                           r19 = _mm512_fmadd_pd (r03, r10, r19);
                           r20 = _mm512_fmadd_pd (r04, r10, r20);
                           r21 = _mm512_fmadd_pd (r05, r10, r21);
                           r22 = _mm512_fmadd_pd (r06, r10, r22);
                           r23 = _mm512_fmadd_pd (r07, r10, r23);

                            r24 = _mm512_set1_pd(a[aik+3]);
                            r25 = _mm512_set1_pd(a[aik+35]);
                            r26 = _mm512_set1_pd(a[aik+67]);
                            r27 = _mm512_set1_pd(a[aik+99]);
                            r28 = _mm512_set1_pd(a[aik+131]);
                            r29 = _mm512_set1_pd(a[aik+163]);
                            r30 = _mm512_set1_pd(a[aik+195]);
                            r31 = _mm512_set1_pd(a[aik+227]);
                           __m512d r11 = _mm512_load_pd ((void const*) (b+bkp+96));
                           r16 = _mm512_fmadd_pd (r24, r11, r16);
                           r17 = _mm512_fmadd_pd (r25, r11, r17);
                           r18 = _mm512_fmadd_pd (r26, r11, r18);
                           r19 = _mm512_fmadd_pd (r27, r11, r19);
                           r20 = _mm512_fmadd_pd (r28, r11, r20);
                           r21 = _mm512_fmadd_pd (r29, r11, r21);
                           r22 = _mm512_fmadd_pd (r30, r11, r22);
                           r23 = _mm512_fmadd_pd (r31, r11, r23);

                            r00 = _mm512_set1_pd(a[aik+4]);
                            r01 = _mm512_set1_pd(a[aik+36]);
                            r02 = _mm512_set1_pd(a[aik+68]);
                            r03 = _mm512_set1_pd(a[aik+100]);
                            r04 = _mm512_set1_pd(a[aik+132]);
                            r05 = _mm512_set1_pd(a[aik+164]);
                            r06 = _mm512_set1_pd(a[aik+196]);
                            r07 = _mm512_set1_pd(a[aik+228]);
                    }
                           _mm512_store_pd ((void*) (c+inn), r16);
                           _mm512_store_pd ((void*) (c+inn+32), r17);
                           _mm512_store_pd ((void*) (c+inn+64), r18);
                           _mm512_store_pd ((void*) (c+inn+96), r19);
                           _mm512_store_pd ((void*) (c+inn+128), r20);
                           _mm512_store_pd ((void*) (c+inn+160), r21);
                           _mm512_store_pd ((void*) (c+inn+192), r22);
                           _mm512_store_pd ((void*) (c+inn+224), r23);
                }
            }
       */     data::mulcount++;
        }

        void blockMulOneAvx32(double a[], double b[], double c[]) // 32
        {
            const int n = 32;
            int m;
        /*    #pragma omp parallel for
                  for (m = 0; m < 8; m++)
            {
                for (int p = 0; p < 4; p++)
                {
                    int i = m * 4 + p;
                        int inn = i*n;

                           __m512d r16 = _mm512_load_pd ((void const*) (c+inn));
                           __m512d r17 = _mm512_load_pd ((void const*) (c+inn+8));
                           __m512d r18 = _mm512_load_pd ((void const*) (c+inn+16));
                           __m512d r19 = _mm512_load_pd ((void const*) (c+inn+24));

                        for (int k = 0; k < n; k+=4)
                    {
                        double aink = a[i*n+k];
                        double aink1 = a[i*n+k+1];
                        double aink2 = a[i*n+k+2];
                        double aink3 = a[i*n+k+3];
                        int kn = k*n;
                           __m512d r00 = _mm512_set1_pd(aink);
                           __m512d r01 = _mm512_set1_pd(aink1);
                           __m512d r02 = _mm512_set1_pd(aink2);
                           __m512d r03 = _mm512_set1_pd(aink3);

                           __m512d r08 = _mm512_load_pd ((void const*) (b+kn));
                           __m512d r09 = _mm512_load_pd ((void const*) (b+kn+8));
                           __m512d r10 = _mm512_load_pd ((void const*) (b+kn+16));
                           __m512d r11 = _mm512_load_pd ((void const*) (b+kn+24));
                           __m512d r12 = _mm512_load_pd ((void const*) (b+kn+32));
                           __m512d r13 = _mm512_load_pd ((void const*) (b+kn+40));
                           __m512d r14 = _mm512_load_pd ((void const*) (b+kn+48));
                           __m512d r15 = _mm512_load_pd ((void const*) (b+kn+56));

                           __m512d r24 = _mm512_load_pd ((void const*) (b+kn+64));
                           __m512d r25 = _mm512_load_pd ((void const*) (b+kn+72));
                           __m512d r26 = _mm512_load_pd ((void const*) (b+kn+80));
                           __m512d r27 = _mm512_load_pd ((void const*) (b+kn+88));
                           __m512d r28 = _mm512_load_pd ((void const*) (b+kn+96));
                           __m512d r29 = _mm512_load_pd ((void const*) (b+kn+104));
                           __m512d r30 = _mm512_load_pd ((void const*) (b+kn+112));
                           __m512d r31 = _mm512_load_pd ((void const*) (b+kn+120));
                      
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r00, r09, r17);
                           r18 = _mm512_fmadd_pd (r00, r10, r18);
                           r19 = _mm512_fmadd_pd (r00, r11, r19);
                           r16 = _mm512_fmadd_pd (r01, r12, r16);
                           r17 = _mm512_fmadd_pd (r01, r13, r17);
                           r18 = _mm512_fmadd_pd (r01, r14, r18);
                           r19 = _mm512_fmadd_pd (r01, r15, r19);

                           r16 = _mm512_fmadd_pd (r02, r24, r16);
                           r17 = _mm512_fmadd_pd (r02, r25, r17);
                           r18 = _mm512_fmadd_pd (r02, r26, r18);
                           r19 = _mm512_fmadd_pd (r02, r27, r19);
                           r16 = _mm512_fmadd_pd (r03, r28, r16);
                           r17 = _mm512_fmadd_pd (r03, r29, r17);
                           r18 = _mm512_fmadd_pd (r03, r30, r18);
                           r19 = _mm512_fmadd_pd (r03, r31, r19);
                    }
                           _mm512_store_pd ((void*) (c+inn), r16);
                           _mm512_store_pd ((void*) (c+inn+8), r17);
                           _mm512_store_pd ((void*) (c+inn+16), r18);
                           _mm512_store_pd ((void*) (c+inn+24), r19);
                }
            }
         */   data::mulcount++;
        }

        void blockMulOneNegAvx32(double a[], double b[], double c[]) //32
        {
            const int n = 32;
            //Parallel.For(0, 4, m =>
            int m;
          /*  #pragma omp parallel for
                  for (m = 0; m < 8; m++)
            {
                for (int p = 0; p < 4; p++)
                {
                    int i = m * 4 + p;
                        int inn = i*n;

                           __m512d r16 = _mm512_load_pd ((void const*) (c+inn));
                           __m512d r17 = _mm512_load_pd ((void const*) (c+inn+8));
                           __m512d r18 = _mm512_load_pd ((void const*) (c+inn+16));
                           __m512d r19 = _mm512_load_pd ((void const*) (c+inn+24));

                        for (int k = 0; k < n; k+=4)
                    {
                        double aink = -a[i*n+k];
                        double aink1 = -a[i*n+k+1];
                        double aink2 = -a[i*n+k+2];
                        double aink3 = -a[i*n+k+3];
                        int kn = k*n;
                    //for (int j = 0; j < n; j += 8)
                        {
                           __m512d r00 = _mm512_set1_pd(aink);
                           __m512d r01 = _mm512_set1_pd(aink1);
                           __m512d r02 = _mm512_set1_pd(aink2);
                           __m512d r03 = _mm512_set1_pd(aink3);

                           __m512d r08 = _mm512_load_pd ((void const*) (b+kn));
                           __m512d r09 = _mm512_load_pd ((void const*) (b+kn+8));
                           __m512d r10 = _mm512_load_pd ((void const*) (b+kn+16));
                           __m512d r11 = _mm512_load_pd ((void const*) (b+kn+24));
                           __m512d r12 = _mm512_load_pd ((void const*) (b+kn+32));
                           __m512d r13 = _mm512_load_pd ((void const*) (b+kn+40));
                           __m512d r14 = _mm512_load_pd ((void const*) (b+kn+48));
                           __m512d r15 = _mm512_load_pd ((void const*) (b+kn+56));

                           __m512d r24 = _mm512_load_pd ((void const*) (b+kn+64));
                           __m512d r25 = _mm512_load_pd ((void const*) (b+kn+72));
                           __m512d r26 = _mm512_load_pd ((void const*) (b+kn+80));
                           __m512d r27 = _mm512_load_pd ((void const*) (b+kn+88));
                           __m512d r28 = _mm512_load_pd ((void const*) (b+kn+96));
                           __m512d r29 = _mm512_load_pd ((void const*) (b+kn+104));
                           __m512d r30 = _mm512_load_pd ((void const*) (b+kn+112));
                           __m512d r31 = _mm512_load_pd ((void const*) (b+kn+120));
                      
                           r16 = _mm512_fmadd_pd (r00, r08, r16);
                           r17 = _mm512_fmadd_pd (r00, r09, r17);
                           r18 = _mm512_fmadd_pd (r00, r10, r18);
                           r19 = _mm512_fmadd_pd (r00, r11, r19);
                           r16 = _mm512_fmadd_pd (r01, r12, r16);
                           r17 = _mm512_fmadd_pd (r01, r13, r17);
                           r18 = _mm512_fmadd_pd (r01, r14, r18);
                           r19 = _mm512_fmadd_pd (r01, r15, r19);

                           r16 = _mm512_fmadd_pd (r02, r24, r16);
                           r17 = _mm512_fmadd_pd (r02, r25, r17);
                           r18 = _mm512_fmadd_pd (r02, r26, r18);
                           r19 = _mm512_fmadd_pd (r02, r27, r19);
                           r16 = _mm512_fmadd_pd (r03, r28, r16);
                           r17 = _mm512_fmadd_pd (r03, r29, r17);
                           r18 = _mm512_fmadd_pd (r03, r30, r18);
                           r19 = _mm512_fmadd_pd (r03, r31, r19);
                     //       c[inn + j] += aink * b[kn + j];
                        }
                    }
                           _mm512_store_pd ((void*) (c+inn), r16);
                           _mm512_store_pd ((void*) (c+inn+8), r17);
                           _mm512_store_pd ((void*) (c+inn+16), r18);
                           _mm512_store_pd ((void*) (c+inn+24), r19);
                }
            }
           */ data::mulcount++;
        }


        double* MatrixStdDouble::CopyToStd()
        {
           
            double* stdMat = new double[data::mSize * data::mSize];
            for (int i = 0; i < data::mSize; i++)
                for (int j = 0; j < data::mSize; j++)
                    stdMat[i * data::mSize + j] = 0;
            for (int i = 0; i < data::valcount; i++)
            {
                stdMat[data::indexi[i] * data::mSize + data::indexj[i]] = data::vals[i];
            }

            broker::postMessage("copied to big matrix." );
            return stdMat;
        }
 
        void MatrixStdDouble::printMatrix(double* a, int n)
        {
            int count = 0;
            for(int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double t = a[i * n + j];

             //       if(i==j && i%4==0)
                    if ((!(t < TINY2 && t > -TINY2)) || i==j){
                        std::cout<<i<<" "<<j<<" "<<t<<std::endl;
                        count++;
                    }
                }
            }
            std::cout<<n<<" "<<n<<" non zero "<<count<<std::endl;
        }

        int MatrixStdDouble::countNonZero(double* a, int n)
        {
            int rtn = 0;
            

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double t = a[i * n + j];

                    if (!(t < TINY && t > -TINY))
                        rtn++;

                }
            }

            return rtn;
        }

        int MatrixStdDouble::checkUpperZero(double* a, int n)
        {
            int rtn = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = i+1; j < n; j++)
                {
                    double t = a[i * n + j];

                    if (t > TINY || t < -TINY)
                        rtn++;
                }
            }
            broker::postMessage("number of upper non zero: " + std::to_string(rtn));
            return rtn;
        }

        int MatrixStdDouble::checkLowerZero(double* a, int n)
        {
            int rtn = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double t = a[i * n + j];

                    if (t > TINY || t < -TINY)
                        rtn++;
                }
            }
            broker::postMessage("number of lower non zero: " + std::to_string(rtn));
            return rtn;
        }


        double* MatrixStdDouble::getNonZero(double* a, int n)
        {
            int rtn = 0;
            double* nz = new double[n * 2];
            int cnt = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double t = a[i * n + j];

                    if (t > TINY || t < -TINY)
                    {
                        if (cnt < n*2 && i!=j)
                        {
                            nz[cnt] = t;
                            cnt++;
                        }
                        rtn++;
                    }
                }
            }

            broker::postMessage("found first "+ std::to_string(cnt)+ " non zero ");

            return nz;
        }

        int MatrixStdDouble::countNaN(double* a, int n)
        {
            int rtn = 0;


            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if(std::isnan(a[i*n+j]))
                        rtn++;

                }
            }

            broker::postMessage("number of NAN: " + std::to_string(rtn) + " matrix rows: " + std::to_string(n));
            return rtn;
        }

        void MatrixStdDouble::CopyToStd(int* a, int n, double* std, int limit)
        {
            for(int i = 0; i < n; i++)
            {
                for(int j=0; j<n; j++)
                {
                    int t = a[i * n + j];
                    if(t>0)
                    for(int m=0; m <data::blockSize; m++)
                    {
                        for(int k=0; k<data::blockSize; k++)
                        {
                                if(i * data::blockSize + m < limit && j * data::blockSize + k < limit)
                                std[(i * data::blockSize + m) * limit + j * data::blockSize + k] = data::blockstorage[t][m * data::blockSize + k];
                        }
                    }
                }
            }

            broker::postMessage("copied to single matrix." );
        }

        int LUDcmp(double *a, int n, int *P){
            int i, j, k, imax;
            double maxa, absa;
            for(i=0;i<n;i++)
                P[i] = i;
            for(i=0;i<n;i++){
                maxa = 0.0;
                imax = i;
                for(k=i;k<n;k++){
                    if((absa=fabs(a[k*n+i]))>maxa){
                        maxa = absa;
                        imax = k;
                    }
                }
                if(imax != i){
                   j = P[i];
                   P[i] = P[imax];
                   P[imax] = j;
                   for(k=0;k<n;k++){
                       absa = a[i*n+k];
                       a[i*n+k] = a[imax*n+k];
                       a[imax*n+k] = absa;
                   }
                }
                if(fabs(a[i*n+i])<TINY){
                    if(a[i*n+i]<0)
                       a[i*n+i] = -TINY;
                    else
                       a[i*n+i] = TINY;
                }
                for(j = i+1; j<n; j++){
                   a[j*n+i] /= a[i*n+i];
                   for(k=i+1;k<n;k++)
                       a[j*n+k] -= a[j*n+i] * a[i*n+k];
                }
            }
            return 1;
        }
        int LUDcmp(double *b, int n){
            int i, j, k;
            long double aii,scale;
            long double a[BLOCK64 * BLOCK64];
            for(i=0;i<n*n;i++)
                 a[i] = b[i];
            for(i=0;i<n;i++){
                if(fabs(a[i*n+i])<TINY){
                    if(a[i*n+i]<0)
                       a[i*n+i] = -TINY;
                    else
                       a[i*n+i] = TINY;
                }
                aii = a[i*n+i];
                aii = 1.0 / aii;
                for(j = i+1; j<n; j++){
                   scale = a[j*n+i] * aii;
                   a[j*n+i] = scale;
                   for(k=i+1;k<n;k++)
                       a[j*n+k] -= scale * a[i*n+k];
                }
            }
            for(i=0;i<n*n;i++)
                 b[i] = a[i];
            return 1;
        }

//        void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* u)
        void ludcmpSimple_nri(double* a, int n, double* l, double* u)
        {
            int P[BLOCK64];
            double t[BLOCK64 * BLOCK64];

            long double sum = 0;
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }

            for (int i = 0; i < n*n; i++) {
                    u[i] = a[i];
                    l[i] = 0;
            }

            LUDcmp(u,n,P);
            for (int i=0;i<n;i++){
                for(int j=0;j<P[i];j++){
                    l[i*n+j] = u[P[i]*n+j];
                    u[P[i]*n+j] = 0;
                }
                l[i*n+P[i]] = 1.0;
            }
            LUDcmp(l,n);
            for(int i = 0; i < n*n; i++) {
                    t[i] = u[i];
                    u[i] = 0;
            }
            for(int i=0; i<n;i++){
                for(int j=i;j<n;j++){
                    for(int k=i;k<n;k++){
                        u[i*n+j] += l[i*n+k] * t[k*n+j];
                    }
                }
            }
            for(int i=0; i<n;i++){
                for(int j=i;j<n;j++){
                    l[i*n+j] = 0;
                }
                l[i*n+i] = 1;
            }
        }

        void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* t)
//        void ludcmpSimple_ldouble(double* a, int n, double* l, double* t)
//        void ludcmpSimple_nr(double* a, int n, double* l, double* u)
        {
            long double sum = 0;
            long double u[BLOCK64 * BLOCK64];
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }
            for (int i=0;i < n*n; i++) u[i] = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += l[i * n + k] * u[k * n + j];
                    u[i * n + j] = a[i * n + j] - sum;
                }
                if (u[i * n + i] < TINY2 && u[i * n + i] > -TINY2)
                {
       //             std::cout<<" i " <<i<<" j "<<i<<" val "<<u[i*n+i]<<std::endl;
                    if (u[i * n + i] < 0)
                        u[i * n + i] = -TINY2;
                    else
                        u[i * n + i] = TINY2;
                }
                long double u1 = u[i * n + i];
                u1 = 1 / u1;
                for (int j = i + 1; j < n; j++)
                {
                    sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += l[j * n + k] * u[k * n + i];
                    l[j * n + i] = u1 * (a[j * n + i] - sum);
                }
                l[i * n + i] = 1;
            }

            for (int i=0;i < n*n; i++) t[i] = u[i];
        }

    //    void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* t)
        void ludcmpSimple_iter(double* a, int n, double* l, double* t)
        {
            long double sum = 0;
            long double lw[BLOCK64 * BLOCK64];
            double v[BLOCK64 * BLOCK64];
            double l1[BLOCK64 * BLOCK64];
            int i,j,k;
            std::memset(l, 0, BLOCK64 * BLOCK64 * sizeof(double));
            std::memset(t, 0, BLOCK64 * BLOCK64 * sizeof(double));
            std::memset(lw, 0, BLOCK64 * BLOCK64 * sizeof(long double));
            std::memset(v, 0, BLOCK64 * BLOCK64 * sizeof(double));
            std::memset(l1, 0, BLOCK64 * BLOCK64 * sizeof(double));
         //   ludcmpSimple_ldouble(a,n,l,t);
            MatrixStdDouble::ludcmpSimple(a,n,l,t);
            MatrixStdDouble::inv_lower(l, n, l1);
            for(i=0;i<BLOCK64;i++){
                for(j=0;j<BLOCK64;j++){
                    sum = 0;
                    for(k=0;k<BLOCK64;k++){
                        sum += l[i*BLOCK64+k] * t[k*BLOCK64+j];
                    }
                    lw[i*BLOCK64+j] = a[i*BLOCK64+j] - sum;
                }
            }
            for(i=0;i<BLOCK64;i++){
                    for(j=i;j<BLOCK64;j++){
                        sum = 0;
                        for(k=0;k<BLOCK64;k++){
                            sum += l1[i*BLOCK64+k] * lw[k*BLOCK64+j];
                        }   
                        t[i*BLOCK64+j] = t[i*BLOCK64+j] + sum;
                    }
            }
        }

//        void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* u)
        void ludcmpSimple_npv(double* a, int n, double* l, double* u)
//        void ludcmpSimple_nr(double* a, int n, double* l, double* u)
        {
            long double sum = 0;
            if(a == NULL){
                std::cout<<" a null"<<std::endl;
            }
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += l[i * n + k] * u[k * n + j];
                    u[i * n + j] = a[i * n + j] - sum;
                }
                if (u[i * n + i] < TINY2 && u[i * n + i] > -TINY2)
                {
       //             std::cout<<" i " <<i<<" j "<<i<<" val "<<u[i*n+i]<<std::endl;
                    if (u[i * n + i] < 0)
                        u[i * n + i] = -TINY2;
                    else
                        u[i * n + i] = TINY2;
                }
                long double u1 = u[i * n + i];
                u1 = 1 / u1;
                for (int j = i + 1; j < n; j++)
                {
                    sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += l[j * n + k] * u[k * n + i];
                    l[j * n + i] = u1 * (a[j * n + i] - sum);
                }
                l[i * n + i] = 1;
            }

        }

        void MatrixStdDouble::inv_lower(double* l, int n, double* y)
        {
            int i, j;
            for (i = 0; i < n; i++)
                y[i * n + i] = 1.0;
            for(j=0; j < n; j++)
            {
/*                if (l[j * n + j] < TINY && l[j * n + j] > -TINY)
                {
                    if (l[j * n + j] < 0)
                        l[j * n + j] = -TINY;
                    else
                        l[j * n + j] = TINY;
                }
 */               long double lj1 = l[j * n + j];
                lj1 = 1.0 / lj1;

                for (i = j+1; i < n; i++)
                {
                    double sum = 0;
                    for(int k=0;k<=j;k++)
                        y[i * n + k] -= lj1 * y[j * n + k] * l[i * n + j];
                }
            }
        }

        void MatrixStdDouble::inv_upper(double* u, int n, double* y)
        {
            int i, j;
            for (i = 0; i < n; i++)
                y[i * n + i] = 1.0;
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
                    long double scale = u[j * n + j];
                    scale = u[i * n + j] / scale;
                    for (int k = j; k < n; k++)
                        y[i * n + k] -= y[j * n + k] * scale;
                }

                long double rate = u[j * n + j];
                rate = 1.0 / rate;
                for (i = j; i < n; i++)
                    y[j * n + i] = y[j * n + i] * rate;
            }
        }

        void MatrixStdDouble::ludcmp(double* a, int n, int* indx, double* d)
        {
            int i, imax = 0, j, k;
            double big, dum, sum, temp;
            double* vv;

            vv = new double[n];
            for(i=0;i<n;i++)
                vv[i]=0;
            d[0] = 1.0;
            for (i = 0; i < n; i++)
            {
                big = 0.0;
                for (j = 0; j < n; j++)
                    if ((temp = std::abs(a[i * n + j])) > big) big = temp;

                if (big == 0.0)
                    return;
                vv[i] = 1.0 / big;
            }

            for (j = 0; j < n; j++)
            {
                for (i = 0; i < j; i++)
                {
                    sum = a[i * n + j];
                    for (k = 0; k < i; k++) sum -= a[i * n + k] * a[k * n + j];
                    a[i * n + j] = sum;
                }
                big = 0.0;
                for (i = j; i < n; i++)
                {
                    sum = a[i * n + j];
                    for (k = 0; k < j; k++)
                        sum -= a[i * n + k] * a[k * n + j];
                    a[i * n + j] = sum;
                    if ((dum = vv[i] * std::abs(sum)) >= big)
                    {
                        big = dum;
                        imax = i;
                    }
                }
                if (j != imax)
                {
                    for (k = 0; k < n; k++)
                    {
                        dum = a[imax * n + k];
                        a[imax * n + k] = a[j * n + k];
                        a[j * n + k] = dum;
                    }
                    d[0] = -d[0];
                    dum = vv[imax];
                    vv[imax] = vv[j];
                    vv[j] = dum;
                }
                indx[j] = imax;
                if (a[j * n + j] == 0.0) a[j * n + j] = TINY;
                if (j != n - 1)
                {
                    dum = 1.0 / a[j * n + j];
                    for (i = j + 1; i < n; i++) a[i * n + j] *= dum;
                }
            }
        }

        void MatrixStdDouble::lubksb(double* a, int n, int* indx, double* b)
        {
            int i, ii = 0, ip, j, fd = 0;
            double sum;

            for (i = 0; i < n; i++)
            {
                fd = 0;
                ip = indx[i];
                sum = b[ip];
                b[ip] = b[i];
                if (ii != 0 || fd != 0)
                    for (j = ii; j <= i - 1; j++)
                        sum -= a[i * n + j] * b[j];
                else if (sum != 0)
                {
                    ii = i;
                    fd = 1;
                }
                b[i] = sum;
            }
            for (i = n - 1; i >= 0; i--)
            {
                sum = b[i];
                for (j = i + 1; j < n; j++)
                    sum -= a[i * n + j] * b[j];
                b[i] = sum / a[i * n + i];
            }
        }

        void MatrixStdDouble::mat_decomp_only(double* a, int n)
        {
            double d = 0;
            double* col;
            int i, j;
            int* indx;

            col = new double[n];
            indx = new int[n];
            for(i=0;i<n;i++) {
                col[i] = 0;
                indx[i] = 0;
            }
            ludcmp(a, n, indx, &d);
            broker::postMessage("done decomposition");
        }

//        void MatrixStdDouble::ludcmpSimple(double* a, int n, double* l, double* u)
        void ludcmpSimple_nr(double* a, int n, double* l, double* u)
        {

            double d = 0;
            double* col;
            int i, j;
            int* indx;

            col = new double[n];
            indx = new int[n];
            for(i=0;i<n;i++) {
                col[i] = 0;
                indx[i] = 0;
            }
      //      ludcmp(a, n, indx, &d);

            for(i=0; i<n; i++){
                for(j=0;j<i;j++){
                    l[i*n+j] = a[i*n+j];
                }
                l[i*n+i] = 1;
                for(;j<n;j++){
                    l[i*n+j] = 0;
                }
            }
            for(i=0;i<n;i++){
                for(j=0;j<i;j++){
                    u[i*n+j] = 0;
                }
                for(j=i;j<n;j++){
                    u[i*n+j] =a[i*n+j];
                }
            }
        }


        void MatrixStdDouble::mat_inv(double* a, int n, double* y)
        {
            double d = 0;
            double* col;
            int i, j;
            int* indx;

            col = new double[n];
            indx = new int[n];
            for(i=0;i<n;i++) {
                col[i] = 0;
                indx[i] = 0;
            }
            double* aa= new double[n*n];
           
            for(i=0;i<n;i++)
                for(j=0;j<n;j++)
                    aa[i*n+j] = a[i*n+j];
            ludcmp(aa, n, indx, &d);
     //       broker::postMessage("done decomposition");
            for (j = 0; j < n; j++)
            {
                for (i = 0; i < n; i++) col[i] = 0.0f;
                col[j] = 1.0f;
                lubksb(aa, n, indx, col);
                for (i = 0; i < n; i++) y[i * n + j] = col[i];
            }
      //      broker::postMessage("done inversion");

        }

        void MatrixStdDouble::mat_mul(double* a, double* b, int n, int m, int w, double* y)
        {
            int i, j, k;
            double sum;
            for (i = 0; i < n; i++)
            {
                for (j = 0; j < m; j++)
                {
                    sum = 0.0;
                    for (k = 0; k < w; k++)
                    {
                        sum += a[i * n + k] * b[k * n + j];
                    }
                    y[i * n + j] = sum;
                }
            }
        }

        bool MatrixStdDouble::inv_check_diag(double* a, double* b, int n)
        {
            int i, j, k;
            double sum;
            double min, max;
            min = 1;
            max = 1;
            double tor = 1e-8;
            double* y = new double[n];
            for(i=0;i<n;i++) y[i] = 0;
            for (i = 0; i < n; i++)
            {
                j = i;
                {
                    sum = 0.0f;
                    for (k = 0; k < n; k++)
                    {
                        sum += a[i * n + k] * b[k * n + j];
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
            if( max > 1+tor || max < 1-tor || min > 1+ tor || min < 1-tor){
                broker::postMessage("diag min:" + std::to_string(min) + "  diag max:" + std::to_string(max));
                return false;
            }
            return true;
        }

        void MatrixStdDouble::inv_check_zero(double* a, double* b, int n)
        {
            int i, j, k;
            double sum;
            double min, max;
            min = 0;
            max = 0;
            double* y = new double[n];
            for(i=0;i<n;i++) y[i] = 0;
            for (i = 1; i < n; i++)
            {
                j = i-1;
                {
                    sum = 0.0;
                    for (k = 0; k < n; k++)
                    {
                        sum += a[i * n + k] * b[k * n + j];
                    }
                    y[i] = sum;
                    if (min > sum)
                        min = sum;
                    if (max < sum)
                        max = sum;
                }
            }
            broker::postMessage("diff diag min:" + std::to_string(min) + "  diff max:" + std::to_string(max));

            min = 0;
            max = 0;
            for (i = 2; i < n; i++)
            {
                j = i - 2;
                {
                    sum = 0.0;
                    for (k = 0; k < n; k++)
                    {
                        sum += a[i * n + k] * b[k * n + j];
                    }
                    y[i] = sum;
                    if (min > sum)
                        min = sum;
                    if (max < sum)
                        max = sum;
                }
            }
            broker::postMessage("off diag min:" + std::to_string(min) + "  off diag max:" + std::to_string(max) );

        }

        bool MatrixStdDouble::mat_equ(double* a, double* b, int n)
        {
            int i, j;
            double sum;
            double min, max;
            min = 0;
            max = 0;
            double tor = 1e-6;

            for (i = 0; i < n; i++)
            {
                for(j=0; j<n; j++)
                {
                    
                    sum = a[i * n + j] - b[i * n + j];
                   
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
            if( max > tor || min < -tor){
                broker::postMessage("Mat diff min:" + std::to_string(min) + "  mat diff max:" + std::to_string(max) );
                return false;
            }
            return true;
        }

        void MatrixStdDouble::vec_equ(double* a, double* b, int n)
        {
            int i, j;
            double sum;
            double min, max;
            min = 0;
            max = 0;

            for (i = 0; i < n; i++)
            {
               
                    sum = a[i] - b[i];

                    if (min > sum)
                        min = sum;
                    if (max < sum)
                        max = sum;
                
            }
            broker::postMessage("vector diff min:" + std::to_string(min) + "  vector diff max:" + std::to_string(max) );
        }

        double* MatrixStdDouble::mat_shrink(double* a, int n, int small)
        {
            double* y = new double[small * small];
            for (int i = 0; i < small; i++)
                for (int j = 0; j < small; j++)
                    y[i * small + j] = a[i * n + j];
            return y;
        }

        void MatrixStdDouble::mat_sub(double* a, double* b, int n, double* r)
        {
                            int m;
//            #pragma omp parallel for private(m)
                            for (m = 0; m < 64 ; m++)
                            {
                                int k;
                                int mn = m* n;
                                double* aa = a+mn;
                                double* bb = b+mn;
                                double* cc = r+mn;
 //                       #pragma omp simd aligned(aa, bb, cc: 64)
                                for (k = 0; k < 64; k++)
                                {
                                    cc[k] = aa[k] - bb[k];
                                }
                            }
        }
        void mat_sub512(double* a, double* b, int n, double* r)
        {
             for(int m=0;m<32;m++){
                 int i=m*128;
                 double* aa = a + i;
                 double* bb = b + i;
                 double* cc = r + i;

    /*             __m512d a01 = _mm512_load_pd ((void const*) (aa));
                 __m512d b01 = _mm512_load_pd ((void const*) (bb));
                 a01 = _mm512_sub_pd(a01, b01);
                 _mm512_store_pd ((void*) (cc), a01);

                 __m512d a02 = _mm512_load_pd ((void const*) (aa+8));
                 __m512d b02 = _mm512_load_pd ((void const*) (bb+8));
                 a02 = _mm512_sub_pd(a02, b02);
                 _mm512_store_pd ((void*) (cc+8), a02);

                 __m512d a03 = _mm512_load_pd ((void const*) (aa+16));
                 __m512d b03 = _mm512_load_pd ((void const*) (bb+16));
                 a03 = _mm512_sub_pd(a03, b03);
                 _mm512_store_pd ((void*) (cc+16), a03);

                 __m512d a04 = _mm512_load_pd ((void const*) (aa+24));
                 __m512d b04 = _mm512_load_pd ((void const*) (bb+24));
                 a04 = _mm512_sub_pd(a04, b04);
                 _mm512_store_pd ((void*) (cc+24), a04);

                 __m512d a05 = _mm512_load_pd ((void const*) (aa+32));
                 __m512d b05 = _mm512_load_pd ((void const*) (bb+32));
                 a05 = _mm512_sub_pd(a05, b05);
                 _mm512_store_pd ((void*) (cc+32), a05);

                 __m512d a06 = _mm512_load_pd ((void const*) (aa+40));
                 __m512d b06 = _mm512_load_pd ((void const*) (bb+40));
                 a06 = _mm512_sub_pd(a06, b06);
                 _mm512_store_pd ((void*) (cc+40), a06);

                 __m512d a07 = _mm512_load_pd ((void const*) (aa+48));
                 __m512d b07 = _mm512_load_pd ((void const*) (bb+48));
                 a07 = _mm512_sub_pd(a07, b07);
                 _mm512_store_pd ((void*) (cc+48), a07);

                 __m512d a08 = _mm512_load_pd ((void const*) (aa+56));
                 __m512d b08 = _mm512_load_pd ((void const*) (bb+56));
                 a08 = _mm512_sub_pd(a08, b08);
                 _mm512_store_pd ((void*) (cc+56), a08);

                 __m512d a11 = _mm512_load_pd ((void const*) (aa+64));
                 __m512d b11 = _mm512_load_pd ((void const*) (bb+64));
                 a11 = _mm512_sub_pd(a11, b11);
                 _mm512_store_pd ((void*) (cc+64), a11);

                 __m512d a12 = _mm512_load_pd ((void const*) (aa+8+64));
                 __m512d b12 = _mm512_load_pd ((void const*) (bb+8+64));
                 a12 = _mm512_sub_pd(a12, b12);
                 _mm512_store_pd ((void*) (cc+8+64), a12);

                 __m512d a13 = _mm512_load_pd ((void const*) (aa+16+64));
                 __m512d b13 = _mm512_load_pd ((void const*) (bb+16+64));
                 a13 = _mm512_sub_pd(a13, b13);
                 _mm512_store_pd ((void*) (cc+16+64), a13);

                 __m512d a14 = _mm512_load_pd ((void const*) (aa+24+64));
                 __m512d b14 = _mm512_load_pd ((void const*) (bb+24+64));
                 a14 = _mm512_sub_pd(a14, b14);
                 _mm512_store_pd ((void*) (cc+24+64), a14);

                 __m512d a15 = _mm512_load_pd ((void const*) (aa+32+64));
                 __m512d b15 = _mm512_load_pd ((void const*) (bb+32+64));
                 a15 = _mm512_sub_pd(a15, b15);
                 _mm512_store_pd ((void*) (cc+32+64), a15);

                 __m512d a16 = _mm512_load_pd ((void const*) (aa+40+64));
                 __m512d b16 = _mm512_load_pd ((void const*) (bb+40+64));
                 a16 = _mm512_sub_pd(a16, b16);
                 _mm512_store_pd ((void*) (cc+40+64), a16);

                 __m512d a17 = _mm512_load_pd ((void const*) (aa+48+64));
                 __m512d b17 = _mm512_load_pd ((void const*) (bb+48+64));
                 a17 = _mm512_sub_pd(a17, b17);
                 _mm512_store_pd ((void*) (cc+48+64), a17);

                 __m512d a18 = _mm512_load_pd ((void const*) (aa+56+64));
                 __m512d b18 = _mm512_load_pd ((void const*) (bb+56+64));
                 a18 = _mm512_sub_pd(a18, b18);
                 _mm512_store_pd ((void*) (cc+56+64), a18);
*/
             }
        }
        void MatrixStdDouble::mat_copy(double* a, int n, double* r)
        {
            std::memcpy(r, a, sizeof(double)*n*n);
        }
        void MatrixStdDouble::mat_neg(double* b, int n, double* r)
        {
                            int m;
    //        #pragma omp parallel for 
                            for (m = 0; m < n ; m++)
                            {
                                int k;
                                int mn = m* n;
                                for (k = 0; k < n; k++)
                                {
                                    r[mn + k] = - b[mn + k];
                                }
                            }
        }
}


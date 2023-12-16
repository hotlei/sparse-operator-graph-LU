#include <stdio.h>
#include <cuda_runtime.h>
#include "Matrixcuda.h"

#define WARP33 33
#define WARP8 8
#define WARP16 16
#define WARP48 48
#define WARP64wB 64
#define WARP64wA 64
#define WARP64 64
#define WARP63 63
#define WARP65 65
#define WARP32BLOCK_SIZE 32
#define WARP256 256
#define WARP512 512
#define WARP768 768
#define WARP1024   1024
#define WARP1280   1280
#define WARP1536   1536
#define WARP1792   1792
#define WARP2048   2048
#define WARP4096   4096

#define TINY  1.0e-20

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace vmatrix
{

inline __device__ void matrixTrans64(double *At, double *A, double* As)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    As[ty * WARP33 + tx] = A[WARP64wA * ty + tx];
    __syncthreads();
    At[WARP64wA * ty + tx] = As[tx * WARP33 + ty];
    __syncthreads();

    As[ty * WARP33 + tx] = A[WARP32 + WARP64wA * ty + tx];
    __syncthreads();
    At[WARP32 * WARP64wA + WARP64wA * ty + tx] = As[tx * WARP33 + ty];
    __syncthreads();

    As[ty * WARP33 + tx] = A[WARP32 * WARP64wA + WARP64wA * ty + tx];
    __syncthreads();
    At[WARP32 + WARP64wA * ty + tx] = As[tx * WARP33 + ty];
    __syncthreads();

    As[ty * WARP33 + tx] = A[WARP32 * WARP64wA + WARP32 + WARP64wA * ty + tx];
    __syncthreads();
    At[WARP32 * WARP64wA + WARP32 + WARP64wA * ty + tx] = As[tx * WARP33 + ty];
    __syncthreads();
}

__device__ void matrixMulCUDA_t8(double *C, double *At, double *B, int wA)
{
    __shared__ double As[8][WARP32*2];
    __shared__ double Bs[8][WARP32*2];
    int tx2 = threadIdx.x;
    int ty2 = threadIdx.y;
    int xoff = (ty2 & 1) * WARP32 + tx2;
    int yoff = (ty2 & 15) / 2;

    tx2 = tx2 * 2;
    ty2 = ty2 * 2;

    int ty2tx2 = ty2 * wA + tx2;
    double c00 = C[ty2tx2];
    double c01 = C[ty2tx2 + 1];
    double c10 = C[ty2tx2 + wA];
    double c11 = C[ty2tx2 + wA + 1];
    for(int offset = 0; offset<64; offset += 8){
        if(ty2<32){
            As[yoff][xoff] = At[(offset + yoff) * wA + xoff];
        }else if(ty2<64){
            Bs[yoff][xoff] = B[(offset + yoff) * wA + xoff];
        }
        __syncthreads();

#pragma unroll
        for(int k=0;k<8;k++){
            c00 += As[k][ty2] * Bs[k][tx2]; 
            c01 += As[k][ty2] * Bs[k][tx2+1]; 
            c10 += As[k][ty2+1] * Bs[k][tx2];
            c11 += As[k][ty2+1] * Bs[k][tx2+1];
        }
        __syncthreads();
    }
    C[ty2tx2] = c00;
    C[ty2tx2 + 1] = c01;
    C[ty2tx2 + wA] = c10;
    C[ty2tx2 + wA + 1] = c11;
    __syncthreads();
}

__device__ inline void matrixMulCUDA_t(double *C, double *At, double *B, double *As)
{
    int tx2 = threadIdx.x;
    int ty2 = threadIdx.y;
    int xoff = (ty2 & 1) * WARP32 + tx2;
    int yoff = (ty2 & 7) / 2;

    tx2 = tx2 * 2;
    ty2 = ty2 * 2;

    C += ty2 * WARP64wA + tx2;
    double c00 = C[0];
    double c01 = C[1];
    double c10 = C[WARP64wA];
    double c11 = C[WARP64wA + 1];
    for(int offset = 0; offset<64; offset += 4){
        if(ty2<16){
            As[yoff * WARP64 + xoff] = At[(offset + yoff) * WARP64wA + xoff];
        }else if(ty2<32){
            As[yoff * WARP64 + WARP256 + xoff] = B[(offset + yoff) * WARP64wA + xoff];
        }
        __syncthreads();

        for(int k=0;k<256;k += 64){
            c00 += As[k+ty2] * As[k + WARP256 + tx2]; 
            c01 += As[k+ty2] * As[k + WARP256 + tx2+1]; 
            c10 += As[k+ty2+1] * As[k + WARP256 + tx2];
            c11 += As[k+ty2+1] * As[k + WARP256 + tx2+1];
        }
        __syncthreads();
    }
    C[0] = c00;
    C[1] = c01;
    C[WARP64wA] = c10;
    C[WARP64wA + 1] = c11;
    __syncthreads();
}


__device__ void matrixMulCUDA_t2(double *C, double *At, double *B, int wA)
{
    __shared__ double As[2][WARP32*2];
    __shared__ double Bs[2][WARP32*2];
    int tx2 = threadIdx.x;
    int ty2 = threadIdx.y;
    int xoff = (ty2 & 1) * WARP32 + tx2;
    int yoff = (ty2 & 3) / 2;

    tx2 = tx2 * 2;
    ty2 = ty2 * 2;

    int ty2tx2 = ty2 * wA + tx2;
    double c00 = C[ty2tx2];
    double c01 = C[ty2tx2 + 1];
    double c10 = C[ty2tx2 + wA];
    double c11 = C[ty2tx2 + wA + 1];
    for(int offset = 0; offset<64; offset += 2){
        if(ty2<8){
            As[yoff][xoff] = At[(offset + yoff) * wA + xoff];
        }else if(ty2<16){
            Bs[yoff][xoff] = B[(offset + yoff) * wA + xoff];
        }
        __syncthreads();
        c00 += As[0][ty2] * Bs[0][tx2] + As[1][ty2] * Bs[1][tx2];
        c01 += As[0][ty2] * Bs[0][tx2+1] + As[1][ty2] * Bs[1][tx2+1];
        c10 += As[0][ty2+1] * Bs[0][tx2] + As[1][ty2+1] * Bs[1][tx2];
        c11 += As[0][ty2+1] * Bs[0][tx2+1] + As[1][ty2+1] * Bs[1][tx2+1];
        __syncthreads();
    }
    C[ty2tx2] = c00;
    C[ty2tx2 + 1] = c01;
    C[ty2tx2 + wA] = c10;
    C[ty2tx2 + wA + 1] = c11;
    __syncthreads();
}

__device__ void matrixMulCUDA_d(double *C, double *A, double *B, int wA)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * WARP32 * by;
    int aEnd   = aBegin + wA - 1;

    int aStep  = WARP32;
    int bBegin = WARP32 * bx;
    int bStep  = WARP32 * wA;

    double Csub = 0;
for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        __shared__ double As[WARP32][WARP32];

        __shared__ double Bs[WARP32][WARP32];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wA * ty + tx];

        __syncthreads();

//pragma unroll

        for (int k = 0; k < WARP32; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wA * WARP32 * by + WARP32 * bx;
    C[c + wA * ty + tx] += Csub;
}

__device__ void matrixMulDevice16_8(double* __restrict__ C, const double*  __restrict__ A, const double* __restrict__ B, double* As)
{    // 8 half block square
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 8

    double Csub0 = 0;
    double Csub1 = 0;
    double Csub2 = 0;
    double Csub3 = 0;
    double Csub4 = 0;
    double Csub5 = 0;
    double Csub6 = 0;
    double Csub7 = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;

//        __shared__ double As[WARP16][WARP32BLOCK_SIZE];
//        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= WARP2048 * by + WARP64wA - 1;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        As[ty*WARP32 + tx] = A[a + WARP64wA * ty + tx];
        As[ty*WARP32 + WARP256+ tx] = A[a + WARP64wA * (ty+8) + tx];
        As[ty*WARP32 + WARP512+ tx] = A[a + WARP64wA * (ty+16) + tx];
        As[ty*WARP32 + WARP768+ tx] = A[a + WARP64wA * (ty+24) + tx];
        As[ty*WARP32 + WARP1024 + tx] = B[b + WARP64wB * ty + tx];
        As[ty*WARP32 + WARP1280 + tx] = B[b + WARP64wB * (ty+8) + tx];
        As[ty*WARP32 + WARP1536 + tx] = B[b + WARP64wB * (ty+16) + tx];
        As[ty*WARP32 + WARP1792 + tx] = B[b + WARP64wB * (ty+24) + tx];

        As[ty*WARP32 + 128 + tx] = A[a + WARP64wA * (ty+4) + tx];
        As[ty*WARP32 + WARP256+ 128 + tx] = A[a + WARP64wA * (ty+12) + tx];
        As[ty*WARP32 + WARP512+ 128 + tx] = A[a + WARP64wA * (ty+20) + tx];
        As[ty*WARP32 + WARP768+ 128 + tx] = A[a + WARP64wA * (ty+28) + tx];
        As[ty*WARP32 + WARP1024 + 128 + tx] = B[b + WARP64wB * (ty+4) + tx];
        As[ty*WARP32 + WARP1280 + 128 + tx] = B[b + WARP64wB * (ty+12) + tx];
        As[ty*WARP32 + WARP1536 + 128 + tx] = B[b + WARP64wB * (ty+20) + tx];
        As[ty*WARP32 + WARP1792 + 128 + tx] = B[b + WARP64wB * (ty+28) + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub0 += As[ty*WARP32 + k] * As[k*WARP32 + WARP1024 + tx];
            Csub1 += As[ty*WARP32 + WARP256 + k] * As[k*WARP32 + WARP1024 + tx];
            Csub2 += As[ty*WARP32 + WARP512 + k] * As[k*WARP32 + WARP1024 + tx];
            Csub3 += As[ty*WARP32 + WARP768 + k] * As[k*WARP32 + WARP1024 + tx];

            Csub4 += As[ty*WARP32 + 128 + k] * As[k*WARP32 + WARP1024 + tx];
            Csub5 += As[ty*WARP32 + WARP256 + 128 + k] * As[k*WARP32 + WARP1024 + tx];
            Csub6 += As[ty*WARP32 + WARP512 + 128 + k] * As[k*WARP32 + WARP1024 + tx];
            Csub7 += As[ty*WARP32 + WARP768 + 128 + k] * As[k*WARP32 + WARP1024 + tx];
        }

        __syncthreads();
    }
    *C += Csub0;
    C[WARP512] += Csub1;
    C[WARP1024] += Csub2;
    C[WARP1536] += Csub3;
    C[WARP256] += Csub4;
    C[WARP768] += Csub5;
    C[WARP1280] += Csub6;
    C[WARP1792] += Csub7;
}

__device__ void matrixMulDevice16_4(double* __restrict__ D, const double*  __restrict__ A, const double* __restrict__ B, double* As)
{    // 4 half block square
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 8

    double Csub0 = 0;
    double Csub1 = 0;
    double Csub2 = 0;
    double Csub3 = 0;
    double* C = D + WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;

//        __shared__ double As[WARP16][WARP32BLOCK_SIZE];
//        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= WARP2048 * by + WARP64wA - 1;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        As[ty*WARP32 + tx] = A[a + WARP64wA * ty + tx];
        As[ty*WARP32 + WARP256+ tx] = A[a + WARP64wA * (ty+8) + tx];
        As[ty*WARP32 + WARP512+ tx] = A[a + WARP64wA * (ty+16) + tx];
        As[ty*WARP32 + WARP768+ tx] = A[a + WARP64wA * (ty+24) + tx];
        As[ty*WARP32 + WARP1024 + tx] = B[b + WARP64wB * ty + tx];
        As[ty*WARP32 + WARP1280 + tx] = B[b + WARP64wB * (ty+8) + tx];
        As[ty*WARP32 + WARP1536 + tx] = B[b + WARP64wB * (ty+16) + tx];
        As[ty*WARP32 + WARP1792 + tx] = B[b + WARP64wB * (ty+24) + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            double sk = As[k*WARP32 + WARP1024 + tx];
            Csub0 += As[ty*WARP32 + k] * sk;
            Csub1 += As[ty*WARP32 + WARP256 + k] * sk;
            Csub2 += As[ty*WARP32 + WARP512 + k] * sk;
            Csub3 += As[ty*WARP32 + WARP768 + k] * sk;
        }

        __syncthreads();
    }
    C[0] += Csub0;
    C[WARP512] += Csub1;
    C[WARP1024] += Csub2;
    C[WARP1536] += Csub3;
}

__device__ void matrixMulDevice16(double* __restrict__ C, const double*  __restrict__ A, const double* __restrict__ B, double* As)
{    // 2 half block square
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 16

 //   int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;

//        __shared__ double As[WARP16][WARP32BLOCK_SIZE];
//        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= WARP2048 * by + WARP64wA - 1;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        int ty32tx = ty*WARP32 + tx;
        int ty64tx = WARP64wA * ty + tx;
        As[ty32tx] = A[a + ty64tx];
        As[ty32tx + WARP512] = B[b + ty64tx];
        As[ty32tx + WARP1024] = B[b + WARP1024 + ty64tx];

        __syncthreads();

//#pragma unroll
        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty*WARP32 + k] * As[k*WARP32 + WARP512 + tx];
        }

        __syncthreads();
    }
    *C += Csub;

    C += WARP64wB * 16;
    Csub = 0;
for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= WARP2048 * by + WARP64wA - 1;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {
        int ty32tx = ty*WARP32 + tx;
        int ty64tx = WARP64wA * ty + tx;
        As[ty32tx] = A[a + WARP1024 + ty64tx];
        As[ty32tx + WARP512] = B[b + ty64tx];
        As[ty32tx + WARP1024] = B[b + WARP1024 + ty64tx];
        __syncthreads();

//#pragma unroll
        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty *WARP32 + k] * As[k*WARP32 + WARP512 + tx];
        }
        __syncthreads();
    }
    *C += Csub;
}

__device__ void matrixMulDevice16neg(double* __restrict__ C, const double*  __restrict__ A, const double* __restrict__ B, double* As)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 16

    int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;

//        __shared__ double As[WARP16][WARP32BLOCK_SIZE];
//        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];
    int a = 0;
    int b = 0;
    int k = 0;

for (a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        As[ty*WARP32 + tx] = A[a + WARP64wA * ty + tx];
        As[ty*WARP32 + WARP512 + tx] = B[b + WARP64wB * ty + tx];
        As[ty*WARP32 + WARP1024 + tx] = B[b + WARP64wB * (ty+16) + tx];

        __syncthreads();

//pragma unroll

        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty*WARP32 + k] * As[k*WARP32 + WARP512 + tx];
        }

        __syncthreads();
    }
    C[0] -= Csub;

    C += WARP64wB * 16;
    Csub = 0;
for (a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {
        As[ty*WARP32 + tx] = A[a + WARP64wA * (ty + 16) + tx];
        As[ty*WARP32 + WARP512 + tx] = B[b + WARP64wB * ty + tx];
        As[ty*WARP32 + WARP1024 + tx] = B[b + WARP64wB * (ty+16) + tx];
        __syncthreads();
        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty *WARP32 + k] * As[k*WARP32 + WARP512 + tx];
        }
        __syncthreads();
    }
    C[0] -= Csub;

}


__global__ void matrixMulCUDA16orig(double* __restrict__ C, const double*  __restrict__ A, const double* __restrict__ B)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 16

    int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;

        __shared__ double As[WARP16][WARP32BLOCK_SIZE];
        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];
    int a = 0;
    int b = 0;
    int k = 0;

for (a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        As[ty][tx] = A[a + WARP64wA * ty + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];
        Bs[ty+16][tx] = B[b + WARP64wB * (ty+16) + tx];

        __syncthreads();

//pragma unroll

        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    C[0] += Csub;

    C += WARP64wB * 16;
    Csub = 0;
for (a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {
        As[ty][tx] = A[a + WARP64wA * (ty + 16) + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];
        Bs[ty+16][tx] = B[b + WARP64wB * (ty+16) + tx];
        __syncthreads();
        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[0] += Csub;

}


__global__ void matrixMulCUDANeg16(double *C, double *A, double *B)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 16

    int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;

        __shared__ double As[WARP16][WARP32BLOCK_SIZE];
        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];
    int a = 0;
    int b = 0;

for (a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        As[ty][tx] = A[a + WARP64wA * ty + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];
        Bs[ty+16][tx] = B[b + WARP64wB * (ty+16) + tx];

        __syncthreads();

//pragma unroll

        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    C[0] -= Csub;

    C += WARP64wB * 16;
    Csub = 0;
for (a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {
        As[ty][tx] = A[a + WARP64wA * (ty + 16) + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];
        Bs[ty+16][tx] = B[b + WARP64wB * (ty+16) + tx];
        __syncthreads();
        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[0] -= Csub;

}



__global__ void matrixMulCUDA(double *C, double *A, double *B)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;
for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        __shared__ double As[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

        As[ty][tx] = A[a + WARP64wA * ty + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];

        __syncthreads();

//pragma unroll

        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[0] += Csub;
}

__global__ void matrixMulCUDA2x(double *C, double *A, double *B, double *A1, double *B1)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;
for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {
        __shared__ double As[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];
        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

        As[ty][tx] = A[a + WARP64wA * ty + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];

        __syncthreads();

//pragma unroll

        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();

        As[ty][tx] = A1[a + WARP64wA * ty + tx];
        Bs[ty][tx] = B1[b + WARP64wB * ty + tx];
        __syncthreads();

//pragma unroll

        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[0] += Csub;
}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA2xt(double *C, double *A, double *B, double *A1, double *B1, double *At)
{
    extern __shared__ double ssm[];

    matrixTrans64(At, A, ssm);
    matrixMulCUDA_t(C, At, B, ssm);

    matrixTrans64(At, A1, ssm);
    matrixMulCUDA_t(C, At, B1, ssm);
}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA2yt(double *C, double *A, double *B, double *A1, double *B1, double *C1, double *At)
{
    extern __shared__ double ssm[];

    matrixTrans64(At, A, ssm);
    matrixMulCUDA_t(C, At, B, ssm);

    matrixTrans64(At, A1, ssm);
    matrixMulCUDA_t(C1, At, B1, ssm);
}

__global__ void matrixMulCUDA2y(int C, double *mbase, int A, int B, int A1, int B1, int C1, int tx, int ty)
{
    tx = threadIdx.x;
    ty = threadIdx.y;

    A += WARP2048 * blockIdx.y + WARP64wA * ty + tx;
    B += WARP32BLOCK_SIZE * blockIdx.x + WARP64wA * ty + tx;
    A1 += WARP2048 * blockIdx.y + WARP64wA * ty + tx;
    B1 += WARP32BLOCK_SIZE * blockIdx.x + WARP64wA * ty + tx;

    C += WARP2048 * blockIdx.y + WARP32BLOCK_SIZE * blockIdx.x + WARP64wB * ty + tx;
    C1 += WARP2048 * blockIdx.y + WARP32BLOCK_SIZE * blockIdx.x + WARP64wB * ty + tx;
    double Csub = 0;
     
    int k = 0;

        __shared__ double As[WARP64+WARP32][WARP32BLOCK_SIZE];

        As[ty][tx] = mbase[A];
        //As[ty+WARP32][tx] = mbase[B];
        As[ty][tx+WARP1024] = mbase[B];
        As[ty+WARP64][tx] = 0;

        __syncthreads();

#pragma unroll
        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * As[k+WARP32][tx];
        }

        __syncthreads();
     
        As[ty][tx] = mbase[A1];
        //As[ty+WARP32][tx] = mbase[B1];
        As[ty][tx+WARP1024] = mbase[B1];

        __syncthreads();

#pragma unroll
        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            As[ty+WARP64][tx] += As[ty][k] * As[k+WARP32][tx];
        }

        __syncthreads();


    A += WARP32BLOCK_SIZE; 
    B += WARP2048; 
    A1 += WARP32BLOCK_SIZE; 
    B1 += WARP2048;

        As[ty][tx] = mbase[A];
        //As[ty+WARP32][tx] = mbase[B];
        As[ty][tx+WARP1024] = mbase[B];

        __syncthreads();

#pragma unroll
        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * As[k+WARP32][tx];
        }

        __syncthreads();
     
        As[ty][tx] = mbase[A1];
        //As[ty+WARP32][tx] = mbase[B1];
        As[ty][tx+WARP1024] = mbase[B1];

        __syncthreads();

#pragma unroll
        for (k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            As[ty+WARP64][tx] += As[ty][k] * As[k+WARP32][tx];
        }

        __syncthreads();


    mbase[C] += Csub;
    mbase[C1] += As[ty+WARP64][tx];
}

__global__ void matrixMulCUDANeg(double *C, double *A, double *B)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aEnd   = WARP2048 * by + WARP64wA - 1;

    double Csub = 0;
    C += WARP2048 * by + WARP32BLOCK_SIZE * bx + WARP64wB * ty + tx;
for (int a = WARP2048 * by, b = WARP32BLOCK_SIZE * bx;
         a <= aEnd;
         a += WARP32BLOCK_SIZE, b += WARP2048)
    {

        __shared__ double As[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];
        __shared__ double Bs[WARP32BLOCK_SIZE][WARP32BLOCK_SIZE];

        As[ty][tx] = A[a + WARP64wA * ty + tx];
        Bs[ty][tx] = B[b + WARP64wB * ty + tx];

        __syncthreads();

//pragma unroll

        for (int k = 0; k < WARP32BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    C[0] -= Csub;
}

__device__ void matrixSubDevice16(double* C, double* A, double* B) //const double* __restrict__ B
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c =  WARP2048 * by + WARP32* bx + WARP64 * ty + tx;
    C[c] = A[c] - B[c];
}

__device__ void matrixResetDevice16(double* C) //const double* __restrict__ B
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c =  WARP2048 * by + WARP32* bx + WARP64 * ty + tx;
    C[c] = 0;
}


template <int BLOCK_SIZE> __global__ void
matrixSubCUDA(double *C, double *A, double *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = A[c + wB * ty + tx] - B[c + wB * ty + tx];
}

        void cudasubone(double* d_base, int a, int b, int n, int y, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             double *d_A, *d_B, *d_C;
             int block_size = 32;

             d_A = d_base + a * size_A;
             d_B = d_base + b * size_A;
             d_C = d_base + y * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(n / threads.x, n / threads.y);
              matrixSubCUDA<32><<< grid, threads, 0, stream >>>(d_C, d_A, d_B, n, n);
        }
        void cudamuloneNeg(double* d_base, int a, int b, int n, int y, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             double *d_A, *d_B, *d_C;
             int block_size = 32;

             d_A = d_base + a * size_A;
             d_B = d_base + b * size_A;
             d_C = d_base + y * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(n / threads.x, n / threads.y);
              matrixMulCUDANeg<<< grid, threads ,0,  stream>>>(d_C, d_A, d_B);
        }

        void cudamulone(double* d_base, int a, int b, int n, int y, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             double *d_A, *d_B, *d_C;
             int block_size = 32;

             d_A = d_base + a * size_A;
             d_B = d_base + b * size_A;
             d_C = d_base + y * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(2, 2);
              matrixMulCUDA<<< grid, threads , 0, stream>>>(d_C, d_A, d_B);
        }

        void cudamulone2xt(double* d_base, int a, int b, int n, int y, int a1, int b1, double* d_tmp, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             double *d_A, *d_B, *d_C, *d_A1, *d_B1;
             int block_size = 32;

             d_A = d_base + a * size_A;
             d_B = d_base + b * size_A;
             d_C = d_base + y * size_A;
             d_A1 = d_base + a1 * size_A;
             d_B1 = d_base + b1 * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(1, 1);
              matrixMulCUDA2xt<32><<< grid, threads , 24576, stream>>>(d_C, d_A, d_B, d_A1, d_B1, d_tmp);
            gpuErrchk( cudaPeekAtLastError() );
        }
        void cudamulone2x(double* d_base, int a, int b, int n, int y, int a1, int b1, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             double *d_A, *d_B, *d_C, *d_A1, *d_B1;
             int block_size = 32;

             d_A = d_base + a * size_A;
             d_B = d_base + b * size_A;
             d_C = d_base + y * size_A;
             d_A1 = d_base + a1 * size_A;
             d_B1 = d_base + b1 * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(2, 2);
              matrixMulCUDA2x<<< grid, threads , 0, stream>>>(d_C, d_A, d_B, d_A1, d_B1);
            gpuErrchk( cudaPeekAtLastError() );
        }

        void cudamulone2yt(double* d_base, int a, int b, int n, int y, int a1, int b1, int y1, 
                                           double* d_tmp, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             double *d_A, *d_B, *d_C, *d_A1, *d_B1, *d_C1;
             int block_size = 32;

             d_A = d_base + a * size_A;
             d_B = d_base + b * size_A;
             d_C = d_base + y * size_A;
             d_A1 = d_base + a1 * size_A;
             d_B1 = d_base + b1 * size_A;
             d_C1 = d_base + y1 * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(1, 1);
              matrixMulCUDA2yt<32><<< grid, threads , 24576, stream>>>(d_C, d_A, d_B, d_A1, d_B1, d_C1, d_tmp);
       //       matrixMulCUDAyt<32><<< grid, threads , 0, stream>>>(d_C, d_A, d_B, n, n, d_tmp);
       //     gpuErrchk( cudaPeekAtLastError() );
       //       matrixMulCUDAyt<32><<< grid, threads , 0, stream>>>(d_C1, d_A1, d_B1, n, n, d_tmp);
            gpuErrchk( cudaPeekAtLastError() );
        }
        void cudamulone2y(double* d_base, int a, int b, int n, int y, int a1, int b1, int y1, cudaStream_t stream)
        {
             unsigned int size_A = n * n;
             int d_C, d_C1;
             int d_A, d_B, d_A1, d_B1;
             int block_size = 32;

             d_A = a * size_A;
             d_B = b * size_A;
             d_C = y * size_A;
             d_A1 = a1 * size_A;
             d_B1 = b1 * size_A;
             d_C1 = y1 * size_A;

             dim3 threads(block_size, block_size);
             dim3 grid(2, 2);
             if(y1 >= y){
                 matrixMulCUDA2y<<< grid, threads , 0, stream>>>(d_C, d_base, d_A, d_B, d_A1, d_B1, d_C1,n,n);
             }else{
                 matrixMulCUDA2y<<< grid, threads , 0, stream>>>(d_C1, d_base, d_A1, d_B1, d_A, d_B, d_C,n,n);
             }
            gpuErrchk( cudaPeekAtLastError() );
        }

__global__ void matrixMulCUDA16(double* __restrict__ C, const double*  __restrict__ A, const double* __restrict__ B)
{
    __shared__ double As[WARP48*WARP32BLOCK_SIZE];
    matrixMulDevice16(C, A, B, As);
}

__global__ void matrixMulCUDAlist(double* d_base, int* list, int count)
{

/*    __shared__ int param[432];
    param[threadIdx.y*32+threadIdx.x] = list[threadIdx.y*32+threadIdx.x];
    __syncthreads();
*/
    if( threadIdx.y == 0 && threadIdx.x == 0){

    cudaStream_t gtream[5];
    for(int j=0;j<5;j++){
            cudaStreamCreateWithFlags(&gtream[j],cudaStreamNonBlocking);
        }

        unsigned int size_A = WARP64 * WARP64;
             double *d_A, *d_B, *d_C;
             int block_size = 32;
             dim3 threads(block_size, block_size);
             dim3 threads16(block_size, block_size/2);
             dim3 grid(2, 2);

        for(int i=0;i<count;i++){
             d_A = d_base + list[i*4+1] * size_A;
             d_B = d_base + list[i*4+2] * size_A;
             d_C = d_base + list[i*4+3] * size_A;

             if(list[i*4] == 1){
                 matrixMulCUDA16<<< grid, threads16, 0, gtream[list[i*4+3]%5] >>>(d_C, d_A, d_B);
       //          matrixMulCUDA16<<< grid, threads16 >>>(d_C, d_A, d_B);
             }
             if(list[i*4] == 2){
                 matrixMulCUDANeg16<<< grid, threads16 ,0, gtream[list[i*4+3]%5] >>>(d_C, d_A, d_B);
      //           matrixMulCUDANeg<<< grid, threads >>>(d_C, d_A, d_B);
             }
             if(list[i*4] == 3){
                matrixSubCUDA<32><<< grid, threads, 0, gtream[list[i*4+3]%5] >>>(d_C, d_A, d_B, 64, 64);
     //            matrixSubCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, 64, 64);
             }
        }
        for(int j=0;j<5;j++)
            cudaStreamDestroy(gtream[j]);
        cudaDeviceSynchronize();
    }
}

__global__ void matrixMulParam1(double* d_base, int a, int b, int c)
{
    __shared__ double As[WARP64*WARP32BLOCK_SIZE];
             double *d_A, *d_B, *d_C;
             d_A = d_base + a * WARP4096;
             d_B = d_base + b * WARP4096;
             d_C = d_base + c * WARP4096;
    matrixMulDevice16(d_C, d_A, d_B, As);
}
        void cudamulparam1(double* d_base, int a, int b, int c, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulParam1<<< grid, threads16, 0, stream >>>(d_base, a, b, c);
    }

template <int PARAM_COUNT> __global__ void matrixMulParam(double* d_base, opparam list)
{
    __shared__ double As[WARP64*WARP32BLOCK_SIZE];
             double *d_A, *d_B, *d_C;
#pragma unroll
    for(int i=0;i<PARAM_COUNT;i++){
             d_A = d_base + list.a[i] * WARP4096;
             d_B = d_base + list.b[i] * WARP4096;
             d_C = d_base + list.c[i] * WARP4096;
    matrixMulDevice16(d_C, d_A, d_B, As);
    }
} 

void cudamulparam8(double* d_base, opparam list, cudaStream_t stream)
{
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulParam<8><<< grid, threads16, 0, stream >>>(d_base, list);
}
        void cudamulparam2(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulParam<2><<< grid, threads16, 0, stream >>>(d_base, list);
    }
        void cudamulparam3(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulParam<3><<< grid, threads16, 0, stream >>>(d_base, list);
    }
        void cudamulparam4(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulParam<4><<< grid, threads16, 0, stream >>>(d_base, list);
    }


__global__ void matrixMulnegParam1(double* d_base, int a, int b, int c)
{
    __shared__ double As[WARP48*WARP32BLOCK_SIZE];
             double *d_A, *d_B, *d_C;
             d_A = d_base + a * WARP4096;
             d_B = d_base + b * WARP4096;
             d_C = d_base + c * WARP4096;
    matrixMulDevice16neg(d_C, d_A, d_B, As);
}
void cudamulnegparam1(double* d_base, int a, int b, int c, cudaStream_t stream)
{
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulnegParam1<<< grid, threads16, 0, stream >>>(d_base, a, b, c);
}

template <int PARAM_COUNT> __global__ void matrixMulnegParam(double* d_base, opparam list)
{
    __shared__ double As[WARP64*WARP32BLOCK_SIZE];
             double *d_A, *d_B, *d_C;
#pragma unroll
    for(int i=0;i<PARAM_COUNT;i++){
             d_A = d_base + list.a[i] * WARP4096;
             d_B = d_base + list.b[i] * WARP4096;
             d_C = d_base + list.c[i] * WARP4096;
    matrixMulDevice16neg(d_C, d_A, d_B, As);
    }
} 

void cudamulnegparam8(double* d_base, opparam list, cudaStream_t stream)
{
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulnegParam<8><<< grid, threads16, 0, stream >>>(d_base, list);
}
        void cudamulnegparam2(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulnegParam<2><<< grid, threads16, 0, stream >>>(d_base, list);
    }
        void cudamulnegparam3(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulnegParam<3><<< grid, threads16, 0, stream >>>(d_base, list);
    }
        void cudamulnegparam4(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP16);
             dim3 grid(2, 2);
             matrixMulnegParam<4><<< grid, threads16, 0, stream >>>(d_base, list);
    }


__global__ void matrixsubParam1(double* d_base, int a, int b, int c)
{
             double *d_A, *d_B, *d_C;
             d_A = d_base ;
             d_B = d_base ;
             d_C = d_base ;
             d_A = &d_A[a];
             d_B = &d_B[b];
             d_C = &d_C[c];
    matrixSubDevice16(d_C, d_A, d_B);
}
        void cudasubparam1(double* d_base, int a, int b, int c, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixsubParam1<<< grid, threads16, 0, stream >>>(d_base, a*WARP4096, b*WARP4096, c*WARP4096);
    }

template <int PARAM_COUNT> __global__ void matrixsubParam(double* d_base, opparam list)
{
             double *d_A, *d_B, *d_C;
#pragma unroll
    for(int i=0;i<PARAM_COUNT;i++){
             d_A = d_base + list.a[i] * WARP4096;
             d_B = d_base + list.b[i] * WARP4096;
             d_C = d_base + list.c[i] * WARP4096;
    matrixSubDevice16(d_C, d_A, d_B);
    }
}

        void cudasubparam2(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixsubParam<2><<< grid, threads16, 0, stream >>>(d_base, list);
    }
        void cudasubparam3(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixsubParam<3><<< grid, threads16, 0, stream >>>(d_base, list);
    }
        void cudasubparam4(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixsubParam<4><<< grid, threads16, 0, stream >>>(d_base, list);
    }

        void cudasubparam8(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixsubParam<8><<< grid, threads16, 0, stream >>>(d_base, list);
    }

__device__ void matrixCopyDevice16(double* A, double* B) //const double* __restrict__ B
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c =  WARP2048 * by + WARP32* bx + WARP64 * ty + tx;
    B[c] = A[c];
}
template <int PARAM_COUNT> __global__ void matrixcopyParam(double* d_base, opparam list)
{
             double *d_A, *d_B;
#pragma unroll
    for(int i=0;i<PARAM_COUNT;i++){
         if(list.a[i] > 0 && list.b[i] > 0){
             d_A = d_base + list.a[i] * WARP4096;
             d_B = d_base + list.b[i] * WARP4096;
             matrixCopyDevice16(d_A, d_B);
         }
    }
}
        void cudacopyparam12(double* d_base, opparam list, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixcopyParam<12><<< grid, threads16, 0, stream >>>(d_base, list);
    }

__global__ void matrixResetParam4(double* d_base, int a, int b, int c, int a1, int b1, int c1, 
                                               int a2, int b2, int c2, int a3, int b3, int c3)
{
             double *d_A, *d_B, *d_C;
             d_A = d_base + a * WARP4096;
             d_B = d_base + b * WARP4096;
             d_C = d_base + c * WARP4096;
    if(c>0) matrixResetDevice16(d_C);
    if(a>0) matrixResetDevice16(d_A);
    if(b>0) matrixResetDevice16(d_B);

             d_A = d_base + a1 * WARP4096;
             d_B = d_base + b1 * WARP4096;
             d_C = d_base + c1 * WARP4096;
    if(c1>0) matrixResetDevice16(d_C);
    if(a1>0) matrixResetDevice16(d_A);
    if(b1>0) matrixResetDevice16(d_B);

             d_A = d_base + a2 * WARP4096;
             d_B = d_base + b2 * WARP4096;
             d_C = d_base + c2 * WARP4096;
    if(c2>0) matrixResetDevice16(d_C);
    if(a2>0) matrixResetDevice16(d_A);
    if(b2>0) matrixResetDevice16(d_B);

             d_A = d_base + a3 * WARP4096;
             d_B = d_base + b3 * WARP4096;
             d_C = d_base + c3 * WARP4096;
    if(c3>0) matrixResetDevice16(d_C);
    if(a3>0) matrixResetDevice16(d_A);
    if(b3>0) matrixResetDevice16(d_B);
}

        void cudaresetparam4(double* d_base, int a, int b, int c, int a1, int b1, int c1, 
                                           int a2, int b2, int c2, int a3, int b3, int c3, cudaStream_t stream)
    {
             dim3 threads16(WARP32, WARP32);
             dim3 grid(2, 2);
             matrixResetParam4<<< grid, threads16, 0, stream >>>(d_base, a, b, c, a1, b1, c1, a2, b2, c2,  a3, b3, c3);
    }

__device__ void matrixInvUpperDevice(double* __restrict__ C, const double*  __restrict__ A)
{  //inv lower 32
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 32
    __shared__ double A11[WARP32][WARP32];
    __shared__ double Y11[WARP32][WARP32];
    __shared__ double D11[WARP32][WARP32];
    __shared__ double E11[WARP32][WARP32];
    __shared__ double Bs[WARP32];
    __shared__ double Hs[WARP32];

//D11 = inv(A22)
//Y11 = inv(A11)
        A11[ty][tx] = A[ty*WARP64+tx+2080];
        E11[ty][tx] = A[ty*WARP64+tx];
    __syncthreads();

    D11[ty][tx] = 0;
    Y11[ty][tx] = 0;
    if(tx == ty){
        double bstx = 1.0 / A11[ty][tx];
        Bs[tx] = bstx;
        D11[tx][tx] = bstx;
        double hstx = 1.0 / E11[ty][tx];
        Y11[ty][tx] = hstx;
        Hs[tx] = hstx;
    }
    __syncthreads();
    A11[ty][tx] = A11[ty][tx] * Bs[ty];
    E11[ty][tx] = E11[ty][tx] * Hs[ty];
    __syncthreads();
    

    for (int j=WARP32-1; j>0; j--)
    {
        if(ty<j){
            D11[ty][tx] -= D11[j][tx] * A11[ty][j] ;
            Y11[ty][tx] -= Y11[j][tx] * E11[ty][j];
        }
        __syncthreads();
    }
    C[ty*WARP64+tx+2080] = D11[ty][tx];

    C[ty*WARP64+tx] = Y11[ty][tx];
//    C[ty*WARP64+tx+WARP32] = 0;

//A1B = inv(A11) * A12
    E11[ty][tx] = A[ty*WARP64+tx+WARP32];
    __syncthreads();

    double Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += Y11[ty][k] * E11[k][tx];
        }
    A11[ty][tx] = Csub;

    __syncthreads();
//C12 = -A1BD1
    Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * D11[k][tx];
        }
    C[ty*WARP64+tx+WARP32] = -Csub;
}

__device__ void matrixInvLowerDevice(double* __restrict__ C, const double*  __restrict__ A)
{  //inv lower 32
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 32
    __shared__ double A11[WARP32][WARP32];
    __shared__ double Y11[WARP32][WARP32];
    __shared__ double D11[WARP32][WARP32];
    __shared__ double E11[WARP32][WARP32];

//D11 = inv(A22)
//Y11 = inv(A11)
        A11[ty][tx] = A[ty*WARP64+tx+2080];
        E11[ty][tx] = A[ty*WARP64+tx];
    __syncthreads();

    D11[ty][tx] = 0;
        Y11[ty][tx] = 0;
    if(tx == ty){
        D11[ty][tx] = 1.0;
        Y11[ty][tx] = 1.0;
    }
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty>j){
            D11[ty][tx] -= D11[j][tx] * A11[ty][j]; 
            Y11[ty][tx] -= Y11[j][tx] * E11[ty][j]; 
        }
        __syncthreads();
    }
    C[ty*WARP64+tx+2080] = D11[ty][tx];
    C[ty*WARP64+tx] = Y11[ty][tx];

//    C[ty*WARP64+tx+WARP32] = 0;

//D1C = inv(A22) * A21
    E11[ty][tx] = A[ty*WARP64+tx+WARP2048];
    __syncthreads();

    double Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    A11[ty][tx] = Csub;

    __syncthreads();
//C21 = -D1CA1
    Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * Y11[k][tx];
        }
    C[ty*WARP64+tx+WARP2048] = -Csub;
}

/*   __device__ void matrixInvLowerDevice64(double* __restrict__ C, const double*  __restrict__ A,
                                     double* As, double* Bs)
{    // inv_lower 64 

    int x = threadIdx.x; // 64
    int ty = threadIdx.y; // 16


    for (int a = 0; a < WARP64; a+=16)
    {
        if(x<=a+ty)
            As[(a+ty)*WARP65 + x] = A[(a+ty)*WARP64 + x];
        else
            As[(a+ty)*WARP65 + x] = 0;
    }

    __syncthreads();

    if(ty==0){
    double Csub = 0;
    Csub = As[x * WARP65 + x];
    Bs[x] = 1.0 / Csub;
    As[x * WARP65 + x] = 1.0;
    }

    __syncthreads();

    for (int j=0; j<WARP64; j++)
    {
        int k = x;
        if(ty == 0 && k<=j){
        double lj1 = Bs[j];
      //  if(k<=j)
        for (int i=j+1; i<WARP64; i++)
       {
 //           for(int k=0;k<=j;k++)
                As[WARP65*(WARP63-i)+WARP63-k] -= lj1 * As[WARP65*(WARP63-j)+WARP63-k] * As[WARP65*i + j]; 
        }
        }
        __syncthreads();
    }

    for (int a = 0; a < WARP64; a+=16)
    {
        if(x<=a+ty)
            C[(a+ty)*WARP64 + x] = As[(WARP63-(a+ty))*WARP65 + WARP63-x];
        else
            C[(a+ty)*WARP64 + x] = 0;
    }
}      */

__global__ void matrixinvupper(double* d_base, int a, int c)
{
  //  __shared__ double As[WARP64*WARP65];
  //  __shared__ double Bs[WARP64];
             double *d_A, *d_C;
             d_A = d_base + a * WARP4096;
             d_C = d_base + c * WARP4096;
  //  matrixInvUpperDevice(d_C, d_A, As, Bs);
    matrixInvUpperDevice(d_C, d_A);
}

void cudainvupper(double* d_base, int a, int c, cudaStream_t stream)
{   
 //  dim3 threads(64,16);
   dim3 threads(32,32);
   matrixinvupper<<<1, threads,0,stream>>>(d_base, a, c);
}

__global__ void matrixinvlower(double* d_base, int a, int c)
{
  //  __shared__ double As[WARP64*WARP65];
  //  __shared__ double Bs[WARP64];
             double *d_A, *d_C;
             d_A = d_base + a * WARP4096;
             d_C = d_base + c * WARP4096;
  //  matrixInvLowerDevice(d_C, d_A, As, Bs);
    matrixInvLowerDevice(d_C, d_A);
}


__device__ void matrixLUDevice_npv(double* __restrict__ L, double* __restrict__ U, const double*  __restrict__ A)
{  //inv lower 32
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 32
    __shared__ double A11[WARP32][WARP32];
    __shared__ double Y11[WARP32][WARP32];
    __shared__ double D11[WARP32][WARP32];
    __shared__ double E11[WARP32][WARP32];
    __shared__ double Hs[WARP32];

//P * A11 = L1 * U1
//Y11 = Pt * L1
        E11[ty][tx] = A[ty*WARP64+tx];

    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            if(fabs(E11[j][j])<TINY){
               if(signbit(E11[j][j]))
                   E11[j][j] = -TINY;
               else
                   E11[j][j] = TINY;
            }
            Hs[j] = 1.0 / E11[j][j];
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    Y11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        Y11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    L[ty*WARP64+tx] = Y11[ty][tx];
 
    U[ty*WARP64+tx] = E11[ty][tx];


// D11 = inv( Y11 ) inv (L11)
    D11[ty][tx] = 0;
    __syncthreads();
    if(0 == ty){
        D11[tx][tx] = 1.0;
    }

    __syncthreads();
    A11[ty][tx] = A[ty*WARP64+tx+WARP32];

    for (int j=0; j<WARP32; j++)
    {
        if(ty>j){
            D11[ty][tx] -= D11[j][tx] * Y11[ty][j]; 
        }
        __syncthreads();
    }

    double Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * A11[k][tx];
        }
    U[ty*WARP64+tx+WARP32] = Csub;
    L[ty*WARP64+tx+WARP32] = 0;
    Y11[ty][tx] = Csub;

     __syncthreads();
    A11[ty][tx] = E11[ty][tx];
    __syncthreads();

// D11 = inv( A11 ) inv( U11 )
    D11[ty][tx] = 0;
    __syncthreads();
    if(tx == ty){
            if(fabs(A11[ty][tx])<TINY){
               if(signbit(A11[ty][tx]))
                   A11[ty][tx] = -TINY;
               else
                   A11[ty][tx] = TINY;
            }
        double bstx = 1.0 / A11[ty][tx];
        Hs[tx] = bstx;
        D11[tx][tx] = bstx;
    }
    __syncthreads();
    E11[ty][tx] = A[ty*WARP64+tx+WARP2048];
    A11[ty][tx] = A11[ty][tx] * Hs[ty];
    __syncthreads();
    

    for (int j=WARP32-1; j>0; j--)
    {
        if(ty<j){
            D11[ty][tx] -= D11[j][tx] * A11[ty][j] ;
        }
        __syncthreads();
    }

    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += E11[ty][k] * D11[k][tx];
        }
    L[ty*WARP64+tx+WARP2048] = Csub;
    U[ty*WARP64+tx+WARP2048] = 0;
    A11[ty][tx] = Csub;
//E11 = A22
    __syncthreads();
    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * Y11[k][tx];
        }
    E11[ty][tx] = A[ty*WARP64+tx+2080] - Csub;

    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            if(fabs(E11[j][j])<TINY){
               if(signbit(E11[j][j]))
                   E11[j][j] = -TINY;
               else
                   E11[j][j] = TINY;
            }
            Hs[j] = 1.0 / E11[j][j];
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    L[ty*WARP64+tx+2080] = D11[ty][tx];
 
    U[ty*WARP64+tx+2080] = E11[ty][tx];
}




__device__ void matrixLUDevice_det(double* __restrict__ L, double* __restrict__ U, const double*  __restrict__ A)
{  //inv lower 32
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 32
    __shared__ double A11[WARP32][WARP32];
    __shared__ double Y11[WARP32][WARP32];
    __shared__ double D11[WARP32][WARP32];
    __shared__ double E11[WARP32][WARP32];
    __shared__ double Hs[WARP32];
    __shared__ int    P[WARP32];
    __shared__ int    pivot;
    __shared__ double det[WARP32];

//P * A11 = L1 * U1
//Y11 = Pt * L1
        E11[ty][tx] = A[ty*WARP64+tx];

    if(0 == ty){
        P[tx] = tx;
    }
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            double mx = fabs(E11[j][j]);
            pivot = j;
            for(int i=j+1; i<WARP32; ++i){
                if(fabs(E11[i][j]) > mx){  //absa = fabs(E11[i][j]) > mx
                    mx = fabs(E11[i][j]);
                    pivot = i;
                }
            }
            if(fabs(E11[pivot][j])<TINY){
               if(signbit(E11[pivot][j]))
                   E11[pivot][j] = -TINY;
               else
                   E11[pivot][j] = TINY;
            }
            Hs[j] = 1.0 / E11[pivot][j];
        }
        __syncthreads();
        if(pivot != j && ty == j){
            double tt = E11[j][tx];
            E11[j][tx] = E11[pivot][tx];
            E11[pivot][tx] = tt;
            if(tx==0){
                int i = P[j];
                P[j] = P[pivot];
                P[pivot] = i;
            }
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
               // E11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }
    if(ty==0){
        det[tx] = E11[tx][tx];
    }
    __syncthreads();

//P * A11 = L1 * U1
//Y11 = Pt * L1
        E11[ty][tx] = A[ty*WARP64+tx];

    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            if(fabs(E11[j][j])<TINY){
               if(signbit(E11[j][j]))
                   E11[j][j] = -TINY;
               else
                   E11[j][j] = TINY;
            }
            Hs[j] = 1.0 / E11[j][j];
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }
    if(ty==0 && tx==0){
        double min = fabs(E11[0][0]);
        int mintx = 0;
        for(int k=1; k<WARP32;k++){
            if(min > fabs(E11[k][k])){
                min = fabs(E11[k][k]);
                mintx = k;
            }
        }
        double multi = 1.0;
        for(int k=0;k<WARP32;k++){
            multi = multi * det[k];
            if(k != mintx)
                multi = multi / E11[k][k];
        }
      if(E11[mintx][mintx] / multi > 1.01 || E11[mintx][mintx] / multi < 0.99)
      printf("replace: %g %g \n",E11[mintx][mintx], multi);
//        E11[mintx][mintx] = multi;
    }

    Y11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        Y11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    L[ty*WARP64+tx] = Y11[ty][tx];
 
    U[ty*WARP64+tx] = E11[ty][tx];


// D11 = inv( Y11 ) inv (L11)
    D11[ty][tx] = 0;
    __syncthreads();
    if(0 == ty){
        D11[tx][tx] = 1.0;
    }

    __syncthreads();
    A11[ty][tx] = A[ty*WARP64+tx+WARP32];

    for (int j=0; j<WARP32; j++)
    {
        if(ty>j){
            D11[ty][tx] -= D11[j][tx] * Y11[ty][j]; 
        }
        __syncthreads();
    }

    double Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * A11[k][tx];
        }
    U[ty*WARP64+tx+WARP32] = Csub;
    L[ty*WARP64+tx+WARP32] = 0;
    Y11[ty][tx] = Csub;

     __syncthreads();
    A11[ty][tx] = E11[ty][tx];
    __syncthreads();

// D11 = inv( A11 ) inv( U11 )
    D11[ty][tx] = 0;
    __syncthreads();
    if(tx == ty){
            if(fabs(A11[ty][tx])<TINY){
               if(signbit(A11[ty][tx]))
                   A11[ty][tx] = -TINY;
               else
                   A11[ty][tx] = TINY;
            }
        double bstx = 1.0 / A11[ty][tx];
        Hs[tx] = bstx;
        D11[tx][tx] = bstx;
    }
    __syncthreads();
    E11[ty][tx] = A[ty*WARP64+tx+WARP2048];
    A11[ty][tx] = A11[ty][tx] * Hs[ty];
    __syncthreads();
    

    for (int j=WARP32-1; j>0; j--)
    {
        if(ty<j){
            D11[ty][tx] -= D11[j][tx] * A11[ty][j] ;
        }
        __syncthreads();
    }

    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += E11[ty][k] * D11[k][tx];
        }
    L[ty*WARP64+tx+WARP2048] = Csub;
    U[ty*WARP64+tx+WARP2048] = 0;
    A11[ty][tx] = Csub;


//E11 = A22
    __syncthreads();
    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * Y11[k][tx];
        }
    E11[ty][tx] = A[ty*WARP64+tx+2080] - Csub;

    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            if(fabs(E11[j][j])<TINY){
               if(signbit(E11[j][j]))
                   E11[j][j] = -TINY;
               else
                   E11[j][j] = TINY;
            }
            Hs[j] = 1.0 / E11[j][j];
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    L[ty*WARP64+tx+2080] = D11[ty][tx];
 
    U[ty*WARP64+tx+2080] = E11[ty][tx];

}


__device__ void matrixLUDevice_64(double* __restrict__ L, double* __restrict__ U, const double*  __restrict__ A)
{  //inv lower 32
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 32
    __shared__ double A11[WARP32][WARP32];
    __shared__ double Y11[WARP32][WARP32];
    __shared__ double D11[WARP32][WARP32];
    __shared__ double E11[WARP32][WARP32];
    __shared__ double Hs[WARP32];
    __shared__ int    P[WARP32];
    __shared__ int    pivot;

//P * A11 = L1 * U1
//Y11 = Pt * L1
        E11[ty][tx] = A[ty*WARP64+tx];
/* /*
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            if(fabs(E11[j][j])<TINY){
               if(signbit(E11[j][j]))
                   E11[j][j] = -TINY;
               else
                   E11[j][j] = TINY;
            }
            Hs[j] = 1.0 / E11[j][j];
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    Y11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        Y11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    L[ty*WARP64+tx] = Y11[ty][tx];
 
    U[ty*WARP64+tx] = E11[ty][tx];
    A11[ty][tx] = E11[ty][tx];
    __syncthreads();

*/

    if(0 == ty){
        P[tx] = tx;
    }
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            double mx = fabs(E11[j][j]);
            pivot = j;
            for(int i=j+1; i<WARP32; ++i){
                if(fabs(E11[i][j]) > mx){  //absa = fabs(E11[i][j]) > mx
                    mx = fabs(E11[i][j]);
                    pivot = i;
                }
            }
            if(fabs(E11[pivot][j])<TINY){
               if(signbit(E11[pivot][j]))
                   E11[pivot][j] = -TINY;
               else
                   E11[pivot][j] = TINY;
            }
            Hs[j] = 1.0 / E11[pivot][j];
        }
        __syncthreads();
        if(tx ==0 && ty==0 && pivot != j)
            printf("step: %d, pivot: %d\n",j,pivot);
        if(pivot != j && ty == j){
            double tt = E11[j][tx];
            E11[j][tx] = E11[pivot][tx];
            E11[pivot][tx] = tt;
            if(tx==0){
                int i = P[j];
                P[j] = P[pivot];
                P[pivot] = i;
            }
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
               // E11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    Y11[ty][tx] = D11[P[ty]][tx];
 //   Y11[P[ty]][tx] = D11[ty][tx];
    __syncthreads();
//Y11 = LU( Y11 )
//Y11 = L11
    for (int j=0; j<WARP32; j++)
    {
        if(ty == 0 && tx == 0){
            if(fabs(Y11[j][j])<TINY){
               if(signbit(Y11[j][j]))
                   Y11[j][j] = -TINY;
               else
                   Y11[j][j] = TINY;
            }
            Hs[j] = 1.0 / Y11[j][j];   // Y11[j][j] = 0;
        }
        __syncthreads();
        if(ty == 0 && tx > j){              // __syncthreads();
                Y11[tx][j] *= Hs[j];
         //       Y11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            Y11[ty][tx] -= Y11[ty][j] * Y11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(tx >= ty){
        D11[ty][tx] = Y11[ty][tx];
        Y11[ty][tx] = 0;
    }
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    L[ty*WARP64+tx] = Y11[ty][tx];
 
    __syncthreads();
// A11 = U11
    double Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    A11[ty][tx] = Csub;
    U[ty*WARP64+tx] = A11[ty][tx];

/* */

// D11 = inv( Y11 ) inv (L11)
    D11[ty][tx] = 0;
    __syncthreads();
    if(0 == ty){
        D11[tx][tx] = 1.0;
    }

    __syncthreads();
    E11[ty][tx] = A[ty*WARP64+tx+WARP32];

    for (int j=0; j<WARP32; j++)
    {
        if(ty>j){
            D11[ty][tx] -= D11[j][tx] * Y11[ty][j]; 
        }
        __syncthreads();
    }

    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    __syncthreads();
    U[ty*WARP64+tx+WARP32] = Csub;
    L[ty*WARP64+tx+WARP32] = 0;
    Y11[ty][tx] = Csub;

// D11 = inv( A11 ) inv( U11 )
    D11[ty][tx] = 0;
    __syncthreads();
    if(tx == ty){
            if(fabs(A11[ty][tx])<TINY){
               if(signbit(A11[ty][tx]))
                   A11[ty][tx] = -TINY;
               else
                   A11[ty][tx] = TINY;
            }
        double bstx = 1.0 / A11[ty][tx];
        Hs[tx] = bstx;
        D11[tx][tx] = bstx;
    }
    __syncthreads();
    E11[ty][tx] = A[ty*WARP64+tx+WARP2048];
    A11[ty][tx] = A11[ty][tx] * Hs[ty];
    __syncthreads();
    

    for (int j=WARP32-1; j>0; j--)
    {
        if(ty<j){
            D11[ty][tx] -= D11[j][tx] * A11[ty][j] ;
        }
        __syncthreads();
    }

    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += E11[ty][k] * D11[k][tx];
        }
    L[ty*WARP64+tx+WARP2048] = Csub;
    U[ty*WARP64+tx+WARP2048] = 0;
    A11[ty][tx] = Csub;
/*
//E11 = A22
    __syncthreads();
    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * Y11[k][tx];
        }
    E11[ty][tx] = A[ty*WARP64+tx+2080] - Csub;

    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            if(fabs(E11[j][j])<TINY){
               if(signbit(E11[j][j]))
                   E11[j][j] = -TINY;
               else
                   E11[j][j] = TINY;
            }
            Hs[j] = 1.0 / E11[j][j];
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    L[ty*WARP64+tx+2080] = D11[ty][tx];
 
    U[ty*WARP64+tx+2080] = E11[ty][tx];

*/
//E11 = A22
    __syncthreads();
    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * Y11[k][tx];
        }
    E11[ty][tx] = A[ty*WARP64+tx+2080] - Csub;

    if(0 == ty){
        P[tx] = tx;
    }
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            double mx = fabs(E11[j][j]);
            pivot = j;
            for(int i=j+1; i<WARP32; ++i){
                if(fabs(E11[i][j]) > mx){  //absa = fabs(E11[i][j]) > mx
                    mx = fabs(E11[i][j]);
                    pivot = i;
                }
            }
            if(fabs(E11[pivot][j])<TINY){
               if(signbit(E11[pivot][j]))
                   E11[pivot][j] = -TINY;
               else
                   E11[pivot][j] = TINY;
            }
            Hs[j] = 1.0 / E11[pivot][j];  //
        }
        __syncthreads();
        if(pivot != j && ty == j){
            double tt = E11[j][tx];
            E11[j][tx] = E11[pivot][tx];
            E11[pivot][tx] = tt;
            if(tx==0){
                int i = P[j];
                P[j] = P[pivot];
                P[pivot] = i;
            }
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
       //         E11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    Y11[ty][tx] = D11[P[ty]][tx];
 //   Y11[P[ty]][tx] = D11[ty][tx];
    __syncthreads();
//Y11 = LU( Y11 )
//Y11 = L11
    for (int j=0; j<WARP32; j++)
    {
        if(ty == 0 && tx == 0){
            if(fabs(Y11[j][j])<TINY){
               if(signbit(Y11[j][j]))
                   Y11[j][j] = -TINY;
               else
                   Y11[j][j] = TINY;
            }
            Hs[j] = 1.0 / Y11[j][j];   // Y11[j][j] = 0;
        }
        __syncthreads();
        if(ty == 0 && tx > j){              // __syncthreads();
                Y11[tx][j] *= Hs[j];
           //     Y11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            Y11[ty][tx] -= Y11[ty][j] * Y11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(tx >= ty){
        D11[ty][tx] = Y11[ty][tx];
        Y11[ty][tx] = 0;
    }
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    L[ty*WARP64+tx+2080] = Y11[ty][tx];
 
// A11 = U11
    Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    A11[ty][tx] = Csub;
    U[ty*WARP64+tx+2080] = A11[ty][tx];
/* */
}



__device__ void matrixLUDevice_64_1(double* __restrict__ L, double* __restrict__ U, const double*  __restrict__ A)
{  //inv lower 32
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 32
    __shared__ double A11[WARP32][WARP32];
    __shared__ double Y11[WARP32][WARP32];
    __shared__ double D11[WARP32][WARP32];
    __shared__ double E11[WARP32][WARP32];
    __shared__ double Hs[WARP32];
    __shared__ int    P[WARP32];
    __shared__ int    pivot;
    __shared__ double det[WARP32];

//P * A11 = L1 * U1
//Y11 = Pt * L1
        E11[ty][tx] = A[ty*WARP64+tx];

    if(0 == ty){
        P[tx] = tx;
    }
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            double mx = fabs(E11[j][j]);
            pivot = j;
            for(int i=j+1; i<WARP32; ++i){
                if(fabs(E11[i][j]) > mx){  //absa = fabs(E11[i][j]) > mx
                    mx = fabs(E11[i][j]);
                    pivot = i;
                }
            }
            if(fabs(E11[pivot][j])<TINY){
               if(signbit(E11[pivot][j]))
                   E11[pivot][j] = -TINY;
               else
                   E11[pivot][j] = TINY;
            }
            Hs[j] = 1.0 / E11[pivot][j];
        }
        __syncthreads();
        if(pivot != j && ty == j){
            double tt = E11[j][tx];
            E11[j][tx] = E11[pivot][tx];
            E11[pivot][tx] = tt;
            if(tx==0){
                int i = P[j];
                P[j] = P[pivot];
                P[pivot] = i;
            }
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
               // E11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
        det[tx] = E11[tx][tx];
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    Y11[ty][tx] = D11[P[ty]][tx];
 //   Y11[P[ty]][tx] = D11[ty][tx];
    __syncthreads();
//Y11 = LU( Y11 )
//Y11 = L11
    for (int j=0; j<WARP32; j++)
    {
        if(ty == 0 && tx == 0){
            if(fabs(Y11[j][j])<TINY){
               if(signbit(Y11[j][j]))
                   Y11[j][j] = -TINY;
               else
                   Y11[j][j] = TINY;
            }
            Hs[j] = 1.0 / Y11[j][j];   // Y11[j][j] = 0;
        }
        __syncthreads();
        if(ty == 0 && tx > j){              // __syncthreads();
                Y11[tx][j] *= Hs[j];
         //       Y11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            Y11[ty][tx] -= Y11[ty][j] * Y11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(tx >= ty){
        D11[ty][tx] = Y11[ty][tx];
        Y11[ty][tx] = 0;
    }
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    L[ty*WARP64+tx] = Y11[ty][tx];
 
// A11 = U11
    double Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    A11[ty][tx] = Csub;
    U[ty*WARP64+tx] = A11[ty][tx];

    if(ty==0 && tx==0){
        double min = fabs(A11[0][0]);
        int mintx = 0;
        for(int k=1; k<WARP32;k++){
            if(min > fabs(A11[k][k])){
                min = fabs(A11[k][k]);
                mintx = k;
            }
        }
        double multi = 1.0;
        for(int k=0;k<WARP32;k++){
            multi = multi * det[k];
            if(k != mintx)
                multi = multi / A11[k][k];
        }
      if(A11[mintx][mintx] / multi > 1.01 || A11[mintx][mintx] / multi < 0.99)
      printf("replace: %g %g \n",A11[mintx][mintx], multi);
    }

// D11 = inv( Y11 ) inv (L11)
    D11[ty][tx] = 0;
    __syncthreads();
    if(0 == ty){
        D11[tx][tx] = 1.0;
    }

    __syncthreads();
    E11[ty][tx] = A[ty*WARP64+tx+WARP32];

    for (int j=0; j<WARP32; j++)
    {
        if(ty>j){
            D11[ty][tx] -= D11[j][tx] * Y11[ty][j]; 
        }
        __syncthreads();
    }

    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    __syncthreads();
    U[ty*WARP64+tx+WARP32] = Csub;
    L[ty*WARP64+tx+WARP32] = 0;
    Y11[ty][tx] = Csub;

// D11 = inv( A11 ) inv( U11 )
    D11[ty][tx] = 0;
    __syncthreads();
    if(tx == ty){
            if(fabs(A11[ty][tx])<TINY){
               if(signbit(A11[ty][tx]))
                   A11[ty][tx] = -TINY;
               else
                   A11[ty][tx] = TINY;
            }
        double bstx = 1.0 / A11[ty][tx];
        Hs[tx] = bstx;
        D11[tx][tx] = bstx;
    }
    __syncthreads();
    E11[ty][tx] = A[ty*WARP64+tx+WARP2048];
    A11[ty][tx] = A11[ty][tx] * Hs[ty];
    __syncthreads();
    

    for (int j=WARP32-1; j>0; j--)
    {
        if(ty<j){
            D11[ty][tx] -= D11[j][tx] * A11[ty][j] ;
        }
        __syncthreads();
    }

    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += E11[ty][k] * D11[k][tx];
        }
    L[ty*WARP64+tx+WARP2048] = Csub;
    U[ty*WARP64+tx+WARP2048] = 0;
    A11[ty][tx] = Csub;
//E11 = A22
    __syncthreads();
    Csub = 0;
    for (int k = 0; k < WARP32; ++k)
        {
            Csub += A11[ty][k] * Y11[k][tx];
        }
    E11[ty][tx] = A[ty*WARP64+tx+2080] - Csub;

    if(0 == ty){
        P[tx] = tx;
    }
    __syncthreads();

    for (int j=0; j<WARP32; j++)
    {
        if(ty ==0 && tx == 0){
            double mx = fabs(E11[j][j]);
            pivot = j;
            for(int i=j+1; i<WARP32; ++i){
                if(fabs(E11[i][j]) > mx){  //absa = fabs(E11[i][j]) > mx
                    mx = fabs(E11[i][j]);
                    pivot = i;
                }
            }
            if(fabs(E11[pivot][j])<TINY){
               if(signbit(E11[pivot][j]))
                   E11[pivot][j] = -TINY;
               else
                   E11[pivot][j] = TINY;
            }
            Hs[j] = 1.0 / E11[pivot][j];  //
        }
        __syncthreads();
        if(pivot != j && ty == j){
            double tt = E11[j][tx];
            E11[j][tx] = E11[pivot][tx];
            E11[pivot][tx] = tt;
            if(tx==0){
                int i = P[j];
                P[j] = P[pivot];
                P[pivot] = i;
            }
        }
        __syncthreads();
        if(ty == j && tx > j){              // __syncthreads();
                E11[tx][j] *= Hs[j];
       //         E11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            E11[ty][tx] -= E11[ty][j] * E11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(ty == 0){
        D11[tx][tx] = 1;
    }
    __syncthreads();
    if(tx < ty){
        D11[ty][tx] = E11[ty][tx];
        E11[ty][tx] = 0;
    }
    __syncthreads();
    Y11[ty][tx] = D11[P[ty]][tx];
 //   Y11[P[ty]][tx] = D11[ty][tx];
    __syncthreads();
//Y11 = LU( Y11 )
//Y11 = L11
    for (int j=0; j<WARP32; j++)
    {
        if(ty == 0 && tx == 0){
            if(fabs(Y11[j][j])<TINY){
               if(signbit(Y11[j][j]))
                   Y11[j][j] = -TINY;
               else
                   Y11[j][j] = TINY;
            }
            Hs[j] = 1.0 / Y11[j][j];   // Y11[j][j] = 0;
        }
        __syncthreads();
        if(ty == 0 && tx > j){              // __syncthreads();
                Y11[tx][j] *= Hs[j];
           //     Y11[j][tx] *= Hs[j];
        }
        __syncthreads();
        if(ty>j && tx>j){
            Y11[ty][tx] -= Y11[ty][j] * Y11[j][tx]; 
        }
        __syncthreads();
    }

    D11[ty][tx] = 0;
    __syncthreads();
    if(tx >= ty){
        D11[ty][tx] = Y11[ty][tx];
        Y11[ty][tx] = 0;
    }
    __syncthreads();
    if(ty == 0){
        Y11[tx][tx] = 1;
    }
    __syncthreads();
    L[ty*WARP64+tx+2080] = Y11[ty][tx];
 
// A11 = U11
    Csub = 0;
        for (int k = 0; k < WARP32; ++k)
        {
            Csub += D11[ty][k] * E11[k][tx];
        }
    A11[ty][tx] = Csub;
    U[ty*WARP64+tx+2080] = A11[ty][tx];
}


__global__ void matrixLU64(double* d_base, int a, int l, int u)
{
             double *d_A, *d_L, *d_U;
             d_A = d_base + a * WARP4096;
             d_L = d_base + l * WARP4096;
             d_U = d_base + u * WARP4096;
    matrixLUDevice_64(d_L, d_U, d_A);
}


void cudaLU64(double* d_base, int a, int l, int u, cudaStream_t stream)
{
     dim3 threads(32,32);
   matrixLU64<<<1, threads,0,stream>>>(d_base, a, l, u);
}

void cudainvlower(double* d_base, int a, int c, cudaStream_t stream)
{   
 //  dim3 threads(64,16);
   dim3 threads(32,32);
   matrixinvlower<<<1, threads,0,stream>>>(d_base, a, c);
}


        void cudamullist(double* d_base, int* list, int n, cudaStream_t stream)
        {
            matrixMulCUDAlist<<<1, 1,0,stream>>>(d_base, list, n);
        }

        void mat_mul(double* a, double* b, int n, double* y)
        {
/*             unsigned int size_A = n * n;
             unsigned int mem_size_A = sizeof(double) * size_A;
             double *d_A, *d_B, *d_C;
             cudaError_t error;
             int block_size = 32;

    		error = cudaMalloc((void **) &d_A, mem_size_A);
    		error = cudaMalloc((void **) &d_B, mem_size_A);
    		error = cudaMalloc((void **) &d_C, mem_size_A);

             error = cudaMemcpy(d_A, a, mem_size_A, cudaMemcpyHostToDevice);
             error = cudaMemcpy(d_B, b, mem_size_A, cudaMemcpyHostToDevice);
             error = cudaMemcpy(d_C, y, mem_size_A, cudaMemcpyHostToDevice);

             dim3 threads(block_size, block_size);
             dim3 grid(n / threads.x, n / threads.y);
              matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, n, n);
             cudaDeviceSynchronize();
             cudaEvent_t stop;
    		error = cudaEventCreate(&stop);
              error = cudaEventRecord(stop, NULL);
             error = cudaEventSynchronize(stop);
             error = cudaMemcpy(y, d_C, mem_size_A, cudaMemcpyDeviceToHost);
             if (error != cudaSuccess)
        	printf("cudaGetDevice  %s (code %d)\n", cudaGetErrorString(error), error);
             cudaFree(d_A);
             cudaFree(d_B);
             cudaFree(d_C);
*/
        }

}

// Regarding item 1, cuda dynamic parallelism requires separate compilation and linking (-rdc=true), 
// as well as linking in of the device cudart libraries (-lcudadevrt). Dynamic parallelism that also uses 
// CUBLAS will also require linking in the device CUBLAS library (-lcublas_device).

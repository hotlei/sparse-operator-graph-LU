
#include <cuda_runtime.h>

namespace vmatrix
{
       struct opparam{int count; int a[12]; int b[12]; int c[12];};
       void cudamullist(double* d_base, int* list, int n, cudaStream_t stream);

       void cudamulone(double* d_base, int a, int b, int n, int y, cudaStream_t stream);
       void cudamulone2x(double* d_base, int a, int b, int n, int y, int a1, int b1, cudaStream_t stream);
       void cudamulone2xt(double* d_base, int a, int b, int n, int y, int a1, int b1, double* buf, cudaStream_t stream);
       void cudamulone2y(double* d_base, int a, int b, int n, int y, int a1, int b1, int y1, cudaStream_t stream);
       void cudamulone2yt(double* d_base, int a, int b, int n, int y, int a1, int b1, int y1, double* buf, cudaStream_t stream);
       void cudasubone(double* d_base, int a, int b, int n, int y, cudaStream_t stream);
       void cudamuloneNeg(double* d_base, int a, int b, int n, int y, cudaStream_t stream);
       void mat_mul(double* a, double* b, int n, double* y);

       void cudamulparam1(double* d_base, int a, int b, int c, cudaStream_t stream);
       void cudamulparam8(double* d_base, opparam list, cudaStream_t stream);
       void cudamulparam2(double* d_base, opparam list, cudaStream_t stream);
       void cudamulparam3(double* d_base, opparam list, cudaStream_t stream);
       void cudamulparam4(double* d_base, opparam list, cudaStream_t stream);

       void cudamulnegparam1(double* d_base, int a, int b, int c, cudaStream_t stream);
       void cudamulnegparam8(double* d_base, opparam list, cudaStream_t stream);
       void cudamulnegparam2(double* d_base, opparam list, cudaStream_t stream);
       void cudamulnegparam3(double* d_base, opparam list, cudaStream_t stream);
       void cudamulnegparam4(double* d_base, opparam list, cudaStream_t stream);

       void cudasubparam1(double* d_base, int a, int b, int c, cudaStream_t stream);
       void cudasubparam2(double* d_base, opparam list, cudaStream_t stream);
       void cudasubparam3(double* d_base, opparam list, cudaStream_t stream);
       void cudasubparam4(double* d_base, opparam list, cudaStream_t stream);
       void cudasubparam8(double* d_base, opparam list, cudaStream_t stream);

       void cudacopyparam12(double* d_base, opparam list, cudaStream_t stream);
       void cudaresetparam4(double* d_base, int a, int b, int c, int a1, int b1, int c1,
                                           int a2, int b2, int c2, int a3, int b3, int c3, cudaStream_t stream);
       void cudainvupper(double* d_base, int a, int c, cudaStream_t stream);
       void cudainvlower(double* d_base, int a, int c, cudaStream_t stream);
       void cudaLU64(double* d_base, int a, int l, int u, cudaStream_t stream);
}

#define WARP32 32

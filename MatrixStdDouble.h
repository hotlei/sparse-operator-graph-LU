
namespace vmatrix
{
    class MatrixStdDouble
    {

    public:
        static void blockMulOneBlas(double a[], double b[], double c[]);
        static void blockMulOneAvxBlock(double a[], double b[], double c[], unsigned long msk1);
        static void blockMulOneAvxBlockTrans(double a[], double b[], double c[], unsigned long msk1, unsigned long msk2);
        static void blockMulOneAvx(double a[], double b[], double c[]);
        static void blockMulOneNegAvx(double a[], double b[], double c[]);
        static double* CopyToStd();

        static int countNonZero(double* a, int n);

        static int checkUpperZero(double* a, int n);

        static int checkLowerZero(double* a, int n);

        static double* getNonZero(double* a, int n);

        static int countNaN(double* a, int n);
        static void printMatrix(double* a, int n);

        static void CopyToStd(int* a, int n, double* std, int limit);

        static void ludcmpSimple(double* a, int n, double* l, double* u);

        static void inv_lower(double* l, int n, double* y);

        static void inv_upper(double* u, int n, double* y);

        static void ludcmp(double* a, int n, int* indx, double* d);

        static void lubksb(double* a, int n, int* indx, double* b);

        static void mat_decomp_only(double* a, int n);

        static void mat_inv(double* a, int n, double* y);

        static void mat_mul(double* a, double* b, int n, int m, int w, double* y);

        static bool inv_check_diag(double* a, double* b, int n);

        static void inv_check_zero(double* a, double* b, int n);

        static bool mat_equ(double* a, double* b, int n);

        static void vec_equ(double* a, double* b, int n);

        static double* mat_shrink(double* a, int n, int small);

        static void mat_sub(double* a, double* b, int n, double* r);
        static void mat_copy(double* a, int n, double* r);
        static void mat_neg(double* b, int n, double* r);
    };
}


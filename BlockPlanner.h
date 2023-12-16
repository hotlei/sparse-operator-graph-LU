
namespace vmatrix
{
    class BlockPlanner
    {
        public:
        static int countBlockEntry(int a[], int n);

        static void blockPlan(int saved[], int savedLength, int savedu[], int saveduLength);
        static void copySeqL2(int a[], int l[], int u[], int **bl2, int **bu2, int n);

        static void calculate();

        static void sovle(int bl[], int bu[], double b[], int n);

        static void assignB();

        static void solveUpper(double bu[], double y[], double x[], int n, int offset);
        static void bx(double bu[], double y[], double x[], int n, int ioffset, int joffset);

        static void solveLower(double bl[], double b[], double y[], int n, int offset);

        static void cy(double bl[], double b[], double y[], int n, int ioffset, int joffset);
        
       static int countStrorageEntry(int n, int stage);

        static int  checkMatrixEntry(int a[], int n);
        static double* checkDiagEntry(int a[], int n);

        static void blockLUOne(int a[], int l[], int u[]);
        
        static void blockLU(int a[], int l[], int u[], int n);
        static void blockInvOne(int a[], int y[]);
        
        static void blockInv(int a[], int y[], int n);
        
        static void blockInvLowerOne(int l[], int y[]);
        
        static void blockInvLower(int l[], int y[], int n);

        static void blockInvUpperOne(int u[], int y[]);
        
        static void blockInvUpper(int u[], int y[], int n);

        static void blockScanNAN(int a[], int n);

        static void blockSub(int a[], int b[], int d[], int n);
        static void blockCopy(int a[], int d[], int n);
        static void blockNeg(int b[], int d[], int n);

        static bool isZeroBlock(double tmp[]);

        static void blockMulSparseNeg(int aa[], int bb[], int c[], int n);

        static ushort mask[16]; 

        static void blockMulBit(int aa[], int bb[], int c[], int n);

        static void blockMulBit2(int aa[], int bb[], int c[], int n);

        static void blockMulBitNeg(int aa[], int bb[], int c[], int n);

        static void blockMulSparse(int aa[], int bb[], int c[], int n);

        static void blockMul(int a[], int b[], int c[], int n);

        static void blockMulDiag(int a[], int b[], int c[], int n);

        static void blockMulNeg(int a[], int e[], int c[], int n);

        static void resetOneBlock(double t[]);
        
        static void resetBlocks(int t[], int n);
        
        static void blockMulOne(double a[], double b[], double c[]);
        static void blockMulOneTrans(double a[], double b[], double c[]);
        static void blockMulOneNaive(double a[], double b[], double c[]);
        
        static void blockMulOneNeg(double a[], double b[], double c[]);
        static void blockMulOneNegTrans(double a[], double b[], double c[]);
        static void blockMulOneNegNaive(double a[], double b[], double c[]);
        
        static void putBlockList(int tgt[], int tgtn, int rowStart, int colStart, int n, int val[]);
        
        static int* getBlockList(int src[],int srcn, int rowStart, int colStart, int n);
        
        static void checkBlock();

        static void pushUp();

        static void pushDown();

        static void iniBlockStorage();
        
        static void allocateBlock(int bi, int bj);
        
        static int appendBlockStorageReal(double t[]);
        
        static void appendBlockStorageAgain(double t[], int rtn);
        static void appendBlockStorageAgainL2(int t[], int rtn);
        
        static int appendBlockStorage(double t[]);
        
        static int roundup(int count);
        static void configL2();
        
        static int stageCount(int blockRows);
        static void printInt(int a[], int n);

    };
}

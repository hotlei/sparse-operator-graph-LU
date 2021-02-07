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

namespace SOGLU
{
    class BlockPlanner
    {
        public:

        static ushort mask[16]; 
        static std::vector<int> tbd;
        static long stream[NUMSTREAM*8192];
        static int  streamcount[NUMSTREAM];
        static std::vector<matrix*> blockstorageL2;
        static int storageCountL2;
        static std::vector<operation*> graphL2;

        static void blockPlan(matrix* saved, matrix* savedu, matrix* lhs=NULL);
        static void copyOperatorL2(matrix* a, matrix* l, matrix* u, matrix* bl2, matrix* bu2, int n);
        static void copyOperatorX2(matrix* a, matrix* x1, matrix* y1, matrix* lhs1, matrix* mx2, int n);

        static void calculate();

        static void solve(matrix* bl, matrix* bu, double b[], int n);

        static bool checkSymmetric(matrix* mat, int n, int blocksize, int* indexi, int* indexj, double* vals, int valcount);

        static void blockMatrixLU(matrix* a, matrix* l, matrix* u, int n, matrix* l3, matrix* u3, int group, matrix* x, matrix* y, matrix* lhs);
        static void blockMatrixLU(matrix* a, matrix* l, matrix* u, int n, matrix* hl1, matrix* hu1, int group=0);
        static void blockMatrixLLT(matrix* a, matrix* l, int n, matrix* hl1, int group=0);
        
        static void blockMatrixInv(matrix* a, matrix* y, int n, int group=0);
        
        static void blockMatrixInvLower(matrix* l, matrix* y, int n, matrix* hl1, int group=0);

        static void blockMatrixInvUpper(matrix* u, matrix* y, int n, matrix* hu1, int group=0);

        static int blockScanNAN(matrix* a, int n);

        static void blockMatrixSub(matrix* a, matrix* b, matrix* d, int n, int group=0);
        static void blockMatrixCopy(matrix* a, matrix* d, int n, int group=0);
        static void blockMatrixNeg(matrix* b, matrix* d, int n, int group=0);

        static void blockMatrixMul(matrix* a, matrix*b, matrix* c, int n, int group=0);
        static void blockMatrixMulT(matrix* a, matrix*b, matrix* c, int n, int group=0);
        static void blockMatrixMulNeg(matrix* a, matrix*b, matrix* c, int n, int group=0);

        static void resetOneBlock(double t[]);
        static void resetOneBlockMeta(double t[]);
        
        static void checkBlock();

        static void getFirstColumn(matrix *m, double* b, int n);
        static matrix* iniVectorMatrix(int n);
        static matrix* iniVectorMatrix(double *b, int n);
        static void iniBlockStorage();
        
        static uint64_t claimBlock();
        static uint64_t allocateBlock(int bi, int bj);
        
        static void appendBlockStorageAgain(double t[], int rtn);
        static void appendBlockStorageAgainL2(matrix* t, int rtn);
        
        static int appendBlockStorage(double t[]);
        
        static void printInt(matrix* a, int n);
    };
}

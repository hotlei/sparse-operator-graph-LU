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
    class MatrixStdDouble
    {

    public:
        static uint mask16[16];
        static const double blanckdata[512];

        static void blockMulOneAvxBlock(double a[], double b[], double c[], unsigned long msk1, int coreIdx);
        static void blockMulOneAvxBlockNeg(double a[], double b[], double c[], unsigned long msk1, int coreIdx);
        static void blockMulOneAvxBlock4(double* ab[], double* bb[], double c[], int blocks, unsigned long msk, int coreIdx);

        static void printMatrix(double* a, int n);

        static void ludcmpSimple(double* a, int n, double* l, double* u);
        static void lltdcmpSimple(double* a, int n, double* l);

        static void inv_lower(double* l, int n, double* y);

        static void inv_upper(double* u, int n, double* y);

        static void mat_inv(double* a, int n, double* y);

        static void mat_mult(double* a, double* b, double* y, unsigned long msk1, int coreIdx);
        static void mat_mult(double* a, double* y, unsigned long msk1, int coreIdx);
        static void mat_mult4(double *__restrict ab[], double *__restrict bb[], double *__restrict c, int blocks, unsigned long msk, int coreIdx);

        static bool inv_check_diag(double* a, double* b, int n);

        static void mat_sub(double* a, double* b, int n, double* r, int coreIdx);
        static void mat_copy(double* a, int n, double* r, int coreIdx);
        static void mat_neg(double* b, int n, double* r, int coreIdx);
        static void mat_clear(double* y, int n);
        static void mat_clean(double* y, int n);
        static void mat_clean_lower(double* y, int n);
        static void mat_clean_upper(double* y, int n);
    };
}


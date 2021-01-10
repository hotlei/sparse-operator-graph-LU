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
#include <algorithm>   
#include <vector>     
#include <memory>
#include <string>

#include "const.h"

namespace SOGLU {
    class data
    {
        public:
        static int mSize;
        static int mSize_out;
        static int blockRows;
        static int blockSize;
        static matrix* blocks;
        static bool symmetric;
 
        static bool PlanL2;

        static int *indexi;
        static int *indexj;
        static double *vals;
        static int valcount;
        static std::vector<double*> blockstorage;
        static int storageCount;

        static double *b;
        static double *x;
 
        static std::vector<operation*> graph;
        static std::unique_ptr<int[]> stage;
        static std::unique_ptr<int[]> laststage;

        static void clear();
        static void clearOperations();
    };
}

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

#include <vector>       // std::vector
#include "const.h"

namespace SOGLU {

    class memutil
    {
        static std::vector<std::vector<double*>> blockStorageBuffer;
        static int blockStoragePointer[MAXTHREAD];
        static std::vector<std::vector<double*>> reclaimedStorage;

        static std::vector<std::vector<void*>> memBlobs;
        static std::vector<std::vector<void*>> reclaimedBlobs;
  
        static std::vector<std::vector<void*>> memNugget;
        static int memNuggetPointer[MAXTHREAD];

        static std::vector<std::vector<void*>> matrixBuffer;
        static uint matrixBufferPointer[MAXTHREAD];

        static std::vector<std::vector<void*>> operationBuffer;
        static int operationPointer[MAXTHREAD];

        public:
        static double* newalignedblock(int threadId, int columns);
        static void freeblock(int threadId, double* dmp);
        static void* getMemBlob(int threadId, size_t bytes=MEMBLOB);
        static void freeMemBlob(int threadId, void* vmp);
        static void* getSmallMem(int threadId, size_t bytes);
        static void freeAllSmallMem(); 
        static void freeAllSmallMem(int threadId); 
        static matrix* newmatrix(int threadId);
        static matrix* newmatrix(int threadId, int blockrows, int level);
        static bool clearmemory;

        static operation* newoperation(int src1, int s2, blockOp o, int d1, int d2, int threadId=0, int group=0);

        static void ini();
        static void report();
        static void clear();
        static void releaseOne(void* p);
        static void releaseAllBlockBuffer();
    };
}

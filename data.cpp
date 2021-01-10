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

#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <string>
#include <memory>

#include "operation.h"
#include "matrix.h"
#include "data.h"

#define ALIGN64 64

namespace SOGLU{
        int data::mSize = 0;
        int data::mSize_out = 0;
        matrix* data::blocks = NULL;
        int data::blockRows = 0;
        int data::blockSize = BLOCK64;
        int data::valcount = 0;
        int data::storageCount = 0;
        bool data::symmetric = false;
        
        int* data::indexi = NULL;
        int* data::indexj = NULL;
        double* data::b = NULL;
        double* data::x = NULL;
        double* data::vals = NULL;

        std::vector<double*> data::blockstorage = {};
        std::vector<operation*> data::graph = {};
        std::unique_ptr<int[]> data::stage = nullptr;
        std::unique_ptr<int[]> data::laststage = nullptr;

        bool data::PlanL2=false;

        void data::clear()
        {
            graph.clear();
            blockstorage.clear();
        }

        void data::clearOperations()
        {
            graph.clear();
        }
}

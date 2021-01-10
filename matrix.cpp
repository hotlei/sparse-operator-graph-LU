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

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "operation.h"
#include "matrix.h"
#include "memutil.h"

namespace SOGLU {
    uint64_t matrix::notfound = 0;
    uint64_t matrix::lastkey = -1;
    uint64_t matrix::lastindex = 0;
    uint64_t matrix::hit = 0;
    uint64_t matrix::miss = 0;
    matrix::matrix(){}

    uint64_t& matrix::operator[](uint64_t key)
    {
      //  if(key == lastkey) {
      //     hit++;
      //     return lastindex;
      //  }

      //  miss++;
        matrix* cur = this;
        uint32_t mask = (1u<<level) - 1;
        uint32_t row = key >> level;
        uint32_t col = key & mask;
        int step = level;
        while(step>0){
           step = step - 1;
           uint32_t idx = ((row >> step) << 1) + (col >> step);
           cur = cur -> submatrix[idx];
           mask = mask >> 1;
           row = row & mask;
           col = col & mask;
           if(cur == NULL) {
               lastkey = key;
               lastindex = notfound;
               return notfound;  
           }
        };
        lastkey = key;
        lastindex = cur->blockindex;
        return cur->blockindex;
    }

    void matrix::set(uint64_t key, uint64_t val)
    {
        matrix* cur = this;
        uint32_t mask = (1u<<level) - 1;
        uint32_t row = key >> level;
        uint32_t col = key & mask;
        int step = level;
        while(step>0){
           step = step - 1;
           uint32_t idx = ((row >> step) << 1) + (col >> step);
           if(cur->submatrix[idx] == NULL) {
               cur->submatrix[idx] = memutil::newmatrix(0);
               cur->submatrix[idx]->blockrows2 = cur->blockrows2 / 2; 
               cur->submatrix[idx]->level = cur->level - 1; 
           }
           cur = cur -> submatrix[idx];
           mask = mask >> 1;
           row = row & mask;
           col = col & mask;
        };
        lastkey = key;
        lastindex = val;
        cur->blockindex = val;
    }

    uint64_t matrix::size()
    {
        uint64_t rtn = 1;
        for(int i=0;i<4;i++){
            if(submatrix[i] != NULL) rtn += submatrix[i]->size();
        }
        return rtn;
    }

    uint64_t matrix::datasize()
    {
        uint64_t rtn = 0;
        for(int i=0;i<4;i++){
            if(submatrix[i] != NULL) rtn += submatrix[i]->datasize();
        }
        if(level == 0 && blockindex > 0) rtn = 1;
        return rtn;
    }

    matrix* matrix::clone(matrix* src)
    {
        matrix* rtn = memutil::newmatrix(0);
        rtn->blockrows2 = blockrows2;
        rtn->level = level;
        rtn->blockindex = blockindex;
        for(int i=0;i<4;i++){
            if(submatrix[i] != NULL) 
                rtn->submatrix[i] = clone(submatrix[i]);
            else 
                rtn->submatrix[i] = NULL;
        }
        return rtn;
    }

    matrix* matrix::getBlockList(int sub)
    {
        return submatrix[sub];
    }
    void matrix::putBlockList(int sub, matrix* updated)
    {
        submatrix[sub] = updated;
    }
}

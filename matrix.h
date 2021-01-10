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

#include<stdint.h>

namespace SOGLU {
struct matrix 
{
    public:
    int blockrows2;
    int level;
    uint64_t blockindex;
    matrix* submatrix[4];         //b11, b12, b21, b22
    
    static uint64_t notfound;
    static uint64_t lastkey;
    static uint64_t lastindex;
    static uint64_t hit;
    static uint64_t miss;

    matrix();
    uint64_t& operator[](uint64_t);
    void set(uint64_t key, uint64_t val);
    matrix* clone(matrix* src);
    uint64_t size();
    uint64_t datasize();
    matrix* getBlockList(int sub);
    void putBlockList(int sub, matrix* updated);
};
}

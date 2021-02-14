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

#include "const.h"
#include "config.h"

namespace SOGLU {
    int config::mSize;
    int config::blockRows;
    int config::blockSize;
 
    int config::blockRowsL2;
    int config::blockSizeL2;
   
    int roundup(int multiple)
    {
        int rtn = 1;
        while (multiple >0)
        {
            multiple = multiple / 2;
            rtn = rtn * 2;
        }
        return rtn;
    }

    void config::set(int dim)
    {
        mSize = dim;
        blockSize = BLOCK64;
        blockRows = roundup(mSize / blockSize);

        blockRowsL2 = 2<<(__builtin_popcount(blockRows-1)/2);
        if(blockRows / blockRowsL2 > 4)
           blockRowsL2 *= 2;
        blockSizeL2 = blockSize * blockRows / blockRowsL2;
    }
}

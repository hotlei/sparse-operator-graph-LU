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

#define BLOCK64 128
#define BLOCKCOL 136               // BLOCK64+8
//#define BLOCK64 64
#define BLOCK16 32                 // BLOCK64/4 
#define NUMSTREAM 23
#define MAXTHREAD 16
#define METASIZE 64
#define METAUINT 16
//#define SUB_BLOCK 8
#define SUBROW   16                       // BLOCK64 / SUB_BLOCK
#define SUBCOL   16                       // BLOCK64 / SUB_BLOCK
#define SUBCOL9   17                      // SUBCOL + 1
#define ALLOCDOUBLE 17408                 // SUBROW*SUBCOL9*64     // *SUB_BLOCK*SUB_BLOCK
#define ALLOCBLOCK  139264                // ALLOCDOUBLE*8
#define DETAILOFFSET 128                  // SUBCOL*8             // *SUB_BLOCK
#define DETAILSKIPSHORT   544             // SUBCOL9*8*4
#define METAOFFSET   672                  // DETAILSKIPSHORT+DETAILOFFSET
#define METAMASK 0xffff
#define CACHE64 64
#define ROWSKIP64 BLOCKCOL
#define ROWSKIP128 BLOCKCOL*2
#define ROWSKIP192 BLOCKCOL*3
#define ROWSKIP256 BLOCKCOL*4
#define ROWSKIP320 BLOCKCOL*5
#define ROWSKIP384 BLOCKCOL*6
#define ROWSKIP448 BLOCKCOL*7

#define MEMBLOB 0x2000000   // 32M


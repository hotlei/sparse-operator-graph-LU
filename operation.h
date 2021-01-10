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

namespace SOGLU {
    enum blockOp { inv, lu, lowerInv, upperInv, sub, add, neg, copy, mul, mulneg, llt, mult, noop};

    class position
    {
        public:
        int value;
        int index;
        int hash;
        int count;
        position(int idx, int val);
        position(int idx, int val, int count, int hash);
    };

        bool positionCompare(position x, position y);
        bool positionReverseCompare(position x, position y);
        bool positionHashCompare(position x, position y);

    struct operation
    {
        public:
        int src, src2;
        blockOp op;
        int result, result2;
        bool skip;
        int stage;
        int sequenceNum;
        int groupNum;
        static int seq;

        operation();
        operation(int src1, int s2, blockOp o, int d1, int d2);
        void sett(int src1, int s2, blockOp o, int d1, int d2);
    };
    
        bool operationCompare(operation* x, operation* y);
    
    struct cell
    {
        public:
        int row;
        int col;
        double val;
        cell();
    };
        bool cellCompare(cell x, cell y);
}

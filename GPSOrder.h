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
#include <string>
#include <vector>
#include <set>

namespace SOGLU
{
    class oneRow
    {
        public:
        int origRow;
        std::vector<int> origCols;
        std::set<int> tbd;
    };

    class GOrder
    {
        public:
        static int dim;
        static int valcount;
        static int lastbandwidth;
        static std::vector<oneRow*> rows;
        static int *newOrder;
        static int *reverseOrder;
       
        static void sortInBlock();

        static std::vector<std::vector<int>*>* getLevelList(int startNode);
        static int nextLevel(int level, std::vector<int>* list, std::vector<std::vector<int>*>* levels, int* dis);
        static void setupEdge(int* idx, int* jdx, int n, int vcount);
        static void setupTest();
        static void Reorder();
        static int getLeastConnected();
        static void updatewithlevel(std::vector<std::vector<int>*>* levels);
        static int reorderOneMoreLevel(std::vector<int>*  last, std::vector<int>* cur, int* neworder, int* reverse, int start);
        static int getNodeSum(int node);
        static int addMissing(std::vector<std::vector<int>*>* levels, int* dis);
        static double* reOrderResult(double *x);
        static void clearmost();
        static void clear();
    };
}

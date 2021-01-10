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
#include "operation.h"

namespace SOGLU{
        position::position(int idx, int val)
        {
            index = idx;
            value = val;
            hash = 0;
            count = 0;
        }

        bool positionCompare(position x, position y)
        {
            return (x.value < y.value);
        }

        bool positionReverseCompare(position x, position y)
        {
            return (y.value < x.value);
        }

        position::position(int idx, int val, int counted, int hashvalue)
        {
            index = idx;
            value = val;
            hash = hashvalue;
            count = counted;
        }

  /*      bool positionHashCompare(position x, position y)
        {
            int xb = x.value / data::blockSize;
            int yb = y.value / data::blockSize;
            if(xb == yb){
            if(x.count == y.count){
                if(x.value == y.value){
                    if(x.hash == y.hash){
                    }
               	    return(x.hash < y.hash);
                }
                return (x.value < y.value);
            }
                return (x.count < y.count);
            }
            if(x.value == y.value){
                if(x.count == y.count){
                    return(x.hash < y.hash);
                }
                return (x.count < y.count);
            }
            return (x.value < y.value);
        } /* */
        bool positionHashCompare(position x, position y)
        {
            if(x.value == y.value){
                if(x.count == y.count){
               	    return(x.hash < y.hash);
                }
                return (x.count < y.count);
            }
            return (x.value < y.value);
        } /* */

        cell::cell()
        {
        }
        bool cellCompare(cell x, cell y)
        {
            int tik = __builtin_popcount(BLOCK64-1);
            if((x.row>>tik) == (y.row>>tik))
                return ((x.col>>tik) < (y.col>>tik));
            return ((x.row>>tik) < (y.row>>tik));
        }

        int operation::seq = 0;
        operation::operation(int src1, int s2, blockOp o, int d1, int d2)
        {
            src = src1;
            src2 = s2;
            op = o;
            result = d1;
            result2 = d2;
            skip = false;
            stage = 0;
            sequenceNum = ++seq;
            groupNum = 0;
        }
        void operation::sett(int src1, int s2, blockOp o, int d1, int d2)
        {
            src = src1;
            src2 = s2;
            op = o;
            result = d1;
            result2 = d2;
            skip = false;
            stage = 0;
            groupNum = 0;
            sequenceNum = ++seq;
        }

        operation::operation()
        { }
    
/*        bool operationCompare(operation* x, operation* y)
        {
            if(x->stage == y->stage)
            {
                int xlk = x->result % NUMSTREAM;
                int ylk = y->result % NUMSTREAM;
                if(xlk == ylk){    
                  //  if(x->result == y->result){
                         if(x->op == y->op){
                             return (x->sequenceNum < y->sequenceNum);
                         }
                         return(x->op < y->op);
                  //  }
                  //  return(x->result < y->result);
                }
                return (xlk < ylk);
            }
            return (x->stage < y->stage);
        }
 /* */
 /*
        bool operationCompare(operation* x, operation* y)
        {
            if(x->stage == y->stage)
            {
                if(x->src2 >0 || y->src2 >0){
                if(x->src2 == y->src2){    
                    if(x->result == y->result){
                             return (x->sequenceNum < y->sequenceNum);
                    }
                    return(x->result < y->result);
                }
                return (x->src2 < y->src2);
                }else{
                    return (x->sequenceNum < y->sequenceNum);
                }
            }
            return (x->stage < y->stage);
        } 
/* */
/*
        bool operationCompare(operation* x, operation* y)
        {
            if(x->stage == y->stage)
            {
                if(x->result == y->result){
                    if(x->src == y->src){    
                             return (x->sequenceNum < y->sequenceNum);
                    }
                    return (x->src < y->src);
                }
                return(x->result < y->result);
            }
            return (x->stage < y->stage);
        } 
/* */
//
        bool operationCompare(operation* x, operation* y)
        {
            if(x->stage == y->stage)
            {   
                if(x->groupNum == y->groupNum){
                    if(x->result == y->result){
                        if(x->src == y->src){    
                                 return (x->sequenceNum < y->sequenceNum);
                        }
                        return (x->src < y->src);
                    }
                    return(x->result < y->result);
                }
                return(x->groupNum < y->groupNum);
            }
            return (x->stage < y->stage);
        } 
/* */
/*      
        bool operationCompare(operation* x, operation* y)
        {
            if(x->stage == y->stage)
            {
        //        if(x->sequenceNum == y->sequenceNum){
        //            if(x->src == y->src){    
        //                     return (x->result < y->result);
        //            }
        //            return (x->src < y->src);
        //        }
                return(x->sequenceNum < y->sequenceNum);
            }
            return (x->stage < y->stage);
        }  
/* */
}

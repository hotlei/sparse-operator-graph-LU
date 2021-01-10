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

#include <algorithm>
#include <omp.h>
#include "operation.h"
#include "matrix.h"
#include "data.h"
#include "GPSOrder.h"
#include "BlockPlanner.h"
#include "memutil.h"
#include "config.h"

#define LIST100 100
#define LIST500 500
namespace SOGLU
{

    int GOrder::dim = 0;
    int GOrder::valcount = 0;
    int GOrder::lastbandwidth = 0;
    std::vector<oneRow*> GOrder::rows = {};
    int *GOrder::newOrder = NULL;
    int *GOrder::reverseOrder = NULL;

    double* GOrder::reOrderResult(double *x)
    {
        double *tmp = memutil::getSmallMem(1, sizeof(double) * dim);
        for (int i = 0; i < dim; i++) tmp[i] = 0;
        if (reverseOrder != NULL)
            {
                for (int i = 0; i < dim; i++)
                {
                    tmp[i] = x[GOrder::reverseOrder[i]];
                }
            }
        return tmp;
    }

    void GOrder::sortInBlock(){
            int i, j;
            double TINY = 1e-8;
            double* tmpv = (double*)malloc(sizeof(double) * data::mSize);
            int* tmpi = (int*)malloc(sizeof(int) * data::mSize);
            for(i=0;i<data::mSize;i++){
                tmpv[i] = 0;
                tmpi[i] = -1;
            }
            for(i=0;i<data::valcount;i++){
                if(data::indexi[i] == data::indexj[i]){
                    tmpv[data::indexi[i]] = data::vals[i];
                    tmpi[data::indexi[i]] = data::indexi[i];
                }
            }
            int zcount = 0;
            for(i=0;i<data::mSize;i++){
                if(tmpi[i]==-1){
                    tmpi[i] = i;
                    zcount++;
                }
            }
            for(i =0;i<data::mSize; i+= data::blockSize){

                if(i + data::blockSize >= data::mSize)
                     continue;
                int step = data::blockSize/2; // / 2   8;
                zcount = 0;
                for(j=0;j<step;j++){
                    if(tmpv[i+j]>TINY || tmpv[i+j]<-TINY)
                        continue;
                    zcount++;
                }
                if(zcount == 0)
                    continue;

                while(step > 0){
                    for(j=0;j<step;j++){
                        int k = i + j;
                        int m = i + j + step;
                        if(tmpv[m] > tmpv[k]){
                            double v = tmpv[m];
                            tmpv[m] = tmpv[k];
                            tmpv[k] = v;
                            int t = tmpi[m];
                            tmpi[m] = tmpi[k];
                            tmpi[k] = t;
                        }
                    }
                    step = step / 2;
                }
            }
            int* tmpn = (int*)malloc(sizeof(int) * data::mSize);
            for(i=0;i<data::mSize;i++){
                tmpn[i] = -1;
            }
            for(i=0;i<data::mSize;i++){
                tmpn[tmpi[i]] = i;
            }
            for(i=0;i<data::valcount;i++){
                data::indexi[i] = tmpn[data::indexi[i]];
                data::indexj[i] = tmpn[data::indexj[i]];
            }
            double* bn = (double*)malloc(sizeof(double) * data::mSize);

            for(i=0;i<data::mSize;i++){
                bn[i] = data::b[tmpi[i]];
            }
            for(i=0;i<data::mSize;i++){
                data::b[i] = bn[i];
            }

            if(newOrder == NULL){
                 newOrder = memutil::getSmallMem(1, sizeof(int)*data::mSize);
                 reverseOrder = memutil::getSmallMem(1, sizeof(int)*data::mSize);
                 for (int i = 0; i < data::mSize; i++){
                     reverseOrder[i] = i;
                     newOrder[i] = i;
                 }
            }

            int* norder = (int*)malloc(sizeof(int) * data::mSize);
            for(i=0;i<data::mSize;i++){
                norder[i] = newOrder[tmpi[i]];
            }
            for(i=0;i<data::mSize;i++){
                newOrder[i] = norder[i];
                reverseOrder[norder[i]] = i;
            }

            free(tmpn);
            free(tmpv);
            free(tmpi);
            free(bn);
            free(norder);
          }

    int accountLevel(std::vector<std::vector<int>*>* levels)
    {
        int total = 0;
        for(std::vector<int>* group: *levels){
            total += group->size();
        }
        return total;
    }

    void printLevel(std::vector<std::vector<int>*>* levels)
    {
        int max = 0;
        int last = 0;
        int total = 0;
        for(std::vector<int>* group: *levels){
            total += group->size();
            if(group->size() > max)
                max = group->size();
            if(group->size() > 0)
                last = group->size();
        }
        std::cout<<"GGPS reorder: levels: " <<levels->size()<<" bandwidth: "<<max<<" last level count: "<<last<<
                  " total accounted: "<<total<<" start from "<<levels[0][0][0][0]<<std::endl;
    }

    void clearLevel(std::vector<std::vector<int>*>* levels)
    {
        for(std::vector<int>* group: *levels){
            delete group;
        }
        delete levels;
    }

    std::vector<std::vector<int>*>* GOrder::getLevelList(int startNode)
    {
        int* colored = new int[dim];
        std::vector<std::vector<int>*>* levelList = new std::vector<std::vector<int>*>();
        for(int i=0;i<dim;i++){
            colored[i] = 0;
        }
        std::vector<int>* first=new std::vector<int>();
        int level1 = startNode;

        first->push_back(level1);
        levelList->push_back(first);

        colored[level1] = 1;
        nextLevel(2, first,levelList, colored);

        if(accountLevel(levelList)<dim){
            addMissing(levelList, colored);
        }

        delete[] colored;
        return levelList;
    }

    int GOrder::nextLevel(int level, std::vector<int>* list, std::vector<std::vector<int>*>* levels, int* dis)
    {
        std::vector<int>* next= new std::vector<int>();
        for (int row : *list)
        {
            int deadend = 1;
            for(int col : rows[row]->origCols){
                if(dis[col] == 1) 
                    continue;
                next->push_back(col);
                dis[col] = 1;
                deadend  = 0;
            }	
        }

        if (next->size() > 0)
        {
            levels->push_back(next);
            return nextLevel(level + 1, next, levels, dis);
        }else{
            delete next;
        }

        int nodecount = 0;
        for (std::vector<int>* li : *levels)
            nodecount += li->size();

        int least = dim;
        int vleast = -1;
        for (int row : *list)
        {
            int nodes = rows[row]->origCols.size();
            if (nodes < least)
            {
                least = nodes;
                vleast = row;
            }
        }
        lastbandwidth = least;
        return vleast;
    }

    int GOrder::addMissing(std::vector<std::vector<int>*>* levels, int* dis)
    {
        std::set<int> missed = {};
        for(int i=0;i<dim;i++){
            if(dis[i] == 0) missed.insert(i);
        }
        int lastlevel = levels->size();
        while(!missed.empty()){
            lastlevel = levels->size();
            auto pfirst = missed.begin();
            int efirst = *pfirst;
            std::vector<int>* first=new std::vector<int>();

            first->push_back(efirst);
            levels->push_back(first);

            dis[efirst] = 1;
            nextLevel(lastlevel, first,levels, dis);

            for(int j=lastlevel; j<levels->size(); j++){
                for(int k : levels[0][j][0]){
                    missed.erase(k);
                }
            }
        };
        return 0;
    }

    void GOrder::setupEdge(int* idx, int* jdx, int n, int vcount)
    {
        dim = n;
        valcount = vcount;
        int edgecount = 0, shadowcount = 0;

        for (int i = 0; i < n; i++)
        {
            rows.push_back(new oneRow());
            rows[i]->origRow = i;
            rows[i]->origCols = {};
        }

        std::vector<oneRow*> shadows = {};
        for (int i = 0; i < n; i++)
        {
            shadows.push_back(new oneRow());
            shadows[i]->origRow = i;
            shadows[i]->origCols = {};
        }
        for (int i = 0; i < valcount; i++)
        {
            if (idx[i] != jdx[i])
            {
                    shadows[idx[i]]->origCols.push_back(jdx[i]);
            }
        }
        for (int i = 0; i < n; i++){
            shadowcount += shadows[i]->origCols.size();
        }
        for (int i = 0; i < valcount; i++)
        {
            if (idx[i] != jdx[i])
            {
                    shadows[jdx[i]]->origCols.push_back(idx[i]);
            }
        }
        for(oneRow* r: shadows){
            std::sort (r->origCols.begin(), r->origCols.end());
        }

        for (int i = 0; i < n; i++){
            for(int j=0;j<shadows[i]->origCols.size();j++){
                if(j>0 && rows[i]->origCols.back() == shadows[i]->origCols[j]) continue;
                rows[i]->origCols.push_back(shadows[i]->origCols[j]);
            }
        }
        for (int i = 0; i < n; i++){
            edgecount += rows[i]->origCols.size();
        }
        if(shadowcount == edgecount)
            std::cout<<"structure symetric"<<std::endl;
//        else
//            std::cout<<"edgecount "<<shadowcount<<" filled to "<<edgecount<<std::endl;

        for (int i = 0; i < n; i++)
        {
            delete shadows[i];
        }
    }

    int GOrder::reorderOneMoreLevel(std::vector<int>*  last, std::vector<int>* cur, int* neworder, int* reverse, int start)
    {
        std::vector<position> unclearedLastLevel = {};
        for(int s: *last){
            unclearedLastLevel.push_back(position(s,neworder[s]));
        }
        std::sort(unclearedLastLevel.begin(), unclearedLastLevel.end(), positionReverseCompare);
        std::vector<position> rank = {};
        int cursor = start;
        int shouldAdd = cur->size();
        for(int i=last->size()-1;i>=0;i--){
            int w = unclearedLastLevel[i].index;
            unclearedLastLevel.pop_back();
            if(rows[w]->tbd.empty()) continue;
            rank.clear();
            for(int node: rows[w]->tbd){
                if(neworder[node]<0)
                    rank.push_back(position(node,getNodeSum(node)));
            }
            if(rank.empty()) continue;
            std::sort(rank.begin(),rank.end(),positionCompare);
            for(int j=0;j<rank.size();j++){
                neworder[rank[j].index] = cursor;
                reverse[cursor] = rank[j].index;
                cursor++;
                for(int col: rows[rank[j].index]->origCols){
                    if(neworder[col]>=0 && (!rows[col]->tbd.empty())){
                        rows[col]->tbd.erase(rank[j].index);
                    }
                }
            }
        }
        if(cursor - start != shouldAdd){
            std::vector<position> mank = {};
            for(int node: *cur){
                if(neworder[node]<0)
                    mank.push_back(position(node,getNodeSum(node)));
            }
            std::sort(mank.begin(),mank.end(),positionCompare);
            for(int i=0;i<mank.size();i++){
                neworder[mank[i].index] = cursor;
                reverse[cursor] = mank[i].index;
                cursor++;
                for(int col: rows[mank[i].index]->origCols){
                    if(neworder[col]>=0 && (!rows[col]->tbd.empty())){
                        rows[col]->tbd.erase(mank[i].index);
                    }
                }
            }

        }
        return cursor;
    }

    void GOrder::updatewithlevel(std::vector<std::vector<int>*>* levels)
    {
        int* newOrd = memutil::getSmallMem(1, sizeof(int)*dim);
        int* reverseOrd = memutil::getSmallMem(1, sizeof(int)*dim);
        for(int i=0;i<dim;i++){
            newOrd[i] = -1;
            reverseOrd[i] = -1;
            rows[i]->tbd.clear();
            for(int t:rows[i]->origCols){
                rows[i]->tbd.insert(t);
            }
        }
        std::vector<position> rank = {};
        for(int node: levels[0][0][0]){
            rank.push_back(position(node,getNodeSum(node)));
        }
        std::sort(rank.begin(),rank.end(),positionCompare);
        for(int i=0;i<rank.size();i++){
            newOrd[rank[i].index] = i;
            reverseOrd[i] = rank[i].index;
            for(int col: rows[rank[i].index]->origCols){
                if(newOrd[col]>=0 && (!rows[col]->tbd.empty())){
                    rows[col]->tbd.erase(rank[i].index);
                }
            }
        }
        int cursor = rank.size();
        for(int j=1;j<levels->size();j++){
            cursor = reorderOneMoreLevel((*levels)[j-1],(*levels)[j],newOrd,reverseOrd,cursor);
        }

        for (int i = 0; i < data::valcount; i++)
            {
                int iidx = data::indexi[i] ;

                int jidx = data::indexj[i] ;

                data::indexi[i] = newOrd[iidx] ;
                data::indexj[i] = newOrd[jidx] ;

            }
            double* tmp = data::b;

            int blockrows = config::blockRows;
            data::b = memutil::getSmallMem(1, blockrows * data::blockSize *sizeof(double) );
            for (int i = 0; i < data::mSize; i++)
            {
                int idx = i ;

                data::b[i] = tmp[reverseOrd[idx]];
            }
            for(int i = data::mSize; i< blockrows * data::blockSize; i++){
                data::b[i] = 1;
            }
        GOrder::newOrder = reverseOrd;
        GOrder::reverseOrder = newOrd;
    }
    
    int GOrder::getNodeSum(int node)
    {
        int sum=0;
        for(int j : rows[node]->origCols){
            sum += rows[j]->origCols.size();
        }
        return sum;
    }

    int levelWidth(std::vector<std::vector<int>*>* levels)
    {
        int max = 0;
        for(std::vector<int>* group: *levels){
            if(group->size() > max)
                max = group->size();
        }
        return max;
    }

    int GOrder::getLeastConnected()
    {
        int node=0;
        int count = dim;
        for (int i = 0; i < dim; i++)
        {
            if(rows[i]->origCols.size() < count){
                if(rows[i]->origCols.size() == 0) continue;
                node = i;
                count = rows[i]->origCols.size();
            }	
        }
        return node;
    }
  
    void clearLevelList(std::vector<std::vector<int>*>* lvl)
    {
        for(std::vector<int>* group: *lvl){
            group->clear();
            delete group;
        }
    }

    void GOrder::Reorder()
    {

        newOrder = memutil::getSmallMem(1, sizeof(int)*data::mSize); 
        reverseOrder = memutil::getSmallMem(1, sizeof(int)*data::mSize); 
            for (int i = 0; i < data::mSize; i++){
                reverseOrder[i] = -1;
                newOrder[i] = -1;
            }
        setupEdge(data::indexi, data::indexj, data::mSize, data::valcount);
        int startnode = getLeastConnected();
        std::vector<std::vector<int>*>* levels = getLevelList(startnode);
        int  ppnodeset[LIST500];
        int  ppbandwidth[LIST500];
        std::vector<std::vector<int>*>* pplevelset[LIST500];
        int  ppnodecount = 0;
        int  pplevel;
        int  ppwidth = dim;
        int  u_node = startnode;
        std::set<int> Gset = {};
        for(int g: *(levels->back())){
            Gset.insert(g);
        }
        for(int g=0;g<dim;g++){
            if(rows[g]->origCols.size() == rows[startnode]->origCols.size())
                Gset.insert(g);
        }
        while(1){
            std::vector<int>* lastlevel = levels->back();
            int updated = 0;
            int lastsize = lastlevel->size();
            int lastskip = 0;
            
            if(lastsize>LIST100) {lastskip = (lastsize + LIST100 -1) /  LIST100; lastskip = (lastskip / 2 + 1) * 2 - 1; }
            ppnodecount = 0;
            int maxsize = levels->size();

            for(int t=0;t<lastsize;t++){
                if(lastskip>0 && t%lastskip != 0) continue;
                ppnodeset[ppnodecount] = (*lastlevel)[t];
                pplevelset[ppnodecount] = NULL;
                ppbandwidth[ppnodecount] = dim;
                ppnodecount++;
            }
#pragma omp parallel for
            for(int t=0;t<ppnodecount;t++){
                std::vector<std::vector<int>*>* lvl = getLevelList(ppnodeset[t]);
                ppbandwidth[t] = levelWidth(lvl);
                pplevelset[t] = lvl;
            }

            for(int t=0;t<ppnodecount;t++){
                std::vector<std::vector<int>*>* lvl = pplevelset[t];
                if(lvl->size()<maxsize) continue;
                for(int g: *(lvl->back())){
                    Gset.insert(g);
                }
                if(lvl->size()>levels->size()){
                    clearLevelList(levels);
                    delete levels;
                    levels = lvl;
                    startnode = ppnodeset[t];
                    updated = 1;
                    for(int i=0; i<ppnodecount; i++){
                        if(pplevelset[i] != NULL && i != t) {
                            clearLevelList(pplevelset[i]);
                            delete pplevelset[i];
                        }
                    }
                    ppnodecount = 0;
                    break;
                }
                if(lvl->size()<levels->size()){
                    continue;
                }

                if(ppwidth > levelWidth(lvl)){
                    ppwidth = levelWidth(lvl);
                    u_node = ppnodeset[t];
                }

            }
            if(updated) continue;
            break;
        };
        if(ppnodecount == 0) std::cout<<" can not find ppnode "<<std::endl;
       
        int vdegree = rows[startnode]->origCols.size();
        Gset.insert(dim-1);
        Gset.insert(0);
        int nodeindex = ppnodecount;
        int nodecnt = ppnodecount;
        int nodeskip = 0;
        int cap = LIST500 - ppnodecount;
        if(cap>0)
            {nodeskip = (Gset.size()+cap-1) / cap; nodeskip = (nodeskip / 2 + 1) * 2 - 1;}
        for(int ss: Gset){
            if(rows[ss]->origCols.size() > vdegree && ss != dim-1 && ss!= 0) continue;
            int found = 0;
            for(int qq=0;qq<nodecnt;qq++){
                if(ppnodeset[qq] == ss){
                    found = 1;
                    break;
                }
            }
            nodeindex++;
            if(nodeskip>1 && nodeindex % nodeskip != 0) continue;
            if(found == 1) continue;
            if(nodecnt >= LIST500) break;
            ppnodeset[nodecnt] = ss;
            nodecnt++;
        }

#pragma omp parallel for
            for(int t=ppnodecount;t<nodecnt;t++){
                std::vector<std::vector<int>*>* lvl = getLevelList(ppnodeset[t]);
                ppbandwidth[t] = levelWidth(lvl);
                pplevelset[t] = lvl;
            }
 
        int minmax = -dim;
        int bestset = -1;
        int efflevel = 0;
        for(int i=0; i<nodecnt; i++){
            if(pplevelset[i] == NULL) continue;
            efflevel = pplevelset[i]->size() - levelWidth(pplevelset[i]);
            if(minmax < efflevel){
                minmax = efflevel;
                bestset = i;
            }
        }
        printLevel(pplevelset[bestset]);
        updatewithlevel(pplevelset[bestset]);
        for(int i=0; i<nodecnt; i++){
            if(pplevelset[i] != NULL) {
                clearLevelList(pplevelset[i]);
                delete pplevelset[i];
            }
        }
                    clearLevelList(levels);
                    delete levels;
    }

    void GOrder::setupTest()
    {
    }
    void GOrder::clearmost()
    { 
        for (int i = 0; i < dim; i++)
        {
            delete rows[i];
        }
    }
    void GOrder::clear()
    {
    }
}

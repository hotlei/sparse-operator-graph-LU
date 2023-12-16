#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include "data.h"
#include "BlockReOrder.h"
#include "BlockPlanner.h"
#include "broker.h"

namespace vmatrix
{
          int BlockReOrder::vStart = -1;
          int BlockReOrder::vEnd = -1;
          std::vector<edge*> BlockReOrder::edges;
          int *BlockReOrder::rowOffset = NULL;
          int *BlockReOrder::colored = NULL;
          int *BlockReOrder::newOrder = NULL;
          int *BlockReOrder::reverseOrder = NULL;
          int BlockReOrder::ReOrderBlockSize = 0;
          int BlockReOrder::ReOrderBlockRows = 0;
          int BlockReOrder::levelBandWidth = 0;
          std::vector<int> BlockReOrder::lastLevel = {};
          std::vector<std::vector<int>> BlockReOrder::levelList = {};

          int *BlockReOrder::indexi = NULL;
          int *BlockReOrder::indexj = NULL;
          int BlockReOrder::indexcount = 0;

          void BlockReOrder::clear()
          {
              lastLevel.clear();
              for(std::vector<int> level:levelList){
                   level.clear();
              }
              levelList.clear();
              
//              for(edge* e : edges){
//                if(e != NULL)
//                     delete e;
//              }

//              edges.clear();
              delete[] colored;
              delete[] newOrder;
              delete[] reverseOrder;
              if(rowOffset != NULL)
                  delete[] rowOffset;
              if(indexi != NULL)
                  delete[] indexi;
              if(indexj != NULL)
                  delete[] indexj;
          }

          void BlockReOrder::filltoBlocks()
        {
  /*          int blocks = (data::mSize + data::blockSize - 1) / data::blockSize;
            int count = blocks * data::blockSize;
            int newcount = data::valcount + count - data::mSize;
            std::cout<<"old size "<<data::mSize<<" new size: "<<count<<std::endl;
            int* tmpi = new int[newcount];
            int* tmpj = new int[newcount];
            double* tmpv = new double[newcount];
            for(int i = 0; i < data::valcount; i++)
            {
                tmpi[i] = data::indexi[i];
                tmpj[i] = data::indexj[i];
                tmpv[i] = data::vals[i];
            }
            for(int i= data::valcount; i < newcount; i++)
            {
                tmpi[i] = i - data::valcount + data::mSize;
                tmpj[i] = i - data::valcount + data::mSize;
                tmpv[i] = 1;
            }

            delete [] data::indexi;
            delete [] data::indexj;
            delete [] data::vals;
            data::indexi = tmpi;
            data::indexj = tmpj;
            data::vals = tmpv;
            data::mSize = count;
            data::valcount = newcount;
   */     }

          void BlockReOrder::sortInBlock(){
            int i, j;
            double TINY = 1e-8;
            double* tmpv = (double*)malloc(sizeof(double) * data::mSize);
            int* tmpi = (int*)malloc(sizeof(int) * data::mSize);
            for(i=0;i<data::mSize;i++){
                tmpv[i] = 0;
                tmpi[i] = -1;
            }
            std::cout<<"val count: "<<data::valcount<<" block size: "<<data::blockSize<<std::endl;
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
            if(zcount>0)
                std::cout<<"missing diag: "<<zcount<<std::endl;

            for(i =0;i<data::mSize; i+= data::blockSize){
               
                if(i + data::blockSize >= data::mSize)
                     continue;
                int step = data::blockSize/8; // / 2;
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
                 newOrder = new int[data::mSize];
                 reverseOrder = new int[data::mSize];
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

          int BlockReOrder::getBlockIndexRaw()
        {
            int blocks = 0;
            data::blockRows = (data::mSize + data::blockSize - 1) / data::blockSize;
            data::blockRows = BlockPlanner::roundup(data::blockRows);
            BlockReOrder::ReOrderBlockSize = data::blockSize;
            BlockReOrder::ReOrderBlockRows = data::blockRows;
            std::vector<edge*> slots;
            int lastRow = -1;
            int lastCol = -1;

            for(int i = 0; i < data::valcount; i++)
            {
                int row = data::indexi[i] ;
                int col = data::indexj[i] ;
                lastRow = row;
                lastCol = col;
                slots.push_back(new edge(lastRow, lastCol));
            }
            std::sort(slots.begin(), slots.end(), edgeCompare);
    //        broker::postMessage("slots found: " + std::to_string(slots.size()));

            lastRow = -1;
            lastCol = -1;
            for(int i = 0; i<slots.size(); i++)
            {
                if (lastRow == slots[i]->row && lastCol == slots[i]->col)
                    continue;
                lastRow = slots[i]->row;
                lastCol = slots[i]->col;
                blocks++;
            }

            if(indexi != NULL)
               delete[] indexi;
            if(indexj != NULL)
               delete[] indexj;
            indexi = new int[blocks];
            indexj = new int[blocks];
            indexcount = blocks;
            blocks = 0;
            lastRow = -1;
            lastCol = -1;
            for (int i = 0; i < slots.size() && blocks < indexcount; i++)
            {
                if (lastRow == slots[i]->row && lastCol == slots[i]->col)
                    continue;
                lastRow = slots[i]->row;
                lastCol = slots[i]->col;
                indexi[blocks] = lastRow;
                indexj[blocks] = lastCol;
                blocks++;
            }

            for(edge* e : slots){
                if(e != NULL)
                     delete e;
            }
            slots.clear();
            broker::postMessage("edges found: " + std::to_string(blocks));
            return blocks;
        }
          int BlockReOrder::getBlockIndex()
        {
            int blocks = 0;
    /*        for(int i = 0; i < data::blockRows; i++)
            {
                for(int j = 0; j < data::blockRows; j++)
                {
                    if (data::blocks[i * data::blockRows + j] > 0 || data::blocks[j * data::blockRows + i] > 0)
                        blocks++;
                }
            }

            indexi = new int[blocks];
            indexj = new int[blocks];
            indexcount = blocks;
            blocks = 0;
            for (int i = 0; i < data::blockRows; i++)
            {
                for (int j = 0; j < data::blockRows; j++)
                {
                    if (data::blocks[i * data::blockRows + j] > 0 || data::blocks[j * data::blockRows + i] > 0)
                    {
                        indexi[blocks] = i;
                        indexj[blocks] = j;
                        blocks++;
                    }
                }
            }

     */       broker::postMessage("error here blocks found: " + std::to_string(blocks));
            return blocks;
        }
          int BlockReOrder::checkBandwidth()
        {
            int band = 0;
            int maxi = 0, maxj = 0;
            int i64 = 0, j64=0;
            int* columncount = new int[(data::mSize+data::blockSize-1)/data::blockSize];
            for(int j=0;j<(data::mSize+data::blockSize-1)/data::blockSize;j++) columncount[j] = j;
            for (int i = 0; i < indexcount; i++)
            {
                if(indexj[i] >= indexi[i]) continue;

                i64 = (indexi[i] +data::blockSize-1)/data::blockSize;
                j64 = (indexj[i] +data::blockSize-1)/data::blockSize;
                if(i64 > columncount[j64])
                    columncount[j64] = i64;
            }
            int area = 0;
            for(int j=0;j<(data::mSize+data::blockSize-1)/data::blockSize;j++){
                if(columncount[j] >maxj){
                    int side = columncount[j] - j;
                    area += side * (side + 1) / 2;
                    if(maxi > j){
                        side = maxi - j;
                        area -= side * (side + 1);
                    }
                    maxi = columncount[j];
                    maxj = j;
                }
            }
            delete[] columncount;
            return area;
        }
/*          int checkBandwidth_old()
        {
            int band = 0;
    //        broker::postMessage("indexij found: " + std::to_string(indexcount));
    //        std::cout << "indexi "<< indexi << " val " << std::endl;
            for (int i = 0; i < indexcount; i++)
            {
                int diff = indexi[i] - indexj[i];
                if (band < diff)
                {
                    band = diff;
                    vStart = i;
                }
                else
                {
                    if (band < -diff)
                    {
                        band = -diff;
                        vStart = i;
                    }
                }
            }
            broker::postMessage("vStart: " + std::to_string(vStart));
     //       broker::postMessage(" initial block band width: " + std::to_string(band) + " at Row: " + std::to_string(indexi[vStart]) + "  Col: " + std::to_string(indexj[vStart]));
            return band;
        }
*/
          void BlockReOrder::updateEdges(int proc)
        {
          //  for(edge* e : edges){
          //      if(e != NULL)
          //           delete e;
          //  }

            edges.clear();
            std::vector<edge*> tedges = {};
            tedges.clear();
  //       std::cout<<"start update edge"<<std::endl;
            for (int i = 0; i < indexcount; i++)
            {
                tedges.push_back(data::newedge(indexi[i], indexj[i]));
                tedges.push_back(data::newedge(indexj[i], indexi[i]));
            }
            std::sort(tedges.begin(), tedges.end(), edgeCompare);
            int lrow = -1;
            int lcol = -1;
            for(int i=0;i<tedges.size();i++){
                if(tedges[i]->row == lrow && tedges[i]->col == lcol) continue;
                lrow = tedges[i]->row;
                lcol = tedges[i]->col;
                edges.push_back(tedges[i]);
            }
  //          std::sort(edges.begin(), edges.end(), edgeCompare);
            if(rowOffset != NULL)
                delete[] rowOffset;
            rowOffset = new int[data::mSize];
            int row = edges[0]->row;
            rowOffset[row] = 0;
//         std::cout<<"inserted edge"<<std::endl;
            for (int i = 1; i < edges.size(); i++)
            {
                if (edges[i]->row != row)
                {
                    row = edges[i]->row;
                    rowOffset[row] = i;
                }
            }
 //        std::cout<<"finish update edge"<<std::endl;
        }

/*          void BlockReOrder::updateEdges()
        {
          //  for(edge* e : edges){
          //      if(e != NULL)
          //           delete e;
          //  }

            edges.clear();
         std::cout<<"start update edge"<<std::endl;
            for (int i = 0; i < indexcount; i++)
            {
                edges.push_back(data::newedge(indexi[i], indexj[i]));
            }
            std::sort(edges.begin(), edges.end(), edgeCompare);
            if(rowOffset != NULL)
                delete[] rowOffset;
            rowOffset = new int[data::mSize];
            int row = edges[0]->row;
            rowOffset[row] = 0;
         std::cout<<"inserted edge"<<std::endl;
            for (int i = 1; i < edges.size(); i++)
            {
                if (edges[i]->row != row)
                {
                    row = edges[i]->row;
                    rowOffset[row] = i;
                }
            }
            int count = edges.size();
         std::cout<<"updated edge"<<std::endl;
            for (int i = 0; i < count; i++)   // ????
            {
                if (findEdge(edges[i]->col, edges[i]->row, count) == -1)
                {
                    edges.push_back(data::newedge(edges[i]->col, edges[i]->row));
                }
            }
  //          broker::postMessage(" add " + std::to_string(edges.size() - count) + " new edges, total edges: " + std::to_string(edges.size()));
   //      std::cout<<"found update edge"<<std::endl;
            if (edges.size() > count)
            {
                std::sort(edges.begin(), edges.end(), edgeCompare);
                int row = edges[0]->row;
                rowOffset[row] = 0;
                for (int i = 1; i < edges.size(); i++)
                {
                    if (edges[i]->row != row)
                    {
                        row = edges[i]->row;
                        rowOffset[row] = i;
                    }
                }
            }
         std::cout<<"finish update edge"<<std::endl;
        }
*/
          int BlockReOrder::findMinConnection()
        {
            int min = data::mSize;
            int node = 0;
            for(int i = 0; i < data::mSize; i++)
            {
                int t = connectionCount(i);
                if(min > t && t>1)
                {
                    min = t;
                    node = i;
                }
            }
            vStart = node;
            broker::postMessage(" find node: " + std::to_string(node) + "    connection count: " + std::to_string(min));
            return node;
        }

          int BlockReOrder::connectionCount(int row)
        {
            if (row < data::mSize - 1)
                return rowOffset[row + 1] - rowOffset[row];
            return edges.size() - rowOffset[data::mSize - 1];
        }

          int BlockReOrder::connectionFirst(int row, int start)
        {
            int total = data::mSize;
            int rowend = edges.size();
            if (row < data::mSize - 1)
            {
                rowend = rowOffset[row + 1];
            }

      //      if (row < data::mSize)
      //      {
                for (int i = rowOffset[row] + 1; i < rowend; i++)
                {
                    for(int j = start; j < data::mSize && j < total; j++)
                    {
                        if(newOrder[j] == edges[i]->col)
                        {
                            if (total > j)
                            {
                                total = j;
                            }
                            break;
                        }
                        if(newOrder[j] == -1)
                            break;
                    }
           //         total += edges[i]->col;
                }
                return total;
       //     }

       //     for (int i = rowOffset[data::mSize - 1] + 1; i < edges.size(); i++)
       //         total += edges[i]->col;
       //     return total;
        }

          int BlockReOrder::findEdge(int row, int col, int count)
        {
            int rtn = -1;
            int start = 0;
            if (row > 0 && rowOffset[row] > 0)
                start = rowOffset[row];
   /*         for (int i = 0; i < data::mSize; i++)
            {
                if (i + start >= count)
                    break;
                if (edges[i + start]->row != row)
                    break;
                if (edges[i + start]->col == col)
                    return i + start;
                if (edges[i + start]->col > col)
                    return -1;
            }   */

            int end = start + data::mSize-1;
            if(row + 1 <data::mSize) end = rowOffset[row + 1];
            if(end >= count) end = count -1;
           
            if(edges[start]->row > row)
                   return -1;
            if(edges[start]->col == col && edges[start]->row == row){
                   return(start);
            }
            if (edges[start]->col > col && edges[start]->row == row){
                return -1;
            }
           
            if(edges[end]->row == row && edges[end]->col == col)
                return end;
            if(edges[end]->row < row)
                return -1;
            if(edges[end]->row == row && edges[end]->col < col)
                return -1;

            for(;;){
                if(start == end){
                    if(edges[start]->row == row && edges[start]->col == col)
                        return start;
                    return -1;
                }
                if(start == end-1){
                    if(edges[start]->row == row && edges[start]->col == col)
                        return start;
                    if(edges[end]->row == row && edges[end]->col == col)
                        return end;
                    return -1;
                }
                int st = (start + end)/2;
                if(edges[st]->col == col && edges[st]->row == row){
                   rtn = st;
                   break;
                }
                if(edges[st]->row < row){
                    start = st;
                    continue;
                }
                if(edges[st]->row > row){
                    end = st;
                    continue;
                }
                if(edges[st]->col < col){
                    start = st;
                    continue;
                }
                end = st;
            }
            
            return rtn;
        }

          int BlockReOrder::findSideNode(int startNode)
        {
            std::vector<std::vector<int>> levels;
            int* nodeDis = new int[data::mSize];
     //       std::cout<<data::mSize<<": "<<nodeDis<<std::endl;
            int bestLevel = 0;
            int bestNode = 0;
            int besti = 0;

            for (int i = 0; i < 6; i++)
            {
                int endNode = BlockReOrder::assignLevel(startNode, levels, nodeDis,&levelBandWidth);
                int startLevels = levels.size();
     //    std::cout<<"start Node: "<<startNode<<" start levels "<<startLevels<<" new start: "<<endNode<<" best levels "<< bestLevel<<std::endl;
                if (startLevels > bestLevel)
                {
                    bestLevel = startLevels;
                    bestNode = startNode;
                    besti = i;
                }
                startNode = endNode;
            }

            startNode = bestNode;
            BlockReOrder::assignLevel(startNode, levels, nodeDis, &levelBandWidth);

            int minBandWidth = levelBandWidth;
            int minStart = bestNode;
            int counter = -1;
            int skip = 1;
            std::vector<int> test(lastLevel);
            //test.AddRange(lastLevel);

      std::cout<<"found most leve at:" <<besti<<" test candidate set: "<<test.size()<<std::endl;
            if(test.size()>40){
                skip = test.size()/20;
                test.clear();
                for(int i=0;i<lastLevel.size();i++){
                    if(i%skip==0) test.push_back(lastLevel[i]);
                }
                skip = 1;
            }
            int* testBandWidth = new int[test.size()];
 //          std::cout<<"set size:"<<test.size()<<std::endl;
//#pragma omp parallel for
            for (int i=0; i<test.size(); i++)
            {
                int levelstart = test[i];
                std::vector<std::vector<int>> testlevels;
            int* testnodeDis = new int[data::mSize];
                counter++;
    //       std::cout<<" test start node: "<<levelstart<<std::endl;
                assignLevel(levelstart, testlevels, testnodeDis, testBandWidth+i);
    //     std::cout<<" test start node: "<<levelstart<<" bandwidth "<< testBandWidth[i]<< " levels " << testlevels.size()
      //     <<&testlevels<<" "<<testnodeDis<<std::endl;
            //    if(levels.size()<bestLevel) continue;
     //         std::cout<<" node: "<<levelstart<<" bandwidth: "<<testBandWidth[i]<<std::endl;
                for(std::vector<int> level:testlevels){
                   level.clear();
              }
            testlevels.clear();
                delete[] testnodeDis;
            }

  //          std::cout<<"done parallel" <<std::endl;
            for (int i=0;i<test.size(); i++){
                if (minBandWidth > testBandWidth[i])
                {
                    minBandWidth = testBandWidth[i];
                    minStart = test[i];
                }
            }
        std::cout<<" min start "<<minStart<<" min BandWitdth " << minBandWidth<<std::endl;

            delete[] testBandWidth;
            delete[] nodeDis;
            test.clear();
            return minStart;
        }

          void BlockReOrder::findStartNode()
        {
            int level1 = 0;
            if (vEnd < 0)
            {
                if (connectionCount(indexi[vStart]) < connectionCount(indexj[vStart]))
                    level1 = indexi[vStart];
                else
                    level1 = indexj[vStart];
            }
            else
            {
                level1 = vEnd;
            }

            assignLevel(level1);
            for(int i=0;i<56;i++)
                assignLevel(i);
            assignLevel(vEnd);
            assignLevel(vEnd);

            std::vector<int> test(lastLevel);

            int minBandWidth = data::mSize;
            int minStart = 0;
            for (int levelstart:test)
            {
                assignLevel(levelstart);
                if (minBandWidth > levelBandWidth)
                {
                    minBandWidth = levelBandWidth;
                    minStart = levelstart;
                }
            }
            assignLevel(minStart);
            reOrder();
            test.clear();
        }

          void BlockReOrder::assignLevel(int startNode)
        {
            for(std::vector<int> level:levelList){
                   level.clear();
              }
            levelList.clear();
            levelBandWidth = 0;
            colored = new int[data::mSize];
            for(int i=0;i<data::mSize;i++){
                colored[i] = -1;
            }
            std::vector<int> first;
            int level1 = startNode;

   //         broker::postMessage(" start node: " + std::to_string(startNode));

            first.push_back(level1);
            levelList.push_back(first);

            colored[level1] = 1;
            nextLevel(2, first);
        }

          int BlockReOrder::assignLevel(int startNode, std::vector<std::vector<int>> &levels, int* dis, int* maxBandWidth)
        {
            for(std::vector<int> lev:levels){
                   lev.clear();
              }
            levels.clear();
            *maxBandWidth = 0;
            for (int i = 0; i < data::mSize; i++)
                dis[i] = 0;
            std::vector<int> first;
            int level1 = startNode;

   //         broker::postMessage(" start node: " + std::to_string(startNode));

            first.push_back(level1);
            levels.push_back(first);

            dis[level1] = 1;
            return nextLevel(2, first, levels, dis, maxBandWidth);
        }

          int BlockReOrder::nextLevel(int level, std::vector<int> list, std::vector<std::vector<int>> &levels, int* dis, int* maxBandWidth)
        {
            std::vector<int> next={};
            for (int row : list)
            {
                int offset = rowOffset[row];
                for (int i = 0; i < data::mSize; i++)
                {
                    if (offset + i >= edges.size())
                        break;
                    if (edges[offset + i]->row != row)
                        break;
                    if (dis[edges[offset + i]->col] > 0)
                        continue;
                    dis[edges[offset + i]->col] = level;
                    next.push_back(edges[offset + i]->col);
                }
            }
//            broker::postMessage(" level  " + std::to_string(level) + "  count:  " + std::to_string(next.size()));
            levels.push_back(next);
    //        std::cout<<maxBandWidth<<" next "<<&next<<" list "<<&list<<std::endl;

            if (next.size() > 0)
            {
                if (next.size() > *maxBandWidth)
                {
                    *maxBandWidth = next.size();
                }
                lastLevel = next;
                return nextLevel(level + 1, next, levels, dis, maxBandWidth);
            }

            int nodecount = 0;
            for (std::vector<int> li : levels)
                nodecount += li.size();

     //       broker::postMessage("level bandwith: " + std::to_string(*maxBandWidth) + "      total nodes: " + std::to_string(nodecount) + " level count:  " + std::to_string(levels.size()));
            int least = data::mSize;
            int vleast = -1;
            for (int row : list)
            {
                int nodes = connectionCount(row);
                if (nodes < least)
                {
                    least = nodes;
                    vleast = row;
                }
                //               broker.postMessage(" node: " + row + "   edges: " + nodes);
            }
             //         broker.postMessage(" new start node: " + vEnd + " with " + least + " edge");
            vEnd = vleast;
            return vleast;

        }

          void BlockReOrder::reOrder()
        {
            std::vector<position> seq;
            newOrder = new int[data::mSize];
            reverseOrder = new int[data::mSize];
            for (int i = 0; i < data::mSize; i++){
                reverseOrder[i] = -1;
                newOrder[i] = -1;
            }

            int count = 0;
            int lastcount = 0;
            for (std::vector<int> curLevel : levelList)
            {
                for(position pis : seq){
              //      delete pis;
                }
                seq.clear();
                if (curLevel.size() == 1)
                {
                    newOrder[count] = curLevel[0];
                    reverseOrder[curLevel[0]] = count;
                    count = count + 1;
                    continue;
                }
                if (curLevel.size() == 0)
                {
                    continue;
                }
                for (int idx : curLevel)
                {
                    seq.push_back(position(idx, connectionFirst(idx, lastcount)));
                }
                std::sort(seq.begin(), seq.end(), positionCompare);

                lastcount = count;
                for (int i = 0; i < seq.size(); i++)
                {
 //             std::cout<<seq[i].value<<std::endl;
                    newOrder[count] = seq[i].index;
                    reverseOrder[seq[i].index] = count;
                    count = count + 1;
                }
            }

            for (int i = 0; i < data::mSize; i++)
            {
                if (reverseOrder[i] >= 0)
                    continue;
                newOrder[count] = i;
                reverseOrder[i] = count;
                count = count + 1;
            }

            for (int i = 0; i < data::valcount; i++)
            {
                int iidx = data::indexi[i] ;

                int jidx = data::indexj[i] ;

                data::indexi[i] = reverseOrder[iidx] ;
                data::indexj[i] = reverseOrder[jidx] ;

            }
            double* tmp = data::b;
            
            int rows = (data::mSize + data::blockSize - 1) / data::blockSize;
            rows = BlockPlanner::roundup(rows);
            data::b = new double[rows * data::blockSize];
            for (int i = 0; i < data::mSize; i++)
            {
                int idx = i ;

                data::b[i] = tmp[newOrder[idx]];
            }
            for(int i = data::mSize; i< rows * data::blockSize; i++){
                data::b[i] = 1;
            }
            delete[] tmp;
        }
          void BlockReOrder::nextLevel(int level, std::vector<int> &list)
        {
            std::vector<int> next;
            for (int row : list)
            {
                int offset = rowOffset[row];
                for (int i = 0; i < data::mSize; i++)
                {
                    if (offset + i >= edges.size())
                        break;
                    if (edges[offset + i]->row != row)
                        break;
                    if (colored[edges[offset + i]->col] > 0)
                        continue;
                    colored[edges[offset + i]->col] = level;
                    next.push_back(edges[offset + i]->col);
                }
            }
            //broker.postMessage(" level  " + std::to_string(level + "  count:  " + std::to_string(next.size());
            levelList.push_back(next);

            if (next.size() > 0)
            {
                if (next.size() > levelBandWidth)
                {
                    levelBandWidth = next.size();
                }
                lastLevel = next;
                nextLevel(level + 1, next);
                return;
            }

            int nodecount = 0;
            for (std::vector<int> li:levelList)
                nodecount += li.size();

//            broker::postMessage("level bandwith: " + std::to_string(levelBandWidth) + "      total nodes: " + std::to_string(nodecount) + " level count:  " + std::to_string(levelList.size()));
            int least = data::mSize;
            for (int row : list)
            {
                int nodes = connectionCount(row);
                if (nodes < least)
                {
                    least = nodes;
                    vEnd = row;
                }
                //               broker.postMessage(" node: " + row + "   edges: " + nodes);
            }
            //          broker.postMessage(" new start node: " + vEnd + " with " + least + " edge");
            return;

        }
}

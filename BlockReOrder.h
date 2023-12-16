#include <vector>

namespace vmatrix
{
    class BlockReOrder
    {
         public:
         static int vStart;
         static int vEnd;
         static std::vector<edge*> edges;
         static int *rowOffset;
         static int *colored;
         static int *newOrder;
         static int *reverseOrder;
         static int levelBandWidth;
         static std::vector<int> lastLevel;
         static std::vector<std::vector<int>> levelList;
         static int ReOrderBlockSize;
         static int ReOrderBlockRows;

         static int *indexi;
         static int *indexj;
         static int indexcount;

         static void clear();

         static void filltoBlocks();
         static void sortInBlock();

         static int getBlockIndexRaw();
         static int getBlockIndex();
         static int checkBandwidth();
         static void updateEdges(int proc);

         static int findMinConnection();

         static int connectionCount(int row);

         static int connectionFirst(int row, int start);

         static int findEdge(int row, int col, int count);

         static int findSideNode(int startNode);

         static void findStartNode();

         static void assignLevel(int startNode);

         static int assignLevel(int startNode, std::vector<std::vector<int>> &levels, int* dis, int* bw);

         static int nextLevel(int level, std::vector<int> list, std::vector<std::vector<int>> &levels, int* dis, int* bw);

         static void reOrder();
         static void nextLevel(int level, std::vector<int> &list);
    };
}

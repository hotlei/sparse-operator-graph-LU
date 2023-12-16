#include <iostream>     // std::cout
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <set>       // std::vector
#include <memory>
#include <string>

#define BLOCK64 64
#define BLOCK16 16 
#define NUMSTREAM 23 

namespace vmatrix {
    enum blockOp { inv, lu, lowerInv, upperInv, sub, add, neg, copy, mul, mulneg,noop};

    class position
    {
        
        public:
        int value;
        int index;
        position(int idx, int val);
    };

        bool positionCompare(position x, position y);

    struct edge
    {
        public:
        int row;
        int col;
        edge(int i, int j);
        edge();
        void sett(int i, int j);
    };
        bool edgeCompare(edge* x, edge* y);

    struct operation
    {
        public:
        int src, src2;
        blockOp op;
        int result, result2;
        bool skip;
        int stage;
        int sequenceNum;
        static int seq;

        operation();
        operation(int src1, int s2, blockOp o, int d1, int d2);
        void sett(int src1, int s2, blockOp o, int d1, int d2);
    };
    
        bool operationCompare(operation* x, operation* y);

    class data
    {
        
   //     static {
   //         data::blockSize = 64;
   //         data::storageCount=1;
   //     }

        public:
        static int mSize;
        static int mSize_out;
        static int blockRows;
        static int blockRows_sq;
        static int blockSize;
        static int *blocks;
        static int gpuMB;
 
        static int blockRowsL2;
        static int blockSizeL2;
        static int blockScaleL2;
        static std::vector<int*> blockstorageL2;
        static int storageCountL2;
        static std::vector<operation*> seqL2;
        static bool PlanL2;
        static std::vector<operation*> operationBufferL2;
        static int operationPointerL2;
        static operation* newoperationL2(int src1, int s2, blockOp o, int d1, int d2);

        static int *indexi;
        static int *indexj;
        static double *vals;
        static int valcount;
        static std::vector<double*> blockstorage;
        static int storageCount;

        static int lucount;
        static int linvcount;
        static int uinvcount;
        static int mulcount;
        static int invertcount;
        static int foundzeroblock;
        static int storageResizeCount;
        static int getBlockCount; 
        static int blockMulCount;
        static double *b;
        static double *x;
        static std::string fileName;
 
        static std::vector<operation*> seq;
        static std::unique_ptr<int[]> stage;
        static std::unique_ptr<int[]> laststage;
        static bool UseCPU ;

        static std::vector<operation*> operationBuffer;
        static int operationPointer;
        static operation* newoperation(int src1, int s2, blockOp o, int d1, int d2);

        static std::vector<edge*> edgeBuffer;
        static int edgePointer;
        static edge* newedge(int d1, int d2);

        static std::vector<double*> blockStorageBuffer;
        static int blockStoragePointer;
        static double* newalignedblock();

        static int checkBandwidth();
        static void clear();
        static void clearSeq();
    };
}

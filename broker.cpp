#include <iostream>
#include <string>
#include "broker.h"

namespace vmatrix{
        void broker::postMessage(std::string s)
        {
            std::cout << s <<'\n';
        }
}

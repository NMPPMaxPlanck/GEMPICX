#include <AMReX.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "gmock/gmock.h"

namespace{
    class FooInterface {
        public:
        virtual int foo() = 0;
    };

    TEST(SplittingTest, TimeLoopTest) {
        
    }
}
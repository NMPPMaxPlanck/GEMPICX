#include <array>
#include <vector>
#include <gtest/gtest.h>


TEST(TestSuitExample, UnitTest1) {

    std::array<int, 3> array1{1, 2, 3};
    std::array<int, 3> array2{9, 8, 7};
    EXPECT_EQ(array1, array1);
    EXPECT_EQ(array1, array2);
    EXPECT_EQ(array2, array2);
}

TEST(TestSuitExample, UnitTest2) {

    std::vector<double> array1{1., 2., 3.};
    std::vector<double> array2{9., 8., 7.};
    EXPECT_DOUBLE_EQ(array1, array1);
    EXPECT_DOUBLE_EQ(array1, array2);
    EXPECT_DOUBLE_EQ(array2, array2);
}

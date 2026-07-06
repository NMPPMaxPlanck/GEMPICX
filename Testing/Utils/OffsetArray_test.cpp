/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <array>
#include <type_traits>

#include <gtest/gtest.h>

#include "GEMPIC_OffsetArray.H"

using Gempic::Utils::OffsetArray;

TEST(OffsetArray, ConstructionRoundTrips)
{
    std::array<double, 4> const values{1.0, 2.0, 3.0, 4.0};
    OffsetArray<double, 4> arr{7, values};

    EXPECT_EQ(arr.first_index(), 7);
    EXPECT_EQ(arr.last_index(), 10);
    EXPECT_EQ(arr.size(), 4u);
    EXPECT_EQ(arr.values(), values);
}

TEST(OffsetArray, OperatorBracketUsesGridIndex)
{
    std::array<double, 4> const values{10.0, 20.0, 30.0, 40.0};
    OffsetArray<double, 4> arr{7, values};

    EXPECT_DOUBLE_EQ(arr[7], 10.0);
    EXPECT_DOUBLE_EQ(arr[8], 20.0);
    EXPECT_DOUBLE_EQ(arr[9], 30.0);
    EXPECT_DOUBLE_EQ(arr[10], 40.0);
}

TEST(OffsetArray, OperatorBracketIsMutable)
{
    OffsetArray<double, 3> arr{2, std::array<double, 3>{0.0, 0.0, 0.0}};
    arr[2] = 1.5;
    arr[3] = 2.5;
    arr[4] = 3.5;
    EXPECT_DOUBLE_EQ(arr[2], 1.5);
    EXPECT_DOUBLE_EQ(arr[3], 2.5);
    EXPECT_DOUBLE_EQ(arr[4], 3.5);
}

TEST(OffsetArray, NegativeFirstIndex)
{
    std::array<double, 3> const values{100.0, 200.0, 300.0};
    OffsetArray<double, 3> arr{-1, values};

    EXPECT_EQ(arr.first_index(), -1);
    EXPECT_EQ(arr.last_index(), 1);
    EXPECT_DOUBLE_EQ(arr[-1], 100.0);
    EXPECT_DOUBLE_EQ(arr[0], 200.0);
    EXPECT_DOUBLE_EQ(arr[1], 300.0);
}

TEST(OffsetArray, IsTriviallyCopyable)
{
    static_assert(std::is_trivially_copyable_v<OffsetArray<double, 4>>,
                  "OffsetArray must be trivially copyable for GPU use.");
    static_assert(std::is_trivially_copyable_v<OffsetArray<float, 6>>,
                  "OffsetArray must be trivially copyable for GPU use.");
    SUCCEED();
}

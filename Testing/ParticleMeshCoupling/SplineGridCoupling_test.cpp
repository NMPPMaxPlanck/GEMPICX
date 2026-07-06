/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/

#include <numeric>

#include <gtest/gtest.h>

#include "GEMPIC_BSpline.H"
#include "GEMPIC_ComputationalDomain.H"

using namespace Gempic;
using namespace Gempic::ParticleMeshCoupling;

/***********************************************************/
/* FIXME: Located BSplines are not properly tested         */
/***********************************************************/
TEST(BSplineGridCoupling, Unity)
{
    constexpr int degree{3};
    constexpr int width{degree + 1};
    Utils::OffsetArray<amrex::Real, width> localized{};
    std::array<amrex::Real, width> cardinal{};
    constexpr int idx{5};
    DiscreteAxis cell{3.0, 6.0, 10, DiscreteAxis::IndexPosition::Cell, true};
    LocalizedBSpline<degree> localizedBSplines{};

    localized = localizedBSplines(cell.location(idx) - 0.5 * cell.dx(), cell);
    cardinal = cardinal_b_splines<degree, width, std::array<amrex::Real, width>>(0.5);

    for (int i = 0; i < width; i++)
    {
        EXPECT_NEAR(localized[i + localized.first_index()], cardinal[i], 1.0e-14);
    }
    EXPECT_EQ(localized.first_index(), idx - 2);
}

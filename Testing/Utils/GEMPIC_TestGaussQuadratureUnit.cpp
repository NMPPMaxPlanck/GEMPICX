/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <array>
#include <cmath>
#include <numeric>

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_NumericalIntegrationDifferentiation.H"

namespace
{

template <typename PointsWrapper>
class GaussQuadratureUnitTest : public testing::Test
{
public:
    static constexpr int s_n = PointsWrapper::value;
};

using QuadratureOrders = ::testing::Types<std::integral_constant<int, 1>,
                                          std::integral_constant<int, 2>,
                                          std::integral_constant<int, 3>,
                                          std::integral_constant<int, 4>,
                                          std::integral_constant<int, 5>,
                                          std::integral_constant<int, 6>,
                                          std::integral_constant<int, 7>,
                                          std::integral_constant<int, 8>,
                                          std::integral_constant<int, 9>,
                                          std::integral_constant<int, 10>,
                                          std::integral_constant<int, 11>,
                                          std::integral_constant<int, 12>,
                                          std::integral_constant<int, 14> >;

TYPED_TEST_SUITE(GaussQuadratureUnitTest, QuadratureOrders);

TYPED_TEST(GaussQuadratureUnitTest, ExactnessAndBasicProperties)
{
    constexpr int n = TestFixture::s_n;
    constexpr amrex::Real tol = 1e-13;

    using GL = Gempic::GaussQuadratureUnit<n>;
    constexpr auto nodes = GL::s_nodes;
    constexpr auto weights = GL::s_weights;

    // -------------------------------------------------
    // 1. Weights sum to 1 on [0,1]
    // -------------------------------------------------
    amrex::Real wsum = 0.0;
    for (int i = 0; i < n; ++i) wsum += weights[i];

    EXPECT_NEAR(wsum, 1.0, tol);

    // -------------------------------------------------
    // 2. Nodes are inside (0,1)
    // -------------------------------------------------
    for (int i = 0; i < n; ++i)
    {
        EXPECT_GT(nodes[i], 0.0);
        EXPECT_LT(nodes[i], 1.0);
    }

    // -------------------------------------------------
    // 3. Symmetry around 0.5
    //    x_i + x_{n-1-i} = 1
    //    w_i = w_{n-1-i}
    // -------------------------------------------------
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(nodes[i] + nodes[n - 1 - i], 1.0, tol);
        EXPECT_NEAR(weights[i], weights[n - 1 - i], tol);
    }

    // -------------------------------------------------
    // 4. Polynomial exactness up to degree 2n-1
    // -------------------------------------------------
    for (int k = 0; k <= 2 * n - 1; ++k)
    {
        amrex::Real quadVal = 0.0;

        for (int i = 0; i < n; ++i) quadVal += weights[i] * std::pow(nodes[i], k);

        amrex::Real exact = 1.0 / (k + 1);

        EXPECT_NEAR(quadVal, exact, tol)
            << "Failed for polynomial degree k=" << k << " with n=" << n;
    }
}

template <int n>
amrex::Real compute_error_exp ()
{
    using GL = Gempic::GaussQuadratureUnit<n>;
    static constexpr auto nodes = GL::s_nodes;
    static constexpr auto weights = GL::s_weights;

    amrex::Real approx = 0.0;
    for (int i = 0; i < n; ++i) approx += weights[i] * std::exp(nodes[i]);

    amrex::Real const exact = std::exp(1.0) - 1.0;
    return std::abs(approx - exact);
}

template <int... ns>
std::array<amrex::Real, sizeof...(ns)> compute_errors (std::integer_sequence<int, ns...>)
{
    return {compute_error_exp<ns>()...};
}

TEST(GaussQuadratureSpectralTest, ExponentialConvergence)
{
    constexpr auto orders = std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>{};

    auto errors = compute_errors(orders);

    constexpr amrex::Real tolSlope = 1.0;
    constexpr amrex::Real tolerance = 1e-13;

    // decreasing errors
    for (size_t i = 1; i < errors.size(); ++i)
    {
        if (errors[i] > tolerance)
        {
            EXPECT_LT(errors[i], errors[i - 1]);
        }
    }

    // exponential decay
    for (size_t i = 2; i < errors.size(); ++i)
    {
        amrex::Real slope1 = std::log(errors[i - 1]) - std::log(errors[i - 2]);

        amrex::Real slope2 = std::log(errors[i]) - std::log(errors[i - 1]);

        if (errors[i] > tolerance)
        {
            EXPECT_NEAR(slope1, slope2, tolSlope);
        }
    }
}

} // namespace
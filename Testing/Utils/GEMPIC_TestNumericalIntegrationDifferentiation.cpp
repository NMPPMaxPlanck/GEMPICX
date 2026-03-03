/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <numeric>

#include <gtest/gtest.h>

#include "GEMPIC_NumericalIntegrationDifferentiation.H"

class GaussLegendreQuadrature : public ::testing::TestWithParam<int>
{
public:
    GaussLegendreQuadrature() = default;
};

amrex::Real f (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t) { return x * y * z * t; }
TEST(GaussLegendreQuadratureExample, LineIntegrals)
{
    amrex::Real x0{1.0}, z0{1.0}, t0{1.0};
    amrex::Real y0{0.0}, dy{0.5};
    //! [GaussLegendreQuadratureExample.LineIntegral]
    Gempic::GaussLegendreQuadrature integrate{3};
    auto fy = [=] (amrex::Real y) { return f(x0, y, z0, t0); };
    amrex::Real res = integrate.line(y0, dy, fy);
    //! [GaussLegendreQuadratureExample.LineIntegral]
    EXPECT_LT(std::abs(res), 1.0e-15);
}
TEST(GaussLegendreQuadratureExample, SurfaceIntegrals)
{
    amrex::Real z0{1.0}, t0{1.0};
    amrex::Real x0{0.0}, y0{0.0};
    amrex::Real dx{0.5}, dy{0.5};
    //! [GaussLegendreQuadratureExample.SurfaceIntegral]
    Gempic::GaussLegendreQuadrature integrate{3};
    auto fxy = [=] (amrex::Real x, amrex::Real y) { return f(x, y, z0, t0); };
    amrex::Real res = integrate.surface({x0, y0}, {dx, dy}, fxy);
    //! [GaussLegendreQuadratureExample.SurfaceIntegral]
    EXPECT_LT(std::abs(res), 1.0e-15);
}
TEST(GaussLegendreQuadratureExample, VolumeIntegrals)
{
    amrex::Real t0{1.0};
    amrex::Real x0{0.0}, y0{0.0}, z0{0.0};
    amrex::Real dx{0.5}, dy{0.5}, dz{0.5};
    //! [GaussLegendreQuadratureExample.VolumeIntegral]
    Gempic::GaussLegendreQuadrature integrate{3};
    auto fxyz = [=] (amrex::Real x, amrex::Real y, amrex::Real z) { return f(x, y, z, t0); };
    amrex::Real res = integrate.volume({x0, y0, z0}, {dx, dy, dz}, fxyz);
    //! [GaussLegendreQuadratureExample.VolumeIntegral]
    EXPECT_LT(std::abs(res), 1.0e-15);
}

TEST_P(GaussLegendreQuadrature, LineIntegrals)
{
    auto f = [=] (amrex::Real const& x) { return x + 1; };
    auto fint = [=] (amrex::Real const& min, amrex::Real const& max)
    { return 1.0 / 2.0 * (max * max - min * min) + max - min; };
    Gempic::GaussLegendreQuadrature integrate(GetParam());
    EXPECT_DOUBLE_EQ(integrate.line(2.0, 2.0, f), fint(0.0, 4.0));
}

TEST_P(GaussLegendreQuadrature, SurfaceIntegrals)
{
    auto f = [=] (amrex::Real const& x, amrex::Real const& y) { return x + y + 1; };
    auto fint = [=] (std::array<amrex::Real, 2> const& min, std::array<amrex::Real, 2> const& max)
    {
        amrex::Real a{min[0]}, b{max[0]}, c{min[1]}, d{max[1]};
        return 1.0 / 2.0 * (a - b) * (c - d) * (a + b + c + d + 2);
    };
    Gempic::GaussLegendreQuadrature integrate(GetParam());
    EXPECT_DOUBLE_EQ(integrate.surface(std::array<amrex::Real, 2>{0.0, 0.0},
                                       std::array<amrex::Real, 2>{1.0, 2.0}, f),
                     fint({-1.0, -2.0}, {1.0, 2.0}));
}

TEST_P(GaussLegendreQuadrature, VolumeIntegrals)
{
    auto f = [=] (amrex::Real const& x, amrex::Real const& y, amrex::Real const& z)
    { return x + y + z + 1; };
    auto fint = [=] (std::array<amrex::Real, 3> const& min, std::array<amrex::Real, 3> const& max)
    {
        amrex::Real a{min[0]}, b{max[0]}, c{min[1]}, d{max[1]}, e{min[2]}, f{max[2]};
        return -1.0 / 2.0 * (a - b) * (c - d) * (e - f) * (a + b + c + d + e + f + 2);
    };
    Gempic::GaussLegendreQuadrature integrate(GetParam());
    EXPECT_DOUBLE_EQ(integrate.volume(std::array<amrex::Real, 3>{1.0, 2.0, 2.5},
                                      std::array<amrex::Real, 3>{1.0, 2.0, 2.5}, f),
                     fint({0.0, 0.0, 0.0}, {2.0, 4.0, 5.0}));
}

INSTANTIATE_TEST_SUITE_P(GaussLegendreQuadratureStencils,
                         GaussLegendreQuadrature,
                         ::testing::Range(1, 11));
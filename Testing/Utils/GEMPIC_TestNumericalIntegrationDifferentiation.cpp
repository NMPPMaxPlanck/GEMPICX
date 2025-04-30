#include <numeric>

#include <gtest/gtest.h>

#include "GEMPIC_NumericalIntegrationDifferentiation.H"

class GaussQuadrature : public ::testing::TestWithParam<int>
{
public:
    GaussQuadrature() = default;
};

amrex::Real f (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t) { return x * y * z * t; }
TEST(GaussQuadratureExample, LineIntegrals)
{
    amrex::Real x0{1.0}, z0{1.0}, t0{1.0};
    amrex::Real y0{0.0}, dy{0.5};
    //! [GaussQuadratureExample.LineIntegral]
    Gempic::GaussQuadrature integrate{3};
    auto fy = [=] (amrex::Real y) { return f(x0, y, z0, t0); };
    amrex::Real res = integrate.line(y0, dy, fy);
    //! [GaussQuadratureExample.LineIntegral]
    EXPECT_LT(std::abs(res), 1.0e-15);
}
TEST(GaussQuadratureExample, SurfaceIntegrals)
{
    amrex::Real z0{1.0}, t0{1.0};
    amrex::Real x0{0.0}, y0{0.0};
    amrex::Real dx{0.5}, dy{0.5};
    //! [GaussQuadratureExample.SurfaceIntegral]
    Gempic::GaussQuadrature integrate{3};
    auto fxy = [=] (amrex::Real x, amrex::Real y) { return f(x, y, z0, t0); };
    amrex::Real res = integrate.surface({x0, y0}, {dx, dy}, fxy);
    //! [GaussQuadratureExample.SurfaceIntegral]
    EXPECT_LT(std::abs(res), 1.0e-15);
}
TEST(GaussQuadratureExample, VolumeIntegrals)
{
    amrex::Real t0{1.0};
    amrex::Real x0{0.0}, y0{0.0}, z0{0.0};
    amrex::Real dx{0.5}, dy{0.5}, dz{0.5};
    //! [GaussQuadratureExample.VolumeIntegral]
    Gempic::GaussQuadrature integrate{3};
    auto fxyz = [=] (amrex::Real x, amrex::Real y, amrex::Real z) { return f(x, y, z, t0); };
    amrex::Real res = integrate.volume({x0, y0, z0}, {dx, dy, dz}, fxyz);
    //! [GaussQuadratureExample.VolumeIntegral]
    EXPECT_LT(std::abs(res), 1.0e-15);
}

TEST_P(GaussQuadrature, LineIntegrals)
{
    auto f = [=] (amrex::Real const& x) { return x + 1; };
    auto fint = [=] (amrex::Real const& min, amrex::Real const& max)
    { return 1.0 / 2.0 * (max * max - min * min) + max - min; };
    Gempic::GaussQuadrature integrate(GetParam());
    EXPECT_DOUBLE_EQ(integrate.line(2.0, 2.0, f), fint(0.0, 4.0));
}

TEST_P(GaussQuadrature, SurfaceIntegrals)
{
    auto f = [=] (amrex::Real const& x, amrex::Real const& y) { return x + y + 1; };
    auto fint = [=] (std::array<amrex::Real, 2> const& min, std::array<amrex::Real, 2> const& max)
    {
        amrex::Real a{min[0]}, b{max[0]}, c{min[1]}, d{max[1]};
        return 1.0 / 2.0 * (a - b) * (c - d) * (a + b + c + d + 2);
    };
    Gempic::GaussQuadrature integrate(GetParam());
    EXPECT_DOUBLE_EQ(integrate.surface(std::array<amrex::Real, 2>{0.0, 0.0},
                                       std::array<amrex::Real, 2>{1.0, 2.0}, f),
                     fint({-1.0, -2.0}, {1.0, 2.0}));
}

TEST_P(GaussQuadrature, VolumeIntegrals)
{
    auto f = [=] (amrex::Real const& x, amrex::Real const& y, amrex::Real const& z)
    { return x + y + z + 1; };
    auto fint = [=] (std::array<amrex::Real, 3> const& min, std::array<amrex::Real, 3> const& max)
    {
        amrex::Real a{min[0]}, b{max[0]}, c{min[1]}, d{max[1]}, e{min[2]}, f{max[2]};
        return -1.0 / 2.0 * (a - b) * (c - d) * (e - f) * (a + b + c + d + e + f + 2);
    };
    Gempic::GaussQuadrature integrate(GetParam());
    EXPECT_DOUBLE_EQ(integrate.volume(std::array<amrex::Real, 3>{1.0, 2.0, 2.5},
                                      std::array<amrex::Real, 3>{1.0, 2.0, 2.5}, f),
                     fint({0.0, 0.0, 0.0}, {2.0, 4.0, 5.0}));
}

INSTANTIATE_TEST_SUITE_P(GaussQuadratureStencils, GaussQuadrature, ::testing::Range(1, 11));
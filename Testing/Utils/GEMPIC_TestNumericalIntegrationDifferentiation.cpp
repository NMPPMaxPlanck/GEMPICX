#include <numeric>

#include <gtest/gtest.h>

#include "GEMPIC_NumericalIntegrationDifferentiation.H"

class GaussQuadrature : public ::testing::TestWithParam<int>
{
public:
    GaussQuadrature() = default;
};

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
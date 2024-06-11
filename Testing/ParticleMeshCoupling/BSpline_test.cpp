#include <numeric>

#include <gtest/gtest.h>

#include "GEMPIC_BSpline.H"

template <size_t nIntervalls>
void check_continuity (
    std::function<amrex::GpuArray<amrex::Real, nIntervalls>(amrex::Real)> bSpline,
    std::pair<amrex::Real, amrex::Real> boundaryValues = {0.0, 0.0})
{
    EXPECT_EQ(bSpline(0.0)[0], boundaryValues.first)
        << "Error is at the left most interval boundary (i): " << 0
        << " for B-Spline of degree: " << nIntervalls - 1;
    for (size_t i = 0; i < nIntervalls - 1; i++)
    {
        EXPECT_DOUBLE_EQ(bSpline(1.0)[i], bSpline(0.0)[i + 1])
            << "Error is at interval boundary (i): " << i
            << " for B-Spline of degree: " << nIntervalls - 1;
    }
    EXPECT_EQ(bSpline(1.0)[nIntervalls - 1], boundaryValues.second)
        << "Error is at the left most interval boundary (i): " << nIntervalls
        << " for B-Spline of degree: " << nIntervalls - 1;
}
TEST(BSpline, Continuity)
{
    // B-Splines
    check_continuity<2>(Gempic::cardinal_b_splines<1, 2>);
    check_continuity<3>(Gempic::cardinal_b_splines<2, 3>);
    check_continuity<4>(Gempic::cardinal_b_splines<3, 4>);
    check_continuity<5>(Gempic::cardinal_b_splines<4, 5>);
    check_continuity<6>(Gempic::cardinal_b_splines<5, 6>);
    // Primitives of B-Splines
    check_continuity<1>(Gempic::cardinal_b_splines_primitive<0, 1>, {0.0, 1.0});
    check_continuity<2>(Gempic::cardinal_b_splines_primitive<1, 2>, {0.0, 1.0});
    check_continuity<3>(Gempic::cardinal_b_splines_primitive<2, 3>, {0.0, 1.0});
    check_continuity<4>(Gempic::cardinal_b_splines_primitive<3, 4>, {0.0, 1.0});
    check_continuity<5>(Gempic::cardinal_b_splines_primitive<4, 5>, {0.0, 1.0});
    // Derivatives of B-Splines
    check_continuity<3>(Gempic::cardinal_b_splines_first_derivative<2>);
    check_continuity<4>(Gempic::cardinal_b_splines_first_derivative<3>);
    check_continuity<5>(Gempic::cardinal_b_splines_first_derivative<4>);
    check_continuity<6>(Gempic::cardinal_b_splines_first_derivative<5>);
    check_continuity<4>(Gempic::cardinal_b_splines_second_derivative<3>);
    check_continuity<5>(Gempic::cardinal_b_splines_second_derivative<4>);
    check_continuity<6>(Gempic::cardinal_b_splines_second_derivative<5>);
}

template <size_t nIntervalls>
double integrate (std::function<amrex::GpuArray<double, nIntervalls>(double)> bSpline)
{
    double x{};
    size_t N{1000};
    double dx{1.0 / N};
    double res{};

    // Sxi[i] is S^d(x+i) as declared in the documentation
    amrex::GpuArray<double, nIntervalls> sxi{};
    while (x < 1.0)
    {
        sxi = bSpline(x + dx);
        for (size_t i = 0; i < nIntervalls; i++)
        {
            res += sxi[i] * dx;
        }
        x += dx;
    }
    return res;
}
TEST(BSpline, Norm)
{
    double res{};
    double tol{1.0e-14};
    res = integrate<1>(Gempic::cardinal_b_splines<0, 1>);
    EXPECT_NEAR(res, 1.0, tol);
    res = integrate<2>(Gempic::cardinal_b_splines<1, 2>);
    EXPECT_NEAR(res, 1.0, tol);
    res = integrate<3>(Gempic::cardinal_b_splines<2, 3>);
    EXPECT_NEAR(res, 1.0, tol);
    res = integrate<4>(Gempic::cardinal_b_splines<3, 4>);
    EXPECT_NEAR(res, 1.0, tol);
    res = integrate<5>(Gempic::cardinal_b_splines<4, 5>);
    EXPECT_NEAR(res, 1.0, tol);
    res = integrate<6>(Gempic::cardinal_b_splines<5, 6>);
    EXPECT_NEAR(res, 1.0, tol);
    res = integrate<2>(Gempic::cardinal_b_splines_first_derivative<1>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<3>(Gempic::cardinal_b_splines_first_derivative<2>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<4>(Gempic::cardinal_b_splines_first_derivative<3>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<5>(Gempic::cardinal_b_splines_first_derivative<4>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<6>(Gempic::cardinal_b_splines_first_derivative<5>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<3>(Gempic::cardinal_b_splines_second_derivative<2>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<4>(Gempic::cardinal_b_splines_second_derivative<3>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<5>(Gempic::cardinal_b_splines_second_derivative<4>);
    EXPECT_NEAR(res, 0.0, tol);
    res = integrate<6>(Gempic::cardinal_b_splines_second_derivative<5>);
    EXPECT_NEAR(res, 0.0, tol);
}

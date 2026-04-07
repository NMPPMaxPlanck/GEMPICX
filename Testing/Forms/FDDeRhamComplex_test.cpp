/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

using namespace Gempic;
using namespace Forms;

class FiniteDifferenceExternalDerivativesTest : public testing::Test
{
public:
    FiniteDifferenceExternalDerivativesTest() :
        m_fdDeRhamComplex{GaussLegendreQuadrature{6}, DiscreteGrid{}, amrex::BoxArray{},
                          amrex::DistributionMapping{}}
    {
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.1, 1.2, 1.3)};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(1, 1, 1)};
        amrex::Vector<int> const nCell{AMREX_D_DECL(10, 11, 12)};
        amrex::Vector<int> const maxGridSize{AMREX_D_DECL(10, 11, 12)};
        amrex::Vector<int> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

        m_params.set("ComputationalDomain.domainLo", domainLo);
        m_params.set("k", k);
        m_params.set("ComputationalDomain.nCell", nCell);
        m_params.set("ComputationalDomain.maxGridSize", maxGridSize);
        m_params.set("ComputationalDomain.isPeriodic", isPeriodic);

        m_fdDeRhamComplex = FiniteDifferenceDeRhamSpaces{m_gaussPoints};
    }
    Io::Parameters m_params{};
    int m_gaussPoints{10};
    FiniteDifferenceDeRhamSpaces m_fdDeRhamComplex;
};

struct F
{
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const
    {
        return GEMPIC_D_MULT(std::sin(x), std::sin(2.0 * y), std::sin(3.0 * z));
    };
};

struct GradF
{
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(Direction dir,
                           AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const
    {
        switch (dir)
        {
            case xDir:
                return GEMPIC_D_MULT(std::cos(x), std::sin(2.0 * y), std::sin(3.0 * z));
            case yDir:
#if AMREX_SPACEDIM > 1
                return GEMPIC_D_MULT(std::sin(x), 2.0 * std::cos(2.0 * y), std::sin(3.0 * z));
#else
                return 0.0;
#endif
            case zDir:
#if AMREX_SPACEDIM > 2
                return GEMPIC_D_MULT(std::sin(x), std::sin(2.0 * y), 3.0 * std::cos(3.0 * z));
#else
                return 0.0;
#endif
            default:
                return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
    }
};

struct G
{
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(Direction dir,
                           AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const
    {
        switch (dir)
        {
            case xDir:
                return GEMPIC_D_MULT(std::sin(x), std::sin(y), std::cos(z));
            case yDir:
                return GEMPIC_D_MULT(std::cos(x), std::sin(y), std::sin(z));
            case zDir:
                return GEMPIC_D_MULT(std::sin(x), std::cos(y), std::sin(z));
            default:
                return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
    };
};

struct CurlG
{
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(Direction dir,
                           AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const
    {
        switch (dir)
        {
            case xDir:
#if AMREX_SPACEDIM == 1
                return 0.0;
#elif AMREX_SPACEDIM == 2
                return GEMPIC_D_MULT(std::sin(x), -std::sin(y), std::sin(z));
#elif AMREX_SPACEDIM == 3
                return GEMPIC_D_MULT(std::sin(x), std::sin(y), -std::sin(z)) -
                       GEMPIC_D_MULT(std::cos(x), std::sin(y), std::cos(z));
#endif
            case yDir:

#if AMREX_SPACEDIM < 3
                return -GEMPIC_D_MULT(std::cos(x), std::cos(y), std::sin(z));
#else
                return GEMPIC_D_MULT(-std::sin(x), std::sin(y), std::sin(z)) -
                       GEMPIC_D_MULT(std::cos(x), std::cos(y), std::sin(z));
#endif
            case zDir:
#if AMREX_SPACEDIM == 1
                return GEMPIC_D_MULT(-std::sin(x), std::sin(y), std::sin(z));
#else
                return GEMPIC_D_MULT(-std::sin(x), std::sin(y), std::sin(z)) -
                       GEMPIC_D_MULT(std::sin(x), std::cos(y), std::cos(z));
#endif
            default:
                return std::numeric_limits<amrex::Real>::quiet_NaN();
        }
    };
};

struct DivG
{
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const
    {
        return GEMPIC_D_ADD(GEMPIC_D_MULT(std::cos(x), std::sin(y), std::cos(z)),
                            GEMPIC_D_MULT(std::cos(x), std::cos(y), std::sin(z)),
                            GEMPIC_D_MULT(std::sin(x), std::cos(y), std::cos(z)));
    };
};

void finite_difference_de_rham_complex_grad (Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex)
{
    PrimalZeroForm zf{fdDeRhamComplex.create_primal_zero_form(
        "zf", Gempic::Impl::BoundaryConditionConfiguration{})};
    PrimalOneForm of{fdDeRhamComplex.create_primal_one_form(
        "of", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    PrimalOneForm analytical{fdDeRhamComplex.create_primal_one_form(
        "analytical", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    std::array<amrex::Real, 3> lInfError{};

    project(analytical, GradF{});
    project(zf, F{});
    grad(of, zf);

    lInfError = l_inf_error(of, analytical);
    EXPECT_LE(lInfError[xDir], 1.0e-14);
    EXPECT_LE(lInfError[yDir], 1.0e-14);
    ASSERT_LE(lInfError[zDir], 1.0e-14);

    PrimalTwoForm tf{fdDeRhamComplex.create_primal_two_form(
        "tf", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    curl(tf, of);

    PrimalTwoForm zero{fdDeRhamComplex.create_primal_two_form(
        "zero", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    fill(zero,
         [=] AMREX_GPU_HOST_DEVICE(Direction /*dir*/,
                                   AMREX_D_DECL(amrex::Real /*x*/, amrex::Real /*y*/,
                                                amrex::Real /*z*/)) -> amrex::Real { return 0.0; });
    lInfError = l_inf_error(tf, zero);
    EXPECT_LE(lInfError[xDir], 1.0e-15);
    EXPECT_LE(lInfError[yDir], 1.0e-15);
    ASSERT_LE(lInfError[zDir], 1.0e-15);
}

TEST_F(FiniteDifferenceExternalDerivativesTest, PrimalGrad)
{
    finite_difference_de_rham_complex_grad(m_fdDeRhamComplex);
}

void finite_difference_de_rham_complex_curl (Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex)
{
    PrimalOneForm of{fdDeRhamComplex.create_primal_one_form(
        "of", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    PrimalTwoForm tf{fdDeRhamComplex.create_primal_two_form(
        "tf", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    PrimalTwoForm analytical{fdDeRhamComplex.create_primal_two_form(
        "analytical", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    std::array<amrex::Real, 3> lInfError{};

    project(of, G{});
    project(analytical, CurlG{});
    curl(tf, of);

    lInfError = l_inf_error(tf, analytical);
    EXPECT_LE(lInfError[xDir], 1.0e-15);
    EXPECT_LE(lInfError[yDir], 1.0e-15);
    EXPECT_LE(lInfError[zDir], 1.0e-15);

    PrimalThreeForm threeform{fdDeRhamComplex.create_primal_three_form(
        "threeForm", Gempic::Impl::BoundaryConditionConfiguration{})};
    div(threeform, tf);

    PrimalThreeForm zero{fdDeRhamComplex.create_primal_three_form(
        "zero", Gempic::Impl::BoundaryConditionConfiguration{})};
    fill(zero,
         [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real /*x*/, amrex::Real /*y*/,
                                                amrex::Real /*z*/)) -> amrex::Real { return 0.0; });
    EXPECT_LE(l_inf_error(threeform, zero), 1.0e-15);
}

TEST_F(FiniteDifferenceExternalDerivativesTest, PrimalCurl)
{
    finite_difference_de_rham_complex_curl(m_fdDeRhamComplex);
}

void finite_difference_de_rham_complex_div (Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex)
{
    PrimalTwoForm twoForm{fdDeRhamComplex.create_primal_two_form(
        "twoForm", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    PrimalThreeForm threeForm{fdDeRhamComplex.create_primal_three_form(
        "threeForm", Gempic::Impl::BoundaryConditionConfiguration{})};
    PrimalThreeForm analytical{fdDeRhamComplex.create_primal_three_form(
        "analytical", Gempic::Impl::BoundaryConditionConfiguration{})};

    project(twoForm, G{});
    project(analytical, DivG{});
    div(threeForm, twoForm);

    EXPECT_LE(l_inf_error(threeForm, analytical), 1.0e-13);
}

TEST_F(FiniteDifferenceExternalDerivativesTest, PrimalDiv)
{
    finite_difference_de_rham_complex_div(m_fdDeRhamComplex);
}

void finite_difference_de_rham_complex_dual_grad (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex)
{
    DualZeroForm zf{fdDeRhamComplex.create_dual_zero_form(
        "zf", Gempic::Impl::BoundaryConditionConfiguration{})};
    DualOneForm of{fdDeRhamComplex.create_dual_one_form(
        "of", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    DualOneForm analytical{fdDeRhamComplex.create_dual_one_form(
        "analytical", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    std::array<amrex::Real, 3> lInfError{};

    project(analytical, GradF{});
    project(zf, F{});
    grad(of, zf);

    lInfError = l_inf_error(of, analytical);
    EXPECT_LE(lInfError[xDir], 1.0e-14);
    EXPECT_LE(lInfError[yDir], 1.0e-14);
    ASSERT_LE(lInfError[zDir], 1.0e-14);

    DualTwoForm tf{fdDeRhamComplex.create_dual_two_form(
        "tf", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    curl(tf, of);

    DualTwoForm zero{fdDeRhamComplex.create_dual_two_form(
        "zero", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    fill_zero(zero);
    lInfError = l_inf_error(tf, zero);
    EXPECT_LE(lInfError[xDir], 1.0e-15);
    EXPECT_LE(lInfError[yDir], 1.0e-15);
    ASSERT_LE(lInfError[zDir], 1.0e-15);
}

TEST_F(FiniteDifferenceExternalDerivativesTest, DualGrad)
{
    finite_difference_de_rham_complex_dual_grad(m_fdDeRhamComplex);
}

void finite_difference_de_rham_complex_dual_curl (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex)
{
    DualOneForm of{fdDeRhamComplex.create_dual_one_form(
        "of", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    DualTwoForm tf{fdDeRhamComplex.create_dual_two_form(
        "tf", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    DualTwoForm analytical{fdDeRhamComplex.create_dual_two_form(
        "analytical", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    std::array<amrex::Real, 3> lInfError{};

    project(of, G{});
    project(analytical, CurlG{});
    curl(tf, of);

    lInfError = l_inf_error(tf, analytical);
    EXPECT_LE(lInfError[xDir], 1.0e-15);
    EXPECT_LE(lInfError[yDir], 1.0e-15);
    EXPECT_LE(lInfError[zDir], 1.0e-15);

    DualThreeForm threeform{fdDeRhamComplex.create_dual_three_form(
        "threeForm", Gempic::Impl::BoundaryConditionConfiguration{})};
    div(threeform, tf);

    DualThreeForm zero{fdDeRhamComplex.create_dual_three_form(
        "zero", Gempic::Impl::BoundaryConditionConfiguration{})};
    fill(zero,
         [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real /*x*/, amrex::Real /*y*/,
                                                amrex::Real /*z*/)) -> amrex::Real { return 0.0; });
    EXPECT_LE(l_inf_error(threeform, zero), 1.0e-15);
}

TEST_F(FiniteDifferenceExternalDerivativesTest, DualCurl)
{
    finite_difference_de_rham_complex_dual_curl(m_fdDeRhamComplex);
}

void finite_difference_de_rham_complex_dual_div (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex)
{
    DualTwoForm twoForm{fdDeRhamComplex.create_dual_two_form(
        "twoForm", std::array<Gempic::Impl::BoundaryConditionConfiguration, 3>{})};
    DualThreeForm threeForm{fdDeRhamComplex.create_dual_three_form(
        "threeForm", Gempic::Impl::BoundaryConditionConfiguration{})};
    DualThreeForm analytical{fdDeRhamComplex.create_dual_three_form(
        "analytical", Gempic::Impl::BoundaryConditionConfiguration{})};

    project(twoForm, G{});
    project(analytical, DivG{});
    div(threeForm, twoForm);

    EXPECT_LE(l_inf_error(threeForm, analytical), 1.0e-13);
}

TEST_F(FiniteDifferenceExternalDerivativesTest, DualDiv)
{
    finite_difference_de_rham_complex_dual_div(m_fdDeRhamComplex);
}

// ToDo: If possible remove dependency of test on parameter file
class FiniteDifferenceHodgeTest : public testing::TestWithParam<size_t>
{
public:
    FiniteDifferenceHodgeTest() :
        m_fdDeRhamComplex{GaussLegendreQuadrature{5}, DiscreteGrid{}, amrex::BoxArray{},
                          amrex::DistributionMapping{}}
    {
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.1, 1.2, 1.3)};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(1, 1, 1)};
        amrex::Vector<int> const nCell{AMREX_D_DECL(10, 11, 12)};
        amrex::Vector<int> const maxGridSize{AMREX_D_DECL(10, 11, 12)};
        amrex::Vector<int> const isPeriodic{AMREX_D_DECL(0, 0, 0)};

        m_params.set("ComputationalDomain.domainLo", domainLo);
        m_params.set("ComputationalDomain.domainHi", domainHi);
        m_params.set("ComputationalDomain.nCell", nCell);
        m_params.set("ComputationalDomain.maxGridSize", maxGridSize);
        m_params.set("ComputationalDomain.isPeriodic", isPeriodic);
        m_fdDeRhamComplex = FiniteDifferenceDeRhamSpaces{m_gaussPoints};
    }
    Io::Parameters m_params{};
    int m_gaussPoints{10};
    size_t m_hodgeDegree{GetParam()};
    FiniteDifferenceDeRhamSpaces m_fdDeRhamComplex;
};
INSTANTIATE_TEST_SUITE_P(
    HodgeDegrees,
    FiniteDifferenceHodgeTest,
    testing::Values(2, 4, 6),
    [] (testing::TestParamInfo<FiniteDifferenceHodgeTest::ParamType> const& info)
    { return "Degree" + std::to_string(info.param); });

void finite_difference_de_rham_complex_hodge_primal_to_dual_scalar (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex, size_t hodgeDegree)
{
    Gempic::Impl::BoundaryConditionConfiguration bcConfig{};
    bcConfig.m_extrapolationDegree = hodgeDegree - 1;
    bcConfig.m_bcRec = amrex::BCRec{AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap),
                                    AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap)};
    auto polynom =
        [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        size_t degree{hodgeDegree};
        return GEMPIC_D_ADD(std::pow(x, degree - 1), std::pow(y, degree - 1),
                            std::pow(z, degree - 1)) +
               1;
    };

    PrimalZeroForm pzf{fdDeRhamComplex.create_primal_zero_form("", bcConfig)};
    DualThreeForm hpzf{fdDeRhamComplex.create_dual_three_form("", bcConfig)};
    DualThreeForm dtf{fdDeRhamComplex.create_dual_three_form("", bcConfig)};
    project(pzf, polynom);
    Gempic::Impl::finite_difference_hodge(hpzf, pzf, hodgeDegree);
    project(dtf, polynom);
    EXPECT_LE(l_inf_error(hpzf, dtf), 1.0e-10);

    PrimalThreeForm ptf{fdDeRhamComplex.create_primal_three_form("", bcConfig)};
    DualZeroForm hptf{fdDeRhamComplex.create_dual_zero_form("", bcConfig)};
    DualZeroForm dzf{fdDeRhamComplex.create_dual_zero_form("", bcConfig)};
    project(ptf, polynom);
    Gempic::Impl::finite_difference_hodge(hptf, ptf, hodgeDegree);
    project(dzf, polynom);
    EXPECT_LE(l_inf_error(hptf, dzf), 1.0e-10);
}

TEST_P(FiniteDifferenceHodgeTest, HodgePrimalToDualScalar)
{
    finite_difference_de_rham_complex_hodge_primal_to_dual_scalar(m_fdDeRhamComplex, m_hodgeDegree);
}

void finite_difference_de_rham_complex_hodge_primal_to_dual_vector (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex, size_t hodgeDegree)
{
    Gempic::Impl::BoundaryConditionConfiguration bcConfig{};
    bcConfig.m_extrapolationDegree = hodgeDegree - 1;
    bcConfig.m_bcRec = amrex::BCRec{AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap),
                                    AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap)};
    auto polynom = [=] AMREX_GPU_HOST_DEVICE(
                       Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        size_t degree{hodgeDegree};
        return GEMPIC_D_ADD(std::pow(x, degree - 1), std::pow(y, degree - 1),
                            std::pow(z, degree - 1)) +
               1 + static_cast<double>(dir);
    };

    PrimalOneForm pzf{fdDeRhamComplex.create_primal_one_form("", {bcConfig, bcConfig, bcConfig})};
    DualTwoForm hpzf{fdDeRhamComplex.create_dual_two_form("", {bcConfig, bcConfig, bcConfig})};
    DualTwoForm dtf{fdDeRhamComplex.create_dual_two_form("", {bcConfig, bcConfig, bcConfig})};
    project(pzf, polynom);
    Gempic::Impl::finite_difference_hodge(hpzf, pzf, hodgeDegree);
    project(dtf, polynom);
    EXPECT_LE(l_inf_error(hpzf, dtf)[Direction::xDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hpzf, dtf)[Direction::yDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hpzf, dtf)[Direction::zDir], 1.0e-10);

    PrimalTwoForm ptf{fdDeRhamComplex.create_primal_two_form("", {bcConfig, bcConfig, bcConfig})};
    DualOneForm hptf{fdDeRhamComplex.create_dual_one_form("", {bcConfig, bcConfig, bcConfig})};
    DualOneForm dzf{fdDeRhamComplex.create_dual_one_form("", {bcConfig, bcConfig, bcConfig})};
    project(ptf, polynom);
    Gempic::Impl::finite_difference_hodge(hptf, ptf, hodgeDegree);
    project(dzf, polynom);
    EXPECT_LE(l_inf_error(hptf, dzf)[Direction::xDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hptf, dzf)[Direction::yDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hptf, dzf)[Direction::zDir], 1.0e-10);
}

TEST_P(FiniteDifferenceHodgeTest, HodgePrimalToDualVector)
{
    finite_difference_de_rham_complex_hodge_primal_to_dual_vector(m_fdDeRhamComplex, m_hodgeDegree);
}

void finite_difference_de_rham_complex_hodge_dual_to_primal_scalar (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex, size_t hodgeDegree)
{
    Gempic::Impl::BoundaryConditionConfiguration bcConfig{};
    bcConfig.m_extrapolationDegree = hodgeDegree - 1;
    bcConfig.m_bcRec = amrex::BCRec{AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap),
                                    AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap)};
    auto polynom =
        [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        size_t degree{hodgeDegree};
        return GEMPIC_D_ADD(std::pow(x, degree - 1), std::pow(y, degree - 1),
                            std::pow(z, degree - 1)) +
               1;
    };

    DualThreeForm dtf{fdDeRhamComplex.create_dual_three_form("", bcConfig)};
    PrimalZeroForm hdtf{fdDeRhamComplex.create_primal_zero_form("", bcConfig)};
    PrimalZeroForm pzf{fdDeRhamComplex.create_primal_zero_form("", bcConfig)};
    project(dtf, polynom);
    Gempic::Impl::finite_difference_hodge(hdtf, dtf, hodgeDegree);
    project(pzf, polynom);
    EXPECT_LE(l_inf_error(hdtf, pzf), 1.0e-10);

    DualZeroForm dzf{fdDeRhamComplex.create_dual_zero_form("", bcConfig)};
    PrimalThreeForm hdzf{fdDeRhamComplex.create_primal_three_form("", bcConfig)};
    PrimalThreeForm ptf{fdDeRhamComplex.create_primal_three_form("", bcConfig)};
    project(dzf, polynom);
    Gempic::Impl::finite_difference_hodge(hdzf, dzf, hodgeDegree);
    project(ptf, polynom);
    EXPECT_LE(l_inf_error(hdzf, ptf), 1.0e-10);
}

TEST_P(FiniteDifferenceHodgeTest, HodgeDualToPrimalScalar)
{
    finite_difference_de_rham_complex_hodge_dual_to_primal_scalar(m_fdDeRhamComplex, m_hodgeDegree);
}

void finite_difference_de_rham_complex_hodge_dual_to_primal_vector (
    Gempic::FiniteDifferenceDeRhamSpaces& fdDeRhamComplex, size_t hodgeDegree)
{
    Gempic::Impl::BoundaryConditionConfiguration bcConfig{};
    bcConfig.m_extrapolationDegree = hodgeDegree - 1;
    bcConfig.m_bcRec = amrex::BCRec{AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap),
                                    AMREX_D_DECL(BCType::extrap, BCType::extrap, BCType::extrap)};
    auto polynom = [=] AMREX_GPU_HOST_DEVICE(
                       Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        size_t degree{hodgeDegree};
        return GEMPIC_D_ADD(std::pow(x, degree - 1), std::pow(y, degree - 1),
                            std::pow(z, degree - 1)) +
               1 + static_cast<double>(dir);
    };

    DualOneForm dof{fdDeRhamComplex.create_dual_one_form("", {bcConfig, bcConfig, bcConfig})};
    PrimalTwoForm hdof{fdDeRhamComplex.create_primal_two_form("", {bcConfig, bcConfig, bcConfig})};
    PrimalTwoForm ptf{fdDeRhamComplex.create_primal_two_form("", {bcConfig, bcConfig, bcConfig})};
    project(dof, polynom);
    Gempic::Impl::finite_difference_hodge(hdof, dof, hodgeDegree);
    project(ptf, polynom);
    EXPECT_LE(l_inf_error(hdof, ptf)[Direction::xDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hdof, ptf)[Direction::yDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hdof, ptf)[Direction::zDir], 1.0e-10);

    DualTwoForm dtf{fdDeRhamComplex.create_dual_two_form("", {bcConfig, bcConfig, bcConfig})};
    PrimalOneForm hdtf{fdDeRhamComplex.create_primal_one_form("", {bcConfig, bcConfig, bcConfig})};
    PrimalOneForm pof{fdDeRhamComplex.create_primal_one_form("", {bcConfig, bcConfig, bcConfig})};
    project(dtf, polynom);
    Gempic::Impl::finite_difference_hodge(hdtf, dtf, hodgeDegree);
    project(pof, polynom);
    EXPECT_LE(l_inf_error(hdtf, pof)[Direction::xDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hdtf, pof)[Direction::yDir], 1.0e-10);
    EXPECT_LE(l_inf_error(hdtf, pof)[Direction::zDir], 1.0e-10);
}

TEST_P(FiniteDifferenceHodgeTest, HodgeDualToPrimalVector)
{
    finite_difference_de_rham_complex_hodge_dual_to_primal_vector(m_fdDeRhamComplex, m_hodgeDegree);
}

namespace
{
class FDDeRhamComplexTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;

    FDDeRhamComplexTest() : m_infra{Gempic::Test::Utils::get_default_compdom()}
    {
        // Not checking particles
        int const nGhostExtra{0};

        m_parameters.set("nGhostExtra", nGhostExtra);
    }
};

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg2)
{
    constexpr int hodgeDegree{2};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg4)
{
    constexpr int hodgeDegree{4};
    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg6)
{
    constexpr int hodgeDegree{6};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTest)
{
    constexpr int hodgeDegree{2};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    ASSERT_EQ(0, Utils::gempic_norm(rho.m_data, m_infra, 2));

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entires to be 0
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, 0);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestII)
{
    constexpr int hodgeDegree{2};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entries to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestIII)
{
    constexpr int hodgeDegree{2};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham, func);

    ASSERT_EQ(0, Utils::gempic_norm(rho.m_data, m_infra, 2));

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entires to be 0
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, 0);
    }
    ASSERT_TRUE(loopRun);
}
} // namespace

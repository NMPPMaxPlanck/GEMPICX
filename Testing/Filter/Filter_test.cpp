/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <vector>

#include <gtest/gtest.h>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;

class NoFilterTest : public ::testing::Test
{
public:
    Io::Parameters m_parameters{};
    static constexpr int s_nCell{30};

    NoFilterTest()
    {
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.0, 2.0, 3.0)};
        amrex::Vector<int> nCell{AMREX_D_DECL(s_nCell, s_nCell, s_nCell)};
        amrex::Vector<int> maxGridSize{AMREX_D_DECL(s_nCell, s_nCell, s_nCell)};
        amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        m_parameters.set("ComputationalDomain.nCell", nCell);
        m_parameters.set("ComputationalDomain.maxGridSize", maxGridSize);
        m_parameters.set("ComputationalDomain.domainLo", domainLo);
        m_parameters.set("ComputationalDomain.domainHi", domainHi);
        m_parameters.set("ComputationalDomain.isPeriodic", isPeriodic);
    }

protected:
    using Dof = DiscreteField::DOFCategory;

    DiscreteField make_scalar_field (std::string const& label)
    {
        return DiscreteField{label,
                             m_parameters,
                             DiscreteGrid{m_parameters,
                                          {AMREX_D_DECL(DiscreteAxis::Cell, DiscreteAxis::Cell,
                                                        DiscreteAxis::Cell)}},
                             {AMREX_D_DECL(Dof::PointValue, Dof::PointValue, Dof::PointValue)}};
    }
};

enum ModeType
{
    MinMode,
    GridMode
};

enum class Compensation
{
    True,
    False
};

void fill_mode (DiscreteField& f, ModeType mode)
{
    std::array<amrex::Real, AMREX_SPACEDIM> k0{};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        k0[dir] = 2 * M_PI / f.discrete_grid().length(dir);
        if (ModeType::GridMode == mode)
        {
            k0[dir] *= f.discrete_grid().n_cells(dir) / 2;
        }
    }
    auto func = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        return GEMPIC_D_MULT(std::sin(k0[Direction::xDir] * x), std::sin(k0[Direction::yDir] * y),
                             std::sin(k0[Direction::zDir] * z));
    };
    fill(f, func);
}

TEST_F(NoFilterTest, Copy)
{
    m_parameters.set("Filter.type", "NoFilter");
    Filter::NoFilter noFilter{};
    auto src = make_scalar_field("src");
    auto dst = make_scalar_field("dst");
    auto ref = make_scalar_field("ref");

    fill_mode(src, ModeType::MinMode);
    fill_zero(dst);
    fill_mode(ref, ModeType::MinMode);
    noFilter(dst, src);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);
}

class BilinearFilterTest : public ::testing::TestWithParam<int>
{
public:
    Io::Parameters m_parameters{};
    static constexpr int s_nCell{30};

    BilinearFilterTest()
    {
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.0, 2.0, 3.0)};
        amrex::Vector<int> nCell{AMREX_D_DECL(s_nCell, s_nCell, s_nCell)};
        amrex::Vector<int> maxGridSize{AMREX_D_DECL(s_nCell, s_nCell, s_nCell)};
        amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        m_parameters.set("ComputationalDomain.nCell", nCell);
        m_parameters.set("ComputationalDomain.maxGridSize", maxGridSize);
        m_parameters.set("ComputationalDomain.domainLo", domainLo);
        m_parameters.set("ComputationalDomain.domainHi", domainHi);
        m_parameters.set("ComputationalDomain.isPeriodic", isPeriodic);
    }

protected:
    using Dof = DiscreteField::DOFCategory;

    DiscreteField make_scalar_field (std::string const& label)
    {
        return DiscreteField{label,
                             m_parameters,
                             DiscreteGrid{m_parameters,
                                          {AMREX_D_DECL(DiscreteAxis::Cell, DiscreteAxis::Cell,
                                                        DiscreteAxis::Cell)}},
                             {AMREX_D_DECL(Dof::PointValue, Dof::PointValue, Dof::PointValue)}};
    }
};

void reference (DiscreteField& f,
                ModeType mode,
                std::array<int, AMREX_SPACEDIM> nPass,
                Compensation comp = Compensation::False)
{
    std::array<amrex::Real, AMREX_SPACEDIM> k0{};
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        k0[dir] = 2 * M_PI / f.discrete_grid().length(dir);
        if (ModeType::GridMode == mode)
        {
            k0[dir] *= f.discrete_grid().n_cells(dir) / 2;
        }
    }
    std::array<amrex::Real, AMREX_SPACEDIM> dx{f.discrete_grid().dx()};
    auto func = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        amrex::Real base{1.0};
        base *= GEMPIC_D_MULT(std::sin(k0[Direction::xDir] * x), std::sin(k0[Direction::yDir] * y),
                              std::sin(k0[Direction::zDir] * z));
        for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
        {
            for (int i = 0; i < nPass[dir]; i++)
            {
                base *= 0.5 * (1 + std::cos(k0[dir] * dx[dir]));
            }
            if (comp == Compensation::True)
            {
                amrex::Real alphaComp{nPass[dir] / 2.0 + 1.0};
                base *= alphaComp + (1 - alphaComp) * std::cos(k0[dir] * dx[dir]);
            }
        }
        return base;
    };
    fill(f, func);
}

TEST_P(BilinearFilterTest, AnalyticalComparison)
{
    int const passes{GetParam()};
    std::array<int, AMREX_SPACEDIM> nPass{AMREX_D_DECL(passes, passes, passes)};
    m_parameters.set("Filter.Bilinear.nPass", nPass);
    Filter::BilinearFilter bilinearFilter{m_parameters};
    auto low = make_scalar_field("low");
    auto high = make_scalar_field("high");
    auto dst = make_scalar_field("dst");
    auto ref = make_scalar_field("ref");

    fill_mode(low, ModeType::MinMode);
    fill_zero(dst);
    reference(ref, ModeType::MinMode, nPass);
    bilinearFilter(dst, low);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);

    fill_mode(high, ModeType::GridMode);
    fill_zero(ref);
    bilinearFilter(dst, high);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);
}

TEST_P(BilinearFilterTest, AnalyticalComparisonWithCompensation)
{
    int const passes{GetParam()};
    std::array<int, AMREX_SPACEDIM> nPass{AMREX_D_DECL(passes, passes, passes)};
    m_parameters.set("Filter.Bilinear.nPass", nPass);
    m_parameters.set("Filter.Bilinear.compensate", true);
    Filter::BilinearFilter bilinearFilter{m_parameters};
    auto low = make_scalar_field("low");
    auto high = make_scalar_field("high");
    auto dst = make_scalar_field("dst");
    auto ref = make_scalar_field("ref");

    fill_mode(low, ModeType::MinMode);
    fill_zero(dst);
    reference(ref, ModeType::MinMode, nPass, Compensation::True);
    bilinearFilter(dst, low);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);

    fill_mode(high, ModeType::GridMode);
    fill_zero(ref);
    bilinearFilter(dst, high);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);
}

INSTANTIATE_TEST_SUITE_P(FilterPasses, BilinearFilterTest, testing::Range(1, 5));

#ifdef AMREX_USE_FFT
class FourierFilterTest : public ::testing::Test
{
public:
    Io::Parameters m_parameters{};
    static constexpr int s_nCell{30};

    FourierFilterTest()
    {
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(1.0, 2.0, 3.0)};
        amrex::Vector<int> nCell{AMREX_D_DECL(s_nCell, s_nCell, s_nCell)};
        amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
        amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        m_parameters.set("ComputationalDomain.nCell", nCell);
        m_parameters.set("ComputationalDomain.maxGridSize", maxGridSize);
        m_parameters.set("ComputationalDomain.domainLo", domainLo);
        m_parameters.set("ComputationalDomain.domainHi", domainHi);
        m_parameters.set("ComputationalDomain.isPeriodic", isPeriodic);
    }

protected:
    using Dof = DiscreteField::DOFCategory;

    DiscreteField make_scalar_field (std::string const& label)
    {
        return DiscreteField{label,
                             m_parameters,
                             DiscreteGrid{m_parameters,
                                          {AMREX_D_DECL(DiscreteAxis::Node, DiscreteAxis::Cell,
                                                        DiscreteAxis::Node)}},
                             {AMREX_D_DECL(Dof::PointValue, Dof::PointValue, Dof::PointValue)}};
    }
};

enum class Initialization
{
    NoFilter,
    Filtered
};

void fill (DiscreteField& f, Initialization init)
{
    // wavenumber 2: outside [3, 7], removed
    // wavenumber 5: inside [3, 7], kept
    // wavenumber 10: outside [3, 7], removed
    std::array<amrex::Real, AMREX_SPACEDIM> kKept{}, highMode{}, lowMode{}, nyquistMode;
    for (auto dir : {AMREX_D_DECL(Direction::xDir, Direction::yDir, Direction::zDir)})
    {
        lowMode[dir] = 2 * M_PI * 2 / f.discrete_grid().length(dir);
        kKept[dir] = 2 * M_PI * 5 / f.discrete_grid().length(dir);
        highMode[dir] = 2 * M_PI * 10 / f.discrete_grid().length(dir);
        nyquistMode[dir] =
            2 * M_PI * f.discrete_grid().n_cells(dir) / 2.0 / f.discrete_grid().length(dir);
    }

    fill(f,
         [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
         {
             amrex::Real f{};
             f += GEMPIC_D_MULT(std::sin(kKept[Direction::xDir] * x),
                                std::sin(kKept[Direction::yDir] * y),
                                std::sin(kKept[Direction::zDir] * z));
             if (Initialization::NoFilter == init)
             {
                 f += 1;
                 f += GEMPIC_D_MULT(std::sin(lowMode[Direction::xDir] * x),
                                    std::sin(lowMode[Direction::yDir] * y),
                                    std::sin(lowMode[Direction::zDir] * z));
                 f += GEMPIC_D_MULT(std::sin(highMode[Direction::xDir] * x),
                                    std::sin(highMode[Direction::yDir] * y),
                                    std::sin(highMode[Direction::zDir] * z));
                 f += GEMPIC_D_MULT(std::sin(nyquistMode[Direction::xDir] * x),
                                    std::sin(nyquistMode[Direction::yDir] * y),
                                    std::sin(nyquistMode[Direction::zDir] * z));
             }
             return f;
         });
}

TEST_F(FourierFilterTest, RemoveModesOutOfRange)
{
    amrex::Vector<int> nMinVec{AMREX_D_DECL(3, 3, 3)};
    amrex::Vector<int> nMaxVec{AMREX_D_DECL(7, 7, 7)};
    m_parameters.set("Filter.Fourier.nMin", nMinVec);
    m_parameters.set("Filter.Fourier.nMax", nMaxVec);

    auto src = make_scalar_field("src");
    auto dst = make_scalar_field("dst");
    auto ref = make_scalar_field("ref");

    Filter::FourierFilter fourierFilter{m_parameters, dst.discrete_grid()};

    fill(src, Initialization::NoFilter);
    fill(ref, Initialization::Filtered);

    fourierFilter(dst, src);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);
}

TEST_F(FourierFilterTest, Identity)
{
    auto src = make_scalar_field("src");
    auto dst = make_scalar_field("dst");
    auto ref = make_scalar_field("ref");

    amrex::Vector<int> nMinVec{AMREX_D_DECL(0, 0, 0)};
    amrex::Vector<int> nMaxVec{AMREX_D_DECL(dst.discrete_grid().size(Direction::xDir) / 2,
                                            dst.discrete_grid().size(Direction::yDir) / 2,
                                            dst.discrete_grid().size(Direction::zDir) / 2)};
    m_parameters.set("Filter.Fourier.nMin", nMinVec);
    m_parameters.set("Filter.Fourier.nMax", nMaxVec);

    Filter::FourierFilter fourierFilter{m_parameters, dst.discrete_grid()};

    fill(src, Initialization::NoFilter);
    fill(ref, Initialization::NoFilter);

    fourierFilter(dst, src);
    EXPECT_LT(l_inf_error(ref, dst), 1.0e-13);
}

#endif
} // namespace

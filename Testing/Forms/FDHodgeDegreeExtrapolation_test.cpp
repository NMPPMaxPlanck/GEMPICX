/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
/*------------------------------------------------------------------------------
 Test Finite Difference Hodge convergence rates with non periodic boundaries (extrapolation)

  Computes the max norm ||I_{3-k} H_k R_k f - 1/omega f|| at the cell mid points
  where f(x,y,z) =  cos(2*pi*(x-0.2))*sin(4*pi*(y-0.6))*cos(2*pi*z) + sin(2*pi*x -
0.4)*sin(2*pi*(y-0.1))*sin(4*pi*z + 0.3) for k = 0,3

  or                cos(2*pi*(x-0.2))*sin(4*pi*(y-0.6))*cos(2*pi*z) + sin(2*pi*x -
0.4)*sin(2*pi*(y-0.1))*sin(4*pi*z + 0.3) f(x,y,z) = (sin(2*pi*(x+0.3))*cos(2*pi*(y-0.6))*cos(4*pi*z)
+ cos(2*pi*x - 0.2)*sin(2*pi*(y+0.6))*sin(2*pi*z - 0.4)) for k = 1,2
                    cos(4*pi*(x-0.6))*cos(2*pi*(y+0.3))*sin(2*pi*z) + sin(2*pi*x +
0.3)*sin(2*pi*(y-0.3))*cos(2*pi*z + 0.6)

  for 16 and 32 nodes (1D and 2D: 64/128) in each direction. The convergence rate is estimated
by log_2 (error_coarse / error_fine)
------------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic;
using namespace Forms;

namespace
{
/* Initialize the infrastructure */
ComputationalDomain get_compdom (amrex::IntVect const& nCell)
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(1.3, 1.6, 1.4)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(8, 6, 9)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(0, 0, 0)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

template <int hodgeDegree, Space space>
double hodge_one_two_error (int const n, int maxSplineDegree = 3)
{
    // Initialize computational_domain
    amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
    Gempic::Io::Parameters parameters{};
    ComputationalDomain infra = get_compdom(nCell);

    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

#if (AMREX_SPACEDIM == 1)
    amrex::Array<std::string, 3> const func = {
        "pi = 3.141592653589793; (cos(2*pi*x) + sin(2*pi*x - 0.2))",
        "pi = 3.141592653589793; (sin(2*pi*x) + cos(2*pi*x - 0.2))",
        "pi = 3.141592653589793; (cos(2*2*pi*x) + sin(2*pi*x - 0.2))",
    };
#endif
#if (AMREX_SPACEDIM == 2)
    amrex::Array<std::string, 3> const func = {
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; (sin(2*pi*x)*cos(2*pi*y) + cos(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; (cos(2*2*pi*x)*cos(2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
    };
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::Array<std::string, 3> const func = {
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; (sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z) + cos(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; (cos(2*2*pi*x)*cos(2*pi*y)*sin(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*cos(2*2*pi*z + 1.3))",
    };
#endif

    // project f to a primalOneForm or a primalTwoForm
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3> funcP;
    amrex::Array<amrex::Parser, 3> parser;
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<AMREX_SPACEDIM + 1>();
    }

    double e = 0;

    DeRhamField<Grid::primal, space> primalForm(deRham);
    DeRhamField<Grid::dual, oppositeSpace<space>> dualForm(deRham);

    projection(funcP, 0.0, primalForm);
    primalForm.apply_bc();
    hodge(dualForm, primalForm);
    dualForm.apply_bc();
    hodge(primalForm, dualForm);
    primalForm.apply_bc();

    auto dr{infra.cell_size_array()};
    auto form{static_cast<int>(space)};
    for (int comp = 0; comp < 3; ++comp)
    {
        e += max_error_midpoint<hodgeDegree>(infra.m_geom, funcP[comp], primalForm.m_data[comp], dr,
                                             form, false, comp);
    }
    return e;
}

template <int hodgeDegree, Space space>
double hodge_zero_three_error (int const n, int maxSplineDegree = 3)
{
    // Initialize computational_domain
    amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
    Gempic::Io::Parameters parameters{};
    ComputationalDomain infra = get_compdom(nCell);

    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

#if (AMREX_SPACEDIM == 1)
    std::string const func = "pi = 3.141592653589793; (cos(2*pi*x) + sin(2*pi*x - 0.2))";
#endif
#if (AMREX_SPACEDIM == 2)
    std::string const func =
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))";
#endif
#if (AMREX_SPACEDIM == 3)
    std::string const func =
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))";
#endif

    // project f to a primalZeroForm or a primalThreeForm
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> funcP;
    amrex::Parser parser;

    parser.define(func);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<AMREX_SPACEDIM + 1>();

    DeRhamField<Grid::primal, space> primalForm(deRham);
    DeRhamField<Grid::dual, oppositeSpace<space>> dualForm(deRham);

    projection(funcP, 0.0, primalForm);
    primalForm.apply_bc();
    hodge(dualForm, primalForm);
    dualForm.apply_bc();
    hodge(primalForm, dualForm);
    primalForm.apply_bc();

    auto dr{infra.cell_size_array()};
    auto form{static_cast<int>(space)};
    return max_error_midpoint<hodgeDegree>(infra.m_geom, funcP, primalForm.m_data, dr, form, false);
}

/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
template <typename DegreeSpace>
class FDHodgeDegreeExtrapolationTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};

    inline static int const s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};
    static constexpr int s_hodgeDegree{DegreeSpace::s_hodgeDegree};
    static constexpr Space s_space{DegreeSpace::s_space};
};

class NameGenerator
{
public:
    template <typename T>
    static std::string GetName (int)
    {
        if constexpr (T::s_space == Space::node)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " node";
        }
        if constexpr (T::s_space == Space::edge)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " edge";
        }
        if constexpr (T::s_space == Space::face)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " face";
        }
        if constexpr (T::s_space == Space::cell)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " cell";
        }
    }
};

template <int degree, Space space>
struct DegreeSpace
{
    static constexpr int s_hodgeDegree{degree};
    static constexpr Space s_space{space};
};

using ZeroThreeForms = ::testing::Types<DegreeSpace<4, Space::node>,
                                        DegreeSpace<4, Space::cell>,
                                        DegreeSpace<6, Space::node>,
                                        DegreeSpace<6, Space::cell>>;
TYPED_TEST_SUITE(FDHodgeDegreeExtrapolationTest, ZeroThreeForms, NameGenerator);

// Sillily we have to define a new class for a new typed test suite
template <typename DegreeSpace>
class FDHodgeDegreeExtrapolationTest2 : public FDHodgeDegreeExtrapolationTest<DegreeSpace>
{
};

using OneTwoForms = ::testing::Types<DegreeSpace<4, Space::edge>,
                                     DegreeSpace<4, Space::face>,
                                     DegreeSpace<6, Space::edge>,
                                     DegreeSpace<6, Space::face>>;
TYPED_TEST_SUITE(FDHodgeDegreeExtrapolationTest2, OneTwoForms, NameGenerator);

TYPED_TEST(FDHodgeDegreeExtrapolationTest, ZeroThreeFormTest)
{
    int coarse = 64, fine = 128;
    if (AMREX_SPACEDIM == 3)
    {
        coarse /= 4;
        fine /= 4;
    }

    amrex::Real tol = 0.5;

    constexpr int hodgeDegree{TestFixture::s_hodgeDegree};
    constexpr Space space{TestFixture::s_space};
    amrex::Real errorCoarse{hodge_zero_three_error<hodgeDegree, space>(coarse)};
    amrex::Real errorFine{hodge_zero_three_error<hodgeDegree, space>(fine)};

    amrex::Real rateOfConvergence{std::log2(errorCoarse / errorFine)};

    EXPECT_NEAR(rateOfConvergence, hodgeDegree, tol);
}

TYPED_TEST(FDHodgeDegreeExtrapolationTest2, OneTwoFormTest)
{
    int coarse = 64, fine = 128;
    if (AMREX_SPACEDIM == 3)
    {
        coarse /= 4;
        fine /= 4;
    }

    amrex::Real tol = 0.5;

    constexpr int hodgeDegree{TestFixture::s_hodgeDegree};
    constexpr Space space{TestFixture::s_space};
    amrex::Real errorCoarse{hodge_one_two_error<hodgeDegree, space>(coarse)};
    amrex::Real errorFine{hodge_one_two_error<hodgeDegree, space>(fine)};

    amrex::Real rateOfConvergence{std::log2(errorCoarse / errorFine)};

    EXPECT_NEAR(rateOfConvergence, hodgeDegree, tol);
}
} // namespace

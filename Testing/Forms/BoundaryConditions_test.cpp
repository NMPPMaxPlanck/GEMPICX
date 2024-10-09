#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)

using namespace Gempic;
using namespace Forms;

namespace
{

using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
bool func_cond_x (AMREX_D_DECL(int i, int j, int k)) { return (i == 0 || i == 3); }
bool func_cond_y (AMREX_D_DECL(int i, int j, int k))
{
#if GEMPIC_SPACEDIM >= 2
    return (j == 0 || j == 3);
#else
    return false;
#endif
}
bool func_cond_z (AMREX_D_DECL(int i, int j, int k))
{
#if GEMPIC_SPACEDIM == 3
    return (k == 0 || k == 3);
#else
    return false;
#endif
}
condLambda condX = func_cond_x;
condLambda condY = func_cond_y;
condLambda condZ = func_cond_z;

class BoundaryConditionTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra{false};  // "uninitialized" computational domain

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        // Cells of size 1x1x1
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(3.0, 3.0, 3.0)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(3, 3, 3)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(0, 0, 0)};
        const amrex::Vector<std::string> bcVec(6, "PerfectlyConducting");
        // Not checking particles
        const int nGhostExtra{0};

        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("domainHi", domainHi);
        pp.addarr("nCellVector", nCell);
        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);
        pp.add("nGhostExtra", nGhostExtra);
        pp.addarr("BoundaryCondition.Default", bcVec);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override { m_infra = ComputationalDomain{}; }
};

TEST_F(BoundaryConditionTest, PerfectlyConductingPrimal)
{
    constexpr int hodgeDegree{2};

    const std::string oneFuncScalar = "1.0";
    const amrex::Array<std::string, 3> oneFuncVector = {
        "1.0",
        "1.0",
        "1.0",
    };

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t

    amrex::ParserExecutor<nVar> funcScalar;
    amrex::Parser parserScalar;
    parserScalar.define(oneFuncScalar);
    parserScalar.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcScalar = parserScalar.compile<nVar>();

    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcVector;
    amrex::Array<amrex::Parser, 3> parserVector;
    for (int i = 0; i < 3; ++i)
    {
        parserVector[i].define(oneFuncVector[i]);
        parserVector[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcVector[i] = parserVector[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> zeroForm(deRham, funcScalar);
    DeRhamField<Grid::primal, Space::edge> oneForm(deRham, funcVector);
    DeRhamField<Grid::primal, Space::face> twoForm(deRham, funcVector);
    DeRhamField<Grid::primal, Space::cell> threeForm(deRham, funcScalar);

    zeroForm.apply_bc();
    oneForm.apply_bc();
    twoForm.apply_bc();
    threeForm.apply_bc();

    // Zero form -> expect all boundary nodes to be zero
    bool loopRun{false};
    for (amrex::MFIter mfi(zeroForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((zeroForm.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {condX, condY, condZ},
                    {0, 0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);

    // One form -> expect edges tangential to boundary to be zero
    loopRun = false;
    for (amrex::MFIter mfi(oneForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((oneForm.m_data[xDir][mfi]).array(), m_infra.m_nCell.dim3(), {condY, condZ},
                    {0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(oneForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((oneForm.m_data[yDir][mfi]).array(), m_infra.m_nCell.dim3(), {condX, condZ},
                    {0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(oneForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((oneForm.m_data[zDir][mfi]).array(), m_infra.m_nCell.dim3(), {condX, condY},
                    {0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);

    // Two form -> expect fluxes normal to boundary to be zero
    loopRun = false;
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((twoForm.m_data[xDir][mfi]).array(), m_infra.m_nCell.dim3(), {condX}, {0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((twoForm.m_data[yDir][mfi]).array(), m_infra.m_nCell.dim3(), {condY}, {0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((twoForm.m_data[zDir][mfi]).array(), m_infra.m_nCell.dim3(), {condZ}, {0}, 1);
    }
    ASSERT_TRUE(loopRun);

    // Three form -> expect no change
    loopRun = false;
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((threeForm.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(BoundaryConditionTest, PerfectlyConductingDual)
{
    constexpr int hodgeDegree{2};

    const std::string oneFuncScalar = "1.0";
    const amrex::Array<std::string, 3> oneFuncVector = {
        "1.0",
        "1.0",
        "1.0",
    };

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t

    amrex::ParserExecutor<nVar> funcScalar;
    amrex::Parser parserScalar;
    parserScalar.define(oneFuncScalar);
    parserScalar.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcScalar = parserScalar.compile<nVar>();

    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcVector;
    amrex::Array<amrex::Parser, 3> parserVector;
    for (int i = 0; i < 3; ++i)
    {
        parserVector[i].define(oneFuncVector[i]);
        parserVector[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcVector[i] = parserVector[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::node> zeroForm(deRham, funcScalar);
    DeRhamField<Grid::dual, Space::edge> oneForm(deRham, funcVector);
    DeRhamField<Grid::dual, Space::face> twoForm(deRham, funcVector);
    DeRhamField<Grid::dual, Space::cell> threeForm(deRham, funcScalar);

    zeroForm.apply_bc();
    oneForm.apply_bc();
    twoForm.apply_bc();
    threeForm.apply_bc();

    // All dual forms should not change, i.e., ones for all dofs
    // Zero form
    bool loopRun{false};
    for (amrex::MFIter mfi(zeroForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((zeroForm.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);

    // One and two forms
    for (int dir = 0; dir < GEMPIC_SPACEDIM; ++dir)
    {
        loopRun = false;
        for (amrex::MFIter mfi(oneForm.m_data[dir]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            check_field((oneForm.m_data[dir][mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
        loopRun = false;
        for (amrex::MFIter mfi(twoForm.m_data[dir]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            check_field((twoForm.m_data[dir][mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
    }

    // Three form
    loopRun = false;
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        check_field((threeForm.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}
}  // namespace

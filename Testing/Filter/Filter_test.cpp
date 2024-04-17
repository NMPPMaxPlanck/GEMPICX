#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "GEMPIC_BilinearFilter.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)
#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace Forms;

// Test fixture. Sets up clean environment before each test.
class BilinearFilterTest : public testing::Test
{
public:
    Io::Parameters m_parameters{};

    // Initialize computational_domain
    ComputationalDomain m_infra;

    int m_nComps{1};

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        int gSize{5};
        double k1D{2 * M_PI / gSize};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(k1D, k1D, k1D)};
        // int gSize{static_cast<int>(dSize)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(gSize, gSize, gSize)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(gSize, gSize, gSize)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("nCellVector", nCell);
        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);
    }
};

class BilinearFilterTestParameter : public BilinearFilterTest,
                                    public testing::WithParamInterface<int>
{
public:
    std::vector<int> m_filterNpass;
    amrex::IntVect m_nGhost;

    BilinearFilterTestParameter()
    {
        amrex::ParmParse pp;
        pp.add("Filter.enable", true);
        int filterPass{GetParam()};
        m_filterNpass = std::vector<int>{AMREX_D_DECL(filterPass, filterPass, filterPass)};
        pp.addarr("Filter.nPass", m_filterNpass);

        std::vector<int> nGhostVec{m_filterNpass};
        m_parameters.set("nGhost", nGhostVec);
        m_nGhost = amrex::IntVect{AMREX_D_DECL(nGhostVec[xDir], nGhostVec[yDir], nGhostVec[zDir])};
    }
};

TEST_P(BilinearFilterTestParameter, ConstantTest)
{
    amrex::MultiFab mf;
    mf.define(amrex::convert(m_infra.m_grid, amrex::IntVect{1}), m_infra.m_distriMap, m_nComps,
              m_nGhost);
    double constVal{5.0};
    mf.setVal(constVal);

    amrex::MultiFab mfTmp;
    mfTmp.define(amrex::convert(m_infra.m_grid, amrex::IntVect{1}), m_infra.m_distriMap, m_nComps,
                 m_nGhost);
    double tmpVal{0.5};
    mfTmp.setVal(tmpVal);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(mf, mfTmp, srcCompBegIn, dstCompBegIn, m_nComps);
    // probably fails due to filter expanding box (which then has 0s)

    for (amrex::MFIter mfi(mfTmp); mfi.isValid(); ++mfi)
    {
        // Expect all indices to be constVal
        check_field(mfTmp.array(mfi), m_infra.m_nCell.dim3(), {}, {}, constVal);
    }
}

INSTANTIATE_TEST_SUITE_P(FilterPasses, BilinearFilterTestParameter, testing::Range(0, 4));

TEST_F(BilinearFilterTest, NoFilter)
{
    amrex::ParmParse pp;
    pp.add("Filter.enable", false);

    amrex::IntVect nGhost{AMREX_D_DECL(0, 0, 0)};

    amrex::MultiFab mf;
    mf.define(amrex::convert(m_infra.m_grid, amrex::IntVect{1}), m_infra.m_distriMap, m_nComps,
              nGhost);
    double constVal{5.0};
    mf.setVal(constVal);

    amrex::MultiFab mfTmp;
    mfTmp.define(amrex::convert(m_infra.m_grid, amrex::IntVect{1}), m_infra.m_distriMap, m_nComps,
                 nGhost);
    double tmpVal{0.5};
    mfTmp.setVal(tmpVal);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(mf, mfTmp, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(mfTmp); mfi.isValid(); ++mfi)
    {
        // Expect the all indices to be constVal
        check_field(mfTmp.array(mfi), m_infra.m_nCell.dim3(), {}, {}, tmpVal);
    }
}

TEST_F(BilinearFilterTest, AnalyticalTest)
{
    // Parse analytical field and initialize parserEval.
    const std::string analyticalInit{"sin(kvarx*x)"};
    //  One pass bilinear filter is
    //  f(x) + 0.5*(sum_{n=1}^{infty} df^(2n)/d^(2n)x (x) *(dx)^(2n)/(2n)!)
    //  with dx = 1 and f(x) = sin(k*x), this is
    //  f(x) + 0.5*(cos(k) - 1)*f(x) = 0.5*f(x)(1 + cos(k))
    const std::string analyticalSol{"0.5*sin(kvarx*x)*(1+cos(kvarx))"};
    //  One pass bilinear filter is with compensation is
    //  (alpha+(1-alpha)cos(kvarx))*g(x)
    //  where g(x) is the analytical solution without filter
    const std::string analyticalSolComp{"(0.5 + 0.5*cos(kvarx))*" + analyticalSol};

    amrex::Vector<amrex::Real> k;
    m_parameters.get("k", k);
    const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t

    amrex::Parser parserInit;
    parserInit.define(analyticalInit);
    parserInit.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    parserInit.setConstant("kvarx", k[xDir]);
    auto funcInit = parserInit.compile<nVar>();

    amrex::Parser parserSol;
    parserSol.define(analyticalSol);
    parserSol.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    parserSol.setConstant("kvarx", k[xDir]);
    auto funcSol = parserSol.compile<nVar>();

    amrex::Parser parserSolComp;
    parserSolComp.define(analyticalSolComp);
    parserSolComp.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    parserSolComp.setConstant("kvarx", k[xDir]);
    auto funcSolComp = parserSolComp.compile<nVar>();

    constexpr int hodgeDegree{2};
    constexpr int maxSplineDegree{3};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, maxSplineDegree);
    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcInit);
    DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoSol(deRham, funcSol);
    DeRhamField<Grid::dual, Space::cell> rhoSolComp(deRham, funcSolComp);

    amrex::ParmParse pp;
    pp.add("Filter.enable", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 0, 0)};
    pp.addarr("Filter.nPass", filterNpass);

    // Test uncompensated filter
    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(rho.m_data, rhoTemp.m_data, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        compare_fields(rhoTemp.m_data.array(mfi), rhoSol.m_data.array(mfi), bx);
    }

    // Test compensated filter
    pp.add("Filter.compensate", true);

    std::unique_ptr<Filter::Filter> biFilterComp = std::make_unique<Filter::BilinearFilter>();
    biFilterComp->apply_stencil(rho.m_data, rhoTemp.m_data, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        compare_fields(rhoTemp.m_data.array(mfi), rhoSolComp.m_data.array(mfi), bx);
    }
}
}  // namespace

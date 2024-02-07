#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <BilinearFilter.H>

#include "test_utils/GEMPIC_test_utils.H"

#define checkField(...) GEMPIC_TestUtils::checkField(__FILE__, __LINE__, __VA_ARGS__)
#define compareFields(...) GEMPIC_TestUtils::compareFields(__FILE__, __LINE__, __VA_ARGS__)

namespace {
using namespace Gempic;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

// Test fixture. Sets up clean environment before each test.
class BilinearFilterTest : public testing::Test {
    public:
    Parameters parameters{};

    // Initialize computational_domain
    CompDom::computational_domain infra;

    int nComps{1};

    static void SetUpTestSuite()
    {
        /* Initialize the infrastructure */
        amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        int gSize{5};
        double k1D{2*M_PI/gSize};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(k1D, k1D, k1D)};
        //int gSize{static_cast<int>(dSize)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(gSize, gSize, gSize)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(gSize, gSize, gSize)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        amrex::ParmParse pp;
        pp.addarr("domain_lo", domain_lo);
        pp.addarr("k", k);
        pp.addarr("n_cell_vector", nCell);
        pp.addarr("max_grid_size_vector", maxGridSize);
        pp.addarr("is_periodic_vector", isPeriodic);
    }
};

class BilinearFilterTestParameter : public BilinearFilterTest, public testing::WithParamInterface<int> {
    public:
    std::vector<int> filterNpass;
    amrex::IntVect nGhost;

    BilinearFilterTestParameter () {
        amrex::ParmParse pp;
        pp.add("filter.use_filter", true);
        int filterPass{GetParam()};
        filterNpass = std::vector<int>{AMREX_D_DECL(filterPass, filterPass, filterPass)};
        pp.addarr("filter.filter_npass_each_dir", filterNpass);

        std::vector<int> nGhostVec{filterNpass};
        parameters.set("n_ghost", nGhostVec);
        nGhost = amrex::IntVect{AMREX_D_DECL(nGhostVec[xDir], nGhostVec[yDir], nGhostVec[zDir])};
    }
};

TEST_P(BilinearFilterTestParameter , ConstantTest) {
    amrex::MultiFab mf;
    mf.define(amrex::convert(infra.grid, amrex::IntVect{1}), infra.distriMap, nComps, nGhost);
    double constVal{5.0};
    mf.setVal(constVal);

    amrex::MultiFab mfTmp;
    mfTmp.define(amrex::convert(infra.grid, amrex::IntVect{1}), infra.distriMap, nComps, nGhost);
    double tmpVal{0.5};
    mfTmp.setVal(tmpVal);

    std::unique_ptr<Filter> biFilter = std::make_unique<BilinearFilter> ();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->ApplyStencil(mfTmp, mf, srcCompBegIn, dstCompBegIn, nComps);
    // probably fails due to filter expanding box (which then has 0s)

    for (amrex::MFIter mfi(mfTmp); mfi.isValid(); ++mfi) {
        // Expect all indices to be constVal
        checkField(mfTmp[mfi].array(), infra.n_cell.dim3(), {}, {}, constVal);
    }
}

INSTANTIATE_TEST_SUITE_P(FilterPasses, BilinearFilterTestParameter, testing::Range(0, 4));

TEST_F(BilinearFilterTest, NoFilter) {
    amrex::ParmParse pp;
    pp.add("filter.use_filter", false);

    amrex::IntVect nGhost{AMREX_D_DECL(0, 0, 0)};

    amrex::MultiFab mf;
    mf.define(amrex::convert(infra.grid, amrex::IntVect{1}), infra.distriMap, nComps, nGhost);
    double constVal{5.0};
    mf.setVal(constVal);

    amrex::MultiFab mfTmp;
    mfTmp.define(amrex::convert(infra.grid, amrex::IntVect{1}), infra.distriMap, nComps, nGhost);
    double tmpVal{0.5};
    mfTmp.setVal(tmpVal);

    std::unique_ptr<Filter> biFilter = std::make_unique<BilinearFilter> ();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->ApplyStencil(mfTmp, mf, srcCompBegIn, dstCompBegIn, nComps);

    for (amrex::MFIter mfi(mfTmp); mfi.isValid(); ++mfi) {
        // Expect the all indices to be constVal
        checkField(mfTmp[mfi].array(), infra.n_cell.dim3(), {}, {}, tmpVal);
    }
}

TEST_F(BilinearFilterTest, AnalyticalTest) {
    // Parse analytical field and initialize parserEval.
    const std::string analyticalInit{"sin(kvarx*x)"};
    //const std::string analyticalInit{"x"};
    // One pass bilinear filter is
    // f(x) + 0.5*(sum_{n=1}^{infty} df^(2n)/d^(2n)x (x) *(dx)^(2n)/(2n)!)
    // with dx = 1 and f(x) = sin(k*x), this is
    // f(x) + 0.5*(cos(k) - 1)*f(x) = 0.5*f(x)(1 + cos(k))
    const std::string analyticalSol{"0.5*sin(kvarx*x)*(1+cos(kvarx))"};

    amrex::Vector<amrex::Real> k;
    parameters.get("k", k);
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

    constexpr int hodgeDegree{2};
    constexpr int maxSplineDegree{3};
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree);
    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcInit);
    DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoSol(deRham, funcSol);

    amrex::ParmParse pp;
    pp.add("filter.use_filter", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 0, 0)};
    pp.addarr("filter.filter_npass_each_dir", filterNpass);

    std::unique_ptr<Filter> biFilter = std::make_unique<BilinearFilter> ();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->ApplyStencil(rhoTemp.data, rho.data, srcCompBegIn, dstCompBegIn, nComps);

    for (amrex::MFIter mfi(rho.data); mfi.isValid(); ++mfi) {
        const amrex::Box &bx = mfi.tilebox();
        compareFields(rhoTemp.data[mfi].array(), rhoSol.data[mfi].array(), bx); 
    }
}
}

#include <vector>

#include <gtest/gtest.h>

#include "GEMPIC_BilinearFilter.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;

void set_rho_parallel_for (amrex::Array4<amrex::Real> const &rhoInArr,
                           amrex::Array4<amrex::Real> const &rhoOutArr,
                           const amrex::Box &bx)
{
    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Set value of rhoIn to 1 at given point and result for one pass
            double constVal{1.0};
            if (i == bx.smallEnd(0) + 2 && j == bx.smallEnd(1) + 2 && k == bx.smallEnd(2) + 2)
            {
                rhoInArr(i, j, k) = constVal;
                rhoOutArr(i, j, k) = 0.5 * constVal;
                rhoOutArr(i + 1, j, k) = 0.25 * constVal;
                rhoOutArr(i - 1, j, k) = 0.25 * constVal;
            }
        });
}

// Test fixture. Sets up clean environment before each test.
class BilinearFilterTest : public testing::Test
{
public:
    Io::Parameters m_parameters{};

    // Initialize computational_domain
    ComputationalDomain m_infra;

    int m_nComps{1};

    static constexpr int s_maxSplineDegree{1};

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
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("ComputationalDomain.nCell", nCell);
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
    }
};

class BilinearFilterTestParameter : public BilinearFilterTest,
                                    public testing::WithParamInterface<int>
{
public:
    std::vector<int> m_filterNpass;

    BilinearFilterTestParameter()
    {
        amrex::ParmParse pp;
        pp.add("Filter.enable", true);
        int filterPass{GetParam()};
        m_filterNpass = std::vector<int>{AMREX_D_DECL(filterPass, filterPass, filterPass)};
        pp.addarr("Filter.nPass", m_filterNpass);
    }
};

TEST_P(BilinearFilterTestParameter, ConstantTest)
{
    const int hodgeDegree{2};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rhoIn(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoOut(deRham);

    double constVal{5.0};
    rhoIn.m_data.setVal(constVal);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(rhoOut, rhoIn, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rhoOut.m_data); mfi.isValid(); ++mfi)
    {
        // Expect all indices to be constVal
        CHECK_FIELD(rhoOut.m_data.array(mfi), m_infra.m_nCell.dim3(), {}, {}, constVal);
    }
}

INSTANTIATE_TEST_SUITE_P(FilterPasses, BilinearFilterTestParameter, testing::Range(0, 4));

TEST_F(BilinearFilterTest, NoFilter)
{
    amrex::ParmParse pp;
    pp.add("Filter.enable", false);

    const int hodgeDegree{2};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rhoIn(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoOut(deRham);

    double constVal{5.0};
    rhoIn.m_data.setVal(constVal);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(rhoOut, rhoIn, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rhoOut.m_data); mfi.isValid(); ++mfi)
    {
        // Expect all indices to be constVal
        CHECK_FIELD(rhoOut.m_data.array(mfi), m_infra.m_nCell.dim3(), {}, {}, constVal);
    }
}

TEST_F(BilinearFilterTest, OneNonZeroValueTest)
{
    const int hodgeDegree{2};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rhoIn(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoOut(deRham);

    rhoIn.m_data.setVal(0.0);
    rhoOut.m_data.setVal(0.0);

    amrex::ParmParse pp;
    pp.add("Filter.enable", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 0, 0)};
    pp.addarr("Filter.nPass", filterNpass);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(rhoOut, rhoIn, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rhoOut.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const &rhoInArr = rhoIn.m_data.array(mfi);
        amrex::Array4<amrex::Real> const &rhoOutArr = rhoOut.m_data.array(mfi);

        set_rho_parallel_for(rhoInArr, rhoOutArr, bx);

        COMPARE_FIELDS(rhoInArr, rhoOutArr, bx);
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
    //  One pass bilinear filter with compensation is
    //  (alpha+(1-alpha)cos(kvarx))*g(x)
    //  where g(x) is the analytical solution without filter
    const std::string analyticalSolComp{"(0.5 + 0.5*cos(kvarx))*" + analyticalSol};

    amrex::Vector<amrex::Real> k;
    m_parameters.get("k", k);
    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t

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
    biFilter->apply_stencil(rhoTemp, rho, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        COMPARE_FIELDS(rhoTemp.m_data.array(mfi), rhoSol.m_data.array(mfi), bx);
    }

    // Test compensated filter
    pp.add("Filter.compensate", true);

    std::unique_ptr<Filter::Filter> biFilterComp = std::make_unique<Filter::BilinearFilter>();
    biFilterComp->apply_stencil(rhoTemp, rho, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        COMPARE_FIELDS(rhoTemp.m_data.array(mfi), rhoSolComp.m_data.array(mfi), bx);
    }
}
} // namespace

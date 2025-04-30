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

void set_rho_parallel_for (amrex::Array4<amrex::Real> const& rhoInArr,
                           amrex::Array4<amrex::Real> const& rhoOutArr,
                           amrex::Box const& bx)
{
    auto smallEnd = bx.smallEnd().dim3();
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Set value of rhoIn to 1 at given point and result for one pass
                    double constVal{1.0};
                    if (i == smallEnd.x + 2 && j == smallEnd.y + 2 && k == smallEnd.z + 2)
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

    static constexpr int s_gSize{5};
    static constexpr int s_maxSplineDegree{1};

    BilinearFilterTest() : m_infra{Gempic::Test::Utils::get_compdom(s_gSize)}
    {
        amrex::Real k1D{2 * M_PI / s_gSize};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(k1D, k1D, k1D)};
        m_parameters.set("k", k);
    }
};

class BilinearFilterTestParameter : public BilinearFilterTest,
                                    public testing::WithParamInterface<int>
{
public:
    std::vector<int> m_filterNpass;

    BilinearFilterTestParameter()
    {
        m_parameters.set("Filter.enable", true);
        int filterPass{GetParam()};
        m_filterNpass = std::vector<int>{AMREX_D_DECL(filterPass, filterPass, filterPass)};
        m_parameters.set("Filter.nPass", m_filterNpass);
    }
};

TEST_P(BilinearFilterTestParameter, ConstantTest)
{
    int const hodgeDegree{2};
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
        CHECK_FIELD(rhoOut.m_data.array(mfi), mfi.validbox(), {}, {}, constVal);
    }
}

INSTANTIATE_TEST_SUITE_P(FilterPasses, BilinearFilterTestParameter, testing::Range(0, 4));

TEST_F(BilinearFilterTest, LinearTest)
{
    amrex::Real tolerance{2.0e-14};
    m_parameters.set("Filter.enable", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 1, 1)};
    m_parameters.set("Filter.nPass", filterNpass);

    std::string const linearRho{AMREX_D_PICK("x", "1 + x + 2 * y + x * y", "x + y + z")};
    amrex::Parser parserRho;
    parserRho.define(linearRho);
    parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    auto funcRho = parserRho.compile<nVar>();

    int const hodgeDegree{4}; // need 1 ghost cell per pass
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rhoIn(deRham, funcRho);
    DeRhamField<Grid::dual, Space::cell> rhoOut(deRham, funcRho);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(rhoOut, rhoIn, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rhoIn.m_data); mfi.isValid(); ++mfi)
    {
        // Expect filter not to change linear function
        amrex::Box const& bx = mfi.tilebox();
        amrex::Box const& interiorBox = amrex::grow(bx, -1); // remove boundary terms
        COMPARE_FIELDS(rhoIn.m_data.array(mfi), rhoOut.m_data.array(mfi), interiorBox, tolerance);
    }
}

TEST_F(BilinearFilterTest, NoFilter)
{
    m_parameters.set("Filter.enable", false);

    int const hodgeDegree{2};
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
        CHECK_FIELD(rhoOut.m_data.array(mfi), mfi.validbox(), {}, {}, constVal);
    }
}

TEST_F(BilinearFilterTest, OneNonZeroValueTest)
{
    int const hodgeDegree{2};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rhoIn(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoOut(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoOutExpected(deRham);

    rhoIn.m_data.setVal(0.0);
    rhoOut.m_data.setVal(0.0);

    m_parameters.set("Filter.enable", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 0, 0)};
    m_parameters.set("Filter.nPass", filterNpass);

    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};

    for (amrex::MFIter mfi(rhoOut.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& rhoInArr = rhoIn.m_data.array(mfi);
        amrex::Array4<amrex::Real> const& rhoOutExpArr = rhoOutExpected.m_data.array(mfi);

        set_rho_parallel_for(rhoInArr, rhoOutExpArr, bx);
    }

    biFilter->apply_stencil(rhoOut, rhoIn, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rhoOut.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& rhoOutArr = rhoOut.m_data.array(mfi);
        amrex::Array4<amrex::Real> const& rhoOutExpArr = rhoOutExpected.m_data.array(mfi);

        COMPARE_FIELDS(rhoOutArr, rhoOutExpArr, bx);
    }
}

TEST_F(BilinearFilterTest, AnalyticalTest)
{
    // Parse analytical field and initialize parserEval.
    std::string const analyticalInit{"sin(kvarx*x)"};
    //  One pass bilinear filter is
    //  f(x) + 0.5*(sum_{n=1}^{infty} df^(2n)/d^(2n)x (x) *(dx)^(2n)/(2n)!)
    //  with dx = 1 and f(x) = sin(k*x), this is
    //  f(x) + 0.5*(cos(k) - 1)*f(x) = 0.5*f(x)(1 + cos(k))
    std::string const analyticalSol{"0.5*sin(kvarx*x)*(1+cos(kvarx))"};

    m_parameters.set("Function.rhoInit", analyticalInit);
    m_parameters.set("Function.rhoAnal", analyticalSol);

    [[maybe_unused]] auto [parserInit, funcInit] = Utils::parse_function("rhoInit");
    [[maybe_unused]] auto [parserSol, funcSol] = Utils::parse_function("rhoAnal");

    constexpr int hodgeDegree{2};
    constexpr int maxSplineDegree{3};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, maxSplineDegree);
    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcInit);
    DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoSol(deRham, funcSol);

    m_parameters.set("Filter.enable", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 0, 0)};
    m_parameters.set("Filter.nPass", filterNpass);

    // Test uncompensated filter
    std::unique_ptr<Filter::Filter> biFilter = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegIn{0};
    int dstCompBegIn{0};
    biFilter->apply_stencil(rhoTemp, rho, srcCompBegIn, dstCompBegIn, m_nComps);

    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        COMPARE_FIELDS(rhoTemp.m_data.array(mfi), rhoSol.m_data.array(mfi), bx);
    }
}

TEST_F(BilinearFilterTest, CompensatedAnalyticalTest)
{
    // Parse analytical field and initialize parserEval.
    std::string const analyticalInit{"sin(kvarx*x)"};
    std::string const analyticalSol{"0.5*sin(kvarx*x)*(1+cos(kvarx))"};
    //  One pass bilinear filter with compensation is
    //  (alpha+(1-alpha)cos(kvarx))*g(x)
    //  where g(x) is the analytical solution without filter
    std::string const analyticalSolComp{"(0.5 + 0.5*cos(kvarx))*" + analyticalSol};

    m_parameters.set("Function.rhoInit", analyticalInit);
    m_parameters.set("Function.rhoAnal", analyticalSolComp);

    [[maybe_unused]] auto [parserInit, funcInit] = Utils::parse_function("rhoInit");
    [[maybe_unused]] auto [parserSolComp, funcSolComp] = Utils::parse_function("rhoAnal");

    constexpr int hodgeDegree{2};
    constexpr int maxSplineDegree{3};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, maxSplineDegree);
    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcInit);
    DeRhamField<Grid::dual, Space::cell> rhoTemp(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoSolComp(deRham, funcSolComp);

    m_parameters.set("Filter.enable", true);
    std::vector<int> filterNpass{AMREX_D_DECL(1, 0, 0)};
    m_parameters.set("Filter.nPass", filterNpass);

    // Test compensated filter
    m_parameters.set("Filter.compensate", true);

    std::unique_ptr<Filter::Filter> biFilterComp = std::make_unique<Filter::BilinearFilter>();
    int srcCompBegin{0};
    int dstCompBegin{0};
    biFilterComp->apply_stencil(rhoTemp, rho, srcCompBegin, dstCompBegin, m_nComps);

    for (amrex::MFIter mfi(rho.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        COMPARE_FIELDS(rhoTemp.m_data.array(mfi), rhoSolComp.m_data.array(mfi), bx);
    }
}
} // namespace

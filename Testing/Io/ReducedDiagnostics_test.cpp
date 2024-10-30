#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Sampler.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;

class ReducedDiagnosticsTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    // particle data
    static const int s_vdim{3};

    //
    ComputationalDomain m_infra{false};  // "uninitialized" periodic computational domain
    std::vector<std::shared_ptr<ParticleGroups<s_vdim>>> m_particles;
    //
    Io::Parameters m_parameters{};

    // Setup all the tests in the TestSuite
    static void SetUpTestSuite ()
    {
        // Variables that could come from the input file and are stored by amrex during the whole
        // simulation
        amrex::ParmParse pp;
        /* computational domain */
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        pp.addarr("domainLo", domainLo);
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("k", k);
        const amrex::Vector<int> nCell{AMREX_D_DECL(8, 8, 8)};
        pp.addarr("nCellVector", nCell);
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(4, 4, 4)};
        pp.addarr("maxGridSizeVector", maxGridSize);
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        pp.addarr("isPeriodicVector", isPeriodic);
        // particles (data read by particle_groups constructor)
        std::string speciesNames{"ions"};
        pp.add("Particle.speciesNames", speciesNames);
        std::string samplerName{"PseudoRandom"};
        pp.add("Particle.sampler", samplerName);
        amrex::Real charge{1.0};
        pp.add("Particle.ions.charge", charge);
        amrex::Real mass{1.0};
        pp.add("Particle.ions.mass", mass);
        int nPartPerCell{100};
#if GEMPIC_SPACEDIM == 1
        nPartPerCell = 1000;
#endif
        pp.add("Particle.ions.nPartPerCell", nPartPerCell);
        std::string density{"1.0 + 0.02 * cos(kvarx * x)"};
        pp.add("Particle.ions.density", density);
        int numGaussians{1};
        pp.add("Particle.ions.numGaussians", numGaussians);
        amrex::Real vWeightG0{1.0};
        pp.add("Particle.ions.G0.vWeight", vWeightG0);
        amrex::Vector<amrex::Real> vMean{{-1.0, 1.0, 2.0}};
        pp.addarr("Particle.ions.G0.vMean", vMean);
        amrex::Vector<amrex::Real> vThermal{{1.0, 2.0, 3.0}};
        pp.addarr("Particle.ions.G0.vThermal", vThermal);
        // functions defining fields. Variable y, z not available in 1D, variable y not available in
        // 2D
        std::string Ex = "cos(kvarx * x)";
        std::string Ey = "cos(kvary * y)";
        std::string Ez = "cos(kvarx * x) * cos(kvarz * z)";
        std::string Bx = "2 * cos(kvarx * x)";
        std::string By = "1.0";
        std::string Bz = "1.0";
#if GEMPIC_SPACEDIM == 2
        Ex = "cos(kvarx * x)";
        Ey = "cos(kvary * y)";
        Ez = "cos(kvarx * x) * cos(kvary * y)";
#elif GEMPIC_SPACEDIM == 1
        Ex = "cos(kvarx * x)";
        Ey = "0.7071";
        Ez = "0.5";
#endif

        pp.add("Function.Bx", Bx);
        pp.add("Function.By", By);
        pp.add("Function.Bz", Bz);
        pp.add("Function.Ex", Ex);
        pp.add("Function.Ey", Ey);
        pp.add("Function.Ez", Ez);

        // Parse reduced diagnostics
        amrex::ParmParse ppRedDiag("ReducedDiagnostics");
        amrex::Vector<std::string> reducedDiagsNames{"PartDiag", "FieldElec", "FieldMag",
                                                     "GaussError"};
        int saveReduced{1};
        std::string fieldElecType{"ElecFieldEnergy"};
        std::string fieldMagType{"MagFieldEnergy"};
        std::string partDiagType{"Particle"};
        std::string gaussErrorType{"GaussError"};
        ppRedDiag.addarr("groupNames", reducedDiagsNames);
        ppRedDiag.add("saveInterval", saveReduced);
        ppRedDiag.add("FieldElec.types", fieldElecType);
        ppRedDiag.add("FieldMag.types", fieldMagType);
        ppRedDiag.add("PartDiag.types", partDiagType);
        ppRedDiag.add("GaussError.types", gaussErrorType);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        m_infra = ComputationalDomain{};
        init_particles(m_infra, m_particles);
    }
};

TEST_F(ReducedDiagnosticsTest, ReducedDiags)
{
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    [[maybe_unused]] auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
    [[maybe_unused]] auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

    DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
    DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
    DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
    DeRhamField<Grid::dual, Space::face> D(deRham, funcE, "D");
    DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
    DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");

    // Initialize reduced diagnostics
    Io::MultiReducedDiagnostics<s_vdim, s_degX, s_degY, s_degZ, hodgeDegree, 1> redDiagn(deRham);
    // Compute and write reduced diagnostics
    redDiagn.compute_diags(m_infra, deRham->m_fieldsDiagnostics, m_particles);
    redDiagn.write_to_file(0, 1.0);

    // check electric field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputElec("ReducedDiagnostics/FieldElec.txt");
    std::string line;
    std::getline(inputElec, line);  // first line not used
    std::getline(inputElec, line);
    std::stringstream splitLine(line);
    double step, t, ex2, ey2, ez2;
    splitLine >> step >> t >> ex2 >> ey2 >> ez2;
    // precision is not high as coarse grid is used as well as not periodic functions
    EXPECT_NEAR(ex2, 0.25, 0.01);
    EXPECT_NEAR(ey2, 0.25, 0.01);
    EXPECT_NEAR(ez2, 0.125, 0.01);

    // check magnetic field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputMag("ReducedDiagnostics/FieldMag.txt");
    std::getline(inputMag, line);  // first line not used
    std::getline(inputMag, line);
    std::stringstream splitLineB(line);
    double bx2, by2, bz2;
    splitLineB >> step >> t >> bx2 >> by2 >> bz2;
    EXPECT_NEAR(bx2, 1.0, 0.1);
    EXPECT_NEAR(by2, 0.5, 1e-7);
    EXPECT_NEAR(bz2, 0.5, 1e-7);

    // check particle diagnostics
    // formulas for computing the exact values of the moments that are needed
    // are given in the file test_sampler.cpp
    std::ifstream inputPart("ReducedDiagnostics/PartDiag.txt");
    std::getline(inputPart, line);  // first line not used
    std::getline(inputPart, line);
    std::stringstream splitLinePart(line);
    double px, py, pz, kin;
    splitLinePart >> step >> t >> px >> py >> pz >> kin;
    EXPECT_NEAR(px, -1.0, 0.1);
    EXPECT_NEAR(py, 1.0, 0.1);
    EXPECT_NEAR(pz, 2.0, 0.1);
    EXPECT_NEAR(kin, 10.0, 0.1);

    // Test Gauss error (only that terms are correctly written)
    std::ifstream inputGauss("ReducedDiagnostics/GaussError.txt");
    std::getline(inputGauss, line);  // first line not used
    std::getline(inputGauss, line);
    std::stringstream splitLineGauss(line);
    double error{0.0};
    splitLineGauss >> step >> t >> error;
    EXPECT_NEAR(error, 1.0, 1e-12);  // actual error not checked
}
}  // namespace
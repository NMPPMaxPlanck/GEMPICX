#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Sampler.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;

void reduced_diagnostics_setup (Io::Parameters& parameters, ComputationalDomain const& compDom)
{
    // Variables that could come from the input file and are stored by amrex during the whole
    // simulation
    // 2pi(domainHi - domainLo) = k
    std::array<amrex::Real, AMREX_SPACEDIM> domainSize = {
        AMREX_D_DECL(compDom.geometry().ProbLength(xDir), compDom.geometry().ProbLength(yDir),
                     compDom.geometry().ProbLength(zDir))};
    amrex::Vector<amrex::Real> k{AMREX_D_DECL(
        2 * M_PI * domainSize[xDir], 2 * M_PI * domainSize[yDir], 2 * M_PI * domainSize[zDir])};
    parameters.set("k", k);
    // particles (data read by particle_groups constructor)
    std::string speciesNames{"ions"};
    parameters.set("Particle.speciesNames", speciesNames);
    std::string samplerName{"PseudoRandom"};
    parameters.set("Particle.sampler", samplerName);
    amrex::Real charge{1.0};
    parameters.set("Particle.ions.charge", charge);
    amrex::Real mass{1.0};
    parameters.set("Particle.ions.mass", mass);
    int nPartPerCell{100};
#if AMREX_SPACEDIM == 1
    nPartPerCell = 1000;
#endif
    parameters.set("Particle.ions.nPartPerCell", nPartPerCell);
    std::string density{"1.0 + 0.02 * cos(kvarx * x)"};
    parameters.set("Particle.ions.density", density);
    int numGaussians{1};
    parameters.set("Particle.ions.numGaussians", numGaussians);
    amrex::Real vWeightG0{1.0};
    parameters.set("Particle.ions.G0.vWeight", vWeightG0);
    amrex::Vector<amrex::Real> vMean{{-1.0, 1.0, 2.0}};
    parameters.set("Particle.ions.G0.vMean", vMean);
    amrex::Vector<amrex::Real> vThermal{{1.0, 2.0, 3.0}};
    parameters.set("Particle.ions.G0.vThermal", vThermal);
    // functions defining fields. Variable y, z not available in 1D, variable y not available in 2D
    std::string Ex = "cos(kvarx * x)";
    std::string Ey = "cos(kvary * y)";
    std::string Ez = "cos(kvarx * x) * cos(kvarz * z)";
    std::string Bx = "2 * cos(kvarx * x)";
    std::string By = "1.0";
    std::string Bz = "1.0";
#if AMREX_SPACEDIM == 2
    Ex = "cos(kvarx * x)";
    Ey = "cos(kvary * y)";
    Ez = "cos(kvarx * x) * cos(kvary * y)";
#elif AMREX_SPACEDIM == 1
    Ex = "cos(kvarx * x)";
    Ey = "0.7071";
    Ez = "0.5";
#endif

    parameters.set("Function.Bx", Bx);
    parameters.set("Function.By", By);
    parameters.set("Function.Bz", Bz);
    parameters.set("Function.Ex", Ex);
    parameters.set("Function.Ey", Ey);
    parameters.set("Function.Ez", Ez);
}

ComputationalDomain get_compdom ()
{
    // 2pi(domainHi - domainLo) = k
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(8, 8, 8)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(4, 4, 4)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

class ReducedDiagnosticsTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    // particle data
    static int const s_vdim{3};

    //
    ComputationalDomain m_infra;
    std::vector<std::shared_ptr<ParticleGroups<s_vdim>>> m_particles;
    //
    Io::Parameters m_parameters{};
    //
    amrex::Real const m_backgroundDensity{-1.0}; // so that \int rho dx = 0

    // Setup all the tests in the TestSuite
    ReducedDiagnosticsTest() : m_infra{get_compdom()}
    {
        reduced_diagnostics_setup(m_parameters, m_infra);
        init_particles(m_particles, m_infra);

        // Parse reduced diagnostics
        Io::Parameters ppRedDiag("ReducedDiagnostics");
        amrex::Vector<std::string> reducedDiagsNames{"PartDiag", "FieldElec", "FieldMag",
                                                     "FieldCurrent", "GaussError"};
        int saveReduced{1};
        std::string fieldElecType{"ElecFieldEnergy"};
        std::string fieldMagType{"MagFieldEnergy"};
        std::string fieldCurrentType{"CurrentFieldEnergy"};
        std::string partDiagType{"Particle"};
        std::string gaussErrorType{"GaussError"};
        ppRedDiag.set("groupNames", reducedDiagsNames);
        ppRedDiag.set("saveInterval", saveReduced);
        ppRedDiag.set("FieldElec.types", fieldElecType);
        ppRedDiag.set("FieldMag.types", fieldMagType);
        ppRedDiag.set("FieldCurrent.types", fieldCurrentType);
        ppRedDiag.set("PartDiag.types", partDiagType);
        ppRedDiag.set("GaussError.types", gaussErrorType);
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
    DeRhamField<Grid::primal, Space::edge> jField(deRham, funcE, "JField");
    DeRhamField<Grid::dual, Space::face> jFieldT(deRham, funcE, "JFieldT");

    // Initialize reduced diagnostics
    Io::MultiReducedDiagnostics<s_vdim, s_degX, s_degY, s_degZ, 1> redDiagn(deRham);
    // Compute and write reduced diagnostics
    rho.m_data.setVal(m_backgroundDensity); // give some non zero value to rho
    redDiagn.compute_and_write_to_file(0, 1.0, m_infra, m_particles);

    // check electric field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputElec("ReducedDiagnostics/FieldElec.txt");
    std::string line;
    std::getline(inputElec, line); // first line not used
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
    std::getline(inputMag, line); // first line not used
    std::getline(inputMag, line);
    std::stringstream splitLineB(line);
    double bx2, by2, bz2;
    splitLineB >> step >> t >> bx2 >> by2 >> bz2;
    EXPECT_NEAR(bx2, 1.0, 0.1);
    EXPECT_NEAR(by2, 0.5, 1e-7);
    EXPECT_NEAR(bz2, 0.5, 1e-7);

    // check current field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputCurrent("ReducedDiagnostics/FieldCurrent.txt");
    std::getline(inputCurrent, line); // first line not used
    std::getline(inputCurrent, line);
    std::stringstream splitLineJ(line);
    double jx2, jy2, jz2;
    splitLineJ >> step >> t >> jx2 >> jy2 >> jz2;
    // precision is not high as coarse grid is used as well as not periodic functions
    EXPECT_NEAR(jx2, 0.25, 0.01);
    EXPECT_NEAR(jy2, 0.25, 0.01);
    EXPECT_NEAR(jz2, 0.125, 0.01);

    // check particle diagnostics
    // formulas for computing the exact values of the moments that are needed
    // are given in the file test_sampler.cpp
    std::ifstream inputPart("ReducedDiagnostics/PartDiag.txt");
    std::getline(inputPart, line); // first line not used
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
    std::getline(inputGauss, line); // first line not used
    std::getline(inputGauss, line);
    std::stringstream splitLineGauss(line);
    double error;
    double divDNorm;
    double readRhoNorm;
    splitLineGauss >> step >> t >> error >> divDNorm >> readRhoNorm;
    EXPECT_NEAR(error / readRhoNorm, 1.0, 1e-12); // actual error not checked
}

class ReducedDiagnosticsMissingFieldsTest : public testing::Test
{
protected:
    static constexpr amrex::Real s_densityFieldFactor{0.5};

    // Spline degreesx
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    // particle data
    static int const s_vdim{3};

    ComputationalDomain m_infra;
    std::vector<std::shared_ptr<ParticleGroups<s_vdim>>> m_particles;
    Io::Parameters m_parameters{};
    amrex::Real const m_backgroundDensity{-1.0}; // so that \int rho dx = 0

    ReducedDiagnosticsMissingFieldsTest() : m_infra{get_compdom()}
    {
        reduced_diagnostics_setup(m_parameters, m_infra);
        init_particles(m_particles, m_infra);

        m_parameters.set("Function.DensityField", std::to_string(s_densityFieldFactor));
        m_parameters.set("Function.DensityFieldInv", std::to_string(s_densityFieldFactor));
        // For verification
        std::string solverStr{"ConjugateGradientInverseHodge"};
        m_parameters.set("PoissonSolver.solver", solverStr);

        Io::Parameters ppRedDiag("ReducedDiagnostics");
        ppRedDiag.set("computeMissingFields", 1);
        amrex::Vector<std::string> reducedDiagsNames{"PartDiagMissing", "FieldElecMissing",
                                                     "FieldMagMissing", "FieldCurrentMissing",
                                                     "GaussErrorMissing"};
        ppRedDiag.set("groupNames", reducedDiagsNames);
        std::string fieldElecType{"ElecFieldEnergy"};
        std::string fieldMagType{"MagFieldEnergy"};
        std::string fieldCurrentType{"CurrentFieldEnergy"};
        std::string partDiagType{"Particle"};
        std::string gaussErrorType{"GaussError"};
        ppRedDiag.set("FieldElecMissing.types", fieldElecType);
        ppRedDiag.set("FieldMagMissing.types", fieldMagType);
        ppRedDiag.set("FieldCurrentMissing.types", fieldCurrentType);
        ppRedDiag.set("PartDiagMissing.types", partDiagType);
        ppRedDiag.set("GaussErrorMissing.types", gaussErrorType);
    }
};

TEST_F(ReducedDiagnosticsMissingFieldsTest, ReducedDiagsMissingPrimalFields)
{
    std::string saveFolder{"RDMissingPrimalFields"};
    m_parameters.set("ReducedDiagnostics.saveFolder", saveFolder);
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    [[maybe_unused]] auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
    [[maybe_unused]] auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

    DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
    DeRhamField<Grid::dual, Space::face> D(deRham, funcE, "D");
    DeRhamField<Grid::dual, Space::face> jFieldT(deRham, funcE, "JFieldT");
    DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");

    // Initialize reduced diagnostics
    Io::MultiReducedDiagnostics<s_vdim, s_degX, s_degY, s_degZ, 1> redDiagn{deRham};

    // Compute and write reduced diagnostics
    rho.m_data.setVal(m_backgroundDensity); // give some non zero value to rho
    redDiagn.compute_and_write_to_file(0, 1.0, m_infra, m_particles);

    // check electric field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputElec(saveFolder + "/FieldElecMissing.txt");
    std::string line;
    std::getline(inputElec, line); // first line not used
    std::getline(inputElec, line);
    std::stringstream splitLine(line);
    double step, t, ex2, ey2, ez2;
    splitLine >> step >> t >> ex2 >> ey2 >> ez2;
    // precision is not high as coarse grid is used as well as not periodic functions
    EXPECT_NEAR(ex2, 0.25, 0.01);
    EXPECT_NEAR(ey2, 0.25, 0.02);
    EXPECT_NEAR(ez2, 0.125, 0.02);

    // check magnetic field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputMag(saveFolder + "/FieldMagMissing.txt");
    std::getline(inputMag, line); // first line not used
    std::getline(inputMag, line);
    std::stringstream splitLineB(line);
    double bx2, by2, bz2;
    splitLineB >> step >> t >> bx2 >> by2 >> bz2;
    EXPECT_NEAR(bx2, 1.0, 0.1);
    EXPECT_NEAR(by2, 0.5, 1e-7);
    EXPECT_NEAR(bz2, 0.5, 1e-7);

    // check current field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputCurrent(saveFolder + "/FieldCurrentMissing.txt");
    std::getline(inputCurrent, line); // first line not used
    std::getline(inputCurrent, line);
    std::stringstream splitLineJ(line);
    double jx2, jy2, jz2;
    splitLineJ >> step >> t >> jx2 >> jy2 >> jz2;
    // precision is not high as coarse grid is used as well as not periodic functions
    EXPECT_NEAR(jx2, 0.25 * s_densityFieldFactor, 0.01);
    EXPECT_NEAR(jy2, 0.25 * s_densityFieldFactor, 0.01);
    EXPECT_NEAR(jz2, 0.125 * s_densityFieldFactor, 0.01);

    // check particle diagnostics
    // formulas for computing the exact values of the moments that are needed
    // are given in the file test_sampler.cpp
    std::ifstream inputPart(saveFolder + "/PartDiagMissing.txt");
    std::getline(inputPart, line); // first line not used
    std::getline(inputPart, line);
    std::stringstream splitLinePart(line);
    double px, py, pz, kin;
    splitLinePart >> step >> t >> px >> py >> pz >> kin;
    EXPECT_NEAR(px, -1.0, 0.1);
    EXPECT_NEAR(py, 1.0, 0.1);
    EXPECT_NEAR(pz, 2.0, 0.1);
    EXPECT_NEAR(kin, 10.0, 0.1);

    // Test Gauss error (extra work because analytical field doesn't correspond to particles)
    ///todo: This is not a good test because it focuses too much on internals of GEMPIC_GaussError
    ///      It would be better to provide a D such that divD is equal to rho _with_ background.
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    rho.m_data.setVal(0.0);
    ParticleMeshCoupling::deposit_particle_density<s_degX, s_degY, s_degZ>(rho, m_particles,
                                                                           m_infra);
    rho += m_backgroundDensity * m_infra.cell_volume();

    auto poisson{Gempic::FieldSolvers::make_poisson_solver(deRham, m_infra)};
    poisson->solve(phi, rho);
    grad(E, phi);
    E *= -1.0;
    hodge(D, E);

    redDiagn.compute_and_write_to_file(0, 1.0, m_infra, m_particles);

    std::ifstream inputGauss(saveFolder + "/GaussErrorMissing.txt");
    std::getline(inputGauss, line); // first line not used
    std::getline(inputGauss, line);
    std::getline(inputGauss, line); // skip first time step
    std::stringstream splitLineGauss(line);
    double error;
    double divDNorm;
    double readRhoNorm;
    splitLineGauss >> step >> t >> error >> divDNorm >> readRhoNorm;
    EXPECT_NEAR(error, 0.0, 1e-12);
    EXPECT_NEAR(divDNorm, readRhoNorm, 1e-12);
}

TEST_F(ReducedDiagnosticsMissingFieldsTest, ReducedDiagsMissingDualFields)
{
    std::string saveFolder{"RDMissingDualFields"};
    m_parameters.set("ReducedDiagnostics.saveFolder", saveFolder);
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    [[maybe_unused]] auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
    [[maybe_unused]] auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});

    DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
    DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
    DeRhamField<Grid::primal, Space::edge> jField(deRham, funcE, "JField");
    DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");

    // Initialize reduced diagnostics
    Io::MultiReducedDiagnostics<s_vdim, s_degX, s_degY, s_degZ, 1> redDiagn(deRham);
    // Compute and write reduced diagnostics
    rho.m_data.setVal(m_backgroundDensity); // give some non zero value to rho
    redDiagn.compute_and_write_to_file(0, 1.0, m_infra, m_particles);

    // check electric field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputElec(saveFolder + "/FieldElecMissing.txt");
    std::string line;
    std::getline(inputElec, line); // first line not used
    std::getline(inputElec, line);
    std::stringstream splitLine(line);
    double step, t, ex2, ey2, ez2;
    splitLine >> step >> t >> ex2 >> ey2 >> ez2;
    // precision is not high as coarse grid is used as well as not periodic functions
    EXPECT_NEAR(ex2, 0.25, 0.02);
    EXPECT_NEAR(ey2, 0.25, 0.02);
    EXPECT_NEAR(ez2, 0.125, 0.01);

    // check magnetic field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputMag(saveFolder + "/FieldMagMissing.txt");
    std::getline(inputMag, line); // first line not used
    std::getline(inputMag, line);
    std::stringstream splitLineB(line);
    double bx2, by2, bz2;
    splitLineB >> step >> t >> bx2 >> by2 >> bz2;
    EXPECT_NEAR(bx2, 1.0, 0.1);
    EXPECT_NEAR(by2, 0.5, 1e-7);
    EXPECT_NEAR(bz2, 0.5, 1e-7);

    // check current field diagnostics
    // Compare with the analytical values for the fields that we have defined above
    std::ifstream inputCurrent(saveFolder + "/FieldCurrentMissing.txt");
    std::getline(inputCurrent, line); // first line not used
    std::getline(inputCurrent, line);
    std::stringstream splitLineJ(line);
    double jx2, jy2, jz2;
    splitLineJ >> step >> t >> jx2 >> jy2 >> jz2;
    // precision is not high as coarse grid is used as well as not periodic functions
    EXPECT_NEAR(jx2, 0.25 * s_densityFieldFactor, 0.01);
    EXPECT_NEAR(jy2, 0.25 * s_densityFieldFactor, 0.01);
    EXPECT_NEAR(jz2, 0.125 * s_densityFieldFactor, 0.01);

    // check particle diagnostics
    // formulas for computing the exact values of the moments that are needed
    // are given in the file test_sampler.cpp
    std::ifstream inputPart(saveFolder + "/PartDiagMissing.txt");
    std::getline(inputPart, line); // first line not used
    std::getline(inputPart, line);
    std::stringstream splitLinePart(line);
    double px, py, pz, kin;
    splitLinePart >> step >> t >> px >> py >> pz >> kin;
    EXPECT_NEAR(px, -1.0, 0.1);
    EXPECT_NEAR(py, 1.0, 0.1);
    EXPECT_NEAR(pz, 2.0, 0.1);
    EXPECT_NEAR(kin, 10.0, 0.1);

    // Test Gauss error (extra work because analytical field doesn't correspond to particles)
    ///todo: This is not a good test because it focuses too much on internals of GEMPIC_GaussError
    ///      It would be better to provide a D such that divD is equal to rho _with_ background.
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    rho.m_data.setVal(0.0);
    ParticleMeshCoupling::deposit_particle_density<s_degX, s_degY, s_degZ>(rho, m_particles,
                                                                           m_infra);
    rho += m_backgroundDensity * m_infra.cell_volume();

    auto poisson{Gempic::FieldSolvers::make_poisson_solver(deRham, m_infra)};
    poisson->solve(phi, rho);
    grad(E, phi);
    E *= -1.0;

    redDiagn.compute_and_write_to_file(0, 1.0, m_infra, m_particles);

    std::ifstream inputGauss(saveFolder + "/GaussErrorMissing.txt");
    std::getline(inputGauss, line); // first line not used
    std::getline(inputGauss, line);
    std::getline(inputGauss, line); // skip first time step
    std::stringstream splitLineGauss(line);
    double error;
    double divDNorm;
    double readRhoNorm;
    splitLineGauss >> step >> t >> error >> divDNorm >> readRhoNorm;
    EXPECT_NEAR(error, 0.0, 1e-12);
    EXPECT_NEAR(divDNorm, readRhoNorm, 1e-12);
}

} // namespace
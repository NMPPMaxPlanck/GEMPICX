#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_FullDiagnostics.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Sampler.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)
#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;

void define_expected (amrex::MFIter& mfi,
                      amrex::MultiFab& mfAllDiagExpected,
                      const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>& dx)
{
    const amrex::Box& bx = mfi.tilebox();
    amrex::Array4<amrex::Real> const& expected = mfAllDiagExpected[mfi].array();

    ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            AMREX_D_TERM(amrex::Real x{(i + 0.5) * dx[xDir]};, amrex::Real y{(j + 0.5) * dx[yDir]};
                         , amrex::Real z{(k + 0.5) * dx[zDir]};)
            expected(i, j, k, 0) = 1;                          // rho 3-form needs to be constant
            expected(i, j, k, 1) = GEMPIC_D_ADD(2, y, z);      // Ex 1-form constant in x direction
            expected(i, j, k, 2) = GEMPIC_D_ADD(x, 0, 2 * z);  // Ey 1-form constant in y direction
            expected(i, j, k, 3) = GEMPIC_D_ADD(2 * x, y, 0);  // Ez 1-form constant in z direction
            expected(i, j, k, 4) = 5 + x;  // Bz 2-form constant in y and z directions
            expected(i, j, k, 5) = GEMPIC_D_ADD(6 + x, y, z);  // phi linear
        });
}

class FullDiagnosticsTest : public testing::Test
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
    static const int s_ndata{1};

    //
    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    std::vector<std::shared_ptr<ParticleGroups<s_vdim>>> m_particles;
    // std::unique_ptr<amrex::MultiFab>  mf_all_diag;
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
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("k", k);
        const amrex::Vector<int> nCell{AMREX_D_DECL(8, 8, 8)};
        pp.addarr("ComputationalDomain.nCell", nCell);
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(4, 4, 4)};
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(0, 0, 0)};
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
        // particles (data read by particleGroups constructor)
        std::string speciesNames{"ions"};
        pp.add("Particle.speciesNames", speciesNames);
        std::string samplerName{"PseudoRandom"};
        pp.add("Particle.sampler", samplerName);
        amrex::Real charge{1.0};
        pp.add("Particle.ions.charge", charge);
        amrex::Real mass{1.0};
        pp.add("Particle.ions.mass", mass);
        int nPartPerCell{2};
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
        std::string rho = "1";
        std::string Ex = "2 + y + z";
        std::string Ey = "x + 2 * z";
        std::string Ez = "2 * x + y";
        std::string Bx = "5.0 + x";
        std::string By = "1.0";
        std::string Bz = "1.0";
        std::string phi = "6 + x + y + z";
#if GEMPIC_SPACEDIM == 2
        phi = "6 + x + y";
        Ex = "2 + y";
        Ey = "x";
#elif GEMPIC_SPACEDIM == 1
        phi = "6 + x";
        Ex = "2";
        Ey = "x";
        Ez = "2 * x";
#endif

        pp.add("Function.Bx", Bx);
        pp.add("Function.By", By);
        pp.add("Function.Bz", Bz);
        pp.add("Function.Ex", Ex);
        pp.add("Function.Ey", Ey);
        pp.add("Function.Ez", Ez);
        pp.add("Function.rho", rho);
        pp.add("Function.phi", phi);

        // Full diagnostics
        pp.add("FullDiagnostics.enable", true);  // 1 for true, 0 for false
        amrex::Vector<std::string> diagsNames = {"part", "field"};
        pp.addarr("FullDiagnostics.groupNames", diagsNames);
        std::string particle = "ions";
        pp.add("FullDiagnostics.part.varNames", particle);
        amrex::Vector<std::string> fieldNames = {"rho", "Ex", "Ey", "Ez", "Bx", "phi"};
        pp.addarr("FullDiagnostics.field.varNames", fieldNames);
        std::string cellCenterOutputProcessor = "CellCenter";
        pp.add("FullDiagnostics.field.outputProcessor", cellCenterOutputProcessor);
        int fieldSave{1};
        pp.add("FullDiagnostics.field.saveInterval", fieldSave);
        int partSave{1};
        pp.add("FullDiagnostics.part.saveInterval", partSave);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        m_infra = ComputationalDomain{};
        init_particles(m_particles, m_infra);
    }
};

TEST_F(FullDiagnosticsTest, FullDiagnosticsFields)
{
    constexpr int hodgeDegree{2};

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Initialize fields
    [[maybe_unused]] auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
    [[maybe_unused]] auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});
    [[maybe_unused]] auto [parseRho, funcRho] = Utils::parse_function("rho");
    [[maybe_unused]] auto [parsePhi, funcPhi] = Utils::parse_function("phi");

    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcRho, "rho");
    DeRhamField<Grid::primal, Space::node> phi(deRham, funcPhi, "phi");
    DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
    DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");

    Io::MultiDiagnostics fullDiagn(m_infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling,
                                   m_particles);
    // Compute and write diagnostics
    fullDiagn.filter_compute_pack_flush(0, 0.0);

    // Create multifab containing expected values of cell centered fields
    int ncomp = fullDiagn.get_num_group_members(1);  // nb of field diagnostics
    amrex::IntVect nghost = amrex::IntVect{AMREX_D_DECL(0, 0, 0)};
    amrex::MultiFab mfAllDiagExpected(m_infra.m_grid, m_infra.m_distriMap, ncomp, nghost);
    // Compare read and expected values. Low order interpolation of forms is used
    // conditions for cell center interpolation given for each example
    for (amrex::MFIter mfi(mfAllDiagExpected); mfi.isValid(); ++mfi)
    {
        define_expected(mfi, mfAllDiagExpected, m_infra.geometry().CellSizeArray());
    }

    // read field diagnostics
    amrex::MultiFab mfAllDiag(m_infra.m_grid, m_infra.m_distriMap, ncomp, nghost);
    std::string filename = "FullDiagnostics/plt_field000000/Level_0/Cell";
    amrex::VisMF::Read(mfAllDiag, filename);

    // Compare read and expected values
    for (amrex::MFIter mfi(mfAllDiag); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        compare_fields(mfAllDiag[mfi].array(), mfAllDiagExpected[mfi].array(), bx, ncomp);
    }
}

void create_input_file_to_select_operator (const std::string& operatorId)
{
    // This part belongs in an input file when not doing testing
    amrex::ParmParse pp;
    const std::string groupNames{"field5"};
    pp.add("FullDiagnostics.groupNames", groupNames);
    const std::string saveFolder = {"FullDiagnostics/scalingTest"};
    pp.add("FullDiagnostics.saveFolder", saveFolder);

    const std::string varNames{"rho"};
    pp.add("FullDiagnostics.field5.varNames", varNames);
    pp.add("FullDiagnostics.field5.saveInterval", 1);
    const std::string custom{"Custom"};
    pp.add("FullDiagnostics.field5.outputProcessor", custom);
    pp.add("FullDiagnostics.field5.customID", operatorId);
}

// create and add lambda, (this one multiplies by a constant and cell centers)
void add_custom_processor (const std::string& operatorId, double multiplicationFactor)
{
    Gempic::Io::add_output_processor(
        operatorId,
        [=] AMREX_GPU_DEVICE(amrex::Array4<amrex::Real> dst,
                             const amrex::Array4<const amrex::Real> src, int nSrcComp, int i, int j,
                             int k, double scaling, double ishift, double jshift, double kshift)
        {
            dst(i, j, k) = multiplicationFactor * 0.125 * scaling *
                           (src(i + ishift, j + jshift, k + kshift) + src(i, j, k) +
                            src(i + ishift, j + jshift, k) + src(i, j, k + kshift) +
                            src(i + ishift, j, k + kshift) + src(i, j + jshift, k) +
                            src(i, j + jshift, k + kshift) + src(i + ishift, j, k));
        });
}

TEST_F(FullDiagnosticsTest, FullDiagnosticsCustomOperatorOutputProcessor)
{
    // Details for the input file
    double multiplicationFactor{5.0};
    std::string operatorId{"scaleField" + std::to_string(static_cast<int>(multiplicationFactor))};
    create_input_file_to_select_operator(operatorId);

    // ------------------------------------ setup ------------------------------------
    constexpr int hodgeDegree{2};
    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Initialize fields
    [[maybe_unused]] auto [parseRho, funcRho] = Utils::parse_function("rho");

    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcRho, "rho");

    // create and add lambda, (this one multiplies by a constant and cell centers)
    add_custom_processor(operatorId, multiplicationFactor);

    // Initialize diagnostics
    Io::MultiDiagnostics fullDiagn(m_infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling,
                                   m_particles);
    // Compute and write diagnostics
    fullDiagn.filter_compute_pack_flush(0, 0.0);

    // ------------------------ Verify diagnostics output ------------------------
    int ncomp = 1;
    amrex::IntVect nghost = amrex::IntVect{AMREX_D_DECL(0, 0, 0)};

    // Create expected value MultiFab
    amrex::MultiFab multipliedBy5Expected(rho.m_data.boxArray(), rho.m_data.DistributionMap(),
                                          ncomp, nghost);
    multipliedBy5Expected.setVal(multiplicationFactor);

    // Read field diagnostics
    amrex::MultiFab resultMf(rho.m_data.boxArray(), rho.m_data.DistributionMap(), ncomp, nghost);
    std::string filename5 = "FullDiagnostics/scalingTest/plt_field5000000/Level_0/Cell";
    amrex::VisMF::Read(resultMf, filename5);

    // Compare read and expected values
    for (amrex::MFIter mfi(resultMf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        compare_fields(resultMf[mfi].array(), multipliedBy5Expected[mfi].array(), bx, ncomp);
    }
}

// Silly minimal custom outputProcessor class example
class DumbOutputProcessor : public Gempic::Io::CustomOutputProcessor
{
public:
    static inline int s_counter{0};

    DumbOutputProcessor(const amrex::MultiFab& mfSrc,
                        const amrex::Real scaling,
                        const amrex::IntVect crseRatio) :
        CustomOutputProcessor{mfSrc, scaling, crseRatio}
    {
    }

    void operator()(amrex::MultiFab& mfDst, int dcomp) const final
    {
        s_counter++;
        amrex::Copy(mfDst, m_mfSrc, 0, dcomp, mfDst.nComp(), 0);
    }
};

void create_input_file_to_select_custom_strategy (const std::string& customId)
{
    // This part belongs in an input file when not doing testing
    amrex::ParmParse pp;
    const std::string customEmpty{"customEmpty"};
    pp.add("FullDiagnostics.groupNames", customEmpty);
    std::string saveFolder = {"FullDiagnostics/CustomOutputProcessorTest"};
    pp.add("FullDiagnostics.saveFolder", saveFolder);

    const std::string varNames{"phi"};
    pp.add("FullDiagnostics.customEmpty.varNames", varNames);
    pp.add("FullDiagnostics.customEmpty.saveInterval", 1);

    const std::string custom{"Custom"};
    pp.add("FullDiagnostics.customEmpty.outputProcessor", custom);
    pp.add("FullDiagnostics.customEmpty.customID", customId);
}

TEST_F(FullDiagnosticsTest, FullDiagnosticsCustomOutputProcessor)
{
    // Details for the input file
    std::string customId{"counter"};
    create_input_file_to_select_custom_strategy(customId);

    // ------------------------------------ setup ------------------------------------
    constexpr int hodgeDegree{2};
    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Initialize fields
    [[maybe_unused]] auto [parsePhi, funcPhi] = Utils::parse_function("phi");

    DeRhamField<Grid::primal, Space::node> phi(deRham, funcPhi, "phi");

    // Add custom strategy, (this one does nothing except count how many times it's been used)
    Gempic::Io::add_output_processor<DumbOutputProcessor>(customId);

    // Initialize diagnostics
    Io::MultiDiagnostics fullDiagn(m_infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling,
                                   m_particles);
    // Check that it hasn't been used yet ...
    EXPECT_EQ(DumbOutputProcessor::s_counter, 0);
    // Compute and write diagnostics
    fullDiagn.filter_compute_pack_flush(0, 0.0);

    // Check that a copy operator was used (i.e. that DumbOutputProcessor's operator was called)
    EXPECT_EQ(DumbOutputProcessor::s_counter, 1);

    int ncomp = 1;
    amrex::IntVect nghost = amrex::IntVect{AMREX_D_DECL(0, 0, 0)};

    // read field diagnostics
    amrex::MultiFab resultMf(phi.m_data.boxArray(), phi.m_data.DistributionMap(), ncomp, nghost);
    std::string filename =
        "FullDiagnostics/CustomOutputProcessorTest/plt_customEmpty000000/Level_0/Cell";
    amrex::VisMF::Read(resultMf, filename);

    // Compare read and expected values
    for (amrex::MFIter mfi(resultMf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        compare_fields(resultMf[mfi].array(), phi.m_data[mfi].array(), bx, ncomp);
    }
}
}  // namespace
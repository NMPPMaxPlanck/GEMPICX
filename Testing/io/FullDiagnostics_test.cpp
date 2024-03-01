#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "FullDiagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_computational_domain.H"
#include "GEMPIC_gempic_norm.H"
#include "GEMPIC_parameters.H"
#include "GEMPIC_sampler.H"
#include "MultiFullDiagnostics.H"
#include "test_utils/GEMPIC_test_utils.H"

#define check_field(...) GEMPIC_TestUtils::check_field(__FILE__, __LINE__, __VA_ARGS__)
#define compare_fields(...) GEMPIC_TestUtils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace CompDom;
using namespace Sampling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

void define_expected (amrex::MFIter& mfi,
                      amrex::MultiFab& mfAllDiagExpected,
                      const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>& dx)
{
    const amrex::Box& bx = mfi.tilebox();
    amrex::Array4<amrex::Real> const& expected = mfAllDiagExpected[mfi].array();

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real x = (i + 0.5) * dx[0];
                    amrex::Real y = (j + 0.5) * dx[1];
                    amrex::Real z = (k + 0.5) * dx[2];
                    expected(i, j, k, 0) = 1;  // rho 3-form needs to be constant
#if GEMPIC_SPACEDIM == 3
                    expected(i, j, k, 1) = 2 + y + z;  // Ex 1-form constant in x direction
                    expected(i, j, k, 2) = x + 2 * z;  // Ey 1-form constant in y direction
                    expected(i, j, k, 3) = 2 * x + y;  // Ez 1-form constant in z direction
                    expected(i, j, k, 4) = 5 + x;      // Bz 2-form constant in y and z directions
                    expected(i, j, k, 5) = 6 + x + y + z;  // phi linear
#elif GEMPIC_SPACEDIM == 2
                    expected(i,j,k,1) = 2 + y; // Ex
                    expected(i,j,k,2) = x;     // Ey
                    expected(i,j,k,3) = 2 * x + y; // Ez
                    expected(i,j,k,4) = 5.0 + x;   // Bx
                    expected(i,j,k,5) = 6 + x + y; // phi
#elif GEMPIC_SPACEDIM == 1
                    expected(i,j,k,1) = 2;     // Ex 
                    expected(i,j,k,2) = x;     // Ey
                    expected(i,j,k,3) = 2 * x; // Ez
                    expected(i,j,k,4) = 5.0 + x;   // Bx
                    expected(i,j,k,5) = 6 + x; // phi
#endif
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
    static const int s_numspec{1};
    static const int s_ndata{1};

    //
    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    amrex::GpuArray<std::shared_ptr<ParticleGroups<s_vdim>>, s_numspec> m_particles;
    // std::unique_ptr<amrex::MultiFab>  mf_all_diag;
    //
    Parameters m_parameters{};

    // Setup all the tests in the TestSuite
    static void SetUpTestSuite ()
    {
        // Variables that could come from the input file and are stored by amrex during the whole
        // simulation
        amrex::ParmParse pp;
        /* computational domain */
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        pp.addarr("domain_lo", domainLo);
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("k", k);
        const amrex::Vector<int> nCell{AMREX_D_DECL(8, 8, 8)};
        pp.addarr("n_cell_vector", nCell);
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(4, 4, 4)};
        pp.addarr("max_grid_size_vector", maxGridSize);
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(0, 0, 0)};
        pp.addarr("is_periodic_vector", isPeriodic);
        // particles (data read by particle_groups constructor)
        amrex::Real charge{1.0};
        pp.add("particle.species0.charge", charge);
        amrex::Real mass{1.0};
        pp.add("particle.species0.mass", mass);
        int nPartPerCell{2};
        pp.add("particle.species0.n_part_per_cell", nPartPerCell);
        std::string density{"1.0 + 0.02 * cos(kvarx * x)"};
        pp.add("particle.species0.density", density);
        int numGaussians{1};
        pp.add("particle.species0.num_gaussians", numGaussians);
        amrex::Real vWeightG0{1.0};
        pp.add("particle.species0.vWeight_g0", vWeightG0);
        amrex::Vector<amrex::Real> vMean{{-1.0, 1.0, 2.0}};
        pp.addarr("particle.species0.vMean_g0", vMean);
        amrex::Vector<amrex::Real> vThermal{{1.0, 2.0, 3.0}};
        pp.addarr("particle.species0.vThermal_g0", vThermal);
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

        pp.add("function.Bx", Bx);
        pp.add("function.By", By);
        pp.add("function.Bz", Bz);
        pp.add("function.Ex", Ex);
        pp.add("function.Ey", Ey);
        pp.add("function.Ez", Ez);
        pp.add("function.rho", rho);
        pp.add("function.phi", phi);

        // Full diagnostics
        pp.add("fullDiagnostics.enable", true);  // 1 for true, 0 for false
        amrex::Vector<std::string> diagsNames = {"part", "field"};
        pp.addarr("fullDiagnostics.group_names", diagsNames);
        std::string particle = "particles";
        pp.add("fullDiagnostics.part.var_names", particle);
        amrex::Vector<std::string> fieldNames = {"rho", "Ex", "Ey", "Ez", "Bx", "phi"};
        pp.addarr("fullDiagnostics.field.var_names", fieldNames);
        std::string cellCenterFunctor = "CellCenter";
        pp.add("fullDiagnostics.field.functor", cellCenterFunctor);
        int fieldSave{1};
        pp.add("fullDiagnostics.field.save_interval", fieldSave);
        int partSave{1};
        pp.add("fullDiagnostics.part.save_interval", partSave);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        m_infra = ComputationalDomain{};
        int spec{0};  // only one species
        m_particles[spec] = std::make_unique<ParticleGroups<s_vdim>>(spec, m_infra);
        init_particles_full_domain<s_vdim, s_numspec>(m_infra, m_particles, spec);
    }
};

TEST_F(FullDiagnosticsTest, fullDiagnosticsFields)
{
    constexpr int hodgeDegree{2};

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Initialize fields
    auto [parseB, funcB] = Utils::parse_functions<3>({"Bx", "By", "Bz"});
    auto [parseE, funcE] = Utils::parse_functions<3>({"Ex", "Ey", "Ez"});
    auto [parseRho, funcRho] = Utils::parse_function("rho");
    auto [parsePhi, funcPhi] = Utils::parse_function("phi");

    DeRhamField<Grid::dual, Space::cell> rho(deRham, funcRho, "rho");
    DeRhamField<Grid::primal, Space::node> phi(deRham, funcPhi, "phi");
    DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
    DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");

    amrex::Real dt = 1.0;
    auto nGhost = deRham->get_n_ghost();
    MultiDiagnostics<s_vdim, s_numspec, s_ndata> fullDiagn(dt);
    fullDiagn.init_data(m_infra, deRham->m_fieldsDiagnostics, deRham->m_fieldsScaling, m_particles,
                        nGhost);
    // Compute and write diagnostics
    fullDiagn.filter_compute_pack_flush(0);

    // Create multifab containing expected values of cell centered fields
    int ncomp = fullDiagn.get_ncomp(1);  // nb of field diagnostics
    amrex::IntVect nghost = amrex::IntVect{AMREX_D_DECL(0, 0, 0)};
    amrex::MultiFab mfAllDiagExpected(m_infra.m_grid, m_infra.m_distriMap, ncomp, nghost);
    // Compare read and expected values. Low order interpolation of forms is used
    // conditions for cell center interpolation given for each example
    for (amrex::MFIter mfi(mfAllDiagExpected); mfi.isValid(); ++mfi)
    {
        define_expected(mfi, mfAllDiagExpected, m_infra.m_dx);
    }

    // read field diagnostics
    amrex::MultiFab mfAllDiag(m_infra.m_grid, m_infra.m_distriMap, ncomp, nghost);
    std::string filename = "fullDiagnostics/plt_field000000/Level_0/Cell";
    amrex::VisMF::Read(mfAllDiag, filename);

    // Compare read and expected values
    for (amrex::MFIter mfi(mfAllDiag); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        compare_fields(mfAllDiag[mfi].array(), mfAllDiagExpected[mfi].array(), bx, ncomp);
    }
}
}  // namespace
#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include "GEMPIC_computational_domain.H"
#include "FullDiagnostics.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_sampler.H"
#include "GEMPIC_parameters.H"
#include "GEMPIC_gempic_norm.H"
#include "test_utils/GEMPIC_test_utils.H"
#include "MultiFullDiagnostics.H"

#define checkField(...) GEMPIC_TestUtils::checkField(__FILE__, __LINE__, __VA_ARGS__)
#define compareFields(...) GEMPIC_TestUtils::compareFields(__FILE__, __LINE__, __VA_ARGS__)

namespace {
using namespace Gempic;
using namespace CompDom;
using namespace Sampling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

    void defineExpected(amrex::MFIter& mfi, amrex::MultiFab& mf_all_diag_expected, const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>& dx)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& expected = mf_all_diag_expected[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real x = (i + 0.5) * dx[0];
            amrex::Real y = (j + 0.5) * dx[1];
            amrex::Real z = (k + 0.5) * dx[2];
            expected(i,j,k,0) = 1; // rho 3-form needs to be constant
#if GEMPIC_SPACEDIM == 3            
            expected(i,j,k,1) = 2 + y + z; // Ex 1-form constant in x direction
            expected(i,j,k,2) = x + 2 * z; // Ey 1-form constant in y direction
            expected(i,j,k,3) = 2 * x + y; // Ez 1-form constant in z direction
            expected(i,j,k,4) = 5 + x; // Bz 2-form constant in y and z directions
            expected(i,j,k,5) = 6 + x + y + z; // phi linear           
#elif GEMPIC_SPACEDIM == 2
            expected(i,j,k,1) = 2 + y; // Ex
            expected(i,j,k,2) = x;     // Ey
            expected(i,j,k,3) = 2 * x + y; // Ez
            expected(i,j,k,4) = 5.0 + x;   // Bx
            expected(i,j,k,5) = 6 + x + y; // phi
#elif  GEMPIC_SPACEDIM == 1
           expected(i,j,k,1) = 2;     // Ex 
           expected(i,j,k,2) = x;     // Ey
           expected(i,j,k,3) = 2 * x; // Ez
           expected(i,j,k,4) = 5.0 + x;   // Bx
           expected(i,j,k,5) = 6 + x; // phi 
#endif 
        });
    }

    class FullDiagnosticsTest : public testing::Test {
        protected:

        // Linear splines is ok, and lower dimension Hodge is good enough
        // Spline degreesx
        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        inline static const int maxSplineDegree{std::max(std::max(degX, degY), degZ)};
        // particle data
        static const int vdim{3};
        static const int numspec{1};
        static const int ndata{1};

        // 
        computational_domain infra{false}; // "uninitialized" computational domain
        amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec> particles;
        //std::unique_ptr<amrex::MultiFab>  mf_all_diag;
        //
        Parameters parameters{};
        

        // Setup all the tests in the TestSuite
        static void SetUpTestSuite()
        {
            // Variables that could come from the input file and are stored by amrex during the whole simulation
            amrex::ParmParse pp;
            /* computational domain */
            amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
            pp.addarr("domain_lo", domain_lo);
            amrex::Vector<amrex::Real> k{AMREX_D_DECL(2*M_PI, 2*M_PI, 2*M_PI)};
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
            int n_part_per_cell{2};
            pp.add("particle.species0.n_part_per_cell", n_part_per_cell);
            std::string density{"1.0 + 0.02 * cos(kvarx * x)"};
            pp.add("particle.species0.density", density);
            int num_gaussians{1};
            pp.add("particle.species0.num_gaussians", num_gaussians);
            amrex::Real vWeight_g0{1.0};
            pp.add("particle.species0.vWeight_g0", vWeight_g0);
            amrex::Vector<amrex::Real> vMean{{-1.0, 1.0, 2.0}};
            pp.addarr("particle.species0.vMean_g0", vMean);
            amrex::Vector<amrex::Real> vThermal{{1.0, 2.0, 3.0}};
            pp.addarr("particle.species0.vThermal_g0", vThermal);
            // functions defining fields. Variable y, z not available in 1D, variable y not available in 2D           
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
#elif  GEMPIC_SPACEDIM == 1
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
            pp.add("fullDiagnostics.enable", true); //# 1 for true, 0 for false 
            amrex::Vector<std::string> diags_names = {"part", "field"}; 
            pp.addarr("fullDiagnostics.group_names", diags_names);
            std::string particle = "particles";
            pp.add("fullDiagnostics.part.var_names", particle);
            amrex::Vector<std::string> field_names = {"rho", "Ex", "Ey", "Ez", "Bx", "phi"};
            pp.addarr("fullDiagnostics.field.var_names", field_names);
            std::string CellCenterFunctor = "CellCenter";
            pp.add("fullDiagnostics.field.functor", CellCenterFunctor); 
            int field_save{1};
            pp.add("fullDiagnostics.field.save_interval", field_save);
            int part_save{1};
            pp.add("fullDiagnostics.part.save_interval", part_save);
        }


        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
                infra = computational_domain{};
                int spec{0}; // only one species
                particles[spec] = std::make_unique<particle_groups<vdim>>(spec, infra);
                init_particles_full_domain<vdim, numspec>(infra, particles, spec);
        }
    };
    

    TEST_F(FullDiagnosticsTest, fullDiagnosticsFields) {
        constexpr int hodgeDegree{2};

        // Initialize the De Rham Complex with deg 2
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree, HodgeScheme::FDHodge);

        // Initialize fields
        auto [parseB, funcB] = Utils::parseFunctions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parseFunctions<3>({"Ex", "Ey", "Ez"});
        auto [parseRho, funcRho] = Utils::parseFunction("rho");
        auto [parsePhi, funcPhi] = Utils::parseFunction("phi");
        
        DeRhamField<Grid::dual, Space::cell> rho(deRham, funcRho, "rho");
        DeRhamField<Grid::primal, Space::node> phi(deRham, funcPhi, "phi");
        DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
        DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
  
        amrex::Real dt = 1.0;
        auto nGhost = deRham->getNGhost();
        MultiDiagnostics<vdim, numspec, ndata> fullDiagn(dt);
        fullDiagn.InitData(infra, deRham->fieldsDiagnostics, deRham->fieldsScaling, particles, nGhost);
        // Compute and write diagnostics
        fullDiagn.FilterComputePackFlush(0); 

        // Create multifab containing expected values of cell centered fields
        int ncomp = fullDiagn.getNcomp(1); // nb of field diagnostics
        amrex::IntVect nghost = amrex::IntVect{AMREX_D_DECL(0,0,0)};
        amrex::MultiFab mf_all_diag_expected(infra.grid, infra.distriMap, ncomp, nghost);
        // Compare read and expected values. Low order interpolation of forms is used
        // conditions for cell center interpolation given for each example
        for (amrex::MFIter mfi(mf_all_diag_expected); mfi.isValid(); ++mfi)
        {   
            defineExpected(mfi, mf_all_diag_expected, infra.dx);
        }

        // read field diagnostics
        amrex::MultiFab mf_all_diag(infra.grid, infra.distriMap, ncomp, nghost);
        std::string filename = "fullDiagnostics/plt_field000000/Level_0/Cell";
        amrex::VisMF::Read(mf_all_diag, filename);

        // Compare read and expected values
        for (amrex::MFIter mfi(mf_all_diag); mfi.isValid(); ++mfi)
        {   
            const amrex::Box &bx = mfi.tilebox();
            compareFields(mf_all_diag[mfi].array(), mf_all_diag_expected[mfi].array(), bx, ncomp);
        }
    }
}
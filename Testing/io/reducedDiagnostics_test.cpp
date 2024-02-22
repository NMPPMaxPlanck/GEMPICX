#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include "GEMPIC_computational_domain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_sampler.H"
#include "GEMPIC_parameters.H"
#include "GEMPIC_gempic_norm.H"
#include "test_utils/GEMPIC_test_utils.H"
#include <MultiReducedDiagnostics.H>

namespace {
using namespace Gempic;
using namespace CompDom;
using namespace Sampling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

    class ReducedDiagnosticsTest : public testing::Test {
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
        computational_domain infra{false}; // "uninitialized" periodic computational domain
        amrex::GpuArray<std::shared_ptr<particle_groups<vdim>>, numspec> particles;
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
            const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
            pp.addarr("is_periodic_vector", isPeriodic);
            // particles (data read by particle_groups constructor)
            amrex::Real charge{1.0};
            pp.add("particle.species0.charge", charge); 
            amrex::Real mass{1.0};
            pp.add("particle.species0.mass", mass);
            int n_part_per_cell{100};
#if GEMPIC_SPACEDIM == 1
            n_part_per_cell = 1000;
#endif
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
#elif  GEMPIC_SPACEDIM == 1
           Ex = "cos(kvarx * x)"; 
           Ey = "0.7071"; 
           Ez = "0.5"; 
#endif           

            pp.add("function.Bx", Bx);
            pp.add("function.By", By);
            pp.add("function.Bz", Bz);
            pp.add("function.Ex", Ex);
            pp.add("function.Ey", Ey);
            pp.add("function.Ez", Ez);

            // Parse reduced diagnostics
            amrex::ParmParse pp_redDiag("reducedDiagnostics");
            amrex::Vector<std::string> reduced_diags_names{"PartDiag", "FieldElec", "FieldMag", "GaussError"}; 
            int save_reduced{1}; 
            std::string FieldElec_type{"ElecFieldEnergy"}; 
            std::string FieldMag_type{"MagFieldEnergy"}; 
            std::string PartDiag_type{"Particle"}; 
            std::string GaussError_type{"GaussError"}; 
            pp_redDiag.addarr("group_names", reduced_diags_names);
            pp_redDiag.add("save_interval", save_reduced);
            pp_redDiag.add("FieldElec.types", FieldElec_type);
            pp_redDiag.add("FieldMag.types", FieldMag_type);
            pp_redDiag.add("PartDiag.types", PartDiag_type);
            pp_redDiag.add("GaussError.types", GaussError_type);
        }


        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
                infra = computational_domain{};
                int spec{0}; // only one species
                particles[spec] = std::make_unique<particle_groups<vdim>>(spec, infra);
                init_particles_full_domain<vdim, numspec>(infra, particles, spec);
        }
    };
    
    TEST_F(ReducedDiagnosticsTest, reducedDiags) {
        constexpr int hodgeDegree{2};

        auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree, HodgeScheme::FDHodge);

        auto [parseB, funcB] = Utils::parseFunctions<3>({"Bx", "By", "Bz"});
        auto [parseE, funcE] = Utils::parseFunctions<3>({"Ex", "Ey", "Ez"});

        
        DeRhamField<Grid::primal, Space::face> B(deRham, funcB, "B");
        DeRhamField<Grid::dual, Space::edge> H(deRham, funcB, "H");
        DeRhamField<Grid::primal, Space::edge> E(deRham, funcE, "E");
        DeRhamField<Grid::dual, Space::face> D(deRham, funcE, "D");
        DeRhamField<Grid::dual, Space::cell> rho(deRham, "rho");
        DeRhamField<Grid::dual, Space::cell> divD(deRham, "divD");

        // Initialize reduced diagnostics
        MultiReducedDiagnostics<vdim, numspec, degX, degY, degZ, hodgeDegree, 1> redDiagn(deRham);
        // Compute and write reduced diagnostics
        redDiagn.ComputeDiags(infra, deRham->fieldsDiagnostics, particles);
        redDiagn.WriteToFile(0, 1.0);

        // check electric field diagnostics
        // Compare with the analytical values for the fields that we have defined above
        std::ifstream inputElec("reducedDiagnostics/FieldElec.txt");
        std::string line;
        std::getline(inputElec, line); // first line not used
        std::getline(inputElec, line);
        std::stringstream  splitLine(line);
        double step, t, ex2, ey2, ez2;
        splitLine >> step >> t >> ex2 >> ey2 >> ez2;
        // precision is not high as coarse grid is used as well as not periodic functions
        EXPECT_NEAR(ex2, 0.25, 0.01);
        EXPECT_NEAR(ey2, 0.25, 0.01);
        EXPECT_NEAR(ez2, 0.125, 0.01);

        // check magnetic field diagnostics
        // Compare with the analytical values for the fields that we have defined above
        std::ifstream inputMag("reducedDiagnostics/FieldMag.txt");
        std::getline(inputMag, line); // first line not used
        std::getline(inputMag, line);
        std::stringstream  splitLineB(line);
        double bx2, by2, bz2;
        splitLineB >> step >> t >> bx2 >> by2 >> bz2;
        EXPECT_NEAR(bx2, 1.0, 0.1);
        EXPECT_NEAR(by2, 0.5, 1e-7);
        EXPECT_NEAR(bz2, 0.5, 1e-7);

        // check particle diagnostics
        // formulas for computing the exact values of the moments that are needed
        // are given in the file test_sampler.cpp
        std::ifstream inputPart("reducedDiagnostics/PartDiag.txt");
        std::getline(inputPart, line); // first line not used
        std::getline(inputPart, line);
        std::stringstream  splitLinePart(line);
        double px, py, pz, kin;
        splitLinePart >> step >> t >> px >> py >> pz >> kin;
        EXPECT_NEAR(px, -1.0, 0.1);
        EXPECT_NEAR(py, 1.0, 0.1);
        EXPECT_NEAR(pz, 2.0, 0.1);
        EXPECT_NEAR(kin, 10.0, 0.1);    

        // Test Gauss error (only that terms are correctly written)
        std::ifstream inputGauss("reducedDiagnostics/GaussError.txt");
        std::getline(inputGauss, line); // first line not used
        std::getline(inputGauss, line);
        std::stringstream  splitLineGauss(line);
        double error;
        splitLineGauss >> step >> t >> error;
        EXPECT_NEAR(error, 1.0, 1e-12); // actual error not checked
    }
}
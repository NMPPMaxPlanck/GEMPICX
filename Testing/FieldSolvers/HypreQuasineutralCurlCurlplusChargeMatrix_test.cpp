#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>

//#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_BilinearFilter.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

/**
 * @brief Tests the CurlCurl operator and Charge matrix of the
 * quasineutral solver. The rho*E is deposited from particles.
 */
template <typename SplineDegreeStruct>
class HypreQuasineutralCurlCurlPlusChargeMatrixTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};

    static constexpr int s_degX{std::tuple_element_t<0, SplineDegreeStruct>::value};
    static constexpr int s_degY{std::tuple_element_t<1, SplineDegreeStruct>::value};
    static constexpr int s_degZ{std::tuple_element_t<2, SplineDegreeStruct>::value};

    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{2};

    static const int s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcE;
    amrex::Array<amrex::Parser, 3> m_parserE;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHS;
    amrex::Array<amrex::Parser, 3> m_parserRHS;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::ParmParse pp; // Used instead of input file

        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        pp.addarr("ComputationalDomain.domainLo", domainLo);

        const amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("ComputationalDomain.domainHi", domainHi);

        // Grid parameters
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 8, 8)};
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);

        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);

        // Particle parameters (data read by particle_groups constructor)
        std::string speciesNames{"ions"};
        pp.add("Particle.speciesNames", speciesNames);

        std::string samplerName{"Sobol"};
        pp.add("Particle.sampler", samplerName);

        int nPartPerCell{20000};
        pp.add("Particle.ions.nPartPerCell", nPartPerCell);

        amrex::Real charge{1.0};
        pp.add("Particle.ions.charge", charge);

        amrex::Real mass{1.0};
        pp.add("Particle.ions.mass", mass);

        std::string density{"1.0"};
        pp.add("Particle.ions.density", density);

        int numGaussians{1};
        pp.add("Particle.ions.numGaussians", numGaussians);

        // Gaussian parameters
        amrex::Vector<amrex::Real> vMean{{0.0, 0.0, 0.0}};
        pp.addarr("Particle.ions.G0.vMean", vMean);

        amrex::Vector<amrex::Real> vThermal{{0.0, 0.0, 0.0}};
        pp.addarr("Particle.ions.G0.vThermal", vThermal);

        amrex::Real vWeightG0{1.0};
        pp.add("Particle.ions.G0.vWeight", vWeightG0);

        // Filter
        int filter{1};
        pp.add("Filter.enable", filter);

        amrex::Vector<int> nPass{{3, 3, 3}};
        pp.addarr("Filter.nPass", nPass);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
#if AMREX_SPACEDIM == 2
        const std::string analyticalRho = "2";
        const amrex::Array<std::string, 3> analyticalE = {"sin(y)", "sin(x)", "cos(x)*cos(y)"};
        const amrex::Array<std::string, 3> analyticalRHS = {"(1+2)*sin(y)", "(1+2)*sin(x)",
                                                            "(2+2)*cos(x)*cos(y)"};
#elif AMREX_SPACEDIM == 3
        const std::string analyticalRho = "4";
        const amrex::Array<std::string, 3> analyticalE = {"sin(y)", "sin(z)", "sin(x)"};
        const amrex::Array<std::string, 3> analyticalRHS = {"(1+4)*sin(y)", "(1+4)*sin(z)",
                                                            "(1+4)*sin(x)"};
#endif

        for (int i = 0; i < 3; ++i)
        {
            m_parserE[i].define(analyticalE[i]);
            m_parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcE[i] = m_parserE[i].compile<s_nVar>();

            m_parserRHS[i].define(analyticalRHS[i]);
            m_parserRHS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHS[i] = m_parserRHS[i].compile<s_nVar>();
        }
    }

    template <int n>
    amrex::Real curlcurl_plus_chargematrix_solve ()
    {
        // For studies other than convergence, this should be in SetUpTestSuite under Grid
        // parameters
        Gempic::Io::Parameters parameters{};
        amrex::ParmParse pp;
        const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
        pp.addarr("ComputationalDomain.nCell", nCell);

        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles
                  // randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Adding AMREX_SPACEDIM individual particles starts here
        ions.resize(1);
        for (int spec{0}; spec < 1; spec++)
        {
            ions[0] = std::make_shared<ParticleGroups<s_vdim>>(0, infra);
        }

        const int numParticles{AMREX_SPACEDIM * GEMPIC_D_MULT(n, n, n)};

        amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        // particles on edges or faces both works, although below implementation is with particles
        // on edges
        amrex::Vector<amrex::Real> offset[AMREX_SPACEDIM] = {
            AMREX_D_DECL({AMREX_D_DECL(0.5, 0.0, 0.0)}, {AMREX_D_DECL(0.0, 0.5, 0.0)},
                         {AMREX_D_DECL(0.0, 0.0, 0.5)})};

        int i{0}, j{0}, k{0};
        const auto dx{infra.cell_size_array()};

        for (int ndir = 0; ndir < AMREX_SPACEDIM; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                                for (k = 0; k < n; k++))

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> loc = {AMREX_D_DECL(
                    infra.geometry().ProbLo(xDir) + (i + offset[ndir][xDir]) * dx[xDir],
                    infra.geometry().ProbLo(yDir) + (j + offset[ndir][yDir]) * dx[yDir],
                    infra.geometry().ProbLo(zDir) + (k + offset[ndir][zDir]) * dx[zDir])};

                positions[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};
                velocities[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {0.0, 0.0, 0.0};
#if AMREX_SPACEDIM == 2
                weights[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    ((2.0) / AMREX_SPACEDIM) * infra.cell_volume();
#elif AMREX_SPACEDIM == 3
                weights[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    ((4.0) / AMREX_SPACEDIM) * infra.cell_volume();
#endif
            GEMPIC_D_LOOP_END
        }

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();
        // Adding AMREX_SPACEDIM individual particles ends here

        // Computed fields
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::edge> E(deRham);

        // Analytical fields
        DeRhamField<Grid::dual, Space::face> rhs(deRham, m_funcRHS);
        DeRhamField<Grid::primal, Space::edge> eAn(deRham, m_funcE);

        HypreQuasineutralLinearSystem<DeRhamField<Grid::dual, Space::face>,
                                      DeRhamField<Grid::primal, Space::edge>, s_hodgeDegree, s_vdim,
                                      s_degX, s_degY, s_degZ>
            hypreCurlcurlPlusFieldRho(infra, deRham);

        hypreCurlcurlPlusFieldRho.solve_curlcurl_plus_particle_charge_e(rhs, E, ions);

        E -= eAn;

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                             infra.geometry().InvCellSize(yDir),
                                                             infra.geometry().InvCellSize(zDir))};
        return Utils::gempic_norm (E.m_data[xDir], infra, 1) * dxi[xDir] +
               Utils::gempic_norm(E.m_data[yDir], infra, 1) * dxi[yDir] +
               Utils::gempic_norm(E.m_data[zDir], infra, 1) * dxi[zDir];
    }
};

// All 27 permutations of 1,2,3 work for spline degrees
using MyTypes = ::testing::Types<std::tuple<std::integral_constant<int, 1>,
                                            std::integral_constant<int, 1>,
                                            std::integral_constant<int, 1>>,
                                 std::tuple<std::integral_constant<int, 2>,
                                            std::integral_constant<int, 2>,
                                            std::integral_constant<int, 2>>,
                                 std::tuple<std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>>,
                                 std::tuple<std::integral_constant<int, 1>,
                                            std::integral_constant<int, 2>,
                                            std::integral_constant<int, 3>>>;

TYPED_TEST_SUITE(HypreQuasineutralCurlCurlPlusChargeMatrixTest, MyTypes);

TYPED_TEST(HypreQuasineutralCurlCurlPlusChargeMatrixTest, HypreQuasineutralCurlCurlPlusChargeMatrix)
{
    constexpr int coarse = 8;
    constexpr int fine = 16;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.1;

    amrex::Real rateOfConvergence;
    constexpr int splineDegreeX = TestFixture::s_degX;
    constexpr int splineDegreeY = TestFixture::s_degY;
    constexpr int splineDegreeZ = TestFixture::s_degZ;
    errorCoarse = this->template curlcurl_plus_chargematrix_solve<coarse>();
    amrex::Print() << "errorCoarse: " << errorCoarse << "\n";
    errorFine = this->template curlcurl_plus_chargematrix_solve<fine>();
    amrex::Print() << "errorFine: " << errorFine << "\n";
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, 2, tol);
}
} // namespace
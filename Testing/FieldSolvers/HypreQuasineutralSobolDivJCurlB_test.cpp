#include <chrono>
#include <random>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_Particles.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

/**
 * @brief Corrects the divergence of J for the quasineutral
 * solver, uses J to calculate B and verifies curlB = J.
 * The DivGradPhi matrix solver and the poisson solver
 * are used. The rho*gradV is deposited from particles.
 */
template <typename SplineDegreeStruct>
class HypreQuasineutralSobolDivJCurlBTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};

    static constexpr int s_degX{std::tuple_element_t<0, SplineDegreeStruct>::value};
    static constexpr int s_degY{std::tuple_element_t<1, SplineDegreeStruct>::value};
    static constexpr int s_degZ{std::tuple_element_t<2, SplineDegreeStruct>::value};

    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{2};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    HypreQuasineutralSobolDivJCurlBTest()
    {
        Gempic::Io::Parameters parameters;

        // Particle parameters (data read by particle_groups constructor)
        /*/ SINGLE SPECIES
        std::string speciesNames{"species0"};
        parameters.set("Particle.speciesNames", speciesNames);

        std::string samplerName{"Sobol"};
        parameters.set("Particle.sampler", samplerName);

        int nPartPerCell{2000};
        parameters.set("Particle.species0.nPartPerCell", nPartPerCell);

        std::string density{"1.0"};
        parameters.set("Particle.species0.density", density);

        int numGaussians{1};
        parameters.set("Particle.species0.numGaussians", numGaussians);

        // Gaussian parameters
        std::string vMeanx{"0*sin(x)"};
        parameters.set("Particle.species0.G0.vMean.x", vMeanx);
        std::string vMeany{"0*sin(y)"};
        parameters.set("Particle.species0.G0.vMean.y", vMeany);
        std::string vMeanz{"0.0"};
        parameters.set("Particle.species0.G0.vMean.z", vMeanz);

        amrex::Vector<amrex::Real> vThermal{{1.3, 1.3, 1.3}};
        parameters.set("Particle.species0.G0.vThermal", vThermal);
        amrex::Real vWeightG0{1.0};
        parameters.set("Particle.species0.G0.vWeight", vWeightG0);

        amrex::Real charge{-sqrt(3.0)};
        parameters.set("Particle.species0.charge", charge);

        amrex::Real mass{3.0};
        parameters.set("Particle.species0.mass", mass);//*/

        // DOUBLE SPECIES
        amrex::Vector<std::string> const speciesNames{"species0", "species1"};
        parameters.set("Particle.speciesNames", speciesNames);

        std::string samplerName{"Sobol"};
        parameters.set("Particle.sampler", samplerName);

        int nPartPerCell0{2000};
        parameters.set("Particle.species0.nPartPerCell", nPartPerCell0);

        std::string density0{"1.0"};
        parameters.set("Particle.species0.density", density0);

        int numGaussians0{1};
        parameters.set("Particle.species0.numGaussians", numGaussians0);

        // Gaussian parameters
        std::string vMeanx0{"0.0"};
        parameters.set("Particle.species0.G0.vMean.x", vMeanx0);
        std::string vMeany0{"0.0"};
        parameters.set("Particle.species0.G0.vMean.y", vMeany0);
        std::string vMeanz0{"0.0"};
        parameters.set("Particle.species0.G0.vMean.z", vMeanz0);

        amrex::Vector<amrex::Real> vThermal0{{1.3, 1.3, 1.3}};
        parameters.set("Particle.species0.G0.vThermal", vThermal0);

        amrex::Real vWeightG00{1.0};
        parameters.set("Particle.species0.G0.vWeight", vWeightG00);

        amrex::Real charge0{-sqrt(3.0)};
        parameters.set("Particle.species0.charge", charge0);

        amrex::Real mass0{3.0};
        parameters.set("Particle.species0.mass", mass0);

        int nPartPerCell1{3000};
        parameters.set("Particle.species1.nPartPerCell", nPartPerCell1);

        std::string density1{"1.0"};
        parameters.set("Particle.species1.density", density1);

        int numGaussians1{1};
        parameters.set("Particle.species1.numGaussians", numGaussians1);

        // Gaussian parameters
        std::string vMeanx1{"0*sin(x)"};
        parameters.set("Particle.species1.G0.vMean.x", vMeanx1);
        std::string vMeany1{"0*sin(y)"};
        parameters.set("Particle.species1.G0.vMean.y", vMeany1);
        std::string vMeanz1{"0.0"};
        parameters.set("Particle.species1.G0.vMean.z", vMeanz1);

        amrex::Vector<amrex::Real> vThermal1{{1.3, 1.3, 1.3}};
        parameters.set("Particle.species1.G0.vThermal", vThermal1);

        amrex::Real vWeightG01{1.0};
        parameters.set("Particle.species1.G0.vWeight", vWeightG01);

        amrex::Real charge1{sqrt(3.0)};
        parameters.set("Particle.species1.charge", charge1);

        amrex::Real mass1{2000.0};
        parameters.set("Particle.species1.mass", mass1); //*/

        std::string solver{"GMRES"};
        parameters.set("hypre.hypre_solver", solver);

        std::string prec{"euclid"};
        parameters.set("hypre.hypre_preconditioner", prec);

        int numkry{100};
        parameters.set("hypre.num_krylov", numkry);

        int maxitr{1000};
        parameters.set("hypre.max_iterations", maxitr);

        int euclidlevel{1};
        parameters.set("hypre.euclid_level", euclidlevel);

        int jacobi{0};
        parameters.set("hypre.euclid_use_block_jacobi", jacobi);
    }

    template <int n>
    void divj_correction_curlb_eq_j_solve (amrex::Real& divJFinalNorm,
                                           amrex::Real& divBNorm,
                                           amrex::Real& divANorm,
                                           amrex::Real& jMinusCurlHNorm)
    {
        // For studies other than convergence, this should be in SetUpTestSuite under Grid
        // parameters
        Gempic::Io::Parameters parameters{};

        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};

        // Initialize computational_domain
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        amrex::Real dt = 0.05;

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<s_vdim>>> ions;

        init_particles(ions, infra); // if adding large number of particles randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Computed fields related to correction of divJ
        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::dual, Space::cell> divJ(deRham);
        DeRhamField<Grid::primal, Space::node> phiCorr(deRham);
        DeRhamField<Grid::dual, Space::cell> rho(deRham);

        QuasineutralSolver<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ>
            hypreParticleDivJCurlB(infra, deRham);

        // Correction of divJ
        hypreParticleDivJCurlB.push_particles_and_correct_div_j(J, phiCorr, ions, dt);

        // Calculate final divJ
        div(divJ, J);
        divJFinalNorm = Utils::gempic_norm(divJ.m_data, infra, 2);

        // Correction of Javg
        hypreParticleDivJCurlB.push_particles_and_correct_javg(J, rho, ions);

        // Calculation of non-divergencefree A using divergence-free J begins here
        DeRhamField<Grid::dual, Space::face> A(deRham);
        DeRhamField<Grid::dual, Space::cell> divA(deRham);
        hypreParticleDivJCurlB.solve_negative_poisson_equation(J, A);

        div(divA, A);
        divANorm = Utils::gempic_norm(divA.m_data, infra, 2);
        // Calculation of non-divergencefree A using divergence-free J ends here

        // Computed fields related to calculation of B and error norms
        DeRhamField<Grid::primal, Space::edge> aPrimal(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::primal, Space::cell> divB(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::face> curlH(deRham);
        DeRhamField<Grid::dual, Space::face> jMinusCurlH(deRham);

        hodge(aPrimal, A, deRham->scaling_dto_e());
        curl(B, aPrimal);
        div(divB, B);
        divBNorm = Utils::gempic_norm(divB.m_data, infra, 2);
        hodge(H, B, deRham->scaling_bto_h());
        curl(curlH, H);

        linear_combination(jMinusCurlH, 1.0, J, -1.0, curlH);
        jMinusCurlHNorm = Utils::gempic_norm(jMinusCurlH.m_data[xDir], infra, 1) +
                          Utils::gempic_norm(jMinusCurlH.m_data[yDir], infra, 1) +
                          Utils::gempic_norm(jMinusCurlH.m_data[zDir], infra, 1);
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

TYPED_TEST_SUITE(HypreQuasineutralSobolDivJCurlBTest, MyTypes);

TYPED_TEST(HypreQuasineutralSobolDivJCurlBTest, HypreQuasineutralDivJCurlB)
{
    constexpr int meshSize = 6;

    amrex::Real divJFinalNorm, divBNorm, divANorm, jMinusCurlHNorm;
    amrex::Real tol = 2.0e-15;

    this->template divj_correction_curlb_eq_j_solve<meshSize>(divJFinalNorm, divBNorm, divANorm,
                                                              jMinusCurlHNorm);
    amrex::Print() << "Divergence of J after particle push: " << divJFinalNorm << "\n";
    amrex::Print() << "Divergence of B: " << divBNorm << "\n";
    amrex::Print() << "Divergence of A: " << divANorm << "\n";
    amrex::Print() << "J - CurlH error: " << jMinusCurlHNorm << "\n";

    ASSERT_LT(divJFinalNorm, tol);
    ASSERT_LT(divBNorm, tol);
    ASSERT_LT(divANorm, tol);
    ASSERT_LT(jMinusCurlHNorm, tol);
}

} // namespace

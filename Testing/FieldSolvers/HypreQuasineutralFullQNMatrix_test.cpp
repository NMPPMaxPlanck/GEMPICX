#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace FieldSolvers;

/**
 * @brief Tests the CurlCurl operator, Charge matrix and
 * the J x curl matrix of the quasineutral solver. The
 * rho*E and J x curl(E) are deposited from particles.
 */
template <typename SplineDegreeStruct>
class HypreQuasineutralFullQNMatrixTest : public testing::Test
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

    HypreQuasineutralFullQNMatrixTest()
    {
        const std::string analyticalRho = "4";
#if AMREX_SPACEDIM == 2
        const amrex::Array<std::string, 3> analyticalE = {"sin(y)", "sin(x+y)", "sin(x)"};
        const amrex::Array<std::string, 3> analyticalRHS = {
            "-sin(x+y)+(1+4)*sin(y) + (-0.75)*(sin(x)*cos(x+y) + sin(y-x))",
            "(1+4)*sin(x+y) + (-0.75)*sin(x+y)*(cos(y) - cos(x+y))",
            "(1+4)*sin(x) - (-0.75)*sin(x+y)*cos(x)"};
#elif AMREX_SPACEDIM == 3
        const amrex::Array<std::string, 3> analyticalE = {"sin(y)", "sin(z)", "sin(x)"};
        const amrex::Array<std::string, 3> analyticalRHS = {"(1+4)*sin(y) + (-0.75)*sin(y-x)",
                                                            "(1+4)*sin(z) + (-0.75)*sin(z-y)",
                                                            "(1+4)*sin(x) + (-0.75)*sin(x-z)"};
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

        Io::Parameters parameters;
        // Particle parameters (data read by particle_groups constructor)
        amrex::Real charge{-sqrt(2.5)};
        parameters.set("Particle.species0.charge", charge);

        amrex::Real mass{2.5};
        parameters.set("Particle.species0.mass", mass);
    }

    template <int n>
    amrex::Real curlcurl_plus_chargematrix_solve ()
    {
        // For studies other than convergence, this should be in SetUpTestSuite under Grid
        // parameters
        Gempic::Io::Parameters parameters{};
        const amrex::IntVect nCell{AMREX_D_DECL(n, n, n)};
        // Initialize computational_domain
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles
                  // randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Adding AMREX_SPACEDIM individual particles starts here
        ions.resize(1);
        ions[0] = std::make_shared<ParticleGroups<s_vdim>>(0, infra);

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
                amrex::Real rhoAn = 4.0;
                weights[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (rhoAn / AMREX_SPACEDIM) * GEMPIC_D_MULT(dx[xDir], dx[yDir], dx[zDir]);
                velocities[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
#if AMREX_SPACEDIM == 2
                    sin(loc[xDir] + loc[yDir]) / rhoAn,
                    sin(loc[xDir]) / rhoAn,
                    sin(loc[yDir]) / rhoAn,
#elif AMREX_SPACEDIM == 3
                    sin(loc[zDir]) / rhoAn,
                    sin(loc[xDir]) / rhoAn,
                    sin(loc[yDir]) / rhoAn,
#endif
                };
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

        HypreQuasineutralLinearSystem<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ>
            hypreCurlcurlPlusFieldRho(infra, deRham);

        amrex::Real jcrosscurlCoeff = -0.75;

        hypreCurlcurlPlusFieldRho.solve_curlcurl_plus_particle_charge_plus_jcrosscurl_e(
            rhs, E, ions, jcrosscurlCoeff);

        E -= eAn;

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                             infra.geometry().InvCellSize(yDir),
                                                             infra.geometry().InvCellSize(zDir))};

        return Utils::gempic_norm (E.m_data[xDir], infra, 1) * dxi[xDir] +
               Utils::gempic_norm(E.m_data[yDir], infra, 1) * dxi[yDir] +
               Utils::gempic_norm(E.m_data[zDir], infra, 1) * dxi[zDir];
    }
};

// Among all 27 possible permutations of 1,2,3 work for spline degrees, those with
// atleast two spline degrees equal to 1 give somewhat less than 2nd order convergence
// i.e. (1,1,1), (1,1,2), (1,2,1), (2,1,1), (1,1,3), (1,3,1), (3,1,1).
using MyTypes = ::testing::Types<std::tuple<std::integral_constant<int, 2>,
                                            std::integral_constant<int, 2>,
                                            std::integral_constant<int, 2>>,
                                 std::tuple<std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>>>;

TYPED_TEST_SUITE(HypreQuasineutralFullQNMatrixTest, MyTypes);

TYPED_TEST(HypreQuasineutralFullQNMatrixTest, HypreQuasineutralFullQNMatrix)
{
#if AMREX_SPACEDIM == 2
    constexpr int coarse = 8;
    constexpr int fine = 16;
#elif AMREX_SPACEDIM == 3
    constexpr int coarse = 6;
    constexpr int fine = 12;
#endif
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.2;

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
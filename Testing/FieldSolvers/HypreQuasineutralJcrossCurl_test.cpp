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
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

/**
 * @brief Tests the J x curl matrix of the quasineutral solver.
 * The J x curl(E) is deposited from particles.
 * The rho field is added to avoid singularity.
 */
template <typename SplineDegreeStruct>
class HypreQuasineutralJcrossCurlTest : public testing::Test
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

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcE;
    amrex::Array<amrex::Parser, 3> m_parserE;

    amrex::ParserExecutor<s_nVar> m_funcRho;
    amrex::Parser m_parserRho;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHS;
    amrex::Array<amrex::Parser, 3> m_parserRHS;

    HypreQuasineutralJcrossCurlTest()
    {
#if AMREX_SPACEDIM == 2
        amrex::Array<std::string, 3> const analyticalE = {"sin(y)", "sin(x+y)", "sin(x)"};
        std::string const analyticalRho = "4";
        amrex::Array<std::string, 3> const analyticalRHS = {
            "sin(x)*cos(x+y) + sin(y-x) + 4*sin(y)", "sin(x+y)*(cos(y) - cos(x+y)) + 4*sin(x+y)",
            "-sin(x+y)*cos(x) + 4*sin(x)"};
#elif AMREX_SPACEDIM == 3
        amrex::Array<std::string, 3> const analyticalE = {"sin(y)", "sin(z)", "sin(x)"};
        std::string const analyticalRho = "3";
        amrex::Array<std::string, 3> const analyticalRHS = {
            "sin(y-x) + 3*sin(y)", "sin(z-y) + 3*sin(z)", "sin(x-z) + 3*sin(x)"};
#endif

        for (int i = 0; i < 3; ++i)
        {
            m_parserE[i].define(analyticalE[i]);
            m_parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcE[i] = m_parserE[i].compile<s_nVar>();
        }

        m_parserRho.define(analyticalRho);
        m_parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRho = m_parserRho.compile<s_nVar>();

        for (int i = 0; i < 3; ++i)
        {
            m_parserRHS[i].define(analyticalRHS[i]);
            m_parserRHS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHS[i] = m_parserRHS[i].compile<s_nVar>();
        }

        // Particle parameters (data read by particle_groups constructor)
        Gempic::Io::Parameters parameters;
        amrex::Real charge{-1.5};
        parameters.set("Particle.species0.charge", charge);

        amrex::Real mass{2.25};
        parameters.set("Particle.species0.mass", mass);
    }

    template <int n>
    amrex::Real jcrosscurl_solve ()
    {
        Gempic::Io::Parameters parameters{};

        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
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

        int const numParticles{GEMPIC_D_MULT(n, n, n)};

        amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        int i{0}, j{0}, k{0};
        auto const dx{infra.cell_size_array()};

        GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                            for (k = 0; k < n; k++))

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> loc = {
                AMREX_D_DECL(infra.geometry().ProbLo(xDir) + (i + 0.5) * dx[xDir],
                             infra.geometry().ProbLo(yDir) + (j + 0.5) * dx[yDir],
                             infra.geometry().ProbLo(zDir) + (k + 0.5) * dx[zDir])};

            positions[i + j * n + k * n * n] = {AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};
#if AMREX_SPACEDIM == 2
            velocities[i + j * n + k * n * n] = {sin(loc[xDir] + loc[yDir]), sin(loc[xDir]),
                                                 sin(loc[yDir])};
#elif AMREX_SPACEDIM == 3
            velocities[i + j * n + k * n * n] = {sin(loc[zDir]), sin(loc[xDir]), sin(loc[yDir])};
#endif
            weights[i + j * n + k * n * n] = GEMPIC_D_MULT(dx[xDir], dx[yDir], dx[zDir]);
        GEMPIC_D_LOOP_END

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();
        // Adding AMREX_SPACEDIM individual particles ends here

        // Computed fields
        DeRhamField<Grid::primal, Space::edge> E(deRham);

        // Analytical fields
        DeRhamField<Grid::primal, Space::edge> eAn(deRham, m_funcE);
        DeRhamField<Grid::dual, Space::cell> rho(deRham, m_funcRho);
        DeRhamField<Grid::dual, Space::face> rhs(deRham, m_funcRHS);

        QuasineutralSolver<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ>
            hypreParticleJcrossCurl(infra, deRham);

        hypreParticleJcrossCurl.solve_jcrosscurl_plus_field_charge_e(rhs, E, ions, rho);

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

TYPED_TEST_SUITE(HypreQuasineutralJcrossCurlTest, MyTypes);

TYPED_TEST(HypreQuasineutralJcrossCurlTest, HypreQuasineutralJcrossCurl)
{
#if AMREX_SPACEDIM == 2
    constexpr int coarse = 12;
    constexpr int fine = 24;
#elif AMREX_SPACEDIM == 3
    constexpr int coarse = 8;
    constexpr int fine = 16;
#endif
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.15;

    amrex::Real rateOfConvergence;
    constexpr int splineDegreeX = TestFixture::s_degX;
    constexpr int splineDegreeY = TestFixture::s_degY;
    constexpr int splineDegreeZ = TestFixture::s_degZ;
    errorCoarse = this->template jcrosscurl_solve<coarse>();
    amrex::Print() << "errorCoarse: " << errorCoarse << "\n";
    errorFine = this->template jcrosscurl_solve<fine>();
    amrex::Print() << "errorFine: " << errorFine << "\n";
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, 2, tol);
}

} // namespace
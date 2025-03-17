#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;

/**
 * @brief Tests the deposit_j function using analytical function fields
 */
template <typename SplineDegreeStruct>
class DepositJTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};

    static constexpr int s_degX{SplineDegreeStruct::value};
    static constexpr int s_degY{SplineDegreeStruct::value};
    static constexpr int s_degZ{SplineDegreeStruct::value};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{2};

    static const int s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcJ;
    amrex::Array<amrex::Parser, 3> m_parserJ;

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

        amrex::Real charge{1.0};
        pp.add("Particle.ions.charge", charge);

        amrex::Real mass{1.0};
        pp.add("Particle.ions.mass", mass);

        // Gaussian parameters
        amrex::Vector<amrex::Real> vMean{{0.0, 0.0, 0.0}};
        pp.addarr("Particle.ions.G0.vMean", vMean);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        if constexpr (AMREX_SPACEDIM == 1)
        {
            GTEST_SKIP() << "This function works in 2D and 3D.";
        }
        else
        {
            amrex::Array<std::string, 3> analyticalJ;
            if constexpr (AMREX_SPACEDIM == 2)
            {
                analyticalJ = {"sin(x)", "sin(y)", "sin(x+y)"};
            }
            else if constexpr (AMREX_SPACEDIM == 3)
            {
                analyticalJ = {"sin(x)", "sin(y)", "sin(z)"};
            }

            for (int i = 0; i < 3; ++i)
            {
                m_parserJ[i].define(analyticalJ[i]);
                m_parserJ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcJ[i] = m_parserJ[i].compile<s_nVar>();
            }
        }
    }

    template <int n>
    amrex::Real j_solve ()
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

        // particles
        ions.resize(1);
        for (int spec{0}; spec < 1; spec++)
        {
            ions[0] = std::make_shared<ParticleGroups<s_vdim>>(0, infra);
        }

        const int numParticles{GEMPIC_D_MULT(n, n, n)};

        amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        int i{0}, j{0}, k{0};
        const auto &dx{infra.cell_size_array()};
        GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                            for (k = 0; k < n; k++))

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> loc = {
                AMREX_D_DECL(infra.geometry().ProbLo(xDir) + (i + 0.5) * dx[xDir],
                             infra.geometry().ProbLo(yDir) + (j + 0.5) * dx[yDir],
                             infra.geometry().ProbLo(zDir) + (k + 0.5) * dx[zDir])};

            positions[i + j * n + k * n * n] = {AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};
#if AMREX_SPACEDIM == 2
            velocities[i + j * n + k * n * n] = {sin(loc[xDir]), sin(loc[yDir]),
                                                 sin(loc[xDir] + loc[yDir])};
#elif AMREX_SPACEDIM == 3
            velocities[i + j * n + k * n * n] = {sin(loc[xDir]), sin(loc[yDir]), sin(loc[zDir])};
#endif
            weights[i + j * n + k * n * n] = infra.cell_volume();

        GEMPIC_D_LOOP_END

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();

        // Computed fields
        DeRhamField<Grid::dual, Space::face> J(deRham);

        // Analytical fields
        DeRhamField<Grid::dual, Space::face> jAn(deRham, m_funcJ);

        // Deposit initial charge
        for (auto &particleSpecies : ions)
        {
            amrex::Real charge = particleSpecies->get_charge();

            for (amrex::ParIter<0, 0, s_vdim + s_ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                 ++pti)
            {
                const long np = pti.numParticles();
                auto *const particles = pti.GetArrayOfStructs()().data();
                auto *const weight = pti.GetStructOfArrays().GetRealData(s_vdim).data();

                auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, s_vdim> jA;

                for (int cc = 0; cc < s_vdim; cc++)
                {
                    jA[cc] = (J.m_data[cc])[pti].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }

                        SplineBase<s_degX, s_degY, s_degZ> spline(positionParticle, plo,
                                                                  infra.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, s_vdim> vel{velx[pp], vely[pp], velz[pp]};
                        deposit_j(jA, spline, vel, charge * weight[pp]);
                    });
            }
        }
        J.post_particle_loop_sync();

        J -= jAn;

        amrex::GpuArray<amrex::Real, 3> dxi3d{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                               infra.geometry().InvCellSize(yDir),
                                                               infra.geometry().InvCellSize(zDir))};
        return (Utils::gempic_norm(J.m_data[xDir], infra, 1) * dxi3d[yDir] * dxi3d[zDir] +
                Utils::gempic_norm(J.m_data[yDir], infra, 1) * dxi3d[zDir] * dxi3d[xDir] +
                Utils::gempic_norm(J.m_data[zDir], infra, 1) * dxi3d[xDir] * dxi3d[yDir]);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 3>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 5>>;
TYPED_TEST_SUITE(DepositJTest, MyTypes);

TYPED_TEST(DepositJTest, SingleParticlePerCellJ)
{
    constexpr int coarse = 12;
    constexpr int fine = 24;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.1;

    amrex::Real rateOfConvergence;
    errorCoarse = this->template j_solve<coarse>();
    errorFine = this->template j_solve<fine>();
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    EXPECT_NEAR(rateOfConvergence, 2.0, tol);
}
} // namespace

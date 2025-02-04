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
 * @brief Tests the deposit_deldotS function using analytical function fields
 */
template <typename splineDegreeStruct>
class DepositDelDotSTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};

    static constexpr int s_degX{splineDegreeStruct::value};
    static constexpr int s_degY{splineDegreeStruct::value};
    static constexpr int s_degZ{splineDegreeStruct::value};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{2};

    static const int s_nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcJ;
    amrex::Array<amrex::Parser, 3> m_parserJ;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcDelDotS;
    amrex::Array<amrex::Parser, 3> m_parserDelDotS;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::ParmParse pp;  // Used instead of input file

        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        pp.addarr("domainLo", domainLo);

        const amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("domainHi", domainHi);

        // Grid parameters
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
        pp.addarr("maxGridSizeVector", maxGridSize);

        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        pp.addarr("isPeriodicVector", isPeriodic);

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
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        if constexpr (GEMPIC_SPACEDIM == 1)
        {
            GTEST_SKIP() << "This function works in 2D and 3D.";
        }
        else
        {
            amrex::Array<std::string, 3> analyticalDelDotS;
            if constexpr (GEMPIC_SPACEDIM == 2)
            {
                analyticalDelDotS = {"sin(x)*(2.0*cos(x) + cos(y))", "sin(y)*(cos(x) + 2.0*cos(y))",
                                     "sin(2.0*x+y) + sin(2.0*y+x)"};
            }
            else if constexpr (GEMPIC_SPACEDIM == 3)
            {
                analyticalDelDotS = {"sin(x)*(2.0*cos(x) + cos(y) + cos(z))",
                                     "sin(y)*(cos(x) + 2.0*cos(y) + cos(z))",
                                     "sin(z)*(cos(x) + cos(y) + 2.0*cos(z))"};
            }

            for (int i = 0; i < 3; ++i)
            {
                m_parserDelDotS[i].define(analyticalDelDotS[i]);
                m_parserDelDotS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcDelDotS[i] = m_parserDelDotS[i].compile<s_nVar>();
            }
        }
    }

    template <int n>
    amrex::Real del_dot_s_solve ()
    {
        // For studies other than convergence, this should be in SetUpTestSuite under Grid
        // parameters
        Gempic::Io::Parameters parameters{};
        const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
        parameters.set("nCellVector", nCell);

        // Initialize computational_domain
        ComputationalDomain infra;

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<s_vdim>>>
            ions;  // Use 'init_particles(infra, ions);' if adding large number of particles
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

        const int percelldir{1};

        const int percell{GEMPIC_D_MULT(percelldir, percelldir, percelldir)};
        const int numParticles{percell * GEMPIC_D_MULT(n, n, n)};

        amrex::Vector<amrex::Real> oss[percell];
        amrex::Real off;

        switch (s_maxSplineDegree)
        {
            case 3:
                off = 0.0;
                break;
            case 4:
                off = 0.5;
                break;
            case 5:
                off = 0.0;
                break;
        }

        GEMPIC_D_LOOP_BEGIN(for (int i = 0; i < percelldir; i++),
                            for (int j = 0; j < percelldir; j++),
                            for (int k = 0; k < percelldir; k++))
            oss[AMREX_D_TERM(i, +j * percelldir, +k * percelldir * percelldir)] = {AMREX_D_DECL(
                (i + off) / percelldir, (j + off) / percelldir, (k + off) / percelldir)};
        GEMPIC_D_LOOP_END

        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        int i{0}, j{0}, k{0};

        GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                            for (k = 0; k < n; k++))

            for (int p = 0; p < percell; p++)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> loc = {
                    AMREX_D_DECL(infra.m_geom.ProbLo(xDir) + (i + oss[p][0]) * infra.m_dx[xDir],
                                 infra.m_geom.ProbLo(yDir) + (j + oss[p][1]) * infra.m_dx[yDir],
                                 infra.m_geom.ProbLo(zDir) + (k + oss[p][2]) * infra.m_dx[zDir])};

                positions[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};

#if GEMPIC_SPACEDIM == 2
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    sin(loc[xDir]), sin(loc[yDir]), sin(loc[xDir] + loc[yDir])};
#elif GEMPIC_SPACEDIM == 3
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    sin(loc[xDir]), sin(loc[yDir]), sin(loc[zDir])};
#endif
                weights[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (1.0 / percell) *
                    GEMPIC_D_MULT(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir]);
            }
        GEMPIC_D_LOOP_END

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();

        // Computed fields
        DeRhamField<Grid::dual, Space::face> delDotS(deRham);

        // Analytical fields
        DeRhamField<Grid::dual, Space::face> delDotSAn(deRham, m_funcDelDotS);

        // Deposit initial charge
        for (auto &particleSpecies : ions)
        {
            amrex::Real mass = particleSpecies->get_mass();

            for (amrex::ParIter<0, 0, s_vdim + s_ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                 ++pti)
            {
                const long np = pti.numParticles();
                auto *const particles = pti.GetArrayOfStructs()().data();
                auto *const weight = pti.GetStructOfArrays().GetRealData(s_vdim).data();

                auto *const velx = pti.GetStructOfArrays().GetRealData(0).data();
                auto *const vely = pti.GetStructOfArrays().GetRealData(1).data();
                auto *const velz = pti.GetStructOfArrays().GetRealData(2).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, s_vdim> deldotsA;

                for (int cc = 0; cc < s_vdim; cc++)
                {
                    deldotsA[cc] = (delDotS.m_data[cc])[pti].array();
                }

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }

                        SplineWithFirstDerivative<s_degX, s_degY, s_degZ> splineDeriv(
                            positionParticle, infra.m_plo, infra.m_dxi, infra.m_dx);

                        amrex::GpuArray<amrex::Real, s_vdim> vel{velx[pp], vely[pp], velz[pp]};
                        deposit_deldot_s(deldotsA, splineDeriv, vel, mass * weight[pp]);
                    });
            }
        }
        delDotS.post_particle_loop_sync();

        delDotS -= delDotSAn;

        return (Utils::gempic_norm(delDotS.m_data[xDir], infra, 1) * infra.m_dxi[yDir] *
                    ((GEMPIC_SPACEDIM == 3) ? infra.m_dxi[zDir] : 1) +
                Utils::gempic_norm(delDotS.m_data[yDir], infra, 1) *
                    ((GEMPIC_SPACEDIM == 3) ? infra.m_dxi[zDir] : 1) * infra.m_dxi[xDir] +
                Utils::gempic_norm(delDotS.m_data[zDir], infra, 1) * infra.m_dxi[xDir] *
                    infra.m_dxi[yDir]);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 3>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 5>>;
TYPED_TEST_SUITE(DepositDelDotSTest, MyTypes);

TYPED_TEST(DepositDelDotSTest, SingleParticlePerCellDelDotS)
{
    constexpr int coarse = 20;
    constexpr int fine = 40;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.15;

    amrex::Real rateOfConvergence;
    errorCoarse = this->template del_dot_s_solve<coarse>();
    errorFine = this->template del_dot_s_solve<fine>();
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    EXPECT_NEAR(rateOfConvergence, 2.0, tol);
}
}  // namespace
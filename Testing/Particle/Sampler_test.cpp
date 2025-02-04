#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Sampler.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

using namespace Gempic;
using namespace Particle;

namespace
{
// Calculate 0th order velocity moment \int f(x,v) dx dv
template <unsigned int vDim>
amrex::Real compute_v_moment_0 (const ParticleGroups<vDim>* partGr)
{
    amrex::Real vMomentTmp = amrex::ReduceSum(
        *partGr,
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vDim + 1, 0>& p) -> amrex::Real
        {
            auto w = p.rdata(vDim);  // particle weight
            return (w);
        });
    // reduce sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMomentTmp,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
    return vMomentTmp;
}

// Calculate 1th order velocity moment \int v f(x,v) dx dv
template <unsigned int vDim>
void compute_v_moments_1 (const ParticleGroups<vDim>* partGr,
                          amrex::GpuArray<amrex::Real, vDim>& vMoments)
{
    for (int cmp = 0; cmp < vDim; cmp++)
    {
        // reduce sum over one MPI rank
        amrex::Real vMomentTmp = amrex::ReduceSum(
            *partGr,
            [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vDim + 1, 0>& p) -> amrex::Real
            {
                auto w = p.rdata(vDim);   // particle weight
                auto vel = p.rdata(cmp);  // velocity component
                return (w * vel);
            });
        // reduced sum over MPI ranks
        amrex::ParallelDescriptor::ReduceRealSum(vMomentTmp,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());
        vMoments[cmp] = vMomentTmp;
    }
}

// Calculate 2nd order velocity moment \int v^2 f(x,v) dx dv
template <unsigned int vDim>
amrex::Real compute_v_moment_2 (const ParticleGroups<vDim>* partGr)
{
    // reduce sum over one MPI rank
    amrex::Real vMomentTmp = amrex::ReduceSum(
        *partGr,
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vDim + 1, 0>& p) -> amrex::Real
        {
            auto w = p.rdata(vDim);  // particle weight
            amrex::Real v2{0};
            for (int cmp{0}; cmp < vDim; ++cmp)
            {
                v2 += p.rdata(cmp) * p.rdata(cmp);
            }
            return (w * v2);
        });
    // reduced sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMomentTmp,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
    return vMomentTmp;
}

void add_particle_parameters (const std::string& sampler)
{
    amrex::ParmParse pp;
    pp.add("Particle.speciesNames", sampler);
    const std::string particleInputScope{"Particle." + sampler};
    amrex::ParmParse ppparticle(particleInputScope);
    ppparticle.add("sampler", sampler);
    const int nPartPerCell = 1000;
    ppparticle.add("nPartPerCell", nPartPerCell);
    const amrex::Real charge = 1.0;
    ppparticle.add("charge", charge);
    const amrex::Real mass = 1.0;
    ppparticle.add("mass", mass);
    const std::string density = "1 + 0.5 * sin(kvarx*x + kvary*y + kvarz*z)";
    ppparticle.add("density", density);
    const int numGaussians = 2;
    ppparticle.add("numGaussians", numGaussians);
    const amrex::Vector<amrex::Real> g0vMean = {0.0, 0.0, 0.0};
    ppparticle.addarr("G0.vMean", g0vMean);
    const amrex::GpuArray<std::string, 3> g0vThermal = {"2.0 + 0.0 * sin(kvarx*x)", "2.0", "2.0"};
    ppparticle.add("G0.vThermal.x", g0vThermal[xDir]);
    ppparticle.add("G0.vThermal.y", g0vThermal[yDir]);
    ppparticle.add("G0.vThermal.z", g0vThermal[zDir]);
    const amrex::Real g0vWeight = 0.75;
    ppparticle.add("G0.vWeight", g0vWeight);
    const amrex::Vector<amrex::Real> g1vMean = {2.0, 2.0, 2.0};
    ppparticle.addarr("G1.vMean", g1vMean);
    const amrex::Vector<amrex::Real> g1vThermal = {1.0, 1.0, 1.0};
    ppparticle.addarr("G1.vThermal", g1vThermal);
    const amrex::Real g1vWeight = 0.25;
    ppparticle.add("G1.vWeight", g1vWeight);
}

struct ParticleInputCellwise
{
    inline constexpr static std::string_view s_sampler{"Cellwise"};
};

struct ParticleInputPseudorandom
{
    inline constexpr static std::string_view s_sampler{"PseudoRandom"};
};

struct ParticleInputSobol
{
    inline constexpr static std::string_view s_sampler{"Sobol"};
};

struct ParticleInputGpu
{
    inline constexpr static std::string_view s_sampler{"FullDomainGpu"};
};

template <typename particleInput>
class SamplerTest : public testing::Test
{
protected:
    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;
    amrex::Real m_tol{1e-1};

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(4, 4, 4)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(32, 8, 8)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.nCell", nCell);
        pp.addarr("k", k);
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);

        // Special case by case parameters
        add_particle_parameters(std::string(particleInput::s_sampler));
    }

    void SetUp () override
    {
        // This method is not as accurate
        if (particleInput::s_sampler == "Cellwise")
        {
            m_tol = 3e-1;
        }
    }
};

struct NameGenerator
{
    template <typename T>
    static std::string GetName (int)
    {
        return std::string(T::s_sampler);
    }
};

#ifdef AMREX_USE_GPU
using ParameterInputs = ::testing::
    Types<ParticleInputCellwise, ParticleInputPseudorandom, ParticleInputSobol, ParticleInputGpu>;
#else
using ParameterInputs =
    ::testing::Types<ParticleInputCellwise, ParticleInputPseudorandom, ParticleInputSobol>;
#endif

TYPED_TEST_SUITE(SamplerTest, ParameterInputs, NameGenerator);

TYPED_TEST(SamplerTest, CompareMoments)
{
    using namespace Gempic::Particle::Impl;

    // Initialize Particle Groups
    constexpr int vDim{3};
    std::vector<std::shared_ptr<ParticleGroups<vDim>>> partGr;
    init_particles(partGr, this->m_infra);

    Io::Parameters params("Particle." + partGr[0]->get_name());
    VelocityInitializer<vDim> vInit(params);

    amrex::GpuArray<amrex::Real, vDim> mom1;
    amrex::Real mom2 = 0;

    // assuming constant vThermal functions (if they even exist);
    const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> location{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Real vWeigth;
    for (int i = 0; i < vDim; i++)
    {
        mom1[i] = 0;
        for (int j = 0; j < vInit.m_numGauss; j++)
        {
            params.get("G" + std::to_string(j) + ".vWeight", vWeigth);
            amrex::Real vMean = vInit.m_vMeanPtr[j][i];
            // Extracting vDev is not a use case outside testing, so here we creatively run
            // vThermal = (0.5 * vThermal + vMean) - (-0.5 * vThermal + vMean)
            amrex::Real vThermal = get_gaussian_velocity(vInit, j, 0.5, i, location) -
                                   get_gaussian_velocity(vInit, j, -0.5, i, location);
            mom1[i] += vWeigth * vMean;
            mom2 += vWeigth * (vThermal * vThermal + vMean * vMean);
        }
    }

    amrex::Real tol{this->m_tol};

    EXPECT_NEAR(compute_v_moment_0(partGr[0].get()), 1.0, tol)
        << "0th order velocity moment false!";

    amrex::GpuArray<amrex::Real, vDim> vMoments1;
    compute_v_moments_1(partGr[0].get(), vMoments1);
    EXPECT_NEAR(vMoments1[xDir], mom1[xDir], tol) << "1st order velocity moment x-dir false!";
    EXPECT_NEAR(vMoments1[yDir], mom1[yDir], tol) << "1st order velocity moment y-dir false!";
    EXPECT_NEAR(vMoments1[zDir], mom1[zDir], tol) << "1st order velocity moment z-dir false!";

    EXPECT_NEAR(compute_v_moment_2(partGr[0].get()), mom2, tol)
        << "2nd order velocity moment false!";
}
}  // namespace

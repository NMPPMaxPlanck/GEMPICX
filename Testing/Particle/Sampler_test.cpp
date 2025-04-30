#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Sampler.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

using namespace Gempic;
using namespace Particle;

namespace
{
// Calculate 0th order velocity moment \int f(x,v) dx dv
template <unsigned int vDim>
amrex::Real compute_v_moment_0 (ParticleGroups<vDim> const* partGr)
{
    amrex::Real vMomentTmp = amrex::ReduceSum(
        *partGr,
        [=] AMREX_GPU_HOST_DEVICE(amrex::Particle<vDim + 1, 0> const& p) -> amrex::Real
        {
            auto w = p.rdata(vDim); // particle weight
            return (w);
        });
    // reduce sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMomentTmp,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
    return vMomentTmp;
}

// Calculate 1th order velocity moment \int v f(x,v) dx dv
template <unsigned int vDim>
void compute_v_moments_1 (ParticleGroups<vDim> const* partGr,
                          amrex::GpuArray<amrex::Real, vDim>& vMoments)
{
    for (int cmp = 0; cmp < vDim; cmp++)
    {
        // reduce sum over one MPI rank
        amrex::Real vMomentTmp = amrex::ReduceSum(
            *partGr,
            [=] AMREX_GPU_HOST_DEVICE(amrex::Particle<vDim + 1, 0> const& p) -> amrex::Real
            {
                auto w = p.rdata(vDim);  // particle weight
                auto vel = p.rdata(cmp); // velocity component
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
amrex::Real compute_v_moment_2 (ParticleGroups<vDim> const* partGr)
{
    // reduce sum over one MPI rank
    amrex::Real vMomentTmp = amrex::ReduceSum(
        *partGr,
        [=] AMREX_GPU_HOST_DEVICE(amrex::Particle<vDim + 1, 0> const& p) -> amrex::Real
        {
            auto w = p.rdata(vDim); // particle weight
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

void add_particle_parameters (std::string const& sampler)
{
    Gempic::Io::Parameters parameters{};
    parameters.set("Particle.speciesNames", sampler);
    std::string const particleInputScope{"Particle." + sampler};
    Gempic::Io::Parameters partparams{particleInputScope};
    partparams.set("sampler", sampler);
    int const nPartPerCell = 1000;
    partparams.set("nPartPerCell", nPartPerCell);
    amrex::Real const charge = 1.0;
    partparams.set("charge", charge);
    amrex::Real const mass = 1.0;
    partparams.set("mass", mass);
    std::string const density = "1 + 0.5 * sin(kvarx*x + kvary*y + kvarz*z)";
    partparams.set("density", density);
    int const numGaussians = 2;
    partparams.set("numGaussians", numGaussians);
    amrex::Vector<amrex::Real> const g0vMean = {0.0, 0.0, 0.0};
    partparams.set("G0.vMean", g0vMean);
    amrex::GpuArray<std::string, 3> const g0vThermal = {"2.0 + 0.0 * sin(kvarx*x)", "2.0", "2.0"};
    partparams.set("G0.vThermal.x", g0vThermal[xDir]);
    partparams.set("G0.vThermal.y", g0vThermal[yDir]);
    partparams.set("G0.vThermal.z", g0vThermal[zDir]);
    amrex::Real const g0vWeight = 0.75;
    partparams.set("G0.vWeight", g0vWeight);
    amrex::Vector<amrex::Real> const g1vMean = {2.0, 2.0, 2.0};
    partparams.set("G1.vMean", g1vMean);
    amrex::Vector<amrex::Real> const g1vThermal = {1.0, 1.0, 1.0};
    partparams.set("G1.vThermal", g1vThermal);
    amrex::Real const g1vWeight = 0.25;
    partparams.set("G1.vWeight", g1vWeight);
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

template <typename ParticleInput>
class SamplerTest : public testing::Test
{
protected:
    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;
    amrex::Real m_tol{1e-1};

    SamplerTest() : m_infra{Gempic::Test::Utils::get_default_compdom()}
    {
        // 2pi(domainHi - domainLo) = k
        std::array<amrex::Real, AMREX_SPACEDIM> domainSize = {
            AMREX_D_DECL(m_infra.geometry().ProbLength(xDir), m_infra.geometry().ProbLength(yDir),
                         m_infra.geometry().ProbLength(zDir))};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(
            2 * M_PI * domainSize[xDir], 2 * M_PI * domainSize[yDir], 2 * M_PI * domainSize[zDir])};
        m_parameters.set("k", k);
        // Special case by case parameters
        add_particle_parameters(std::string(ParticleInput::s_sampler));
        // This method is not as accurate
        if (ParticleInput::s_sampler == "Cellwise")
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

    double volumeFactor = this->m_infra.geometry().ProbSize();
    // assuming constant vThermal functions (if they even exist);
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const location{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Real vWeight;
    for (int i = 0; i < vDim; i++)
    {
        mom1[i] = 0;
        for (int j = 0; j < vInit.m_numGauss; j++)
        {
            params.get("G" + std::to_string(j) + ".vWeight", vWeight);
            amrex::Real vMean = vInit.m_vMeanPtr[j][i];
            // Extracting vDev is not a use case outside testing, so here we creatively run
            // vThermal = (0.5 * vThermal + vMean) - (-0.5 * vThermal + vMean)
            amrex::Real vThermal = get_gaussian_velocity(vInit, j, 0.5, i, location) -
                                   get_gaussian_velocity(vInit, j, -0.5, i, location);
            mom1[i] += vWeight * vMean * volumeFactor;
            mom2 += vWeight * (vThermal * vThermal + vMean * vMean) * volumeFactor;
        }
    }

    amrex::Real tol{this->m_tol * volumeFactor};

    EXPECT_NEAR(compute_v_moment_0(partGr[0].get()), volumeFactor, tol)
        << "0th order velocity moment false!";

    amrex::GpuArray<amrex::Real, vDim> vMoments1;
    compute_v_moments_1(partGr[0].get(), vMoments1);
    EXPECT_NEAR(vMoments1[xDir], mom1[xDir], tol) << "1st order velocity moment x-dir false!";
    EXPECT_NEAR(vMoments1[yDir], mom1[yDir], tol) << "1st order velocity moment y-dir false!";
    EXPECT_NEAR(vMoments1[zDir], mom1[zDir], tol) << "1st order velocity moment z-dir false!";

    EXPECT_NEAR(compute_v_moment_2(partGr[0].get()), mom2, tol)
        << "2nd order velocity moment false!";
}
} // namespace

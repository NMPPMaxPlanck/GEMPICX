/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
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
amrex::Real compute_v_moment_0 (ParticleSpecies<vDim> const* particles)
{
    auto const indices = particles->get_data_indices();
    using PTDType = typename ParticleSpecies<vDim>::ParticleTileType::ConstParticleTileDataType;
    amrex::Real vMomentTmp =
        amrex::ReduceSum(*particles, 0, particles->finestLevel(),
                         [=] AMREX_GPU_HOST_DEVICE(PTDType const& ptd, int const pp) -> amrex::Real
                         {
                             return ptd.rdata(indices.m_iweight)[pp]; // particle weight
                         });
    // reduce sum over MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(vMomentTmp,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
    return vMomentTmp;
}

// Calculate 1th order velocity moment \int v f(x,v) dx dv
template <unsigned int vDim>
void compute_v_moments_1 (ParticleSpecies<vDim> const* particles,
                          amrex::GpuArray<amrex::Real, vDim>& vMoments)
{
    using PTDType = typename ParticleSpecies<vDim>::ParticleTileType::ConstParticleTileDataType;
    auto const indices = particles->get_data_indices();
    auto const idxw = indices.m_iweight;
    auto const idxv = indices.m_ivelx;
    for (int cmp = 0; cmp < vDim; cmp++)
    {
        // reduce sum over one MPI rank
        amrex::Real vMomentTmp = amrex::ReduceSum(
            *particles, 0, particles->finestLevel(),
            [=] AMREX_GPU_HOST_DEVICE(PTDType const& ptd, int const pp) -> amrex::Real
            {
                auto const w = ptd.rdata(idxw)[pp];         // weight
                auto const vel = ptd.rdata(idxv + cmp)[pp]; // velocity component
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
amrex::Real compute_v_moment_2 (ParticleSpecies<vDim> const* particles)
{
    // reduce sum over one MPI rank
    using PTDType = typename ParticleSpecies<vDim>::ParticleTileType::ConstParticleTileDataType;
    auto const indices = particles->get_data_indices();
    auto const idxw = indices.m_iweight;
    auto const idxv = indices.m_ivelx;
    amrex::Real vMomentTmp =
        amrex::ReduceSum(*particles, 0, particles->finestLevel(),
                         [=] AMREX_GPU_HOST_DEVICE(PTDType const& ptd, int const pp) -> amrex::Real
                         {
                             auto const w = ptd.rdata(idxw)[pp]; // particle weight
                             amrex::Real v2{0};
                             for (int cmp{0}; cmp < vDim; ++cmp)
                             {
                                 v2 += ptd.rdata(idxv + cmp)[pp] * ptd.rdata(idxv + cmp)[pp];
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
    if (sampler == "PseudoRandom" || sampler == "PseudoRandomBox" || sampler == "PseudoRandomGPU" ||
        sampler == "Sobol")
    {
        Gempic::Io::Parameters parameters{};
        std::string const speciesName = "species0";
        parameters.set("Particle.speciesNames", speciesName);
        std::string const particleInputScope{"Particle." + speciesName};
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
        amrex::GpuArray<std::string, 3> const g0vThermal = {"2.0 + 0.0 * sin(kvarx*x)", "2.0",
                                                            "2.0"};
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
    else if (sampler == "Rejection" || sampler == "RejectionSobol")
    {
        Gempic::Io::Parameters parameters{};
        // dummy particle species to calculate the exact moments
        std::vector<std::string> const speciesNames = {"dummy", "real"};
        parameters.set("Particle.speciesNames", speciesNames);
        std::string const particleInputScopeDummy{"Particle.dummy"};
        Gempic::Io::Parameters partparamsDummy{particleInputScopeDummy};
        std::string const dummySampler = "PseudoRandom";
        partparamsDummy.set("sampler", dummySampler);
        int const nPartPerCellDummy = 0;
        partparamsDummy.set("nPartPerCell", nPartPerCellDummy);
        amrex::Real const charge = 1.0;
        partparamsDummy.set("charge", charge);
        amrex::Real const mass = 1.0;
        partparamsDummy.set("mass", mass);
        std::string const densityDummy = "1 + 0.5 * sin(kvarx*x + kvary*y + kvarz*z)";
        partparamsDummy.set("density", densityDummy);
        int const numGaussiansDummy = 2;
        partparamsDummy.set("numGaussians", numGaussiansDummy);
        amrex::Vector<amrex::Real> const g0vMeanDummy = {0.0, 0.0, 0.0};
        partparamsDummy.set("G0.vMean", g0vMeanDummy);
        amrex::GpuArray<std::string, 3> const g0vThermalDummy = {"2.0 + 0.0 * sin(kvarx*x)", "2.0",
                                                                 "2.0"};
        partparamsDummy.set("G0.vThermal.x", g0vThermalDummy[xDir]);
        partparamsDummy.set("G0.vThermal.y", g0vThermalDummy[yDir]);
        partparamsDummy.set("G0.vThermal.z", g0vThermalDummy[zDir]);
        amrex::Real const g0vWeightDummy = 0.75;
        partparamsDummy.set("G0.vWeight", g0vWeightDummy);
        amrex::Vector<amrex::Real> const g1vMean = {2.0, 2.0, 2.0};
        partparamsDummy.set("G1.vMean", g1vMean);
        amrex::Vector<amrex::Real> const g1vThermal = {1.0, 1.0, 1.0};
        partparamsDummy.set("G1.vThermal", g1vThermal);
        amrex::Real const g1vWeight = 0.25;
        partparamsDummy.set("G1.vWeight", g1vWeight);

        // Actual particle species for the rejection sampler
        // Trying to replicate the same distribution by rejection sampling
        std::string const particleInputScope{"Particle.real"};
        Gempic::Io::Parameters partparams{particleInputScope};
        partparams.set("sampler", sampler);
        int const nPartPerCell = 1000;
        partparams.set("nPartPerCell", nPartPerCell);
        partparams.set("charge", charge);
        partparams.set("mass", mass);
        std::string const density = "1";
        partparams.set("density", density);
        int const numGaussians = 1;
        partparams.set("numGaussians", numGaussians);
        amrex::Vector<amrex::Real> const g0vMean = {0.0, 0.0, 0.0};
        partparams.set("G0.vMean", g0vMean);
        amrex::Vector<amrex::Real> const g0vThermal = {2.0, 2.0, 2.0};
        partparams.set("G0.vThermal", g0vThermal);
        amrex::Real const g0vWeight = 1.0;
        partparams.set("G0.vWeight", g0vWeight);
        int const nMaxRejections = 1000;
        partparams.set("nMaxRejections", nMaxRejections);
        amrex::Real const rejectionScaling = 25;
        partparams.set("rejectionScaling", rejectionScaling);
        std::string const rejectionDistribution =
            "(1 + 0.5 * sin(kvarx*x + kvary*y + kvarz*z)) * (0.75 / (sqrt(2*3.14159265*4)^3) * "
            "exp(-(vx^2 + vy^2 + vz^2)/(2*4)) + 0.25 / (sqrt(2*3.14159265)^3) * exp(-((vx-2)^2 + "
            "(vy-2)^2 + (vz-2)^2)/(2)))";
        partparams.set("rejectionDistribution", rejectionDistribution);
    }
}

struct ParticleInputPseudorandom
{
    inline constexpr static std::string_view s_sampler{"PseudoRandom"};
};

struct ParticleInputPseudorandomBox
{
    inline constexpr static std::string_view s_sampler{"PseudoRandomBox"};
};

struct ParticleInputPseudorandomGpu
{
    inline constexpr static std::string_view s_sampler{"PseudoRandomGPU"};
};

struct ParticleInputSobol
{
    inline constexpr static std::string_view s_sampler{"Sobol"};
};

struct ParticleInputRejection
{
    inline constexpr static std::string_view s_sampler{"Rejection"};
};

struct ParticleInputRejectionSobol
{
    inline constexpr static std::string_view s_sampler{"RejectionSobol"};
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

using ParameterInputs = ::testing::Types<ParticleInputPseudorandom,
                                         ParticleInputPseudorandomBox,
                                         ParticleInputPseudorandomGpu,
                                         ParticleInputSobol,
                                         ParticleInputRejection,
                                         ParticleInputRejectionSobol>;

TYPED_TEST_SUITE(SamplerTest, ParameterInputs, NameGenerator);

TYPED_TEST(SamplerTest, CompareMoments)
{
    using namespace Gempic::Particle::Impl;

    // Initialize particles
    constexpr int vDim{3};
    std::vector<std::shared_ptr<ParticleSpecies<vDim>>> particles;
    init_particles(particles, this->m_infra);

    amrex::Vector<int> const writeRealComp(vDim + 1, 1);
    amrex::Vector<int> const& writeIntComp = {};
    amrex::Vector<std::string> const& intCompNames = {};
    amrex::Vector<std::string> const& realCompNames = {"vx", "vy", "vz", "weight"};

    testing::TestInfo const* const testInfo = testing::UnitTest::GetInstance()->current_test_info();
    amrex::ignore_unused(testInfo);
#ifndef NDEBUG
    GEMPIC_DEBUG("Writing particles from test " + testInfo->name() + " of test suite " +
                 testInfo->test_suite_name() + ".");
    for (int gg = 0; gg < particles.size(); gg++)
    {
        particles[gg]->WritePlotFile(
            ("particle_test_" + std::string(testInfo->name()) + "_group" + std::to_string(gg)),
            "particles", writeRealComp, writeIntComp, realCompNames, intCompNames);
    }
#endif

    // in case of rejection sampling, use first particle species (dummy) to compute analytic moments
    // of distribution function
    Io::Parameters params("Particle." + particles[0]->get_name());
    VelocityInitializer<vDim> vInit(params);

    std::array<amrex::Real, vDim> mom1;
    amrex::Real mom2 = 0;

    double volumeFactor = this->m_infra.geometry().ProbSize();
    // assuming constant vThermal functions (if they even exist);
    std::array<amrex::Real, AMREX_SPACEDIM> const location{AMREX_D_DECL(0.0, 0.0, 0.0)};
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

    // In case of rejection sampling, use second/last particle species (real) to compute actual
    // moments of sampled particles
    EXPECT_NEAR(compute_v_moment_0(particles.back().get()), volumeFactor, tol)
        << "0th order velocity moment false!";

    amrex::GpuArray<amrex::Real, vDim> vMoments1;
    compute_v_moments_1(particles.back().get(), vMoments1);
    EXPECT_NEAR(vMoments1[xDir], mom1[xDir], tol) << "1st order velocity moment x-dir false!";
    EXPECT_NEAR(vMoments1[yDir], mom1[yDir], tol) << "1st order velocity moment y-dir false!";
    EXPECT_NEAR(vMoments1[zDir], mom1[zDir], tol) << "1st order velocity moment z-dir false!";

    EXPECT_NEAR(compute_v_moment_2(particles.back().get()), mom2, tol)
        << "2nd order velocity moment false!";
}

class SamplerTestMultiLevel : public testing::Test
{
protected:
    Io::Parameters m_parameters{};
    amrex::Vector<amrex::Geometry> m_geom;
    amrex::Vector<amrex::BoxArray> m_ba;
    amrex::Vector<amrex::DistributionMapping> m_dm;
    amrex::Real m_tol{1e-1};

    SamplerTestMultiLevel()
    {
        Gempic::Test::Utils::init_multilevel_domain(m_geom, m_ba, m_dm);
        // 2pi(domainHi - domainLo) = k
        std::array<amrex::Real, AMREX_SPACEDIM> domainSize = {AMREX_D_DECL(
            m_geom[0].ProbLength(xDir), m_geom[0].ProbLength(yDir), m_geom[0].ProbLength(zDir))};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(
            2 * M_PI * domainSize[xDir], 2 * M_PI * domainSize[yDir], 2 * M_PI * domainSize[zDir])};
        m_parameters.set("k", k);

        // Special case by case parameters
        add_particle_parameters("PseudoRandom");
    }
};

TEST_F(SamplerTestMultiLevel, CompareMoments)
{
    using namespace Gempic::Particle::Impl;

    // Initialize particles
    constexpr int vDim{3};
    amrex::Vector<int> rr{2, 1};
    auto particles = std::make_shared<Particle::ParticleSpecies<vDim, 1>>(
        "species0", this->m_geom, this->m_dm, this->m_ba, rr);
    EXPECT_EQ(particles->finestLevel(), 1) << "ParticleSpecies does not have two levels!";

    Particle::Impl::sample_species(particles, this->m_geom[0], 0, "PseudoRandom", 123);
    // sampling done on level 0 needs to be followed by redistribute
    EXPECT_EQ(particles->NumberOfParticlesAtLevel(1), 0)
        << "Particles on fine level before Redistribute!";
    particles->Redistribute();
    EXPECT_GT(particles->NumberOfParticlesAtLevel(1), 0)
        << "No particles on fine level after Redistribute!";

    amrex::Vector<int> const writeRealComp(vDim + 1, 1);
    amrex::Vector<int> const& writeIntComp = {};
    amrex::Vector<std::string> const& intCompNames = {};
    amrex::Vector<std::string> const& realCompNames = {"vx", "vy", "vz", "weight"};

    Io::Parameters params("Particle." + particles->get_name());
    VelocityInitializer<vDim> vInit(params);

    std::array<amrex::Real, vDim> mom1;
    amrex::Real mom2 = 0;

    double volumeFactor = this->m_geom[0].ProbSize();
    // assuming constant vThermal functions (if they even exist);
    std::array<amrex::Real, AMREX_SPACEDIM> const location{AMREX_D_DECL(0.0, 0.0, 0.0)};
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

    EXPECT_NEAR(compute_v_moment_0(particles.get()), volumeFactor, tol)
        << "0th order velocity moment false!";

    amrex::GpuArray<amrex::Real, vDim> vMoments1;
    compute_v_moments_1(particles.get(), vMoments1);
    EXPECT_NEAR(vMoments1[xDir], mom1[xDir], tol) << "1st order velocity moment x-dir false!";
    EXPECT_NEAR(vMoments1[yDir], mom1[yDir], tol) << "1st order velocity moment y-dir false!";
    EXPECT_NEAR(vMoments1[zDir], mom1[zDir], tol) << "1st order velocity moment z-dir false!";

    EXPECT_NEAR(compute_v_moment_2(particles.get()), mom2, tol)
        << "2nd order velocity moment false!";
}

class RelativisticSamplerTest : public testing::Test
{
protected:
    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;
    amrex::Real m_tol{1e-1};

    RelativisticSamplerTest() : m_infra{Gempic::Test::Utils::get_default_compdom()}
    {
        std::string const speciesName = "species0";
        m_parameters.set("Particle.speciesNames", speciesName);
        amrex::Real c = 1;
        m_parameters.set("c", c);
        std::string const particleInputScope{"Particle." + speciesName};
        Gempic::Io::Parameters partparams{particleInputScope};
        std::string sampler = "Juettner";
        partparams.set("sampler", sampler);
        int const nPartPerCell = 1000;
        partparams.set("nPartPerCell", nPartPerCell);
        amrex::Real const charge = 1.0;
        partparams.set("charge", charge);
        amrex::Real const mass = 1.0;
        partparams.set("mass", mass);
        std::string const density = "1";
        partparams.set("density", density);
        int const numGaussians = 1;
        partparams.set("numGaussians", numGaussians);
        amrex::Vector<amrex::Real> const vMean = {0.5, 0.0, 0.0};
        partparams.set("G0.vMean", vMean);
        amrex::Vector<amrex::Real> const vThermal = {1.1, 1.1, 1.1};
        partparams.set("G0.vThermal", vThermal);
        amrex::Real const vWeight = 1;
        partparams.set("G0.vWeight", vWeight);
    }
};

TEST_F(RelativisticSamplerTest, CompareMoments)
{
    // Initialize particles
    constexpr int vDim{3};
    std::vector<std::shared_ptr<ParticleSpecies<vDim>>> particles;
    init_particles(particles, this->m_infra);

    std::array<amrex::Real, vDim> mom1;
    amrex::Real mom2 = 0;

    double volumeFactor = this->m_infra.geometry().ProbSize();
    amrex::Real tol{this->m_tol * volumeFactor};

    // In case of rejection sampling, use second/last particle species (real) to compute actual
    // moments of sampled particles
    EXPECT_NEAR(compute_v_moment_0(particles.back().get()), volumeFactor, tol)
        << "0th order velocity moment false!";

    // moments are obtained from a numerical integration of the analytical distribution function
    // in the python file RelativisticDistribution.py
    amrex::GpuArray<amrex::Real, vDim> vMoments1;
    compute_v_moments_1(particles.back().get(), vMoments1);
    EXPECT_NEAR(vMoments1[xDir], volumeFactor * 2.9816739396049625, tol)
        << "1st order velocity moment x-dir false!";
    EXPECT_NEAR(vMoments1[yDir], 0, tol) << "1st order velocity moment y-dir false!";
    EXPECT_NEAR(vMoments1[zDir], 0, tol) << "1st order velocity moment z-dir false!";

    EXPECT_NEAR(compute_v_moment_2(particles.back().get()),
                volumeFactor * (2 * 6.248937013549826 + 19.080144373982794), tol)
        << "2nd order velocity moment false!";
}
} // namespace

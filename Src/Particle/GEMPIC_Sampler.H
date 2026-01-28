#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <cassert>
#include <numeric>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Parser.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <AMReX_RealBox.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleUtils.H"
#include "GEMPIC_SobolSampler.H"

namespace Gempic::Particle
{
namespace Impl
{
enum class VelocityInitScheme
{
    constant,
    function,
};

template <std::size_t numValues, typename RandomDistribution, typename RandomGenerator>
inline void calculate_random_values (std::array<amrex::Real, numValues>& randomArray,
                                     RandomDistribution&& rnd,
                                     RandomGenerator& gen)
{
    for (int i = 0; i < numValues; i++)
    {
        randomArray[i] = rnd(gen);
    }
}

template <std::size_t numValues>
inline void calculate_sobol_values (std::array<amrex::Real, numValues>& randomArray,
                                    long long int& seed)
{
    double quasi[numValues];
    Gempic::Particle::sobol(numValues, &seed, quasi);
    for (int i = 0; i < numValues; i++)
    {
        randomArray[i] = quasi[i];
    }
}

template <std::size_t vDim>
class VelocityInitializer;

template <template <std::size_t> class VInit, std::size_t vDim, std::size_t numVal>
AMREX_GPU_HOST_DEVICE inline int get_gauss_id (VInit<vDim> const& vIniter,
                                               std::array<amrex::Real, numVal> const& randomArray)
{
    amrex::Real gaussChoice = randomArray[AMREX_SPACEDIM + 2 * vDim]; // uniform in [0, 1]
    int gaussIdx = 0;

    for (int gg = 1; gg < vIniter.m_numGauss; gg++)
    {
        if ((vIniter.m_velCumPtr[gg - 1] - gaussChoice < 0) &&
            (vIniter.m_velCumPtr[gg] - gaussChoice >= 0))
        {
            gaussIdx = gg;
        }
    }
    return gaussIdx;
}

AMREX_GPU_HOST_DEVICE inline amrex::Real get_function_or_constant (
    VelocityInitScheme scheme,
    amrex::Real constant,
    amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
    std::array<amrex::Real, AMREX_SPACEDIM> const& location)
{
    if (scheme == VelocityInitScheme::constant)
    {
        return constant;
    }
    else // (scheme == VelocityInitScheme::function)
    {
        return func(AMREX_D_DECL(location[xDir], location[yDir], location[zDir]), 0.0);
    }
}

template <template <std::size_t> class VInit, std::size_t vDim>
AMREX_GPU_HOST_DEVICE inline amrex::Real get_gaussian_velocity (
    VInit<vDim> const& vIniter,
    int gaussId,
    amrex::Real z0,
    int cc,
    std::array<amrex::Real, AMREX_SPACEDIM> const& location)
{
    amrex::Real vMean =
        get_function_or_constant(vIniter.m_vMeanSchemePtr[gaussId], vIniter.m_vMeanPtr[gaussId][cc],
                                 vIniter.m_funcvMeanPtr[gaussId][cc], location);
    amrex::Real vDev =
        get_function_or_constant(vIniter.m_vDevSchemePtr[gaussId], vIniter.m_vDevPtr[gaussId][cc],
                                 vIniter.m_funcvDevPtr[gaussId][cc], location);
    return vDev * z0 + vMean;
}

template <template <std::size_t> class VInit, std::size_t vDim, std::size_t numValues>
AMREX_GPU_HOST_DEVICE inline void get_velocities (
    VInit<vDim> const& vIniter,
    std::array<amrex::Real, vDim>& velocities,
    std::array<amrex::Real, numValues> const& randomArray,
    std::array<amrex::Real, AMREX_SPACEDIM> const& location)
{
    int gaussIdx{get_gauss_id(vIniter, randomArray)};

    auto epsilon{std::numeric_limits<amrex::Real>::epsilon()};
    for (int cc = 0; cc < vDim; cc++)
    {
        amrex::Real u1 = randomArray[AMREX_SPACEDIM + 2 * cc];
        amrex::Real u2 = randomArray[AMREX_SPACEDIM + 2 * cc + 1];
        // make sure epsilon large enough to take log
        // first value from Box-Muller algorithm
        u1 = u1 < epsilon ? epsilon : u1;
        amrex::Real z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
        velocities[cc] = get_gaussian_velocity(vIniter, gaussIdx, z0, cc, location);
    }
}

// Gpu compatible version of VelocityInitializer (see below)
template <std::size_t vDim>
struct VelocityInitializerGpu
{
    friend class VelocityInitializer<vDim>;

    std::array<amrex::Real, vDim>* m_vMeanPtr;
    std::array<amrex::Real, vDim>* m_vDevPtr;
    std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* m_funcvMeanPtr;
    std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* m_funcvDevPtr;
    VelocityInitScheme* m_vMeanSchemePtr;
    VelocityInitScheme* m_vDevSchemePtr;
    amrex::Real* m_velCumPtr;
    size_t m_numGauss;

private:
    VelocityInitializerGpu(std::array<amrex::Real, vDim>* vMean,
                           std::array<amrex::Real, vDim>* vDev,
                           std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* funcvMean,
                           std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* funcvDev,
                           VelocityInitScheme* vMeanScheme,
                           VelocityInitScheme* vDevScheme,
                           amrex::Real* velCum,
                           size_t numGauss) :
        m_vMeanPtr{vMean},
        m_vDevPtr{vDev},
        m_funcvMeanPtr{funcvMean},
        m_funcvDevPtr{funcvDev},
        m_vMeanSchemePtr{vMeanScheme},
        m_vDevSchemePtr{vDevScheme},
        m_velCumPtr{velCum},
        m_numGauss{numGauss}
    {
    }
};

template <std::size_t vDim>
inline void read_function_or_constant (
    Gempic::Io::Parameters& params,
    int nGauss,
    VelocityInitScheme& scheme,
    std::array<amrex::Real, vDim>& constant,
    std::array<amrex::Parser, vDim>& parser,
    std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>& func,
    std::string const& name)
{
    std::string gaussianNumber = 'G' + std::to_string(nGauss) + '.';
    amrex::Vector<amrex::Real> vm, vt;
    std::string inputString = gaussianNumber + name;
    if (params.exists(inputString + ".x"))
    {
        scheme = VelocityInitScheme::function;
        std::tie(parser[xDir], func[xDir]) = Utils::parse_function(inputString + ".x", params);
        if constexpr (vDim > 1)
        {
            GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(params.exists(inputString + ".y"),
                                              name + " must be given in all directions");
            std::tie(parser[yDir], func[yDir]) = Utils::parse_function(inputString + ".y", params);
            if constexpr (vDim > 2)
            {
                GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(params.exists(inputString + ".z"),
                                                  name + " must be given in all directions");
                std::tie(parser[zDir], func[zDir]) =
                    Utils::parse_function(inputString + ".z", params);
            }
        }
        if (params.exists(inputString))
        {
            amrex::Warning("Warning: " + name + " constant string not read if functions are given");
        }
    }
    else
    {
        scheme = VelocityInitScheme::constant;
        params.get(inputString, vm);
        for (int i{0}; i < vDim; ++i) constant[i] = vm[i];
    }
}

// Helper to read input parameters for velocity initialization and later initialize velocities
template <std::size_t vDim>
class VelocityInitializer
{
private:
    amrex::Gpu::DeviceVector<std::array<amrex::Real, vDim>> m_vMeanGpu, m_vDevGpu;
    amrex::Gpu::DeviceVector<std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>>
        m_funcvMeanGpu;
    amrex::Gpu::DeviceVector<std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>>
        m_funcvDevGpu;
    amrex::Gpu::DeviceVector<VelocityInitScheme> m_vMeanSchemeGpu;
    amrex::Gpu::DeviceVector<VelocityInitScheme> m_vDevSchemeGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_velCumGpu;

    amrex::Vector<std::array<amrex::Real, vDim>> m_vMean, m_vDev;
    amrex::Vector<std::array<amrex::Parser, vDim>> m_funcvMeanParser;
    amrex::Vector<std::array<amrex::Parser, vDim>> m_funcvDevParser;
    amrex::Vector<std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>> m_funcvMean;
    amrex::Vector<std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>> m_funcvDev;
    amrex::Vector<VelocityInitScheme> m_vMeanScheme;
    amrex::Vector<VelocityInitScheme> m_vDevScheme;
    amrex::Vector<amrex::Real> m_velCum;

public:
    std::array<amrex::Real, vDim>* m_vMeanPtr;
    std::array<amrex::Real, vDim>* m_vDevPtr;
    std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* m_funcvMeanPtr;
    std::array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* m_funcvDevPtr;
    VelocityInitScheme* m_vMeanSchemePtr;
    VelocityInitScheme* m_vDevSchemePtr;
    amrex::Real* m_velCumPtr;

    size_t m_numGauss;

    VelocityInitializer() {}

    VelocityInitializer(Gempic::Io::Parameters& params)
    {
        int numGauss{1};
        params.get_or_set("numGaussians", numGauss);
        m_numGauss = numGauss;
        m_vMean.resize(numGauss);
        m_funcvMeanParser.resize(numGauss);
        m_funcvMean.resize(numGauss);
        m_vMeanScheme.resize(numGauss);
        m_vDev.resize(numGauss);
        m_funcvDevParser.resize(numGauss);
        m_funcvDev.resize(numGauss);
        m_funcvMeanParser.resize(numGauss);
        m_funcvMean.resize(numGauss);
        m_vDevScheme.resize(numGauss);
        m_vMeanScheme.resize(numGauss);
        m_velCum.resize(numGauss);
        amrex::Vector<amrex::Real> vWeight(numGauss);
        for (int ng = 0; ng < numGauss; ng++)
        {
            read_function_or_constant(params, ng, m_vMeanScheme[ng], m_vMean[ng],
                                      m_funcvMeanParser[ng], m_funcvMean[ng], "vMean");
            read_function_or_constant(params, ng, m_vDevScheme[ng], m_vDev[ng],
                                      m_funcvDevParser[ng], m_funcvDev[ng], "vThermal");
            std::string vwString = 'G' + std::to_string(ng) + '.' + "vWeight";
            amrex::Real vw;
            params.get(vwString, vw);
            vWeight[ng] = vw;
        }
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(1.0 - std::reduce(vWeight.begin(), vWeight.end())) < 1e-12,
            "Gaussian weights must sum to 1.0!");
        std::partial_sum(vWeight.begin(), vWeight.end(), m_velCum.begin());

        m_vMeanPtr = m_vMean.data();
        m_funcvMeanPtr = m_funcvMean.data();
        m_vDevPtr = m_vDev.data();
        m_funcvMeanPtr = m_funcvMean.data();
        m_funcvDevPtr = m_funcvDev.data();
        m_vMeanSchemePtr = m_vMeanScheme.data();
        m_vDevSchemePtr = m_vDevScheme.data();
        m_velCumPtr = m_velCum.data();
    }

    VelocityInitializerGpu<vDim> get_gpu_initializer ()
    {
        m_vMeanGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vMean.begin(), m_vMean.end(),
                              m_vMeanGpu.begin());
        m_funcvMeanGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_funcvMean.begin(), m_funcvMean.end(),
                              m_funcvMeanGpu.begin());
        m_vMeanSchemeGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vMeanScheme.begin(), m_vMeanScheme.end(),
                              m_vMeanSchemeGpu.begin());
        m_vDevGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vDev.begin(), m_vDev.end(),
                              m_vDevGpu.begin());
        m_funcvMeanGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_funcvMean.begin(), m_funcvMean.end(),
                              m_funcvMeanGpu.begin());
        m_funcvDevGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_funcvDev.begin(), m_funcvDev.end(),
                              m_funcvDevGpu.begin());
        m_vMeanSchemeGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vMeanScheme.begin(), m_vMeanScheme.end(),
                              m_vMeanSchemeGpu.begin());
        m_vDevSchemeGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vDevScheme.begin(), m_vDevScheme.end(),
                              m_vDevSchemeGpu.begin());
        m_velCumGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_velCum.begin(), m_velCum.end(),
                              m_velCumGpu.begin());
        return VelocityInitializerGpu<vDim>(m_vMeanGpu.dataPtr(), m_vDevGpu.dataPtr(),
                                            m_funcvMeanGpu.dataPtr(), m_funcvDevGpu.dataPtr(),
                                            m_vMeanSchemeGpu.dataPtr(), m_vDevSchemeGpu.dataPtr(),
                                            m_velCumGpu.dataPtr(), m_numGauss);
    }
};

template <std::size_t vDim, std::size_t nData>
class ParticleBuffer
{
public:
    void add_particle (std::array<amrex::Real, AMREX_SPACEDIM> position,
                       std::array<amrex::Real, vDim> velocity,
                       amrex::Real weight)
    {
        m_id.push_back(amrex::SoAParticle<AMREX_SPACEDIM + vDim + nData, 0>::NextID());
        m_cpu.push_back(amrex::ParallelDescriptor::MyProc());
        for (int cc = 0; cc < AMREX_SPACEDIM; cc++)
        {
            m_pos[cc].push_back(position[cc]);
        }

        for (int cc = 0; cc < vDim; cc++)
        {
            m_rdata[cc].push_back(velocity[cc]);
        }
        m_rdata[vDim].push_back(weight);
        for (int nd = 1; nd < nData; nd++)
        {
            m_rdata[vDim + nd].push_back(0.0);
        }
    }

    // move our AoS particles to an outside SoA particle tile object.
    void flush (amrex::ParticleTile<amrex::SoAParticle<AMREX_SPACEDIM + vDim + nData, 0>,
                                    AMREX_SPACEDIM + vDim + nData,
                                    0>& particles)
    {
        auto const oldSize = particles.numParticles();
        auto const newSize = oldSize + m_id.size();
        particles.resize(newSize);
        // the `ptd` object is necessary to write "cpu" and "id" information
        auto ptd = particles.getParticleTileData();
        // the `soa` object is necessary to write the rest of the data
        // using ptd.rdata leads to failed tests.
        auto& soa = particles.GetStructOfArrays();

        // "buffer" objects to copy data to GPU
        // The point of using "DeviceVector" is to avoid separate host and device code
        // as of AMReX 041f225d declaring `bufferId(m_id)` leads to compilation errors,
        // hence the manual calls to `copy` are necessary.
        amrex::Gpu::DeviceVector<int> bufferId(m_id.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_id.begin(), m_id.end(), bufferId.begin());
        amrex::Gpu::DeviceVector<int> bufferCpu(m_cpu.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_cpu.begin(), m_cpu.end(), bufferCpu.begin());

        auto* const ptrId = bufferId.dataPtr();
        auto* const ptrCpu = bufferCpu.dataPtr();

        amrex::ParallelFor(m_id.size(),
                           [=] AMREX_GPU_DEVICE(int ii)
                           {
                               int const iii = ii + oldSize;
                               ptd.id(iii) = ptrId[ii];
                               ptd.cpu(iii) = ptrCpu[ii];
                           });

        // position
        for (int cc = 0; cc < AMREX_SPACEDIM; cc++)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_pos[cc].begin(), m_pos[cc].end(),
                                  soa.GetRealData(cc).begin() + oldSize);
        }
        // velocity
        for (int cc = 0; cc < vDim; cc++)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_rdata[cc].begin(), m_rdata[cc].end(),
                                  soa.GetRealData(cc + AMREX_SPACEDIM).begin() + oldSize);
        }
        // weight
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_rdata[vDim].begin(), m_rdata[vDim].end(),
                              soa.GetRealData(vDim + AMREX_SPACEDIM).begin() + oldSize);
        // everything else
        for (int cc = 1; cc < nData; cc++)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_rdata[cc + vDim].begin(),
                                  m_rdata[cc + vDim].end(),
                                  soa.GetRealData(cc + vDim + AMREX_SPACEDIM).begin() + oldSize);
        }
        amrex::Gpu::streamSynchronize();
        clear();
    }

    void reserve (int numParticles)
    {
        m_id.reserve(numParticles);
        m_cpu.reserve(numParticles);
        for (int comp = 0; comp < AMREX_SPACEDIM; comp++)
        {
            m_pos[comp].reserve(numParticles);
        }
        for (int comp = 0; comp < vDim + nData; comp++)
        {
            m_rdata[comp].reserve(numParticles);
        }
    }

    void clear ()
    {
        m_id.clear();
        m_cpu.clear();
        for (int comp = 0; comp < AMREX_SPACEDIM; comp++)
        {
            m_pos[comp].clear();
        }
        for (int comp = 0; comp < vDim + nData; comp++)
        {
            m_rdata[comp].clear();
        }
    }

    std::size_t size () const { return m_id.size(); }

private:
    amrex::Gpu::HostVector<int> m_id;
    amrex::Gpu::HostVector<int> m_cpu;
    std::array<amrex::Gpu::HostVector<amrex::Real>, AMREX_SPACEDIM> m_pos;
    std::array<amrex::Gpu::HostVector<amrex::Real>, vDim + nData> m_rdata;
};

enum class RandomScheme
{
    pseudo,
    sobol,
};

enum class SamplerRegion
{
    cell,
    box,
    global,
};

enum class SamplerType
{
    directMaxwellian,
    rejectionMaxwellian,
    rejectionJuettner,
};

/**
 * @brief Particle sampler class
 *
 * @tparam RandomScheme : Switching between pseudo and quasi random number generator
 * @tparam SamplerRegion : Sampling on local boxes or on the full domain followed by communication
 * @tparam SamplerType : Selecting the type of sampling, e.g., rejection
 * @tparam extraRandomVar : Number of additional random variables
 * @tparam vDim : Number of velocity components associated with the particles
 * @tparam nData : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] ppg : Particle species
 * @param geom : Geometry
 * @param species : Species number
 * @param rSeed : Random seed
 *
 */
template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
class ParticleSampler
{
public:
    ParticleSampler(std::shared_ptr<ParticleSpecies<vDim, nData>>& ppg,
                    amrex::Geometry geom,
                    int species,
                    int rSeed = 925);
    void sample(std::shared_ptr<ParticleSpecies<vDim, nData>>& ppg);

private:
    void sample_region(int const nPart,
                       std::mt19937& gen,
                       std::uniform_real_distribution<>& unif,
                       long long int& sobolSeed);
    void direct_maxwellian_kernel();
    bool rejection_maxwellian_kernel();
    bool rejection_juettner_kernel();

    amrex::Geometry m_geom;
    int m_species;
    int m_rSeed;
    int m_nPartPerCell;

    ParticleBuffer<vDim, nData> m_buffer;
    std::array<amrex::Real, AMREX_SPACEDIM> m_position;
    std::array<amrex::Real, vDim> m_velocity;
    amrex::Real m_weight;

    std::array<amrex::Real, AMREX_SPACEDIM + 2 * vDim + 1 + extraRandomVar> m_randArray;
    VelocityInitializer<vDim> m_vInit;
    std::array<amrex::Real, AMREX_SPACEDIM> m_regionLo;
    std::array<amrex::Real, AMREX_SPACEDIM> m_regionLength;
    amrex::Parser m_densityParse;
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> m_densityEval;

    int m_nMaxRejections{100};
    int m_nTargetHigherThanProposal{0};
    int m_nSamples{0};
    amrex::Real m_c{1.0};
    amrex::Real m_rejectionScaling{1};
    amrex::Real m_minRejectionScaling{0};
    amrex::Parser m_rejectionParse;
    amrex::ParserExecutor<AMREX_SPACEDIM + 3> m_rejectionEval;
};

template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
ParticleSampler<scheme, region, type, extraRandomVar, vDim, nData>::ParticleSampler(
    std::shared_ptr<ParticleSpecies<vDim, nData>>& ppg,
    amrex::Geometry geom,
    int species,
    int rSeed) :
    m_geom(geom), m_species(species), m_rSeed(rSeed)
{
    std::string speciesName{ppg->get_name()};
    std::string specString = "Particle." + speciesName;
    Io::Parameters params(specString, "object ParticleSampler (" + speciesName + ')');

    params.get("nPartPerCell", m_nPartPerCell);
    std::tie(m_densityParse, m_densityEval) = Utils::parse_function("density", params);

    m_vInit = VelocityInitializer<vDim>(params);

    if constexpr (type == SamplerType::rejectionMaxwellian)
    {
        // check that the settings for the Maxwellian are valid for rejection sampling
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(
            m_vInit.m_numGauss == 1,
            "Cannot apply rejection sampling with more than one Gaussian as proposal");

        // read in additional information for rejection samplers
        params.get("nMaxRejections", m_nMaxRejections);
        params.get("rejectionScaling", m_rejectionScaling);
        std::string parseString;
        params.get("rejectionDistribution", parseString);
        m_rejectionParse = amrex::Parser(parseString);
        m_rejectionParse.registerVariables({AMREX_D_DECL("x", "y", "z"), "vx", "vy", "vz"});
        if (params.exists("k"))
        {
            amrex::Vector<amrex::Real> k;
            params.get("k", k);
            AMREX_D_TERM(m_rejectionParse.setConstant("kvarx", k[xDir]);
                         , m_rejectionParse.setConstant("kvary", k[yDir]);
                         , m_rejectionParse.setConstant("kvarz", k[zDir]);)
        }
        m_rejectionEval = m_rejectionParse.template compile<AMREX_SPACEDIM + 3>();
    }

    else if constexpr (type == SamplerType::rejectionJuettner)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_vInit.m_numGauss == 1,
            "Cannot apply Juettner rejection sampling with more than one Gaussian");

        // read in additional information for relativistic samplers
        params.get("nMaxRejections", m_nMaxRejections);
        // read in speed of light from input file
        params.get("c", m_c);
    }
}

template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
void ParticleSampler<scheme, region, type, extraRandomVar, vDim, nData>::sample (
    std::shared_ptr<ParticleSpecies<vDim, nData>>& ppg)
{
    for (amrex::MFIter mfi = ppg->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        int nPart = m_nPartPerCell * bx.numPts();
        std::mt19937 gen(GEMPIC_D_ADD(m_rSeed + bx.smallEnd(xDir), bx.smallEnd(yDir) * 10,
                                      bx.smallEnd(zDir) * 100) +
                         m_species * 1000);
        std::uniform_real_distribution<> unif(0.0, 1.0);
        long long int sobolSeed = gen();
        m_buffer.reserve(nPart);

        if constexpr (region == SamplerRegion::cell)
        {
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};
            GEMPIC_D_LOOP_BEGIN(for (int l = lo[xDir]; l <= hi[xDir]; l++),
                                for (int j = lo[yDir]; j <= hi[yDir]; j++),
                                for (int k = lo[zDir]; k <= hi[zDir]; k++))
                amrex::Vector<amrex::Real> loNode;
                m_geom.LoNode(amrex::IntVect{AMREX_D_DECL(l, j, k)}, loNode);
                for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                {
                    m_regionLo[dir] = loNode[dir];
                    m_regionLength[dir] = m_geom.CellSize(dir);
                }
                sample_region(m_nPartPerCell, gen, unif, sobolSeed);
            GEMPIC_D_LOOP_END
        }
        else
        {
            for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
            {
                if constexpr (region == SamplerRegion::box)
                {
                    m_regionLo[dir] = m_geom.ProbLo(dir) + bx.smallEnd(dir) * m_geom.CellSize(dir);
                    m_regionLength[dir] = bx.length(dir) * m_geom.CellSize(dir);
                }
                else if constexpr (region == SamplerRegion::global)
                {
                    m_regionLo[dir] = m_geom.ProbLo(dir);
                    m_regionLength[dir] = m_geom.ProbLength(dir);
                }
            }
            sample_region(nPart, gen, unif, sobolSeed);
        }

        m_buffer.flush(ppg->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())]);
    }
    if constexpr (type == SamplerType::rejectionMaxwellian or
                  type == SamplerType::rejectionJuettner)
    {
        amrex::ParallelAllReduce::Max(m_minRejectionScaling,
                                      amrex::ParallelContext::CommunicatorAll());
        std::array<int, 2> tmp{m_nTargetHigherThanProposal, m_nSamples};
        amrex::ParallelAllReduce::Sum(tmp.data(), tmp.size(),
                                      amrex::ParallelContext::CommunicatorAll());
        amrex::Real acceptanceRate = 1.0 * m_geom.Domain().numPts() * m_nPartPerCell / tmp[1];

        amrex::Print() << "Rejection sampling: Acceptance rate                = " << acceptanceRate
                       << "\n";
        amrex::Print() << "Rejection sampling: Samples with target > proposal = " << tmp[0] << "\n";
        amrex::Print() << "Rejection sampling: Minimum scaling necessary      = "
                       << m_minRejectionScaling << "\n";
    }

    if constexpr (region == SamplerRegion::global)
    {
        ppg->Redistribute();
    }
}

template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
void ParticleSampler<scheme, region, type, extraRandomVar, vDim, nData>::sample_region (
    int const nPart,
    std::mt19937& gen,
    std::uniform_real_distribution<>& unif,
    long long int& sobolSeed)
{
    for (int pp = 0; pp < nPart; pp++)
    {
        if constexpr (type == SamplerType::directMaxwellian)
        {
            if constexpr (scheme == RandomScheme::pseudo)
            {
                calculate_random_values(m_randArray, unif, gen);
            }
            else if constexpr (scheme == RandomScheme::sobol)
            {
                calculate_sobol_values(m_randArray, sobolSeed);
            }
            direct_maxwellian_kernel();
        }
        else if constexpr (type == SamplerType::rejectionMaxwellian)
        {
            bool accepted = false;
            for (int i = 0; i < m_nMaxRejections; i++)
            {
                if constexpr (scheme == RandomScheme::pseudo)
                {
                    calculate_random_values(m_randArray, unif, gen);
                }
                else if constexpr (scheme == RandomScheme::sobol)
                {
                    calculate_sobol_values(m_randArray, sobolSeed);
                }
                accepted = rejection_maxwellian_kernel();
                if (accepted)
                {
                    m_nSamples += i + 1;
                    break;
                }
            }
            if (not accepted)
            {
                amrex::Error(
                    "Rejection sampling: Did not accept a sample in nMaxRejections "
                    "tries.");
            }
        }
        else if constexpr (type == SamplerType::rejectionJuettner)
        {
            bool accepted = false;
            for (int i = 0; i < m_nMaxRejections; i++)
            {
                static_assert(scheme == RandomScheme::pseudo,
                              "Only the pseudorandom scheme is "
                              "implemented for Juettner rejection sampling.");
                calculate_random_values(m_randArray, unif, gen);
                accepted = rejection_juettner_kernel();
                if (accepted)
                {
                    m_nSamples += i + 1;
                    break;
                }
            }
            if (not accepted)
            {
                amrex::Error(
                    "Juettner rejection sampling: Did not accept a sample in nMaxRejections "
                    "tries.");
            }
        }
        m_weight *= m_geom.CellSizeArray().product() / m_nPartPerCell;
        m_buffer.add_particle(m_position, m_velocity, m_weight);
    }
}

template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
inline void
ParticleSampler<scheme, region, type, extraRandomVar, vDim, nData>::direct_maxwellian_kernel()
{
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        m_position[i] = m_regionLo[i] + m_randArray[i] * m_regionLength[i];
    }
    get_velocities(m_vInit, m_velocity, m_randArray, m_position);
    m_weight =
        m_densityEval(AMREX_D_DECL(m_position[xDir], m_position[yDir], m_position[zDir]), 0.);
}

template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
bool ParticleSampler<scheme, region, type, extraRandomVar, vDim, nData>::
    rejection_maxwellian_kernel()
{
    direct_maxwellian_kernel(); // calculate position, Gaussian velocities, and density
    std::array<amrex::Real, vDim> vMean, vDev;
    for (int cc = 0; cc < vDim; cc++)
    {
        vMean[cc] = get_function_or_constant(m_vInit.m_vMeanSchemePtr[0], m_vInit.m_vMeanPtr[0][cc],
                                             m_vInit.m_funcvMeanPtr[0][cc], m_position);
        vDev[cc] = get_function_or_constant(m_vInit.m_vDevSchemePtr[0], m_vInit.m_vDevPtr[0][cc],
                                            m_vInit.m_funcvDevPtr[0][cc], m_position);
    }
    amrex::Real proposal = Utils::eval_maxwellian(m_velocity, m_weight, vDev, vMean);
    amrex::Real target;
    if constexpr (vDim == 1)
    {
        target = m_rejectionEval(AMREX_D_DECL(m_position[xDir], m_position[yDir], m_position[zDir]),
                                 m_velocity[xDir], 0., 0.);
    }
    else if constexpr (vDim == 2)
    {
        target = m_rejectionEval(AMREX_D_DECL(m_position[xDir], m_position[yDir], m_position[zDir]),
                                 m_velocity[xDir], m_velocity[yDir], 0.);
    }
    else
    {
        target = m_rejectionEval(AMREX_D_DECL(m_position[xDir], m_position[yDir], m_position[zDir]),
                                 m_velocity[xDir], m_velocity[yDir], m_velocity[zDir]);
    }

    amrex::Real probability = target / (m_rejectionScaling * proposal);
    m_minRejectionScaling = std::max(m_minRejectionScaling, target / proposal);
    if (probability > 1.)
    {
        m_nTargetHigherThanProposal += 1;
    }

    return probability > m_randArray[AMREX_SPACEDIM + 2 * vDim + 1];
}

template <RandomScheme scheme,
          SamplerRegion region,
          SamplerType type,
          std::size_t extraRandomVar,
          std::size_t vDim,
          std::size_t nData>
bool ParticleSampler<scheme, region, type, extraRandomVar, vDim, nData>::rejection_juettner_kernel ()
{
    // position and density same as for the direct kernel
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        m_position[i] = m_regionLo[i] + m_randArray[i] * m_regionLength[i];
    }

    m_weight =
        m_densityEval(AMREX_D_DECL(m_position[xDir], m_position[yDir], m_position[zDir]), 0.);

    Io::Parameters params("Particle");

    amrex::Real eta{0.0};
    amrex::Real u{2.0};
    amrex::Real checksum{0.0};
    amrex::Real gamma{0.0};
    std::array<amrex::Real, vDim> vThermal, vMean;

    // implementation for now: just one Maxwell-Juettner sampling
    for (int cc = 0; cc < vDim; cc++)
    {
        vMean[cc] = get_function_or_constant(m_vInit.m_vMeanSchemePtr[0], m_vInit.m_vMeanPtr[0][cc],
                                             m_vInit.m_funcvMeanPtr[0][cc], m_position);
        vThermal[cc] =
            get_function_or_constant(m_vInit.m_vDevSchemePtr[0], m_vInit.m_vDevPtr[0][cc],
                                     m_vInit.m_funcvDevPtr[0][cc], m_position);
    }

    // always use the x-value for vThermal
    u = -vThermal[xDir] * vThermal[xDir] * log(m_randArray[3] * m_randArray[4] * m_randArray[5]);
    eta = -vThermal[xDir] * vThermal[xDir] *
          log(m_randArray[3] * m_randArray[4] * m_randArray[5] * m_randArray[6]);

    checksum = eta * eta - u * u;

    if (checksum > 1)
    {
        m_velocity[xDir] = u * (2 * m_randArray[7] - 1);
        m_velocity[yDir] = 2 * u * sqrt(m_randArray[7] * (1 - m_randArray[7]));
        m_velocity[zDir] = m_velocity[yDir] * sin(2 * M_PI * m_randArray[8]);
        m_velocity[yDir] = m_velocity[yDir] * cos(2 * M_PI * m_randArray[8]);

        gamma = sqrt(1 + u * u / (m_c * m_c));

        if (-vMean[xDir] * m_velocity[xDir] / gamma > m_randArray[9])
        {
            m_velocity[xDir] = -m_velocity[xDir];
        }

        m_velocity[xDir] =
            (m_velocity[xDir] + vMean[0] * gamma) / sqrt(1 - vMean[0] * vMean[0] / (m_c * m_c));

        return true;
    }
    else
    {
        return false;
    }
}

// Get global location of a cell corner given the cell indices assuming uniform mesh
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inline get_cell_corner(
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lowerCorner,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
    AMREX_D_DECL(int i, int j, int k))
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellCorner;

    AMREX_D_TERM(cellCorner[xDir] = lowerCorner[xDir] + i * dx[xDir];
                 , cellCorner[yDir] = lowerCorner[yDir] + j * dx[yDir];
                 , cellCorner[zDir] = lowerCorner[zDir] + k * dx[zDir];)
    return cellCorner;
}

AMREX_GPU_HOST_DEVICE inline int get_cellid (
    int i, int j, int k, amrex::IntVect lo, amrex::IntVect n)
{
    AMREX_D_TERM(int ix = i - lo[xDir];, int iy = j - lo[yDir];, int iz = k - lo[zDir];)
    AMREX_D_PICK(return ix;, return iy * n[xDir] + ix;, return (iz * n[yDir] + iy) * n[xDir] + ix;)
}

/**
 * @brief Random particle initializer for gpu.
 *
 * @details Particle positions are generated using a uniform random
 * distribution giving position on each cell and then remapped to full domain.
 * Velocity distribution is generated from uniform distribution using the Box-Muller algorithm
 * as random_normal is machine dependent. This enables for a given seed to always get the same
 * random sequence
 *
 * @tparam vDim Number of velocity components associated with the particles
 * @tparam nData Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] ppg Particle species
 * @param geom Geometry
 * @param species Index of species in vector. Only used for the random seed
 * @param rSeed Seed for the random generator
 *
 */
template <std::size_t vDim, std::size_t nData = 1>
void init_particles_full_domain_gpu (std::shared_ptr<ParticleSpecies<vDim, nData>>& ppg,
                                     amrex::Geometry geom,
                                     int species,
                                     int rSeed = 925)
{
    using namespace Impl;
    BL_PROFILE("Gempic::Particle::init_particles_full_domain_gpu()");

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probLo{geom.ProbLoArray()};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{geom.CellSizeArray()};

    std::string speciesName{ppg->get_name()};
    std::string specString = "Particle." + speciesName;
    Io::Parameters params(specString,
                          "function init_particles_full_domain_gpu (" + speciesName + ')');

    int nPartPerCell;
    params.get("nPartPerCell", nPartPerCell);
    // we capture by std::tie instead of by structured binding because of idiosyncracies with lambda
    // expressions. May or may not be resolved in C++20, but not in e.g. Clang.
    amrex::Parser densityParse;
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> densityEval;
    std::tie(densityParse, densityEval) = Utils::parse_function("density", params);

    VelocityInitializer<vDim> vInit(params);
    auto vInitGpu{vInit.get_gpu_initializer()};

    for (amrex::MFIter mfi = ppg->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect boxSize{bx.size()};
        int npart = nPartPerCell * bx.numPts();
        auto& particles = ppg->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        // global domain box for determining unique particle id
        amrex::IntVect domainLo = {geom.Domain().smallEnd()};
        amrex::IntVect domainSize{geom.Domain().size()};

        // resize particles structure before launching on GPU
        auto oldSize = particles.numParticles();
        auto newSize = oldSize + npart;
        particles.resize(newSize);

        auto pstruct = particles.getParticleTileData();
        auto pdi = ppg->get_data_indices();
        std::array<amrex::ParticleReal*, vDim> velArray;
        if (vDim > 0) velArray[0] = pstruct.rdata(pdi.m_ivelx);
        if (vDim > 1) velArray[1] = pstruct.rdata(pdi.m_ively);
        if (vDim > 2) velArray[2] = pstruct.rdata(pdi.m_ivelz);
        amrex::ParticleReal* weightArray = pstruct.rdata(pdi.m_iweight);
        int procID = amrex::ParallelDescriptor::MyProc();

        int seed = rSeed + GEMPIC_D_ADD(lo[xDir], lo[yDir] * 10, lo[zDir] * 100) + species * 1000;
        amrex::InitRandom(seed, amrex::ParallelDescriptor::NProcs(),
                          seed); // pass seed also for GPU (third argument)

        ParallelForRNG(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const& gen) noexcept
            {
                int globalid = get_cellid(i, j, k, domainLo, domainSize);
                int cellid = get_cellid(i, j, k, lo, boxSize);

                // globally unique particle id
                int pidx = globalid * nPartPerCell + 1;
                // local index for ParticleTile data starting at 0 on each Tile
                int localidx = cellid * nPartPerCell;
                auto cellCorner{get_cell_corner(probLo, dx, AMREX_D_DECL(i, j, k))};

                std::array<amrex::Real, AMREX_SPACEDIM> position;
                std::array<amrex::Real, vDim> velocity;
                amrex::Real t = 0.0; // initial time
                std::array<amrex::Real, AMREX_SPACEDIM + 2 * vDim + 1> randArray;

                for (int pp = 0; pp < nPartPerCell; pp++)
                {
                    for (auto& randomValue : randArray)
                    {
                        randomValue = amrex::Random(gen);
                    }
                    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                    {
                        position[dir] = cellCorner[dir] + randArray[dir] * dx[dir];
                    }
                    get_velocities(vInitGpu, velocity, randArray, position);

                    double const factor = densityEval(
                        AMREX_D_DECL(position[xDir], position[yDir], position[zDir]), t);
                    amrex::Real weight =
                        factor / nPartPerCell * GEMPIC_D_MULT(dx[xDir], dx[yDir], dx[zDir]);
                    pstruct.id(localidx) = pidx;
                    pstruct.cpu(localidx) = procID;
                    AMREX_D_TERM(pstruct.pos(xDir, localidx) = position[xDir];
                                 , pstruct.pos(yDir, localidx) = position[yDir];
                                 , pstruct.pos(zDir, localidx) = position[zDir];)
                    for (int i = 0; i < vDim; i++)
                    {
                        velArray[i][localidx] = velocity[i];
                    }
                    weightArray[localidx] = weight;

                    ++pidx;
                    ++localidx;
                }
            });
    }
}

/**
 * @brief
 *
 * @tparam vDim : Number of velocity components associated with the particles
 * @tparam nData : Number of further amount of data components to be stored, e.g. particle
 * weight
 *
 * @param[out] particles Particle species
 * @param geom Geometry
 * @param species Index of species in vector. Only used for the random seed
 * @param globalSamplerName Default sampler for all species
 * @param rSeed Seed for the random generator
 */
template <unsigned int vDim, unsigned int nData>
void sample_species (std::shared_ptr<ParticleSpecies<vDim, nData>>& particles,
                     amrex::Geometry geom,
                     int species,
                     std::string globalSamplerName,
                     int rSeed)
{
    BL_PROFILE("Gempic::Particle::sample_species(particles,geom,species,globalSamplerName,rSeed)");

    std::string speciesName{particles->get_name()};
    Io::Parameters params("Particle." + speciesName);
    std::string samplerName{globalSamplerName};
    params.get_or_set("sampler", samplerName);

    if (samplerName == "PseudoRandom")
    {
        ParticleSampler<RandomScheme::pseudo, SamplerRegion::cell, SamplerType::directMaxwellian, 0,
                        vDim, nData>
            sampler(particles, geom, species, rSeed);
        sampler.sample(particles);
    }
    else if (samplerName == "PseudoRandomBox")
    {
        ParticleSampler<RandomScheme::pseudo, SamplerRegion::box, SamplerType::directMaxwellian, 0,
                        vDim, nData>
            sampler(particles, geom, species, rSeed);
        sampler.sample(particles);
    }
    else if (samplerName == "Sobol")
    {
        ParticleSampler<RandomScheme::sobol, SamplerRegion::box, SamplerType::directMaxwellian, 0,
                        vDim, nData>
            sampler(particles, geom, species, rSeed);
        sampler.sample(particles);
    }
    else if (samplerName == "Rejection")
    {
        ParticleSampler<RandomScheme::pseudo, SamplerRegion::global,
                        SamplerType::rejectionMaxwellian, 1, vDim, nData>
            sampler(particles, geom, species, rSeed);
        sampler.sample(particles);
    }
    else if (samplerName == "RejectionSobol")
    {
        ParticleSampler<RandomScheme::sobol, SamplerRegion::global,
                        SamplerType::rejectionMaxwellian, 1, vDim, nData>
            sampler(particles, geom, species, rSeed);
        sampler.sample(particles);
    }
    else if (samplerName == "RejectionJuettner")
    {
        ParticleSampler<RandomScheme::pseudo, SamplerRegion::global, SamplerType::rejectionJuettner,
                        0, vDim, nData>
            sampler(particles, geom, species, rSeed);
        sampler.sample(particles);
    }
    else if (samplerName == "PseudoRandomGPU")
    {
        init_particles_full_domain_gpu<vDim, nData>(particles, geom, species, rSeed);
    }
    else
    {
        GEMPIC_ERROR("Unknown particle initialization method: " + samplerName);
    }
}
} //namespace Impl

/**
 * @brief particle initializer
 *
 * @tparam vdim : Number of velocity components associated with the particles
 * @tparam ndata : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] particles : particles that are initialised
 * @param geom : Geometry
 *
 */
template <unsigned int vDim, unsigned int nData>
void init_particles (std::vector<std::shared_ptr<ParticleSpecies<vDim, nData>>>& particles,
                     ComputationalDomain infra)
{
    BL_PROFILE("Gempic::Particle::init_particles(particles,infra)");
    Io::Parameters params("Particle");
    amrex::Vector<std::string> speciesNames;
    params.get("speciesNames", speciesNames);
    std::string globalSamplerName{"PseudoRandom"};
    params.get_or_set("sampler", globalSamplerName);
    int rSeed{925};
    params.get_or_set("seed", rSeed);
    params.print_class_parameters();

    particles.resize(speciesNames.size());

    for (int species{0}; species < speciesNames.size(); ++species)
    {
        particles[species] =
            std::make_shared<ParticleSpecies<vDim, nData>>(speciesNames[species], infra);

        Impl::sample_species(particles[species], infra.geometry(), species, globalSamplerName,
                             rSeed);
        GEMPIC_DEBUG("species " + std::to_string(species) + " has " +
                     std::to_string(particles[species]->TotalNumberOfParticles()) + " particles");
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(particles[species]->OK(),
                                          (std::string("Initialization of particle species ") +
                                           std::to_string(species) + " failed.\n"));
    }
}

/**
 * @brief particle initializer
 *
 * @tparam vdim : Number of velocity components associated with the particles
 * @tparam ndata : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] particles : particles that are initialised
 * @param[out] particlesLinVlasov :
 * @param geom : Geometry
 *
 */
template <unsigned int vDim, unsigned int nData>
void init_particles (
    std::vector<std::shared_ptr<ParticleSpeciesLinVlasov<vDim, nData>>>& particlesLinVlasov,
    std::vector<std::shared_ptr<ParticleSpecies<vDim, nData>>>& particles,
    ComputationalDomain infra)
{
    BL_PROFILE("Gempic::Particle::init_particles(particlesLinVlasov,particles,infra)");
    Io::Parameters params("Particle");
    amrex::Vector<std::string> speciesNames;
    params.get("speciesNames", speciesNames);
    std::string globalSamplerName{"PseudoRandom"};
    params.get_or_set("sampler", globalSamplerName);
    int rSeed{925};
    params.get_or_set("seed", rSeed);
    params.print_class_parameters();

    particlesLinVlasov.resize(speciesNames.size());
    particles.resize(speciesNames.size());

    for (int species{0}; species < speciesNames.size(); ++species)
    {
        particlesLinVlasov[species] =
            std::make_shared<ParticleSpeciesLinVlasov<vDim, nData>>(speciesNames[species], infra);
        particles[species] =
            std::static_pointer_cast<ParticleSpecies<vDim, nData>>(particlesLinVlasov[species]);

        Impl::sample_species(particles[species], infra.geometry(), species, globalSamplerName,
                             rSeed);
    }
}

} // namespace Gempic::Particle

#endif

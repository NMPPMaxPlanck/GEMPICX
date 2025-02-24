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
#include "GEMPIC_ParticleGroups.H"
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

template <unsigned int vDim,
          typename RandomDistribution,
          typename RandomGenerator,
          typename RandomDistribution2,
          typename RandomGenerator2>
inline amrex::GpuArray<amrex::Real, AMREX_SPACEDIM + 3 * vDim> calculate_random_values (
    RandomDistribution&& rnd,
    RandomGenerator& gen,
    RandomDistribution2&& rndv,
    RandomGenerator2& genv)
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM + 3 * vDim> randomArray;
    // position random variables
    for (int dir{0}; dir < AMREX_SPACEDIM; ++dir)
    {
        randomArray[dir] = rnd(gen);
    }
    // velocity random variables
    for (int dir{AMREX_SPACEDIM}; dir < AMREX_SPACEDIM + 3 * vDim; ++dir)
    {
        randomArray[dir] = rndv(genv);
    }
    return randomArray;
}

template <template <unsigned int> class VInit, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline amrex::GpuArray<int, vDim> get_gauss_ids (
    const VInit<vDim>& vIniter,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM + 3 * vDim>& randomArray)
{
    amrex::GpuArray<amrex::Real, vDim> gaussChoice;
    amrex::GpuArray<int, vDim> gaussIdx;
    for (int i = 0; i < vDim; i++)
    {
        gaussChoice[i] = randomArray[AMREX_SPACEDIM + 2 * vDim + i]; // uniform in [0, 1]
        gaussIdx[i] = 0;
    }

    for (int i = 0; i < vDim; i++)
    {
        for (int gg = 1; gg < vIniter.m_numGauss; gg++)
        {
            if ((vIniter.m_velCumPtr[gg - 1] - gaussChoice[i] < 0) &&
                (vIniter.m_velCumPtr[gg] - gaussChoice[i] >= 0))
            {
                gaussIdx[i] = gg;
            }
        }
    }
    return gaussIdx;
}

template <template <unsigned int> class VInit, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline amrex::Real get_gaussian_velocity (
    const VInit<vDim>& vIniter,
    int gaussId,
    amrex::Real z0,
    int cc,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& location)
{
    amrex::Real vDev;
    if (vIniter.m_vSchemePtr[gaussId] == VelocityInitScheme::constant)
    {
        vDev = vIniter.m_vDevPtr[gaussId][cc];
    }
    else if (vIniter.m_vSchemePtr[gaussId] == VelocityInitScheme::function)
    {
        vDev = vIniter.m_funcvDevPtr[gaussId][cc](
            AMREX_D_DECL(location[xDir], location[yDir], location[zDir]), 0.0);
    }
    else
    {
        amrex::Abort("Unknown vThermal input format. Must be either constants or functions.");
        return std::numeric_limits<amrex::Real>::signaling_NaN(); // won't be reached.
    }
    return vDev * z0 + vIniter.m_vMeanPtr[gaussId][cc];
}

template <unsigned int vDim>
class VelocityInitializer;

// Gpu compatible version of VelocityInitializer (see below)
template <unsigned int vDim>
struct VelocityInitializerGpu
{
    friend class VelocityInitializer<vDim>;

    amrex::GpuArray<amrex::Real, vDim>* m_vMeanPtr;
    amrex::GpuArray<amrex::Real, vDim>* m_vDevPtr;
    amrex::GpuArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* m_funcvDevPtr;
    VelocityInitScheme* m_vSchemePtr;
    amrex::Real* m_velCumPtr;
    size_t m_numGauss;

private:
    VelocityInitializerGpu(
        amrex::GpuArray<amrex::Real, vDim>* vMean,
        amrex::GpuArray<amrex::Real, vDim>* vDev,
        amrex::GpuArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* funcvDev,
        VelocityInitScheme* vScheme,
        amrex::Real* velCum,
        size_t numGauss) :
        m_vMeanPtr{vMean},
        m_vDevPtr{vDev},
        m_funcvDevPtr{funcvDev},
        m_vSchemePtr{vScheme},
        m_velCumPtr{velCum},
        m_numGauss{numGauss}
    {
    }
};

// Helper to read input parameters for velocity initialization and later initialize velocities
template <unsigned int vDim>
class VelocityInitializer
{
private:
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::Real, vDim>> m_vMeanGpu, m_vDevGpu;
    amrex::Gpu::DeviceVector<amrex::GpuArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>>
        m_funcvDevGpu;
    amrex::Gpu::DeviceVector<VelocityInitScheme> m_vSchemeGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_velCumGpu;

    amrex::Vector<amrex::GpuArray<amrex::Real, vDim>> m_vMean, m_vDev;
    amrex::Vector<amrex::GpuArray<amrex::Parser, vDim>> m_funcvDevParser;
    amrex::Vector<amrex::GpuArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>> m_funcvDev;
    amrex::Vector<VelocityInitScheme> m_vScheme;
    amrex::Vector<amrex::Real> m_velCum;

public:
    amrex::GpuArray<amrex::Real, vDim>* m_vMeanPtr;
    amrex::GpuArray<amrex::Real, vDim>* m_vDevPtr;
    amrex::GpuArray<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, vDim>* m_funcvDevPtr;
    VelocityInitScheme* m_vSchemePtr;
    amrex::Real* m_velCumPtr;

    size_t m_numGauss;

    VelocityInitializer(Gempic::Io::Parameters& params)
    {
        int numGauss{1};
        params.get_or_set("numGaussians", numGauss);
        m_numGauss = numGauss;
        m_vMean.resize(numGauss);
        m_vDev.resize(numGauss);
        m_funcvDevParser.resize(numGauss);
        m_funcvDev.resize(numGauss);
        m_vScheme.resize(numGauss);
        m_velCum.resize(numGauss);
        amrex::Vector<amrex::Real> vWeight(numGauss);
        for (int ng = 0; ng < numGauss; ng++)
        {
            std::string gaussianNumber = 'G' + std::to_string(ng) + '.';
            amrex::Vector<amrex::Real> vm, vt;
            std::string vmString = gaussianNumber + "vMean";
            params.get(vmString, vm);
            for (int i{0}; i < vDim; ++i) m_vMean[ng][i] = vm[i];
            std::string vtString = gaussianNumber + "vThermal";
            if (params.exists(vtString + ".x"))
            {
                m_vScheme[ng] = VelocityInitScheme::function;
                std::tie(m_funcvDevParser[ng][xDir], m_funcvDev[ng][xDir]) =
                    Utils::parse_function(vtString + ".x", params);
                if constexpr (vDim > 1)
                {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.exists(vtString + ".y"),
                                                     "vThermal must be given in all directions");
                    std::tie(m_funcvDevParser[ng][yDir], m_funcvDev[ng][yDir]) =
                        Utils::parse_function(vtString + ".y", params);
                    if constexpr (vDim > 2)
                    {
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                            params.exists(vtString + ".z"),
                            "vThermal must be given in all directions");
                        std::tie(m_funcvDevParser[ng][zDir], m_funcvDev[ng][zDir]) =
                            Utils::parse_function(vtString + ".z", params);
                    }
                }
                if (params.exists(vtString))
                {
                    amrex::Warning("vThermal constant string not read if functions are given");
                }
            }
            else
            {
                m_vScheme[ng] = VelocityInitScheme::constant;
                params.get(vtString, vt);
                for (int i{0}; i < vDim; ++i) m_vDev[ng][i] = vt[i];
            }
            std::string vwString = gaussianNumber + "vWeight";
            amrex::Real vw;
            params.get(vwString, vw);
            vWeight[ng] = vw;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(1.0 - std::reduce(vWeight.begin(), vWeight.end())) < 1e-12,
            "Gaussian weights must sum to 1.0!");
        std::partial_sum(vWeight.begin(), vWeight.end(), m_velCum.begin());

        m_vMeanPtr = m_vMean.data();
        m_vDevPtr = m_vDev.data();
        m_funcvDevPtr = m_funcvDev.data();
        m_vSchemePtr = m_vScheme.data();
        m_velCumPtr = m_velCum.data();
    }

    VelocityInitializerGpu<vDim> get_gpu_initializer ()
    {
        m_vMeanGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vMean.begin(), m_vMean.end(),
                              m_vMeanGpu.begin());
        m_vDevGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vDev.begin(), m_vDev.end(),
                              m_vDevGpu.begin());
        m_funcvDevGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_funcvDev.begin(), m_funcvDev.end(),
                              m_funcvDevGpu.begin());
        m_vSchemeGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_vScheme.begin(), m_vScheme.end(),
                              m_vSchemeGpu.begin());
        m_velCumGpu.resize(m_numGauss);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_velCum.begin(), m_velCum.end(),
                              m_velCumGpu.begin());
        return VelocityInitializerGpu<vDim>(m_vMeanGpu.dataPtr(), m_vDevGpu.dataPtr(),
                                            m_funcvDevGpu.dataPtr(), m_vSchemeGpu.dataPtr(),
                                            m_velCumGpu.dataPtr(), m_numGauss);
    }
};

} //namespace Impl

template <template <unsigned int> class VInit, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void get_velocities (
    const VInit<vDim>& vIniter,
    amrex::GpuArray<amrex::Real, vDim>& velocities,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM + 3 * vDim>& randomArray,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& location)
{
    using namespace Impl;
    auto gaussIdx{get_gauss_ids(vIniter, randomArray)};

    auto epsilon{std::numeric_limits<amrex::Real>::epsilon()};
    for (int cc = 0; cc < vDim; cc++)
    {
        amrex::Real u1 = randomArray[AMREX_SPACEDIM + 2 * cc];
        amrex::Real u2 = randomArray[AMREX_SPACEDIM + 2 * cc + 1];
        // make sure epsilon large enough to take log
        // first value from Box-Muller algorithm
        u1 = u1 < epsilon ? epsilon : u1;
        amrex::Real z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
        velocities[cc] = get_gaussian_velocity(vIniter, gaussIdx[cc], z0, cc, location);
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

AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inline cell_center_from_corner(
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellCorner,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellCenter;

    AMREX_D_TERM(cellCenter[xDir] = cellCorner[xDir] + 0.5 * dx[xDir];
                 , cellCenter[yDir] = cellCorner[yDir] + 0.5 * dx[yDir];
                 , cellCenter[zDir] = cellCorner[zDir] + 0.5 * dx[zDir];)
    return cellCenter;
}

template <typename Type, unsigned int length>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<Type, length> inline operator+(
    amrex::GpuArray<Type, length> a, amrex::GpuArray<Type, length> b)
{
    amrex::GpuArray<Type, length> c;
    for (int i{0}; i < length; ++i)
    {
        c[i] = a[i] + b[i];
    }
    return c;
}

/**
 * @brief particle initializer
 *
 * @tparam vdim : Number of velocity components associated with the particles
 * @tparam ndata : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] particles : particles that are initialised
 * @param infra : Computational Domain
 *
 */
template <unsigned int vDim, unsigned int nData>
inline void init_particles (std::vector<std::shared_ptr<ParticleGroups<vDim, nData>>>& particles,
                            ComputationalDomain infra)
{
    BL_PROFILE("Gempic::Particle::init_particles(infra,particles)");
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
            std::make_shared<ParticleGroups<vDim, nData>>(speciesNames[species], infra);

        sample_species(particles[species], infra, species, globalSamplerName, rSeed);
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
 * @param infra : Computational Domain
 *
 */
template <unsigned int vDim, unsigned int nData>
inline void init_particles (
    std::vector<std::shared_ptr<ParticleGroupsLinVlasov<vDim, nData>>>& particlesLinVlasov,
    std::vector<std::shared_ptr<ParticleGroups<vDim, nData>>>& particles,
    ComputationalDomain infra)
{
    BL_PROFILE("Gempic::Particle::init_particles(infra,particlesLinVlasov,particles)");
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
            std::make_shared<ParticleGroupsLinVlasov<vDim, nData>>(speciesNames[species], infra);
        particles[species] =
            std::static_pointer_cast<ParticleGroups<vDim, nData>>(particlesLinVlasov[species]);

        sample_species(particles[species], infra, species, globalSamplerName, rSeed);
    }
}

/**
 * @brief
 *
 * @tparam vdim : Number of velocity components associated with the particles
 * @tparam ndata : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] partGr : Particle group
 * @param infra : Computational Domain
 * @param species :
 * @param globalSamplerName :
 * @param rSeed :
 *
 */
template <unsigned int vDim, unsigned int nData>
inline void sample_species (std::shared_ptr<ParticleGroups<vDim, nData>>& partGr,
                            ComputationalDomain infra,
                            int species,
                            std::string globalSamplerName,
                            int rSeed)
{
    BL_PROFILE("Gempic::Particle::sample_species(infra,partGr,species,globalSamplerName,rSeed)");

    std::string speciesName{partGr->get_name()};
    Io::Parameters params("Particle." + speciesName);
    std::string samplerName{globalSamplerName};
    params.get_or_set("sampler", samplerName);

    if (samplerName == "Cellwise")
    {
        init_particles_cellwise(partGr, infra, rSeed);
    }
    else if (samplerName == "PseudoRandom")
    {
        init_particles_full_domain(partGr, infra, species, rSeed);
    }
    else if (samplerName == "Sobol")
    {
        init_particles_full_domain_sobol(partGr, infra, species, rSeed);
    }
    else if (samplerName == "FullDomainGpu")
    {
        init_particles_full_domain_gpu(partGr, infra, species, rSeed);
    }
    else
    {
        amrex::Abort("Unknown particle initialization method: " + samplerName);
    }
}

/**
 * @brief
 *
 * @tparam vdim : Number of velocity components associated with the particles
 * @tparam ndata : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] ppg : Particle group
 * @param infra : Computational Domain
 * @param rSeed :
 *
 */
using ParticleType = amrex::Particle<0, 0>; // For SoA attributes
template <unsigned int vDim, unsigned int nData>
void init_particles_cellwise (std::shared_ptr<ParticleGroups<vDim, nData>>& ppg,
                              ComputationalDomain infra,
                              int rSeed = 925)
{
    BL_PROFILE("Gempic::Particle::init_particles_cellwise(infra,ppg,rSeed)");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> internalPosition; // coordinates within each cell
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;         // global coordinates
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellCorner;       // cell-corner coordinates
    amrex::GpuArray<amrex::Real, vDim> velocity;
    amrex::Real weight;

    std::string speciesName{ppg->get_name()};
    std::string specString = "Particle." + speciesName;
    Io::Parameters params(specString, "function init_particles_cellwise (" + speciesName + ')');

    int nPartPerCell;
    params.get("nPartPerCell", nPartPerCell);
    auto [densityParse, densityEval] = Utils::parse_function("density", params);

    Impl::VelocityInitializer<vDim> vInit(params);

    // (0,1)-normally distributed random number generator:
    // srand(rSeed);
    std::mt19937 gen(rSeed);
    std::mt19937 genv(rSeed + 17);
    // std::normal_distribution<> norm01(0,1);
    std::uniform_real_distribution<> unif(0.0, 1.0);
    std::uniform_real_distribution<> unifv(0.0, 1.0);

    for (int pp = 0; pp < nPartPerCell; pp++)
    {
        auto precomputedRand{Impl::calculate_random_values<vDim>(unif, gen, unifv, genv)};
        // internalPosition in model cell [0,dx]x[0,dy]x[0,dz] (uniform in [0, dx]):
        AMREX_D_TERM(
            internalPosition[xDir] = precomputedRand[xDir] * infra.geometry().CellSize(xDir);
            , internalPosition[yDir] = precomputedRand[yDir] * infra.geometry().CellSize(yDir);
            , internalPosition[zDir] = precomputedRand[zDir] * infra.geometry().CellSize(zDir);)

        // Adding particle from the model cell to all cells
        for (amrex::MFIter mfi = ppg->MakeMFIter(0); mfi.isValid(); ++mfi)
        {
            amrex::ParticleTile<ParticleType, vDim + nData, 0>& particles =
                ppg->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            const amrex::Box& bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};

            GEMPIC_D_LOOP_BEGIN(for (int l = lo[xDir]; l <= hi[xDir]; l++),
                                for (int j = lo[yDir]; j <= hi[yDir]; j++),
                                for (int k = lo[zDir]; k <= hi[zDir]; k++))
                cellCorner =
                    get_cell_corner(infra.geometry().ProbLoArray(),
                                    infra.geometry().CellSizeArray(), AMREX_D_DECL(l, j, k));
                position = internalPosition + cellCorner;

                amrex::Real result =
                    densityEval(AMREX_D_DECL(position[xDir], position[yDir], position[zDir]), 0.);
                get_velocities(
                    vInit, velocity, precomputedRand,
                    cell_center_from_corner(cellCorner, infra.geometry().CellSizeArray()));
                weight = result / nPartPerCell * infra.geometry().CellSizeArray().product();

                ppg->add_particle(position, velocity, weight, particles);
            GEMPIC_D_LOOP_END
        }
    }
}

AMREX_GPU_HOST_DEVICE inline int get_cellid (
    int i, int j, int k, amrex::IntVect lo, amrex::IntVect n)
{
    AMREX_D_TERM(int ix = i - lo[xDir];, int iy = j - lo[yDir];, int iz = k - lo[zDir];)
    AMREX_D_PICK(return ix;, return iy * n[xDir] + ix;, return (iz * n[yDir] + iy) * n[xDir] + ix;)
}

/**
 * @brief
 *
 * @tparam vdim : Number of velocity components associated with the particles
 * @tparam ndata : Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] ppg : Particle group
 * @param infra : Computational Domain
 * @param species :
 * @param rSeed :
 *
 */
template <unsigned int vDim, unsigned int nData = 1>
void init_particles_full_domain_gpu (std::shared_ptr<ParticleGroups<vDim, nData>>& ppg,
                                     ComputationalDomain infra,
                                     int species,
                                     int rSeed = 925)
{
    BL_PROFILE("Gempic::Particle::init_particles_full_domain_gpu()");

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probLo{infra.m_geom.ProbLoArray()};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{infra.geometry().CellSizeArray()};

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

    Impl::VelocityInitializer<vDim> vInit(params);
    auto vInitGpu{vInit.get_gpu_initializer()};

    for (amrex::MFIter mfi = ppg->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect boxSize{bx.size()};
        int nCellsTile = GEMPIC_D_MULT(boxSize[xDir], boxSize[yDir], boxSize[zDir]);
        int npart = nPartPerCell * nCellsTile;
        auto& particles = ppg->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        // resize particles structure before launching on GPU
        auto oldSize = particles.GetArrayOfStructs().size();
        auto newSize = oldSize + npart;
        particles.resize(newSize);

        ParticleType* AMREX_RESTRICT pstruct = particles.GetArrayOfStructs()().data();
        amrex::GpuArray<amrex::ParticleReal*, vDim> velArray;
        for (int i = 0; i < vDim; i++)
        {
            velArray[i] = particles.GetStructOfArrays().realarray()[i];
        }
        amrex::ParticleReal* weightArray = particles.GetStructOfArrays().realarray()[vDim];
        int procID = amrex::ParallelDescriptor::MyProc();

        amrex::InitRandom(rSeed + GEMPIC_D_ADD(lo[xDir], lo[yDir] * 10, lo[zDir] * 100) +
                          species * 1000);

        ParallelForRNG(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const& gen) noexcept
            {
                int cellid = get_cellid(i, j, k, lo, boxSize);

                int pidx = cellid * nPartPerCell;
                auto cellCorner{get_cell_corner(probLo, dx, AMREX_D_DECL(i, j, k))};
                auto cellCenter{cell_center_from_corner(cellCorner, dx)};

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
                amrex::GpuArray<amrex::Real, vDim> velocity;
                amrex::Real t = 0.0; // initial time
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM + 3 * vDim> randArray;

                for (int pp = 0; pp < nPartPerCell; pp++)
                {
                    for (auto& randomValue : randArray)
                    {
                        randomValue = amrex::Random(gen);
                    }
                    for (int comp = 0; comp < AMREX_SPACEDIM; comp++)
                    {
                        position[comp] = cellCorner[comp] + randArray[comp] * dx[comp];
                    }
                    get_velocities(vInitGpu, velocity, randArray, cellCenter);

                    const double factor = densityEval(
                        AMREX_D_DECL(position[xDir], position[yDir], position[zDir]), t);
                    amrex::Real weight = factor / nPartPerCell * dx.product();
                    ParticleType& p = pstruct[pidx];
                    p.id() = pidx;
                    p.cpu() = procID;
                    AMREX_D_TERM(p.pos(xDir) = position[xDir];, p.pos(yDir) = position[yDir];
                                 , p.pos(zDir) = position[zDir];)
                    for (int i = 0; i < vDim; i++)
                    {
                        velArray[i][pidx] = velocity[i];
                    }
                    weightArray[pidx] = weight;

                    ++pidx;
                }
            });
    }
    amrex::Print() << "init_full_domain_gpu OK\n";
}

/**
 * @brief Random number generator. Particle positions are generated using a uniform random
 * distribution giving position on each cell and then remapped to full domain.
 * Velocity distribution is generated from uniform distribution using the Box-Muller algorithm
 * as random_normal is machine dependent. This enables for a given seed to always get the same
 * random sequence
 *
 * @tparam vDim Number of velocity components associated with the particles
 * @tparam nData Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] ppg : Particle group
 * @param infra : Computational Domain
 * @param species :
 * @param rSeed :
 */
template <unsigned int vDim, unsigned int nData>
void init_particles_full_domain (std::shared_ptr<ParticleGroups<vDim, nData>>& ppg,
                                 ComputationalDomain infra,
                                 int species,
                                 int rSeed = 925)
{
    BL_PROFILE("Gempic::Particle::init_particles_full_domain()");
    amrex::Real t{0.0};

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
    amrex::GpuArray<amrex::Real, vDim> velocity;
    amrex::Real weight;

    std::string speciesName{ppg->get_name()};
    std::string specString = "Particle." + speciesName;
    Io::Parameters params(specString, "function init_particles_full_domain (" + speciesName + ')');

    int nPartPerCell;
    params.get("nPartPerCell", nPartPerCell);
    auto [densityParse, densityEval] = Utils::parse_function("density", params);

    Impl::VelocityInitializer<vDim> vInit(params);

    //#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    for (amrex::MFIter mfi = ppg->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        std::mt19937 gen(GEMPIC_D_ADD(rSeed + lo[xDir], lo[yDir] * 10, lo[zDir] * 100) +
                         species * 1000);
        std::mt19937 genv(GEMPIC_D_ADD(rSeed + lo[xDir], lo[yDir] * 10, lo[zDir] * 100) +
                          species * 1000 + 17);
        std::uniform_real_distribution<> unif(0.0, 1.0);
        std::uniform_real_distribution<> unifv(0.0, 1.0);

        using myParticleTile = amrex::ParticleTile<ParticleType, vDim + nData, 0>;
        myParticleTile& particles =
            ppg->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        GEMPIC_D_LOOP_BEGIN(for (int l = lo[xDir]; l <= hi[xDir]; l++),
                            for (int j = lo[yDir]; j <= hi[yDir]; j++),
                            for (int k = lo[zDir]; k <= hi[zDir]; k++))
            auto cellCorner{get_cell_corner(infra.m_geom.ProbLoArray(),
                                            infra.geometry().CellSizeArray(),
                                            AMREX_D_DECL(l, j, k))};
            auto cellCenter{cell_center_from_corner(cellCorner, infra.geometry().CellSizeArray())};
            for (int pp = 0; pp < nPartPerCell; pp++)
            {
                auto randArray{Impl::calculate_random_values<vDim>(unif, gen, unifv, genv)};
                for (int i = 0; i < AMREX_SPACEDIM; i++)
                {
                    position[i] = cellCorner[i] + randArray[i] * infra.geometry().CellSize(i);
                }
                get_velocities(vInit, velocity, randArray, cellCenter);

                const double factor =
                    densityEval(AMREX_D_DECL(position[xDir], position[yDir], position[zDir]), t);
                weight = factor / nPartPerCell * infra.geometry().CellSizeArray().product();
                ppg->add_particle(position, velocity, weight, particles);
            }
        GEMPIC_D_LOOP_END
    }
}

/**
 * @brief Random number generator. Particle positions are generated using quasirandom
 * numbers from the Sobol sequence giving position on each cell and then remapped to full domain.
 * Velocity distribution is generated from quasirandom numbers from the Sobol sequence using the
 * Box-Muller algorithm
 *
 * @tparam vDim Number of velocity components associated with the particles
 * @tparam nData Number of further amount of data components to be stored, e.g. particle weight
 *
 * @param[out] ppg : Particle group
 * @param infra : Computational Domain
 * @param species :
 * @param rSeed :
 */
template <unsigned int vDim, unsigned int nData>
void init_particles_full_domain_sobol (std::shared_ptr<ParticleGroups<vDim, nData>>& ppg,
                                       ComputationalDomain infra,
                                       int species,
                                       int rSeed = 925)
{
    BL_PROFILE("Gempic::Particle::init_particles_fulldomain_sobol()");
    amrex::Real t{0.0};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM + 3 * vDim> randArray;
    amrex::GpuArray<amrex::Real, vDim> velocity;
    amrex::Real weight;

    std::string speciesName{ppg->get_name()};
    std::string specString = "Particle." + speciesName;
    Io::Parameters params(specString, "function init_particles_full_domain (" + speciesName + ')');

    int nPartPerCell;
    params.get("nPartPerCell", nPartPerCell);
    auto [densityParse, densityEval] = Utils::parse_function("density", params);

    Impl::VelocityInitializer<vDim> vInit(params);

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probLo = infra.geometry().ProbLoArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellSize = infra.geometry().CellSizeArray();

    //#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    for (amrex::MFIter mfi = ppg->MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boxLo;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boxLength;

        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            boxLo[dir] = probLo[dir] + cellSize[dir] * lo[dir];
            boxLength[dir] = cellSize[dir] * bx.length(dir);
        }

        // Pseudorandom number generator to switch between multiple Gaussians
        std::mt19937 gen(GEMPIC_D_ADD(rSeed + lo[xDir], lo[yDir] * 10, lo[zDir] * 100) +
                         species * 1000);
        std::uniform_real_distribution<> unif(0.0, 1.0);

        long long int sobolSeed = gen();
        constexpr int numDim = AMREX_SPACEDIM + 2 * vDim;
        double quasi[numDim];

        using myParticleTile = amrex::ParticleTile<ParticleType, vDim + nData, 0>;
        myParticleTile& particles =
            ppg->GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        // sample on box and not cell to get a low-discrepancy sequence on each box
        // and prevent correlations between neighboring cells
        int nPartBox = nPartPerCell * bx.numPts();

        for (int pp = 0; pp < nPartBox; pp++)
        {
            Gempic::Particle::sobol(numDim, &sobolSeed, quasi);
            // two random variables per vDim are calculated for the velocity distribution...
            for (int i{0}; i < numDim; ++i)
            {
                randArray[i] = quasi[i];
            }
            // ... one needs to be generated for Gaussian selection.
            for (int i{0}; i < vDim; ++i)
            {
                randArray[AMREX_SPACEDIM + 2 * vDim + i] = unif(gen);
            }

            for (int i = 0; i < AMREX_SPACEDIM; i++)
            {
                position[i] = boxLo[i] + randArray[i] * boxLength[i];
            }

            get_velocities(vInit, velocity, randArray, position);
            const double factor =
                densityEval(AMREX_D_DECL(position[xDir], position[yDir], position[zDir]), t);
            weight = factor / nPartPerCell * infra.geometry().CellSizeArray().product();
            ppg->add_particle(position, velocity, weight, particles);
        }
    }
}

template <unsigned int vDim>
void init_one_particle_cellwise (amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                                 ParticleGroups<vDim>& pg,
                                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> internalPosition,
                                 amrex::GpuArray<amrex::Real, vDim> velocity = {1., 1., 1.})
{
    BL_PROFILE("Gempic::Particle::init_one_particle_cellwise()");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellCorner;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;

    // MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi = pg.MakeMFIter(0); mfi.isValid(); ++mfi)
    { // 0 is level
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        auto& particles = pg.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        GEMPIC_D_LOOP_BEGIN(for (int i = lo[xDir]; i <= hi[xDir]; i++),
                            for (int j = lo[yDir]; j <= hi[yDir]; j++),
                            for (int k = lo[zDir]; k <= hi[zDir]; k++))
            cellCorner = get_cell_corner(plo, dx, AMREX_D_DECL(i, j, k));
            position = internalPosition + cellCorner;

            ParticleType p;
            p.id() = ParticleType::NextID();
            p.cpu() = amrex::ParallelDescriptor::MyProc();

            AMREX_D_TERM(p.pos(xDir) = position[xDir];, p.pos(yDir) = position[yDir];
                         , p.pos(zDir) = position[zDir];)

            // p.rdata(0) = velocity[xDir];
            particles.push_back_real(0, velocity[xDir]);
            if (vDim > 1)
            {
                // p.rdata(1) = velocity[yDir];
                particles.push_back_real(1, velocity[yDir]);
            }
            if (vDim > 2)
            {
                // p.rdata(2) = velocity[zDir];
                particles.push_back_real(2, velocity[zDir]);
            }

            // p.rdata(vDim) = 1.0/64.0; //weight
            particles.push_back_real(vDim, 1.0 / 64.0); // weight

            particles.push_back(p);
        GEMPIC_D_LOOP_END
    }
}

} // namespace Gempic::Particle

#endif

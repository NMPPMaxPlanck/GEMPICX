#ifndef PARTICLE_GROUPS_H_
#define PARTICLE_GROUPS_H_

#include <AMReX.H>
#include <AMReX_ArrayOfStructs.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Parser.H>
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_StructOfArrays.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_ParticleDataIndices.H"

namespace Gempic::Particle
{
template <unsigned int vDim, unsigned int nData = 1>
class ParticleGroups;

namespace Impl
{
// Convenience iterators for ParticleGroups. Allows for range-based for loops.
template <unsigned int vDim, unsigned int nData>
struct ParticleGroupIterator
{
    // When using SoA particles, the number of reals must explicitly include AMREX_SPACEDIM
    using ParIterType = amrex::ParIterSoA<vDim + nData + AMREX_SPACEDIM, 0>;
    std::unique_ptr<ParIterType> m_particleIter;

    // While references to pointers are normally dangerous, the iterators are only expected to last
    // for the duration of the loop.
    ParIterType& operator*() { return *m_particleIter; }

    void operator++() { m_particleIter->operator++(); }

    bool operator==(ParticleGroupIterator const& other) const
    {
        if (m_particleIter && m_particleIter->isValid())
        {
            // if both iterators are valid, check if they point to the same tile
            return (other.m_particleIter && other.m_particleIter->isValid() &&
                    (m_particleIter == other.m_particleIter));
        }
        else
        {
            // if both iterators are empty/invalid, they are equal
            return !(other.m_particleIter && other.m_particleIter->isValid());
        }
    }

    bool operator!=(ParticleGroupIterator const& other) const { return not(*this == other); }

private:
    // ParticleGroupIterator should only be constructed by the ParticleGroups class
    friend class ParticleGroups<vDim, nData>;
    ParticleGroupIterator(ParticleGroups<vDim, nData>& particles)
    {
        m_particleIter = std::make_unique<ParIterType>(particles, 0);
    }

    ParticleGroupIterator() {}
};
} // namespace Impl

/**
 * Class containing all particle information. The array of particle containers has @p n_species
 * number of particles containers, that is each responsible for one species of particles. For each
 * particle position, velocity and weight are stored in a ArrayOfStructs manner. The charge and
 * mass of the particle species are stored in a separate array.
 */
template <unsigned int vDim, unsigned int nData>
class ParticleGroups : public amrex::ParticleContainerPureSoA<AMREX_SPACEDIM + vDim + nData, 0>
{
private:
    amrex::ParticleReal m_charge = 1;
    amrex::ParticleReal m_mass = 1;
    std::string m_name = "particles";
    amrex::Vector<std::string> m_compNames = {"vx", "vy", "vz", "weight"};

public:
    amrex::ParticleReal get_charge () const { return m_charge; }
    amrex::ParticleReal get_mass () const { return m_mass; }
    std::string get_name () const { return m_name; }
    amrex::Vector<std::string> get_comp_names () const { return m_compNames; }
    /**
     * Default Constructor.
     */
    ParticleGroups() {}
    ParticleGroups(amrex::Geometry const& geom,
                   amrex::DistributionMapping const& distriMap,
                   amrex::BoxArray const& grid) :
        amrex::ParticleContainerPureSoA<AMREX_SPACEDIM + vDim + nData, 0>(geom, distriMap, grid)
    {
        m_compNames.resize(vDim + nData);
        if constexpr (vDim + nData > 0 && vDim + nData < 4)
        {
            m_compNames[vDim + nData - 1] = "weight";
        }
    }
    /**
     * Explicit Constructors: set up the empty particle object.
     * The particles themselves need to be added separately. This can be done one by one (see
     * method add_particle) or with the particle samplers
     * \ref init_particles(std::vector<std::shared_ptr<ParticleGroups<vDim, nData>>>&
       particles, ComputationalDomain infra)
     */
    ParticleGroups(amrex::Real charge,
                   amrex::Real mass,
                   std::string const& name,
                   ComputationalDomain const& ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_charge = charge;
        m_mass = mass;
        m_name = name;
    }
    ParticleGroups(amrex::Real charge, amrex::Real mass, ComputationalDomain const& ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_charge = charge;
        m_mass = mass;
    }

    ParticleGroups(int species, ComputationalDomain const& ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        std::string specString = "Particle.species" + std::to_string(species);
        Io::Parameters params(specString);
        params.get_or_set("charge", m_charge);
        params.get_or_set("mass", m_mass);
        params.get_or_set("name", m_name);
    }

    ParticleGroups(std::string const& speciesName, ComputationalDomain const& ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_name = speciesName;
        std::string specString = "Particle." + speciesName;
        Io::Parameters params(specString, "class ParticleGroups (" + speciesName + ')');
        params.get_or_set("charge", m_charge);
        params.get_or_set("mass", m_mass);
        params.get_or_set("compNames", m_compNames);
    }

    /**
     * adds one particle with the space coordinates @p position, the velocity coordinates @p
     * velocity and the weight @p weight to the Particle Tile @p particles.
     */

    void add_particle (amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position,
                       amrex::GpuArray<amrex::Real, vDim> velocity,
                       amrex::Real weight,
                       amrex::ParticleTile<amrex::SoAParticle<AMREX_SPACEDIM + vDim + nData, 0>,
                                           AMREX_SPACEDIM + vDim + nData,
                                           0>& particles)
    {
        auto oldNumParticles = particles.numParticles();
        particles.resize(oldNumParticles + 1);
        particles.id(oldNumParticles) =
            amrex::SoAParticle<AMREX_SPACEDIM + vDim + nData, 0>::NextID();
        particles.cpu(oldNumParticles) = amrex::ParallelDescriptor::MyProc();
        //auto ptd = particles.getParticleTileData();

        // add new position
        for (int cc = 0; cc < AMREX_SPACEDIM; cc++)
        {
            particles.GetStructOfArrays().GetRealData(cc)[oldNumParticles] = position[cc];
        }
        // add new velocity
        for (int cc = 0; cc < vDim; cc++)
        {
            particles.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + cc)[oldNumParticles] =
                velocity[cc];
        }
        // add new weight
        particles.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + vDim)[oldNumParticles] = weight;
        for (int cc = 1; cc < nData; cc++)
        {
            particles.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + vDim + cc)[oldNumParticles] =
                0.0;
        }
        amrex::Gpu::streamSynchronize();
    }

    void copy (ParticleGroups<vDim, nData>& source, bool local = false)
    {
        BL_PROFILE("Gempic::Particle::ParticleGroups::copy()");
        this->m_charge = source.get_charge();
        this->m_mass = source.get_mass();
        this->m_name = source.get_name();
        this->m_compNames = source.get_comp_names();
        this->copyParticles(source, local);
    }

    /**
     * Resets the total weight of the particle group to the specified target weight.
     * This is needed to correct sampling errors for low particle counts and non-symmetric velocity
     * distributions.
     *
     * @param targetWeight the new total weight of the particle group
     */
    void reset_total_weight (amrex::Real targetWeight)
    {
        BL_PROFILE("Gempic::Particle::ParticleGroups::reset_total_weight(totalWeight)");

        // Compute total weight of particle group
        amrex::ReduceOps<amrex::ReduceOpSum> reduceOps;
        auto ii = this->get_data_indices();
        using PTDType = typename amrex::ParticleContainerPureSoA<
            AMREX_SPACEDIM + vDim + nData, 0>::ParticleTileType::ConstParticleTileDataType;
        auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<amrex::Real>>(
            *this,
            [=] AMREX_GPU_HOST_DEVICE(PTDType const& pt,
                                      int const pp) -> amrex::GpuTuple<amrex::Real>
            {
                auto const weight = pt.rdata(ii.m_iweight)[pp];
                return weight;
            },
            reduceOps);

        amrex::Real W = amrex::get<0>(particleTmp);

        // reduced sum over mpi ranks
        amrex::ParallelDescriptor::ReduceRealSum(W);

        // Remove average weight from all particles
        auto totalNumberOfParticles = this->TotalNumberOfParticles();
        amrex::Real averageDiff = (targetWeight - W) / totalNumberOfParticles;
        for (auto& particleGrid : *this)
        {
            long const np = particleGrid.numParticles();
            auto* const weight = particleGrid.GetStructOfArrays().GetRealData(vDim).data();
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) { weight[pp] += averageDiff; });
        }
    }

    Impl::ParticleGroupIterator<vDim, nData> begin ()
    {
        return Impl::ParticleGroupIterator<vDim, nData>(*this);
    }

    Impl::ParticleGroupIterator<vDim, nData> end ()
    {
        return Impl::ParticleGroupIterator<vDim, nData>();
    }

    /** \brief Provides named array indices for SoA list of arrays.
     *
     * This method shall hide verbose expressions such as
     * `AMREX_SPACEDIM + vDim`, to help provide "clean physics code blocks".
     *
     * This method establishes the following convention for "physical quantity" --- "array of SoA":
     *
     *     * x component of position = rdata(0)
     *     * y component of position = rdata(1)
     *     * z component of position = rdata(2)
     *     * x component of velocity = rdata(AMREX_SPACEDIM)
     *     * y component of velocity = rdata(AMREX_SPACEDIM + 1)
     *     * z component of velocity = rdata(AMREX_SPACEDIM + 2)
     *     * particle weight         = rdata(AMREX_SPACEDIM + vDim)
     *     * s0                      = rdata(AMREX_SPACEDIM + vDim + 1)
     *     * sqrtf0                  = rdata(AMREX_SPACEDIM + vDim + 2)
     *     * s2X                     = rdata(AMREX_SPACEDIM + vDim + 1)
     *     * s2Y                     = rdata(AMREX_SPACEDIM + vDim + 2)
     *     * s2Z                     = rdata(AMREX_SPACEDIM + vDim + 3)
     *     * s2Vx                    = rdata(AMREX_SPACEDIM + vDim + 4)
     *     * s2Vy                    = rdata(AMREX_SPACEDIM + vDim + 5)
     *     * s2Vz                    = rdata(AMREX_SPACEDIM + vDim + 6)
     *
     * See also `GEMPIC_ParticleDataIndices.H`.
     */

    Gempic::Particle::ParticleDataIndices get_data_indices () const
    {
        Gempic::Particle::ParticleDataIndices dd;
        // set position arrays
        dd.m_iposx = 0;
        if (AMREX_SPACEDIM > 1) dd.m_iposy = 1;
        if (AMREX_SPACEDIM > 2) dd.m_iposz = 2;
        // set velocity arrays
        if (vDim > 0) dd.m_ivelx = AMREX_SPACEDIM;
        if (vDim > 1)
        {
            dd.m_ively = AMREX_SPACEDIM + 1;
        }
        if (vDim > 2)
        {
            dd.m_ivelz = AMREX_SPACEDIM + 2;
        }
        // set weight
        if (nData > 0)
        {
            dd.m_iweight = AMREX_SPACEDIM + vDim;
        }
        // set other
        if (nData > 1)
        {
            dd.m_is0 = AMREX_SPACEDIM + vDim + 1;
            dd.m_is2X = AMREX_SPACEDIM + vDim + 1;
        }
        if (nData > 2)
        {
            dd.m_isqrtf0 = AMREX_SPACEDIM + vDim + 2;
            dd.m_is2Y = AMREX_SPACEDIM + vDim + 2;
        }
        if (nData > 3)
        {
            dd.m_is2Z = AMREX_SPACEDIM + vDim + 3;
        }
        if (nData > 4)
        {
            dd.m_is2Vx = AMREX_SPACEDIM + vDim + 4;
        }
        if (nData > 5)
        {
            dd.m_is2Vy = AMREX_SPACEDIM + vDim + 5;
        }
        if (nData > 6)
        {
            dd.m_is2Vz = AMREX_SPACEDIM + vDim + 6;
        }
        return dd;
    }
};

/**
 * Subclass for linearized Vlasov simulation. Every species needs density and thermal velocity of
 * the equilibrium distribution function around which linearization was performed.
 */
template <unsigned int vDim, unsigned int nData = 1>
class ParticleGroupsLinVlasov : public ParticleGroups<vDim, nData>
{
private:
    amrex::Parser m_densityBackground;
    amrex::ParticleReal m_vThermalBackground = 1;

public:
    amrex::Parser get_density_background () { return m_densityBackground; }
    amrex::ParticleReal get_v_thermal_background () { return m_vThermalBackground; }

    ParticleGroupsLinVlasov(std::string const& speciesName, ComputationalDomain ifr) :
        ParticleGroups<vDim, nData>(speciesName, ifr)
    {
        std::string specString = "Particle." + speciesName;
        Io::Parameters params(specString, "class ParticleGroups (" + speciesName + ')');
        auto [parser, eval] = Gempic::Utils::parse_function("densityBackground", params);
        m_densityBackground = parser;
        params.get_or_set("vThermalBackground", m_vThermalBackground);
    }
};
} // namespace Gempic::Particle

#endif

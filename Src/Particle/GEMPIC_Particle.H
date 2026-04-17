/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
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
class ParticleSpecies;

namespace Impl
{
// Convenience iterators for ParticleSpecies. Allows for range-based for loops.
template <unsigned int vDim, unsigned int nData>
struct ParticleSpeciesIterator
{
    // When using SoA particles, the number of reals must explicitly include AMREX_SPACEDIM
    using ParIterType = amrex::ParIterSoA<vDim + nData + AMREX_SPACEDIM, 0>;
    std::unique_ptr<ParIterType> m_particleIter;

    // While references to pointers are normally dangerous, the iterators are only expected to last
    // for the duration of the loop.
    ParIterType& operator*() { return *m_particleIter; }

    void operator++() { m_particleIter->operator++(); }

    bool operator==(ParticleSpeciesIterator const& other) const
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

    bool operator!=(ParticleSpeciesIterator const& other) const { return not(*this == other); }

private:
    // ParticleSpeciesIterator should only be constructed by the ParticleSpecies class
    friend class ParticleSpecies<vDim, nData>;
    ParticleSpeciesIterator(ParticleSpecies<vDim, nData>& particles, int level)
    {
        m_particleIter = std::make_unique<ParIterType>(particles, level);
    }

    ParticleSpeciesIterator() {}
};
} // namespace Impl

/**
 * Class containing all particle information. The array of particle containers has @p n_species
 * number of particles containers, that is each responsible for one species of particles. For each
 * particle position, velocity and weight are stored in a ArrayOfStructs manner. The charge and
 * mass of the particle species are stored in a separate array.
 */
template <unsigned int vDim, unsigned int nData>
class ParticleSpecies : public amrex::ParticleContainerPureSoA<AMREX_SPACEDIM + vDim + nData, 0>
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
    ParticleSpecies() {}
    ParticleSpecies(amrex::Geometry const& geom,
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
    ParticleSpecies(amrex::Vector<amrex::Geometry> const& geom,
                    amrex::Vector<amrex::DistributionMapping> const& distriMap,
                    amrex::Vector<amrex::BoxArray> const& grid,
                    amrex::Vector<int> refinementRatio) :
        amrex::ParticleContainerPureSoA<AMREX_SPACEDIM + vDim + nData, 0>(
            geom, distriMap, grid, refinementRatio)
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
     * \ref init_particles(std::vector<std::shared_ptr<ParticleSpecies<vDim, nData>>>&
       particles, ComputationalDomain infra)
     */
    ParticleSpecies(amrex::Real charge,
                    amrex::Real mass,
                    std::string const& name,
                    ComputationalDomain const& ifr) :
        ParticleSpecies(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_charge = charge;
        m_mass = mass;
        m_name = name;
    }
    ParticleSpecies(amrex::Real charge, amrex::Real mass, ComputationalDomain const& ifr) :
        ParticleSpecies(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_charge = charge;
        m_mass = mass;
    }

    ParticleSpecies(int species, ComputationalDomain const& ifr) :
        ParticleSpecies(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        std::string specString = "Particle.species" + std::to_string(species);
        Io::Parameters params(specString);
        params.get_or_set("charge", m_charge);
        params.get_or_set("mass", m_mass);
        params.get_or_set("name", m_name);
    }

    ParticleSpecies(std::string const& speciesName, ComputationalDomain const& ifr) :
        ParticleSpecies(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_name = speciesName;
        std::string specString = "Particle." + speciesName;
        Io::Parameters params(specString, "class ParticleSpecies (" + speciesName + ')');
        params.get_or_set("charge", m_charge);
        params.get_or_set("mass", m_mass);
        params.get_or_set("compNames", m_compNames);
    }

    // constructors for multi level simulations
    ParticleSpecies(amrex::Real charge,
                    amrex::Real mass,
                    std::string const& name,
                    amrex::Vector<amrex::Geometry> const& geom,
                    amrex::Vector<amrex::DistributionMapping> const& dm,
                    amrex::Vector<amrex::BoxArray> const& ba,
                    amrex::Vector<int> rr) :
        ParticleSpecies(geom, dm, ba, rr)
    {
        m_charge = charge;
        m_mass = mass;
        m_name = name;
    }

    ParticleSpecies(std::string const& speciesName,
                    amrex::Vector<amrex::Geometry> const& geom,
                    amrex::Vector<amrex::DistributionMapping> const& dm,
                    amrex::Vector<amrex::BoxArray> const& ba,
                    amrex::Vector<int> rr) :
        ParticleSpecies(geom, dm, ba, rr)
    {
        m_name = speciesName;
        std::string specString = "Particle." + speciesName;
        Io::Parameters params(specString, "class ParticleSpecies (" + speciesName + ')');
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
        for (int cc = 0; cc < static_cast<int>(vDim); cc++)
        {
            particles.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + cc)[oldNumParticles] =
                velocity[cc];
        }
        // add new weight
        particles.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + vDim)[oldNumParticles] = weight;
        for (unsigned int cc = 1; cc < nData; cc++)
        {
            particles.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + vDim + cc)[oldNumParticles] =
                0.0;
        }
        amrex::Gpu::streamSynchronize();
    }

    void copy (ParticleSpecies<vDim, nData>& source, bool local = false)
    {
        BL_PROFILE("Gempic::Particle::ParticleSpecies::copy()");
        this->m_charge = source.get_charge();
        this->m_mass = source.get_mass();
        this->m_name = source.get_name();
        this->m_compNames = source.get_comp_names();
        this->copyParticles(source, local);
    }

    Impl::ParticleSpeciesIterator<vDim, nData> begin (int level = 0)
    {
        return Impl::ParticleSpeciesIterator<vDim, nData>(*this, level);
    }

    Impl::ParticleSpeciesIterator<vDim, nData> end ()
    {
        return Impl::ParticleSpeciesIterator<vDim, nData>();
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
     *     * s2Vx                    = rdata(2 * AMREX_SPACEDIM + vDim + 1)
     *     * s2Vy                    = rdata(2 * AMREX_SPACEDIM + vDim + 2)
     *     * s2Vz                    = rdata(2 * AMREX_SPACEDIM + vDim + 3)
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
            if (AMREX_SPACEDIM > 1)
            {
                dd.m_is2Y = AMREX_SPACEDIM + vDim + 2;
            }
        }
        if (nData > 3 and AMREX_SPACEDIM > 2)
        {
            dd.m_is2Z = AMREX_SPACEDIM + vDim + 3;
        }
        if (nData > AMREX_SPACEDIM + 1 and vDim > 0)
        {
            dd.m_is2Vx = 2 * AMREX_SPACEDIM + vDim + 1;
        }
        if (nData > AMREX_SPACEDIM + 2 and vDim > 1)
        {
            dd.m_is2Vy = 2 * AMREX_SPACEDIM + vDim + 2;
        }
        if (nData > AMREX_SPACEDIM + 3 and vDim > 2)
        {
            dd.m_is2Vz = 2 * AMREX_SPACEDIM + vDim + 3;
        }
        return dd;
    }

    /**
     * @brief Overloads restarting to watch out for existing particles
     */
    void restart (std::string const& dir, std::string const& file, bool keepOldParticles = false)
    {
        // By default, AMReX's particle container Restart function keeps old particles.
        if (not keepOldParticles and this->TotalNumberOfParticles() > 0)
        {
            amrex::Warning(
                "Warning: Particles exist before restart operation. They will be deleted");
            this->clearParticles();
        }
        this->Restart(dir, file);
    }
};

/** \brief Helper struct for range-based for loops on multi-level particle containers
 *
 * For a given level `lev` the for loop can be written like
 * `for (auto& particleGrid : SelectLevel(particleSpecies, lev)) {...}`
 */
template <unsigned int vDim, unsigned int nData>
struct SelectLevel
{
    SelectLevel(std::shared_ptr<ParticleSpecies<vDim, nData>> particles, int level) :
        m_particles{particles}, m_level{level}
    {
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(0 <= m_level && m_level <= m_particles->finestLevel(),
                                          "Trying to iterate over level " +
                                              std::to_string(m_level) +
                                              " of particle container which does not exist.");
    }

    auto begin () { return m_particles->begin(m_level); }
    auto end () { return m_particles->end(); }

private:
    std::shared_ptr<ParticleSpecies<vDim, nData>> m_particles;
    int m_level{0};
};

/**
 * Subclass for linearized Vlasov simulation. Every species needs density and thermal velocity of
 * the equilibrium distribution function around which linearization was performed.
 */
template <unsigned int vDim, unsigned int nData = 1>
class ParticleSpeciesLinVlasov : public ParticleSpecies<vDim, nData>
{
private:
    amrex::Parser m_densityBackground;
    amrex::ParticleReal m_vThermalBackground = 1;

public:
    amrex::Parser get_density_background () { return m_densityBackground; }
    amrex::ParticleReal get_v_thermal_background () { return m_vThermalBackground; }

    ParticleSpeciesLinVlasov(std::string const& speciesName, ComputationalDomain ifr) :
        ParticleSpecies<vDim, nData>(speciesName, ifr)
    {
        std::string specString = "Particle." + speciesName;
        Io::Parameters params(specString, "class ParticleSpecies (" + speciesName + ')');
        auto [parser, eval] = Gempic::Utils::parse_function("densityBackground", params);
        m_densityBackground = parser;
        params.get_or_set("vThermalBackground", m_vThermalBackground);
    }
};
} // namespace Gempic::Particle

#endif

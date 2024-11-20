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

namespace Gempic::Particle
{
using ParticleType = amrex::Particle<0, 0>;  // SoA particle type
/**
 * Class containing all particle information. The array of particle containers has @p n_species
 * number of particles containers, that is each responsible for one species of particles. For each
 * particle the particle position, velocity and weight are stored in a ArrayOfStructs manner. The
 * charge and mass of the particle species are stored in a separate array.
 */
template <unsigned int vdim, unsigned int ndata = 1>
class ParticleGroups : public amrex::ParticleContainer<0, 0, vdim + ndata, 0>
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
    ParticleGroups(const amrex::Geometry &geom,
                   const amrex::DistributionMapping &distriMap,
                   const amrex::BoxArray &grid) :
        amrex::ParticleContainer<0, 0, vdim + ndata, 0>(geom, distriMap, grid)
    {
        if constexpr (vdim < 3)
        {
            m_compNames.resize(vdim + ndata);
            if constexpr (vdim + ndata > 0)
            {
                m_compNames[vdim + ndata - 1] = "weight";
            }
        }
    }
    // Explicit const
    /**
     * Explicit Constructor: By passing an object of class vlasov maxwell and infrastructure one can
     * set up the empty particle object. The particles themselves need to be added separately. This
     * can be done one by one (see method add_particle) or with the particle samples in the folder
     * sampler.
     */
    ParticleGroups(amrex::Real charge,
                   amrex::Real mass,
                   const std::string &name,
                   const ComputationalDomain &ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_charge = charge;
        m_mass = mass;
        m_name = name;
    }
    ParticleGroups(amrex::Real charge, amrex::Real mass, const ComputationalDomain &ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        m_charge = charge;
        m_mass = mass;
    }

    ParticleGroups(int species, const ComputationalDomain &ifr) :
        ParticleGroups(ifr.m_geom, ifr.m_distriMap, ifr.m_grid)
    {
        std::string specString = "Particle.species" + std::to_string(species);
        Io::Parameters params(specString);
        params.get_or_set("charge", m_charge);
        params.get_or_set("mass", m_mass);
        params.get_or_set("name", m_name);
    }

    ParticleGroups(const std::string &speciesName, const ComputationalDomain &ifr) :
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
     * Method adds one particle with the space coordinates @p position, the velocity coordinates @p
     * velocity and the weight @p weight to the Particle Tile @p particles.
     */

    void add_particle (amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position,
                       amrex::GpuArray<amrex::Real, vdim> velocity,
                       amrex::Real weight,
                       amrex::ParticleTile<ParticleType, vdim + ndata, 0> &particles)
    // amrex::ParticleTile<0, 0, vdim + ndata, 0>& particles)
    {
        //BL_PROFILE("Gempic::Particle::ParticleGroups::add_particle()");
        // Array of Structures
        // new particle:
        ParticleType p;
        p.id() = ParticleType::NextID();
        p.cpu() = amrex::ParallelDescriptor::MyProc();

        for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
        {
            p.pos(cc) = position[cc];
        }

        // Creating attributes for Structure of Arrays container
        // amrex::GpuArray<amrex::Real,vdim+1> particleAttributes{};

        for (int cc = 0; cc < vdim; cc++)
        {
            // p.rdata(cc) = velocity[cc];
            // particleAttributes[cc] = velocity[cc];
            particles.push_back_real(cc, velocity[cc]);
        }
        // p.rdata(vdim) = weight; //weight
        // particleAttributes[vdim] = weight; //weight
        particles.push_back_real(vdim, weight);
        for (int nd = 1; nd < ndata; nd++) particles.push_back_real(vdim + nd, 0.0);

        particles.push_back(p);
    }

    void copy (ParticleGroups<vdim, ndata> &source, bool local = false)
    {
        BL_PROFILE("Gempic::Particle::ParticleGroups::copy()");
        this->charge = source.get_charge();
        this->mass = source.get_mass();
        this->copyParticles(source, local);
    }
    /**
     * Resets the total weight of the particle group to the specified target weight.
     * This is needed to correct sampling errorsfor low particle counts and non-symmetric velocity
     * distributions.
     *
     * @param targetWeight the new total weight of the particle group
     *
     * @returns None
     */
    void reset_total_weight (amrex::Real targetWeight)
    {
        BL_PROFILE("Gempic::Particle::ParticleGroups::reset_total_weight(totalWeight)");

        // Compute total weight of particle group
        amrex::ReduceOps<amrex::ReduceOpSum> reduceOps;
        auto particleTmp = amrex::ParticleReduce<amrex::ReduceData<amrex::Real>>(
            *this,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::ConstParticleTileData<ParticleType, vdim + ndata, 0> &pt,
                const int pp) -> amrex::GpuTuple<amrex::Real>
            {
                auto weight = pt.m_rdata[vdim][pp];
                return weight;
            },
            reduceOps);

        amrex::Real W = amrex::get<0>(particleTmp);

        // reduced sum over mpi ranks
        amrex::ParallelDescriptor::ReduceRealSum(W);

        // Remove average weight from all particles
        auto totalNumberOfParticles = this->TotalNumberOfParticles();
        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*this, 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            auto *const weight = pti.GetStructOfArrays().GetRealData(vdim).data();
            amrex::Real averageDiff = (targetWeight - W) / totalNumberOfParticles;
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) { weight[pp] += averageDiff; });
        }
    }
};

/**
 * Subclass for linearized Vlasov simulation. Every species needs density and thermal velocity of
 * the equilibrium distribution function around which linearization was performed.
 */
template <unsigned int vdim, unsigned int ndata = 1>
class ParticleGroupsLinVlasov : public ParticleGroups<vdim, ndata>
{
private:
    amrex::Parser m_densityBackground;
    amrex::ParticleReal m_vThermalBackground = 1;

public:
    amrex::Parser get_density_background () { return m_densityBackground; }
    amrex::ParticleReal get_v_thermal_background () { return m_vThermalBackground; }

    ParticleGroupsLinVlasov(const std::string &speciesName, ComputationalDomain ifr) :
        ParticleGroups<vdim, ndata>(speciesName, ifr)
    {
        std::string specString = "Particle." + speciesName;
        Io::Parameters params(specString, "class ParticleGroups (" + speciesName + ')');
        auto [parser, eval] = Gempic::Utils::parse_function("densityBackground", params);
        m_densityBackground = parser;
        params.get_or_set("vThermalBackground", m_vThermalBackground);
    }
};

// /**
//  * Method adds one particle with the space coordinates @p position, the velocity coordinates @p
//  * velocity and the weight @p weight to the Particle Tile @p particles.
//  */
// template <typename T, int vdim>
// void add_particle(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position,
//                   amrex::GpuArray<amrex::Real, vdim> velocity, amrex::Real weight, T& particles)
// {                                                // Array of Structures
//     using ParticleType = amrex::Particle<0, 0>;  // SoA particle type

//     // new particle:
//     ParticleType p;
//     p.id() = ParticleType::NextID();
//     p.cpu() = amrex::ParallelDescriptor::MyProc();

//     for (int cc = 0; cc < GEMPIC_SPACEDIM; cc++)
//     {
//         p.pos(cc) = position[cc];
//     }

//     // Creating attributes for Structure of Arrays container
//     // amrex::GpuArray<amrex::Real,vdim+1> particleAttributes{};

//     for (int cc = 0; cc < vdim; cc++)
//     {
//         // p.rdata(cc) = velocity[cc];
//         // particleAttributes[cc] = velocity[cc];
//         particles.push_back_real(cc, velocity[cc]);
//     }
//     // p.rdata(vdim) = weight; //weight
//     // particleAttributes[vdim] = weight; //weight
//     particles.push_back_real(vdim, weight);

//     particles.push_back(p);
// }
}  // namespace Gempic::Particle

#endif

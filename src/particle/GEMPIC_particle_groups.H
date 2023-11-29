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
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_StructOfArrays.H>
#include <GEMPIC_computational_domain.H>

namespace Gempic
{
namespace Particles
{
using namespace CompDom;
using ParticleType = amrex::Particle<0, 0>;  // SoA particle type
/**
 * Class containing all particle information. The array of particle containers has @p n_species
 * number of particles containers, that is each responsible for one species of particles. For each
 * particle the particle position, velocity and weight are stored in a ArrayOfStructs manner. The
 * charge and mass of the particle species are stored in a separate array.
 */
template <unsigned int vdim, unsigned int ndata = 1>
class particle_groups : public amrex::ParticleContainer<0, 0, vdim + ndata, 0>
{
private:
    amrex::ParticleReal m_charge = 1;
    amrex::ParticleReal m_mass = 1;
    std::string m_name = "electrons";

public:
    amrex::ParticleReal getCharge() { return m_charge; }
    amrex::ParticleReal getMass() { return m_mass; }
    std::string getName() { return m_name; }

    /**
     * Default Constructor.
     */
    particle_groups() {}
    particle_groups(amrex::Geometry geom, amrex::DistributionMapping distriMap,
                    amrex::BoxArray grid) :
        amrex::ParticleContainer<0, 0, vdim + ndata, 0>(geom, distriMap, grid)
    {
    }
    // Explicit const
    /**
     * Explicit Constructor: By passing an object of class vlasov maxwell and infrastructure one can
     * set up the empty particle object. The particles themselves need to be added separately. This
     * can be done one by one (see method add_particle) or with the particle samples in the folder
     * sampler.
     */
    // particle_groups (amrex::GpuArray<amrex::Real, numspec> charge_in,
    //                 amrex::GpuArray<amrex::Real, numspec> mass_in,
    particle_groups(amrex::Real charge, amrex::Real mass, std::string& name,
                    computational_domain ifr) :
        amrex::ParticleContainer<0, 0, vdim + ndata, 0>(ifr.geom, ifr.distriMap, ifr.grid),
        m_charge(charge),
        m_mass(mass),
        m_name(name)
    {
    }
    particle_groups(amrex::Real charge, amrex::Real mass, computational_domain ifr) :
        amrex::ParticleContainer<0, 0, vdim + ndata, 0>(ifr.geom, ifr.distriMap, ifr.grid),
        m_charge(charge),
        m_mass(mass)
    {
    }

    particle_groups(int species, computational_domain ifr) :
        amrex::ParticleContainer<0, 0, vdim + ndata, 0>(ifr.geom, ifr.distriMap, ifr.grid)
    {
        std::string specString = "particle.species" + std::to_string(species);
        Parameters params(specString);
        params.getOrSet("charge", m_charge);
        params.getOrSet("mass", m_mass);
        params.getOrSet("name", m_name);
    }

    /**
     * Method adds one particle with the space coordinates @p position, the velocity coordinates @p
     * velocity and the weight @p weight to the Particle Tile @p particles.
     */
    
    void add_particle(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position,
                      amrex::GpuArray<amrex::Real, vdim> velocity, amrex::Real weight,
                      amrex::ParticleTile<ParticleType, vdim + ndata, 0>& particles)
                      //amrex::ParticleTile<0, 0, vdim + ndata, 0>& particles)
    {                                                // Array of Structures
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

    void copy(particle_groups<vdim, ndata>& source, bool local = false)
    {
        this->charge = source.getCharge();
        this->mass = source.getMass();
        this->copyParticles(source, local);
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
}  // namespace Particles
}  // namespace Gempic

#endif

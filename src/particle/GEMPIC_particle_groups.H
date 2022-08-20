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
namespace Vlasov_Maxwell
{
template <int vdim, int numspec>
class vlasov_maxwell;  // forward declaration
}
}  // namespace Gempic

using namespace Gempic;
using namespace CompDom;
using namespace Vlasov_Maxwell;

namespace Gempic
{
namespace Particles
{
/**
 * Class containing all particle information. The array of particle containers has @p n_species
 * number of particles containers, that is each responsible for one species of particles. For each
 * particle the particle position, velocity and weight are stored in a ArrayOfStructs manner. The
 * charge and mass of the particle species are stored in a separate array.
 */
// template <int vdim, int numspec, int n_data=1>
template <int vdim, int n_data = 1>
class particle_groups : public amrex::ParticleContainer<0, 0, vdim + n_data, 0>
{
private:
    amrex::ParticleReal charge = 1;
    amrex::ParticleReal mass = 1;

public:
    amrex::ParticleReal getCharge() { return charge; }
    amrex::ParticleReal getMass() { return mass; }
    // std::unique_ptr<amrex::ParticleContainer<0, 0, vdim+n_data, 0>> mypc;

    /**
     * Default Constructor.
     */
    particle_groups() {}
    particle_groups(amrex::Geometry geom, amrex::DistributionMapping distriMap,
                    amrex::BoxArray grid) :
        amrex::ParticleContainer<0, 0, vdim + n_data, 0>(geom, distriMap, grid)
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
    particle_groups(amrex::Real charge_in, amrex::Real mass_in, computational_domain ifr) :
        amrex::ParticleContainer<0, 0, vdim + n_data, 0>(ifr.geom, ifr.distriMap, ifr.grid),
        charge(charge_in),
        mass(mass_in)
    {
#if GEMPIC_GPU
        amrex::ParticleContainerBase::do_tiling = false;
#else
        amrex::ParticleContainerBase::do_tiling = true;
#endif
#if (GEMPIC_SPACEDIM > 1)
        amrex::ParticleContainerBase::tile_size = ifr.max_grid_size /= 2;
//         amrex::Print() << "particle tile " << amrex::ParticleContainerBase::tile_size << std::endl;
//         amrex::Print() << "mfiter_tile_size " << amrex::FabArrayBase::mfiter_tile_size << std::endl;
#else
        amrex::ParticleContainerBase::tile_size[0] = ifr.max_grid_size[0];
#endif
    }

    /**
     * Method adds one particle with the space coordinates @p position, the velocity coordinates @p
     * velocity and the weight @p weight to the Particle Tile @p particles.
     */
    void add_particle(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position,
                      amrex::GpuArray<amrex::Real, vdim> velocity, amrex::Real weight,
                      amrex::ParticleTile<0, 0, vdim + n_data, 0>& particles)
    {                                                // Array of Structures
        using ParticleType = amrex::Particle<0, 0>;  // SoA particle type

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

        particles.push_back(p);
    }
};
/**
 * Method adds one particle with the space coordinates @p position, the velocity coordinates @p
 * velocity and the weight @p weight to the Particle Tile @p particles.
 */
template <typename T, int vdim>
void add_particle(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position,
                  amrex::GpuArray<amrex::Real, vdim> velocity, amrex::Real weight, T& particles)
{                                                // Array of Structures
    using ParticleType = amrex::Particle<0, 0>;  // SoA particle type

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

    particles.push_back(p);
}
}  // namespace Particles
}  // namespace Gempic

#endif

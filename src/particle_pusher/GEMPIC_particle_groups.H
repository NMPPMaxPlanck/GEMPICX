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

#include <GEMPIC_initializer.H>

using namespace Gempic;
using namespace Infra;
using namespace Init;

#ifndef PARTICLE_GROUPS_H_
#define PARTICLE_GROUPS_H_

namespace Gempic {
namespace Particles {

/**
 * Class containing all particle information. The array of particle containers has @p n_species number of particles containers, that is each responsible for one species of particles.
 * For each particle the particle position, velocity and weight are stored in a ArrayOfStructs manner. The charge and mass of the particle species are stored in a separate array.
 */
template <int vdim, int numspec, int n_data=1>
class particle_groups
{
public:

    // array of ParticleContainers (one container per species)
    std::array<amrex::Real, numspec> charge;
    std::array<amrex::Real, numspec> mass;
    std::array<std::unique_ptr<amrex::ParticleContainer<vdim+n_data, 0, 0, 0>>, numspec> mypc;

    // The default constructor.
    /**
     * Default Constructor.
     */
    particle_groups ();

    // Explicit const
    /**
     * Explicit Constructor: By passing an object of class initializer and infrastructure one can set up the empty particle object. The particles themselves need to be added separately. This
     * can be done one by one (see method add_particle) or with the particle samples in the folder sampler.
     */
    particle_groups (initializer<vdim, numspec> init,
                     infrastructure ifr){

        charge = init.charge;
        mass = init.mass;

        // set infrastructure for each species
        for (int i=0;i<mypc.size();i++) {
            mypc[i].reset(new amrex::ParticleContainer<vdim+n_data, 0, 0, 0> (ifr.geom, ifr.distriMap, ifr.grid));
            (*mypc[i]).do_tiling = true;
# if (GEMPIC_SPACEDIM > 1)
            (*mypc[i]).tile_size = {AMREX_D_DECL(ifr.max_grid_size,ifr.max_grid_size,ifr.max_grid_size)};
#else
            (*mypc[i]).tile_size[0] = ifr.max_grid_size;
#endif
        }
    }

    /**
     * Method adds one particle with the space coordinates @p position, the velocity coordinates @p velocity and the weight @p weight to the Particle Tile @p particles.
     */
    void add_particle(std::array<amrex::Real,GEMPIC_SPACEDIM> position,
                      std::array<amrex::Real,vdim> velocity,
                      amrex::Real weight,
                      amrex::ParticleTile<vdim+n_data, 0, 0, 0>& particles){

        using ParticleType = amrex::Particle<vdim+n_data, 0>; // Particle template

        // new particle:
        ParticleType p;
        p.id() = ParticleType::NextID();
        p.cpu() = amrex::ParallelDescriptor::MyProc();

        for (int cc=0; cc<GEMPIC_SPACEDIM; cc++) {
            p.pos(cc) = position[cc];
        }

        for (int cc=0; cc<vdim; cc++) {
            p.rdata(cc) = velocity[cc];
        }
        p.rdata(vdim) = weight; //weight

        (particles).push_back(p);
    }

};
}
}

#endif


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

namespace Gempic {
namespace Vlasov_Maxwell {
template<int vdim, int numspec>
class vlasov_maxwell; // forward declaration
}
}

using namespace Gempic;
using namespace CompDom;
using namespace Vlasov_Maxwell;

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
    amrex::GpuArray<amrex::Real, numspec> charge;
    amrex::GpuArray<amrex::Real, numspec> mass;
    amrex::GpuArray<std::unique_ptr<amrex::ParticleContainer<0, 0, vdim+n_data, 0>>, numspec> mypc;
    //amrex::Real charge;
    //amrex::Real mass;
    //std::unique_ptr<amrex::ParticleContainer<vdim+n_data, 0, 0, 0>> mypc;

    // The default constructor.
    /**
     * Default Constructor.
     */
    particle_groups (){}

    // Explicit const
    /**
     * Explicit Constructor: By passing an object of class vlasov maxwell and infrastructure one can set up the empty particle object. The particles themselves need to be added separately. This
     * can be done one by one (see method add_particle) or with the particle samples in the folder sampler.
     */
    particle_groups (amrex::GpuArray<amrex::Real, numspec> charge_in,
                     amrex::GpuArray<amrex::Real, numspec> mass_in,
    //particle_groups (amrex::Real charge_in, amrex::Real mass_in,
                     computational_domain ifr){

        charge = charge_in;
        mass = mass_in;

        // set infrastructure for each species
        for (int i=0;i<mypc.size();i++) {
            mypc[i].reset(new amrex::ParticleContainer<0, 0, vdim+n_data, 0> (ifr.geom, ifr.distriMap, ifr.grid));
        //    mypc.reset(new amrex::ParticleContainer<vdim+n_data, 0, 0, 0> (ifr.geom, ifr.distriMap, ifr.grid));
#if GEMPIC_GPU
	    (*mypc[i]).do_tiling = false;
        //*mypc.do_tiling = false;
#else
	    (*mypc[i]).do_tiling = true;
        //*mypc.do_tiling = true;
#endif
# if (GEMPIC_SPACEDIM > 1)
            (*mypc[i]).tile_size = ifr.max_grid_size;
            //*mypc.tile_size = ifr.max_grid_size;
#else
            (*mypc[i]).tile_size[0] = ifr.max_grid_size[0];
            //*mypc.tile_size[0] = ifr.max_grid_size[0];
#endif
        }
    }

    void reinit (amrex::GpuArray<amrex::Real, numspec> charge_in,
                     amrex::GpuArray<amrex::Real, numspec> mass_in,
    //void reinit (amrex::Real charge_in, amrex::Real mass_in,
                     computational_domain ifr){

        charge = charge_in;
        mass = mass_in;

        // set infrastructure for each species
        for (int i=0;i<mypc.size();i++) {
            mypc[i].reset(new amrex::ParticleContainer<0, 0, vdim+n_data, 0> (ifr.geom, ifr.distriMap, ifr.grid));

        //    mypc.reset(new amrex::ParticleContainer<vdim+n_data, 0, 0, 0> (ifr.geom, ifr.distriMap, ifr.grid));
#if GEMPIC_GPU
        (*mypc[i]).do_tiling = false;
        //*mypc.do_tiling = false;
#else
        (*mypc[i]).do_tiling = true;
        //*mypc.do_tiling = true;
#endif
# if (GEMPIC_SPACEDIM > 1)
            (*mypc[i]).tile_size = ifr.max_grid_size;
            //*mypc.tile_size = ifr.max_grid_size;
#else
            (*mypc[i]).tile_size[0] = ifr.max_grid_size[0];
            //*mypc.tile_size[0] = ifr.max_grid_size[0];
#endif
        }
    }


    /**
     * Method adds one particle with the space coordinates @p position, the velocity coordinates @p velocity and the weight @p weight to the Particle Tile @p particles.
     */
    void add_particle(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> position,
                      amrex::GpuArray<amrex::Real,vdim> velocity,
                      amrex::Real weight,
                      amrex::ParticleTile<0, 0, vdim+n_data, 0>& particles){ //Array of Structures

        //using ParticleType = amrex::Particle<vdim+n_data, 0>; // Particle template
        using ParticleType = amrex::Particle<0, 0>; // Particle template

        // new particle:
        ParticleType p;
        p.id() = ParticleType::NextID();
        p.cpu() = amrex::ParallelDescriptor::MyProc();

        for (int cc=0; cc<GEMPIC_SPACEDIM; cc++) {
            p.pos(cc) = position[cc];
        }
        
        // Creating attributes for Structure of Arrays container
        //amrex::GpuArray<amrex::Real,vdim+1> particleAttributes{};

        for (int cc=0; cc<vdim; cc++) {
            //p.rdata(cc) = velocity[cc];
            //particleAttributes[cc] = velocity[cc];
            particles.push_back_real(cc,velocity[cc]);
        }
        //p.rdata(vdim) = weight; //weight
        //particleAttributes[vdim] = weight; //weight
        particles.push_back_real(vdim,weight);

        particles.push_back(p);
        //particles.push_back_real(particleAttributes);
        // if this works, we can try passing velocity and weight separately and avoid creating temp array
    }

};
}
}

#endif


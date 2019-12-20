#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>

#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_ArrayOfStructs.H>
#include <AMReX_StructOfArrays.H>

#include <infrastructure.H>

#ifndef PARTICLE_GROUPS_H_
#define PARTICLE_GROUPS_H_

class particle_groups
{
  public:
  
  // array of ParticleContainers (one container per species)
  int n_species;
  std::array<amrex::Real, GEMPIC_NUMSPEC> charge;
  std::array<amrex::Real, GEMPIC_NUMSPEC> mass;
  std::array<std::unique_ptr<amrex::ParticleContainer<GEMPIC_VDIM+1, 0, 0, 0>>, GEMPIC_NUMSPEC> mypc;

  // The default constructor.
  particle_groups ();

  // Explicit const
  particle_groups (initializer init,
                   infrastructure ifr){

    n_species = init.n_species;
    charge = init.charge;
    mass = init.mass;

    // set infrastructure for each species
    for (int i=0;i<mypc.size();i++) {
      mypc[i].reset(new amrex::ParticleContainer<GEMPIC_VDIM+1, 0, 0, 0> (ifr.geom, ifr.distriMap, ifr.grid));
      (*mypc[i]).do_tiling = true;
      (*mypc[i]).tile_size = {AMREX_D_DECL(ifr.max_grid_size,ifr.max_grid_size,ifr.max_grid_size)};
    }
  };

  void add_particle(int species,
            std::array<amrex::Real,GEMPIC_SPACEDIM> position,
            std::array<amrex::Real,GEMPIC_VDIM> velocity,
		    amrex::Real weight){

    using ParticleType = amrex::Particle<GEMPIC_VDIM+1, 0>; // Particle template

    auto& particles = (*mypc[species]).GetParticles(0)[std::make_pair(0,0)];
    
	// new particle:
	ParticleType p;
	p.id() = ParticleType::NextID();
	p.cpu() = amrex::ParallelDescriptor::MyProc();
    p.pos(0) = position[0]; //x
#if (GEMPIC_SPACEDIM > 1)
    p.pos(1) = position[1]; //y
#endif
#if (GEMPIC_SPACEDIM > 2)
    p.pos(2) = position[2]; //z
#endif

	// AoS real data
	p.rdata(0) = velocity[0]; //vx
#if (GEMPIC_VDIM > 1)
    p.rdata(1)  = velocity[1]; //vy
#endif
#if (GEMPIC_VDIM > 2)
    p.rdata(2) = velocity[2]; //vz
#endif
    p.rdata(GEMPIC_VDIM) = weight; //weight

	particles.push_back(p);

  };
  
};

#endif


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
#include <gempic_Particle_mod_K.H>

#ifndef PARTICLE_GROUPS_H_
#define PARTICLE_GROUPS_H_

class particle_groups
{
  public:

  //TODO: change 1 for n_species
  
  // array of ParticleContainers (one container per species)
  int n_species;
  std::array<amrex::Real, 1> charge;
  std::array<amrex::Real, 1> mass;
  std::array<std::unique_ptr<amrex::ParticleContainer<4, 0, 0, 0>>, 1> mypc;

  // The default constructor.
  particle_groups ();

  // Explicit const
  particle_groups (infrastructure ifr,
		   int n_species_in,
		   std::array<amrex::Real, 1> charge_in,
		   std::array<amrex::Real, 1> mass_in
		   ){

    n_species = n_species_in;
    charge = charge_in;
    mass = mass_in;

    // set infrastructure for each species
    for (int i=0;i<mypc.size();i++) {
      mypc[i].reset(new amrex::ParticleContainer<4, 0, 0, 0> (ifr.geom, ifr.distriMap, ifr.grid));
      (*mypc[i]).do_tiling = true;
      (*mypc[i]).tile_size = {ifr.max_grid_size,ifr.max_grid_size,ifr.max_grid_size};
    }
  };

  void add_particle(int species,
		    std::array<amrex::Real,3> position,
		    std::array<amrex::Real,3> velocity,
		    amrex::Real weight){

    using ParticleType = amrex::Particle<4, 0>; // Particle template 

    auto& particles = (*mypc[species]).GetParticles(0)[std::make_pair(0,0)];
    
	// new particle:
	ParticleType p;
	p.id() = ParticleType::NextID();
	p.cpu() = amrex::ParallelDescriptor::MyProc();
        p.pos(0) = position[0]; //x
	p.pos(1) = position[1]; //y
	p.pos(2) = position[2]; //z

	// AoS real data
	p.rdata(0) = velocity[0]; //vx
	p.rdata(1)  = velocity[1]; //vy
	p.rdata(2) = velocity[2]; //vz
	p.rdata(3) = weight; //weight

	particles.push_back(p);

  };
  
};

#endif


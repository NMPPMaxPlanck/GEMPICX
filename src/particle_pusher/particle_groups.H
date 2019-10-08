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
		   std::array<amrex::Real, 1> mass_in){

    n_species = n_species_in;
    charge = charge_in;
    mass = mass_in;

    using ParticleType = amrex::Particle<4, 0>; // Particle template 

    // set infrastructure for each species
    for (int i=0;i<mypc.size();i++) {
      mypc[i].reset(new amrex::ParticleContainer<4, 0, 0, 0> (ifr.geom, ifr.distriMap, ifr.grid));
      //(*mypc[i]).Define(ifr.geom, ifr.distriMap, ifr.grid);
    }

    // ``particles'' starts off empty
    auto& particles = (*mypc[0]).GetParticles(0)[std::make_pair(0,0)];

    double pi = 3.14159265359;
    // new particle:
    ParticleType p1;
    p1.id()   = ParticleType::NextID();
    p1.cpu()  = amrex::ParallelDescriptor::MyProc();
    p1.pos(0) = 0.0; //x
    p1.pos(1) = 0.0; //y
    p1.pos(2) = 0.0; //z

    // AoS real data
    p1.rdata(0) = 0.0; //vx
    p1.rdata(1)  = 0.0; //vy
    p1.rdata(2) = 0.0; //vz
    p1.rdata(3) = 1.0; //weight

    particles.push_back(p1);

    // new particle:
    ParticleType p2;
    p2.id()   = ParticleType::NextID();
    p2.cpu()  = amrex::ParallelDescriptor::MyProc();
    p2.pos(0) = 1.0; //x
    p2.pos(1) = 0.0; //y
    p2.pos(2) = 0.0; //z

    // AoS real data
    p2.rdata(0) = 0.0; //vx
    p2.rdata(1)  = 0.0; //vy
    p2.rdata(2) = 0.0; //vz
    p2.rdata(3) = 1.0; //weight

    particles.push_back(p2);


    // new particle:
    ParticleType p3;
    p3.id()   = ParticleType::NextID();
    p3.cpu()  = amrex::ParallelDescriptor::MyProc();
    p3.pos(0) = 0.0; //x
    p3.pos(1) = pi; //y
    p3.pos(2) = 0.0; //z

    // AoS real data
    p3.rdata(0) = 0.0; //vx
    p3.rdata(1)  = 0.0; //vy
    p3.rdata(2) = 0.0; //vz
    p3.rdata(3) = 1.0; //weight

    particles.push_back(p3);

  };
  
};

#endif


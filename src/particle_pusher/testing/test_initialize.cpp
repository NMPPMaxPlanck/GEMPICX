
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

void main_main ()
{
//------------------------------------------------------------------------------
  // Initialize Infrastructure
  
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int is_periodic[3]; // periodicity: 1 -> periodic

  n_cell = 64;
  max_grid_size = 32;
  // periodic in all directions:
  is_periodic[0] = 1;
  is_periodic[1] = 1;
  is_periodic[2] = 1;
  Real dt = 0.01;

  // physical box (geometry)  
  double twopi = 2.0*3.14159265359;
  RealBox real_box({0.0, 0.0, 0.0},
		   {twopi, twopi, twopi});
  // build infrastructure
  infrastructure infra(n_cell, max_grid_size, is_periodic, real_box);

  //need a multifab to be able to iterate later:
  maxwell_yee mw_yee(0, real_box, infra, dt);

//------------------------------------------------------------------------------
  //Initialize Particle Groups
  const int n_species = 1;
  array<Real, n_species> charge = {1.0};
  array<Real, n_species> mass = {1.0};
  particle_groups part_gr(infra, n_species, charge, mass);

  //set particles for first cell (and copies in remaining cells)
  int Np_cell = 100; //number of particles per cell
  int species = 0; // all particles are same species for now
  array<Real,3> position;
  array<Real,3> shifted_position;
  array<Real,3> velocity;
  Real weight;

  // normally distributed random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normD(0,1);

  Real x,y,z;
  
  for (int pp=0;pp<Np_cell;pp++) {
    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = ((Real) rand() / (RAND_MAX))/infra.dx[0];
    position[1] = ((Real) rand() / (RAND_MAX))/infra.dx[1];
    position[2] = ((Real) rand() / (RAND_MAX))/infra.dx[2];

    velocity[0] = normD(gen);
    velocity[1] = normD(gen);
    velocity[2] = normD(gen);
    weight = 1.0;

    //MFI that adds particle from the modell cell to all cells
    for ( MFIter mfi(*mw_yee.B_Array[0]); mfi.isValid(); ++mfi ){
	const amrex::Box& bx = mfi.validbox();
	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};
	for(int k=lo[2]; k<=hi[2]; k++){
	  z = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
	  for(int j=lo[1]; j<=hi[1]; j++){
	    y = infra.geom.ProbLo()[1] + (double)k*infra.dx[1];
	    for(int l=lo[0]; l<=hi[0]; l++){
	      x = infra.geom.ProbLo()[0] + (double)k*infra.dx[0];
	      shifted_position[0] = position[0] + x;
	      shifted_position[1] = position[1] + y;
	      shifted_position[2] = position[2] + z;
	      part_gr.add_particle(species, shifted_position, velocity, weight);
	    }
	  }
	}
    }
  }
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




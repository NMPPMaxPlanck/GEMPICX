
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
//#include <initializer.H>
#include <time_loop.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;
double weight_fun(double x,double y,double z,double v_x,double v_y,double v_z,int Np){
  double alpha = 0.5;
  double k = 0.5;
  return((1.0 + alpha*cos(k*x))/Np);
}
void main_main ()
{
  double k = 0.5;
  //------------------------------------------------------------------------------
  // Initialize Infrastructure
  
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int is_periodic[3]; // periodicity: 1 -> periodic

  n_cell = 8;
  max_grid_size = 4;
  Real dt = 0.01;
  int n_steps = 5;
  // periodic in all directions:
  is_periodic[0] = 1;
  is_periodic[1] = 1;
  is_periodic[2] = 1;

  // physical box (geometry)
  double twopi = 2.0*3.14159265359;
  RealBox real_box({0.0, 0.0, 0.0},
		   {twopi/k, twopi/k, twopi/k});
  // build infrastructure
  infrastructure infra(n_cell, max_grid_size, is_periodic, real_box);
  //infrastructure infra = initialize_ifr();

  // empty cell-centered FAB for MFIter
  MultiFab IteratorFab(infra.grid, infra.distriMap, 1, 0);

  //------------------------------------------------------------------------------
  //Initialize Maxwell Yee
  maxwell_yee mw_yee(n_steps, real_box, infra, dt);
  
  //------------------------------------------------------------------------------
  //Initialize Particle Groups
  const int n_species = 1;
  array<Real, n_species> charge = {1.0};
  array<Real, n_species> mass = {1.0};

  particle_groups part_gr(infra, n_species, charge, mass);
  //ppg.n_species;
  int Np_cell = 1000; //number of particles per cell
  int species = 0; // all particles are same species for now
  init_particles_cellwise(infra, &part_gr, Np_cell, species, {0.0}, {1.0}, {1.0}, weight_fun,
			  &IteratorFab);

  time_loop(infra, &mw_yee, &part_gr);
  
//------------------------------------------------------------------------------
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




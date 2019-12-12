
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
#include <initializer.H>
#include <time_loop.H>
#include <time_loop_gobal.H>
#include <time_loop_avg.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (double x,double y,double z,double v_x,double v_y,double v_z,int Np) {
  double k = 0.5;
  double alpha = 0.;
  return((1.0 + alpha*cos(k*x))/Np);
}

void main_main ()
{
  //------------------------------------------------------------------------------
  //build objects:

  //initializer
  initializer init;
  int is_periodic[3] = {1,1,1};
  int n_cell[3] = {32,4,4};
  std::array<std::array<amrex::Real, 1>, 3> VM{};
  std::array<std::array<amrex::Real, 1>, 3> VD{};
  std::array<std::array<amrex::Real, 1>, 3> VW{};
  VM[0][0] = 0.0;
  VM[1][0] = 0.0;
  VM[2][0] = 0.0;

  VD[0][0] = 0.02/sqrt(2);
  VD[1][0] = sqrt(12)*VD[0][0];
  VD[2][0] = VD[1][0];

  VW[0][0] = 1.0;
  VW[1][0] = 1.0;
  VW[2][0] = 1.0;

  init.initialize_from_parameters(n_cell,4,is_periodic,0.1,300,1,{1.0},{1.0},500,0.5,
                  VW,VD,VW,WF); //{{0.0},{0.0},{0.0}},{{1.0},{1.0},{1.0}},{{1.0},{1.0},{1.0}}
  //n_cell, max_grid_size, periodic, dt, n_steps, n_species, charge, mass, n_part_per_cell, k, vel_mean, vel_dev, vel_weight, weight_fun
    
  // infrastructure
  infrastructure infra(init);

  // empty cell-centered FAB for MFIter
  MultiFab IteratorFab(infra.grid, infra.distriMap, 1, 0);

  // maxwell_yee
  maxwell_yee mw_yee(init, infra);
  
  // particles
  particle_groups part_gr(init, infra);

  //------------------------------------------------------------------------------
  // initialize particles:
  int species = 0; // all particles are same species for now
  init_particles_cellwise(infra, &part_gr, init, species, &IteratorFab);
  
  std::ofstream ofss("PIC_particle.output", std::ofstream::out);
  for (amrex::ParIter<4,0,0,0> pti(*part_gr.mypc[0], 0); pti.isValid(); ++pti) {

    const auto& particles = pti.GetArrayOfStructs();
    const long np = pti.numParticles();
    for (int pp=0;pp<np;pp++) {
      amrex::Print(ofss) << particles[pp].pos(0) << "," << particles[pp].pos(1) << "," << particles[pp].pos(2) << std::endl;
    }
  }
  ofss.close();

  //------------------------------------------------------------------------------
  // solve:
  time_loop_avg(infra, &mw_yee, &part_gr);
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}





#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
#include <initializer.H>
#include <time_loop.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (double x,double y,double z,double v_x,double v_y,double v_z,int Np) {
  double k = 0.5;
  double alpha = 0.5;
  return((1.0 + alpha*cos(k*x))/Np);
};

void main_main ()
{
  //------------------------------------------------------------------------------
  //build objects:

  //initializer
  initializer init;
  int is_periodic[3] = {1,1,1};
  init.initialize_from_parameters(8,4,is_periodic,0.01,1,1,{1.0},{1.0},1000,0.5,
				  {0.0},{1.0},{1.0},WF);
    
  // infrastructure
  infrastructure infra(init.n_cell, init.max_grid_size, init.is_periodic, init.real_box);

  // empty cell-centered FAB for MFIter
  MultiFab IteratorFab(infra.grid, infra.distriMap, 1, 0);

  // maxwell_yee
  maxwell_yee mw_yee(init.n_steps, init.real_box, infra, init.dt);
  
  // particles
  particle_groups part_gr(infra, init.n_species, init.charge, init.mass);

  //------------------------------------------------------------------------------
  // initialize particles:
  int species = 0; // all particles are same species for now
  init_particles_cellwise(infra, &part_gr, init.Np_cell, species, init.Vmean, init.Vdev,
  			  init.Vweight, init.weight_fun, &IteratorFab);
  
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
  time_loop(infra, &mw_yee, &part_gr);
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




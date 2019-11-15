
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
#include <initializer.H>
#include <initializer2.H>
#include <time_loop.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

void main_main ()
{
  //------------------------------------------------------------------------------
  //build objects:
    
  // infrastructure
  infrastructure infra(n_cell, max_grid_size, is_periodic, real_box);

  // empty cell-centered FAB for MFIter
  MultiFab IteratorFab(infra.grid, infra.distriMap, 1, 0);

  // maxwell_yee
  maxwell_yee mw_yee(n_steps, real_box, infra, dt);
  
  // particles
  particle_groups part_gr(infra, n_species, charge, mass);

  //------------------------------------------------------------------------------
  // initialize particles:
  int species = 0; // all particles are same species for now
  init_particles_cellwise(infra, &part_gr, Np_cell, species, {0.0}, {1.0}, {1.0}, weight_fun,
			  &IteratorFab);
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




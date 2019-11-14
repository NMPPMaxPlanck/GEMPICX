
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
#include <initializerExample.H>
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
  part_gr.add_particle(0, {0.5,0.0,0.0}, {0.0,0.0,0.0}, 0.25);
  part_gr.add_particle(0, {0.0,0.5,0.0}, {0.0,0.0,0.0}, 0.25);
  part_gr.add_particle(0, {0.5,0.5,0.5}, {0.0,0.0,0.0}, 0.25);
  //part_gr.add_particle(0, {10.996,10.996,10.996}, {0.0,0.0,0.0}, 0.25);
  part_gr.add_particle(0, {12.56637,12.56637,12.56637}, {0.0,0.0,0.0}, 0.25);
  //part_gr.add_particle(0, {0.5*12.56637,0.5*12.56637,0.5*12.56637}, {0.0,0.0,0.0}, 0.25);

  //check positions
  for (amrex::ParIter<4,0,0,0> pti(*part_gr.mypc[0], 0); pti.isValid(); ++pti) {

    const auto& particles = pti.GetArrayOfStructs();
    const long np = pti.numParticles();
    for (int pp=0;pp<np;pp++) {
      std::cout << particles[pp].pos(0) << "," << particles[pp].pos(1) << "," << particles[pp].pos(2) << std::endl;
    }
  }

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




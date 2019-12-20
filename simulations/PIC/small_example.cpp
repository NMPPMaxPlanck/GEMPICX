
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>
#include <initializer.H>
#include <time_loop.H>
#include <time_loop_avg.H>
#include <gempic_Config.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,int Np,double k) {
  double alpha = 0.5;
  return((1.0 + alpha*cos(k*x[0]))/Np);
};

double B_x(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#if (GEMPIC_BDIM > 1)
double B_y(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif
#if (GEMPIC_BDIM > 2)
double B_z(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif

void main_main ()
{
  //------------------------------------------------------------------------------
  //build objects:

    double (*initB[GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x,double k);
    initB[0] = B_x;
  #if (GEMPIC_BDIM > 1)
    initB[1] = B_y;
  #endif
  #if (GEMPIC_BDIM > 2)
    initB[2] = B_z;
  #endif

  //initializer
  initializer init;
  amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
  amrex::IntVect n_cell(AMREX_D_DECL(8,8,8));
  init.initialize_from_parameters(n_cell,4,is_periodic,0.01,0,{1.0},{1.0},1000,0.5,
				  {0.0},{1.0},{1.0},WF);
    
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
  part_gr.add_particle(0, {AMREX_D_DECL(0.5,0.0,0.0)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
  part_gr.add_particle(0, {AMREX_D_DECL(2.0,0.5,0.0)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
  part_gr.add_particle(0, {AMREX_D_DECL(7.0,0.5,0.5)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
  //part_gr.add_particle(0, {AMREX_D_DECL(10.996,10.996,10.996)}, {AMREX_D_DECL(0.0,0.0,0.0)}, 0.25);
  part_gr.add_particle(0, {AMREX_D_DECL(12.56637,12.56637,12.56637)}, {AMREX_V_DECL(0.0,0.0,0.0)}, 0.25);
  //part_gr.add_particle(0, {AMREX_D_DECL(0.5*12.56637,0.5*12.56637,0.5*12.56637)}, {AMREX_D_DECL(0.0,0.0,0.0)}, 0.25);

  //check positions
  for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*part_gr.mypc[0], 0); pti.isValid(); ++pti) {

    const auto& particles = pti.GetArrayOfStructs();
    const long np = pti.numParticles();
    for (int pp=0;pp<np;pp++) {
      std::cout << particles[pp].pos(0) << "," <<
             #if (GEMPIC_SPACEDIM > 1)
                   particles[pp].pos(1) << "," <<
             #endif
             #if (GEMPIC_SPACEDIM > 2)
                   particles[pp].pos(2) <<
             #endif
                   std::endl;
    }
  }

  //------------------------------------------------------------------------------
  // solve:
  time_loop_avg(infra, &mw_yee, &part_gr, initB, init.k);
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




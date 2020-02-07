
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
#include <gempic_Config.H>
#include <particle_positions.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,int Np,double k) {
  double alpha = 0.5;
  return((1.0 + alpha*cos(k*x[0]))/Np);
}

double B_x(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#if (GEMPIC_BDIM > 1)
double B_y(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif
#if (GEMPIC_BDIM > 2)
double B_z(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif

double phi_fun(std::array<double,GEMPIC_SPACEDIM> x){return(4*0.5*cos(0.5*x[0]));} // 1/k^2*alpha = 4*0.5
double rho_fun(std::array<double,GEMPIC_SPACEDIM> x){return(0);}

void main_main ()
{
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------
    amrex::IntVect n_cell(AMREX_D_DECL(24,8,8)); //number of cells in the three dimensions, ratio should not be too big
    int n_part_per_cell = 1000; // number of particles per cell
    int n_steps = 2000; // number of steps

    // ------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------

    //std::cout << "x-dim: " << AMREX_SPACEDIM << std::endl;
    //std::cout << "v-dim: " << GEMPIC_SPACEDIM << std::endl;

  //------------------------------------------------------------------------------

    bool output_bool = false;

    //initializer
    initializer init;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));

    double (*initB[GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x,double k);
    initB[0] = B_x;
#if (GEMPIC_BDIM > 1)
    initB[1] = B_y;
#endif
#if (GEMPIC_BDIM > 2)
    initB[2] = B_z;
#endif

    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VM{};
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VD{};
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VW{};

    VM[0][0] = 0.0;
    VD[0][0] = 1.0;
    VW[0][0] = 1.0;
#if (GEMPIC_VDIM > 1)
    VM[1][0] = 0.0;
    VD[1][0] = 1.0;
    VW[1][0] = 1.0;
#endif
#if (GEMPIC_VDIM > 2)
    VM[2][0] = 0.0;
    VD[2][0] = 1.0;
    VW[2][0] = 1.0;
#endif

  init.initialize_from_parameters(n_cell,4,is_periodic,1,0.01,n_steps,{-1.0},{1.0},n_part_per_cell,0.5,
                  VM,VD,VW,WF);
  //n_cell, max_grid_size, periodic, Nghost, dt, n_steps, charge, mass, n_part_per_cell, k, vel_mean, vel_dev, vel_weight, weight_fun
    
  // infrastructure
  infrastructure infra(init);

  // maxwell_yee
  maxwell_yee mw_yee(init, infra, init.Nghost);
  mw_yee.init_rho_phi(phi_fun, rho_fun, infra);
  
  // particles
  particle_groups part_gr(init, infra);

  //------------------------------------------------------------------------------
  // initialize particles:
  int species = 0; // all particles are same species for now
  init_particles_cellwise(infra, &part_gr, init, species);

  if (output_bool){
      save_particle_positions(&part_gr);
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





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
#include <loop_preparation.H>
#include <gempic_Config.H>
#include <particle_positions.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF_Weibel (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v, double k) {
    double alpha = 0.;
    return((1.0 + alpha*cos(k*x[0])));
}

double WF_Landau (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v, double k) {
    double alpha = 0.5;
    return((1.0 + alpha*cos(k*x[0])));
}

double B_x(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#if (GEMPIC_BDIM > 1)
double B_y(std::array<double,GEMPIC_SPACEDIM> x,double k){return(0);}
#endif
#if (GEMPIC_BDIM > 2)
double B_z(std::array<double,GEMPIC_SPACEDIM> x,double k){
    amrex::Real beta = 1e-3;
    return(beta*cos(k*x[0]));
}
#endif

double phi_fun(std::array<double,GEMPIC_SPACEDIM> x){return(4*0.5*cos(0.5*x[0]));}
double rho_fun(std::array<double,GEMPIC_SPACEDIM> x){return(0.);}

void main_main ()
{
    int testcase = 0; // 0 -> Weibel, 1 -> Landau

    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------
    amrex::IntVect n_cell(AMREX_D_DECL(24,8,8)); //number of cells in the three dimensions, ratio should not be too big
    int n_part_per_cell = 100; // number of particles per cell
    int n_steps = 100000; // number of steps

    // for running on SUPER-MUC:
    //amrex::IntVect n_cell(AMREX_D_DECL(32,32,32)); //number of cells in the three dimensions, ratio should not be too big
    //int n_part_per_cell = 1000; // number of particles per cell
    //int n_steps = 10; // number of steps

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
    switch (testcase) {
    case 0:
        initB[2] = B_z;
        break;
    case 1:
        initB[2] = B_x;
        break;
    }

#endif

    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VM{};
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VD{};
    std::array<std::array<amrex::Real, GEMPIC_NUMGAUS>, GEMPIC_VDIM> VW{};

    VM[0][0] = 0.0;
    switch (testcase) {
    case 0:
        VD[0][0] = 0.02/sqrt(2);
        break;
    case 1:
        VD[0][0] = 1.0;
        break;
    }
    VW[0][0] = 1.0;
#if (GEMPIC_VDIM > 1)
    VM[1][0] = 0.0;
    switch (testcase) {
    case 0:
        VD[1][0] = sqrt(12)*VD[0][0];
        break;
    case 1:
        VD[1][0] = 1.0;
        break;
    }
    VW[1][0] = 1.0;
#endif
#if (GEMPIC_VDIM > 2)
    VM[2][0] = 0.0;
    VD[2][0] = VD[1][0];
    VW[2][0] = 1.0;
#endif

    switch (testcase) {
    case 0:
        init.initialize_from_parameters(n_cell,4,is_periodic,3,0.02,n_steps,{-1.0},{1.0},n_part_per_cell,1.25,
                                        VM,VD,VW,WF_Weibel);
        break;
    case 1:
        init.initialize_from_parameters(n_cell,4,is_periodic,3,0.02,n_steps,{-1.0},{1.0},n_part_per_cell,0.5,
                                        VM,VD,VW,WF_Landau);
        break;
    }
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
    diagnostics diagn(mw_yee.nsteps);
    loop_preparation(infra, &mw_yee, &part_gr, &diagn, initB, init.k, output_bool);
    time_loop_avg(infra, &mw_yee, &part_gr, &diagn);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}




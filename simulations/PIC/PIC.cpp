#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_initializer.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_avg.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Init;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;

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

    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // compile parameters
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(GEMPIC_DEG_X, GEMPIC_DEG_Y, GEMPIC_DEG_Z)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));
    int Nghost = maxdeg;

    // initialize parameters
    int testcase;
    std::string sim_name;
    bool ctest;
    std::array<int,3> n_cell_vector;
    int n_part_per_cell;
    int n_steps;
    int freq_x;
    int freq_v;
    int freq_slice;
    std::array<int,3> is_periodic_vector;
    int max_grid_size;
    amrex::Real dt;
    std::array<amrex::Real, GEMPIC_NUMSPEC> charge;
    std::array<amrex::Real, GEMPIC_NUMSPEC> mass;
    amrex::Real k;
    amrex::Real alpha;
    std::string WF;

    // parse parameters
    amrex::ParmParse pp;
    pp.get("testcase",testcase);
    pp.get("sim_name",sim_name);
    pp.get("ctest",ctest);
    pp.get("n_cell_vector",n_cell_vector);
    pp.get("n_part_per_cell",n_part_per_cell);
    pp.get("n_steps",n_steps);
    pp.get("freq_x",freq_x);
    pp.get("freq_v",freq_v);
    pp.get("freq_slice",freq_slice);
    pp.get("is_periodic_vector",is_periodic_vector);
    pp.get("max_grid_size",max_grid_size);
    pp.get("dt",dt);
    pp.get("charge",charge);
    pp.get("mass",mass);
    pp.get("k",k);
    pp.get("alpha", alpha);
    pp.get("WF",WF);

    // initialize amrex data structures from parameters
    amrex::IntVect n_cell(AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2]));
    amrex::IntVect is_periodic(AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2]));

    // functions
    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvar", &k}, {"alpha", &alpha}};
    int varcount = 5;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);

    // parameters not yet parsed in
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

    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VM{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VD{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VW{};

    VM[0].push_back(0.0);
    switch (testcase) {
    case 0:
        VD[0].push_back(0.02/sqrt(2));
        break;
    case 1:
        VD[0].push_back(1.0);
        break;
    }
    VW[0].push_back(1.0);
#if (GEMPIC_VDIM > 1)
    VM[1].push_back(0.0);
    switch (testcase) {
    case 0:
        VD[1].push_back(sqrt(12)*VD[0][0]);
        break;
    case 1:
        VD[1].push_back(1.0);
        break;
    }
    VW[1].push_back(1.0);
#endif
#if (GEMPIC_VDIM > 2)
    VM[2].push_back(0.0);
    VD[2].push_back(VD[1][0]);
    VW[2].push_back(1.0);
#endif

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    //initializer
    initializer init;
    init.initialize_from_parameters(n_cell,max_grid_size,is_periodic,Nghost,dt,n_steps,charge,mass,n_part_per_cell,k,
                                        VM,VD,VW);
    
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
    init_particles_cellwise(infra, part_gr, init, species, WF_parse, &x, &y, &z);

    save_particle_positions(&part_gr, sim_name + "_0");
    save_particle_velocities(&part_gr, sim_name + "_0");


    //------------------------------------------------------------------------------
    // solve:
    diagnostics diagn(mw_yee.nsteps, freq_x, freq_v, freq_slice, sim_name);
    loop_preparation(infra, &mw_yee, &part_gr, &diagn, initB, init.k);
    std::ofstream ofs("PIC.output", std::ofstream::out);
    amrex::Print(ofs) << endl;
    time_loop_avg(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}




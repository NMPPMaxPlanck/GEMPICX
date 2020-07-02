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

void main_main ()
{

    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // compile parameters
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(GEMPIC_DEG_X, GEMPIC_DEG_Y, GEMPIC_DEG_Z)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));
    int Nghost = maxdeg;

    // initialize parameters
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
    std::string WF;
    std::string Bx;
    std::string By;
    std::string Bz;
    std::string phi;
    std::string rho;
    int num_gaussians;

    // parse parameters
    amrex::ParmParse pp;
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
    pp.get("WF",WF);
    pp.get("Bx",Bx);
    pp.get("By",By);
    pp.get("Bz",Bz);
    pp.get("rho",rho);
    pp.get("phi",phi);
    pp.get("num_gaussians",num_gaussians);

    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VM{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VD{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VW{};
    std::array<double, GEMPIC_VDIM> read_tmp_M;
    std::array<double, GEMPIC_VDIM> read_tmp_D;
    std::array<double, GEMPIC_VDIM> read_tmp_W;

    for (int i=0; i<num_gaussians; i++) {
        std::string name_str_M = "velocity_mean_" +  std::to_string(i);
        std::string name_str_D = "velocity_deviation_" +  std::to_string(i);
        std::string name_str_W = "velocity_weight_" +  std::to_string(i);
        const char *name_char_M = name_str_M.c_str();
        const char *name_char_D = name_str_D.c_str();
        const char *name_char_W = name_str_W.c_str();
        pp.get(name_char_M,read_tmp_M);
        pp.get(name_char_D,read_tmp_D);
        pp.get(name_char_W,read_tmp_W);
        for (int j=0; j<GEMPIC_VDIM; j++) {
            VM[j].push_back(read_tmp_M[j]);
            VD[j].push_back(read_tmp_D[j]);
            VW[j].push_back(read_tmp_W[j]);
        }
    }

    // initialize amrex data structures from parameters
    amrex::IntVect n_cell(AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2]));
    amrex::IntVect is_periodic(AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2]));

    // functions
    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvar", &k}};
    int varcount = 4;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);

    te_expr *Bx_parse = te_compile(Bx.c_str(), read_vars, varcount, &err);
    te_expr *By_parse = te_compile(By.c_str(), read_vars, varcount, &err);
    te_expr *Bz_parse = te_compile(Bz.c_str(), read_vars, varcount, &err);

    te_variable read_vars_poi[] = {{"x", &x}, {"y", &y}, {"z", &z}};
    varcount = 3;
    te_expr *rho_parse = te_compile(rho.c_str(), read_vars_poi, varcount, &err);
    te_expr *phi_parse = te_compile(phi.c_str(), read_vars_poi, varcount, &err);

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
    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);

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
    loop_preparation(infra, &mw_yee, &part_gr, &diagn, Bx_parse, By_parse, Bz_parse, &x, &y, &z);
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




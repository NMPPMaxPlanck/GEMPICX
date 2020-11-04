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
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Init;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main (bool ctest)
{
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // compile parameters
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));
    int Nghost = maxdeg;

    // initialize parameters
    std::string sim_name;
    std::array<int,GEMPIC_SPACEDIM> n_cell_vector;
    int n_part_per_cell;
    int n_steps;
    int freq_x;
    int freq_v;
    int freq_slice;
    std::array<int,GEMPIC_SPACEDIM> is_periodic_vector;
    int max_grid_size;
    amrex::Real dt;
    std::array<amrex::Real, numspec> charge;
    std::array<amrex::Real, numspec> mass;
    amrex::Real k;
    std::string WF;
    std::string Bx;
    std::string By;
    std::string Bz;
    std::string phi;
    std::string rho = "0.0";
    int num_gaussians;
    int propagator;
    bool time_staggered;
    amrex::Real tolerance_particles;
    int restart;
    std::string checkpoint_file;
    int curr_step;

    // parse parameters
    amrex::ParmParse pp;

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    if (ctest) {
        sim_name = "Weibel";
        n_cell_vector[0] = 24;
        n_cell_vector[1] = 8;
        n_cell_vector[2] = 8;
        n_part_per_cell = 100;
        n_steps = 10;
        freq_x = 11;
        freq_v = 11;
        freq_slice = 11;
        is_periodic_vector[0] = 1;
        is_periodic_vector[1] = 1;
        is_periodic_vector[2] = 1;
        max_grid_size = 4;
        dt = 0.02;
        charge[0] = -1.0;
        mass[0] = 1.0;
        k = 1.25;
        WF = "1.0 + 0.0 * cos(kvar * x)";
        Bx = "0.0";
        By = "0.0";
        Bz = "1e-3 * cos(kvar * x)";
        phi = "4 * 0.5 * cos(0.5 * x)";
        num_gaussians = 1;
        tolerance_particles = 1.e-10;

        for (int j=0; j<vdim; j++) {
            VM[j].push_back(0.0);
            VW[j].push_back(1.0);
        }
        VD[0].push_back(0.02/sqrt(2));
        VD[1].push_back(sqrt(12)*VD[0][0]);
        VD[2].push_back(VD[1][0]);
        restart = 0;
        checkpoint_file = "";
        curr_step = 0;

    } else {
        pp.get("sim_name",sim_name);
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
        pp.get("phi",phi);
        pp.get("num_gaussians",num_gaussians);
        pp.get("propagator",propagator);
        pp.get("tolerance_particles", tolerance_particles);
        pp.get("restart", restart);
        pp.get("checkpoint_file", checkpoint_file);
        pp.get("curr_step", curr_step);

        std::array<double, vdim> read_tmp_M;
        std::array<double, vdim> read_tmp_D;
        std::array<double, vdim> read_tmp_W;

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
            for (int j=0; j<vdim; j++) {
                VM[j].push_back(read_tmp_M[j]);
                VD[j].push_back(read_tmp_D[j]);
                VW[j].push_back(read_tmp_W[j]);
            }
        }

        // Depending on which propagator is chosen, staggering in time is needed or not
        switch (propagator) {
        case 0:
            time_staggered = true;
            break;
        case 1:
            time_staggered = false;
            Nghost += 2;
            break;
        default:
            break;
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
    initializer<vdim, numspec> init;
    init.initialize_from_parameters(n_cell,max_grid_size,is_periodic,Nghost,dt,n_steps,charge,mass,n_part_per_cell,k,
                                    VM,VD,VW,tolerance_particles);

    // infrastructure
    infrastructure infra;
    init.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(init, infra, init.Nghost);
    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);

    // particles
    particle_groups<vdim, numspec> part_gr(init, infra);

    diagnostics<vdim, numspec,degx,degy,degz> diagn(mw_yee.nsteps, freq_x, freq_v, freq_slice, sim_name);

    //------------------------------------------------------------------------------
    // initialize particles:
    if (restart == 0) {
        int species = 0; // all particles are same species for now
        init_particles_full_domain(infra, part_gr, init, species, WF_parse, &x, &y, &z);

        //------------------------------------------------------------------------------
        // solve:
        loop_preparation<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, Bx_parse, By_parse, Bz_parse, &x, &y, &z,time_staggered);
    } else {
        Gempic_ReadCheckpointFile (&mw_yee, &part_gr, &infra, checkpoint_file, curr_step);
    }
    std::ofstream ofs("vlasov_maxwell.output", std::ofstream::out);
    if (ctest) AllPrintToFile("test_output_pre_rename.output") << endl;
    switch (propagator) {
    case 0:
        time_loop_boris_fd<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 1:
        time_loop_hs_fem<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    default:
        break;
    }
    if (ctest & (ParallelDescriptor::MyProc()==0)) std::rename("test_output_pre_rename.output.0", "vlasov_maxwell.output");
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main<3, 1, 1, 1, 1>(argc==1);

    amrex::Finalize();
}




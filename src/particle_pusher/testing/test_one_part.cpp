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

template< int vdim>
void main_main (bool ctest)
{
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // compile parameters
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(GEMPIC_DEG_X, GEMPIC_DEG_Y, GEMPIC_DEG_Z)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));
    int Nghost = maxdeg;

    // initialize parameters
    std::string sim_name = "One_Particle";
    amrex::IntVect n_cell(AMREX_D_DECL(4,4,4));
    int n_part_per_cell = 1;
    int n_steps = 1;
    int freq_x = 2;
    int freq_v = 2;
    int freq_slice = 1;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
    int max_grid_size = 4;
    amrex::Real dt = 0.02;
    std::array<amrex::Real, GEMPIC_NUMSPEC> charge = {-1.0};
    std::array<amrex::Real, GEMPIC_NUMSPEC> mass = {1.0};
    amrex::Real k = 1.25;
    std::string WF = "1.0";
    std::string Bx = "0.0";
    std::string By = "0.0";
    std::string Bz = "1e-3 * cos(kvar * x)";
    std::string phi = "4 * 0.5 * cos(0.5 * x)";
    std::string rho = "0.0";
    int propagator = 1;
    bool time_staggered = true;
    amrex::Real tolerance_particles = 1.e-10;

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD = {};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VW[j].push_back(1.0);
    }
    VD[0].push_back(0.02/sqrt(2));
    VD[1].push_back(sqrt(12)*VD[0][0]);
    VD[2].push_back(VD[1][0]);


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
    initializer<vdim> init;
    init.initialize_from_parameters(n_cell,max_grid_size,is_periodic,Nghost,dt,n_steps,charge,mass,n_part_per_cell,k,
                                    VM,VD,VW,tolerance_particles);
    
    // infrastructure
    infrastructure infra;
    init.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(init, infra, init.Nghost);
    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);

    // particles
    particle_groups<vdim> part_gr(init, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0;
    for(amrex::MFIter mfi=(*(part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {
        if(mfi.index() == 0) {
            using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
            amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            (part_gr).add_particle({AMREX_D_DECL(2.511, 2.2, 2.3)}, {0.1, 0.1, 0.1}, 1.0, particles);
        }
    }

    //------------------------------------------------------------------------------
    // solve:
    diagnostics<vdim> diagn(mw_yee.nsteps, freq_x, freq_v, freq_slice, sim_name);
    loop_preparation(infra, &mw_yee, &part_gr, &diagn, Bx_parse, By_parse, Bz_parse, &x, &y, &z,time_staggered);
    std::ofstream ofs("PIC.output", std::ofstream::out);
    amrex::Print(ofs) << endl;
    switch (propagator) {
    case 0:
        time_loop_boris_fd(infra, &mw_yee, &part_gr, &diagn, false, &ofs);
        break;
    case 1:
        time_loop_hs_fem(infra, &mw_yee, &part_gr, &diagn, false, &ofs);
        break;
    default:
        break;
    }

    AllPrintToFile("test_output_pre_rename.output") << std::endl;
    AllPrintToFile("test_output_pre_rename.output") << "Jx" << std::endl;
    for (amrex::MFIter mfi(*(mw_yee).J_Array[0]); mfi.isValid(); ++mfi ) {
        AllPrintToFile("test_output_pre_rename.output") << (*(mw_yee).J_Array[0])[mfi] << std::endl;
    }
    AllPrintToFile("test_output_pre_rename.output") << "Jy" << std::endl;
    for (amrex::MFIter mfi(*(mw_yee).J_Array[1]); mfi.isValid(); ++mfi ) {
        AllPrintToFile("test_output_pre_rename.output") << (*(mw_yee).J_Array[1])[mfi] << std::endl;
    }
    if (ParallelDescriptor::MyProc()==0) std::rename("test_output_pre_rename.output.0", "test_one_part.output");
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main<3>(argc==1);

    amrex::Finalize();
}




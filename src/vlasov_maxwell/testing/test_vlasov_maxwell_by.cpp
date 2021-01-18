#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hsall_fem.H>
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_particle_groups.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params();
    VlMa.set_computed_params();

    std::array<std::vector<amrex::Real>, vdim> VM{}, VD{}, VW{};
    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VW[j].push_back(1.0);
    }
    VD[0].push_back(0.02/sqrt(2));
    VD[1].push_back(sqrt(12)*VD[0][0]);
    VD[2].push_back(VD[1][0]);
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    // infrastructure
    infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    mw_yee.init_rho_phi(infra, VlMa);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    diagnostics<vdim, numspec,degx,degy,degz> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0);
    loop_preparation<vdim, numspec>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered);


    //------------------------------------------------------------------------------
    // timeloop
std::ofstream ofs("vlasov_maxwell.output", std::ofstream::out);
AllPrintToFile("test_output_pre_rename.output") << std::endl;
switch (VlMa.propagator) {
    case 0:
        time_loop_boris_fd<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 1:
        time_loop_hs_fem<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 2:
        time_loop_hsall_fem<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    default:
        break;
}

if (ParallelDescriptor::MyProc()==0) std::rename("test_output_pre_rename.output.0", "test_vlasov_maxwell_by.output");
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
#if (GEMPIC_SPACEDIM == 1)
#elif (GEMPIC_SPACEDIM == 2)
     std::cout << "A" << std::endl;
    main_main<2, 1, 1, 1, 1>();
    std::cout << "B" << std::endl;
    main_main<2, 1, 1, 1, 1>();
     std::cout << "C" << std::endl;
    main_main<3, 1, 1, 1, 1>();
     std::cout << "D" << std::endl;
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif

    amrex::Finalize();
}




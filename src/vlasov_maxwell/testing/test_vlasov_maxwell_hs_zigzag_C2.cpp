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
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
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
    VlMa.propagator = 3;
    VlMa.set_prop_related();
    if (int(vdim/2.5)*2+1 < 3) {
        VlMa.Bx = VlMa.Bz;
        VlMa.Bz = "0.0";
    }
    if (GEMPIC_SPACEDIM==1 && vdim==1) {
        // For 1D1V change parameters to make a Landau damping
        VlMa.sim_name = "Landau";
        VlMa.n_part_per_cell = {10000};
        VlMa.k = {0.5,0.5,0.5};
        VlMa.WF = "1.0 + 0.5 * cos(kvarx * x)";
        VlMa.Bz = "0.0";
    }
    VlMa.n_steps = 10;
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
AllPrintToFile("test_vlasov_maxwell_hs_zigzag_C2.tmp") << std::endl;
// time_loop_hsall_fem<vdim, numspec>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_zigzag_C2.tmp", &ofs);
time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz>(infra,&mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_zigzag_C2.tmp", &ofs);

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 1, 1, 2, 3>();
    main_main<3, 1, 4, 3, 2>();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_hs_zigzag_C2.tmp.0", "test_vlasov_maxwell_hs_zigzag_C2.output");
    amrex::Finalize();
}




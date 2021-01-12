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

template<int vdim, int numspec, int degx, int degy, int degz, bool electromagnetic=true>
void main_main (bool ctest)
{
    bool readinfile = false;
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.read_pp_params();
    VlMa.set_computed_params();

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
    if (VlMa.restart == 0) {
        if (readinfile) {
            for (int spec=0; spec<numspec; spec++) {
                VlMa.init_particles_from_file(&part_gr, spec, "particle_input.txt");
            }
        } else {
            for (int spec=0; spec<numspec; spec++) {
                VlMa.read_particle_spec(spec);
                init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, spec);
            }
        }

        loop_preparation<vdim, numspec>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered);
    } else {
        Gempic_ReadCheckpointFile (&mw_yee, &part_gr, &infra, VlMa.checkpoint_file, VlMa.curr_step);
    }

    //------------------------------------------------------------------------------
    // timeloop


std::ofstream ofs("vlasov_maxwell.output", std::ofstream::out);
switch (VlMa.propagator) {
    case 0:
      time_loop_boris_fd<vdim, numspec, degx, degy, degz, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 1:
      time_loop_hs_fem<vdim, numspec, degx, degy, degz, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    case 2:
      time_loop_hsall_fem<vdim, numspec, degx, degy, degz, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, &ofs);
        break;
    default:
        break;
}
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<2, GEMPIC_NUMSPEC, 1, 1, 1, GEMPIC_ELECTROMAGNETIC>(argc==1);
#elif (GEMPIC_SPACEDIM == 2)
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, GEMPIC_ELECTROMAGNETIC>(argc==1);
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, GEMPIC_ELECTROMAGNETIC>(argc==1);
#endif

    amrex::Finalize();
}




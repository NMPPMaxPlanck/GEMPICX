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
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
#include <GEMPIC_time_loop_particles.H>
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_particle_groups.H>

using namespace std;
using namespace std::chrono;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

template<int vdim, int numspec, int degx, int degy, int degz, int degvm, bool electromagnetic=true>
void main_main (bool ctest)
{
    bool readinfile = false;
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.read_pp_params();
    VlMa.set_computed_params();

    std::array<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_B[0] = VlMa.Bx;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[1] = VlMa.By;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[2] = VlMa.Bz;
    }

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

    diagnostics<vdim, numspec,degx,degy,degz,degvm> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    if (VlMa.propagator==100) {
        for (int spec=0; spec<numspec; spec++) {
            for(amrex::MFIter mfi=(*(part_gr).mypc[spec]).MakeMFIter(0); mfi.isValid(); ++mfi) {
                if(mfi.index() == 0) {
                    using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
                    amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(part_gr).mypc[spec]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                    (part_gr).add_particle({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(0.0, 0.0, 0.0)}, 1.0, particles);
                }
            }
        }
        loop_preparation<vdim, numspec, degx, degy, degz, degvm>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, fields_B);
    } else {
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

        loop_preparation<vdim, numspec, degx, degy, degz, degvm>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, fields_B);
    } else {
        Gempic_ReadCheckpointFile (&mw_yee, &part_gr, &infra, VlMa.checkpoint_file, VlMa.curr_step);
    }
    }

    //------------------------------------------------------------------------------
    // timeloop


std::ofstream ofs("vlasov_maxwell.output", std::ofstream::out);
auto start = high_resolution_clock::now();
switch (VlMa.propagator) {
    case 0:
      time_loop_boris_fd<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, VlMa.sim_name, &ofs);
        break;
    case 1:
      time_loop_hs_fem<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, VlMa.sim_name, &ofs);
        break;
    case 2:
      time_loop_hsall_fem<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, VlMa.sim_name, &ofs);
        break;      
    case 3:
      time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degvm>(infra,&mw_yee, &part_gr, &diagn, ctest, VlMa.sim_name, &ofs);
        break;
    case 100:
      time_loop_particles<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(infra, &mw_yee, &part_gr, &diagn, ctest, VlMa.sim_name, &ofs);
        break;
    default:
        break;
}
auto stop = high_resolution_clock::now();
auto duration = duration_cast<seconds>(stop - start);
cout << "execution lasted: " <<  duration.count() << " s" << endl;

Gempic_WritePlotFile(&part_gr, &mw_yee, &infra, "Edipole", 10);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<2, GEMPIC_NUMSPEC, 1, 1, 1, 2, GEMPIC_ELECTROMAGNETIC>(argc==1);
#elif (GEMPIC_SPACEDIM == 2)
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, 2, GEMPIC_ELECTROMAGNETIC>(argc==1);
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, 2, GEMPIC_ELECTROMAGNETIC>(argc==1);
#endif

    amrex::Finalize();
}




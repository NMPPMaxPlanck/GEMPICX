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
#include <GEMPIC_profiling.H>
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
using namespace Profiling;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z)
{
    amrex::Real val = 1.0 ;//+ 0.5 * std::cos(0.5 * x);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real zero(amrex::Real , amrex::Real , amrex::Real , amrex::Real )
{
    amrex::Real val = 0.0;
    return val;
}


template<int vdim, int numspec, int degx, int degy, int degz, int degmw>
void main_main ()
{
    const int strang_order = 2;
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("test_vlasov_maxwell_by", {12,8,8});

    VlMa.n_steps = 5;
    VlMa.set_computed_params();

    std::array<std::vector<amrex::Real>, vdim> VM{}, VD{}, VW{};
    if (GEMPIC_SPACEDIM==1 && vdim==1) {
        VM[0].push_back(0.0);
        VD[0].push_back(1.0);
        VW[0].push_back(1.0);
    } else {
        for (int j=0; j<vdim; j++) {
            VM[j].push_back(0.0);
            VW[j].push_back(1.0);
        }
        VD[0].push_back(0.02/sqrt(2));
        VD[1].push_back(sqrt(12)*VD[0][0]);
        VD[2].push_back(VD[1][0]);
    }
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    // infrastructure
    computational_domain infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    amrex::GpuArray<std::string, 2> fields = {VlMa.rho, VlMa.phi};
    mw_yee.template init_rho_phi<degmw>(zero, zero, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec,degx,degy,degz,degmw> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);
    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0,wave_function);
    loop_preparation<vdim, numspec, degx, degy,degz, degmw, true>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, zero, zero, zero);

    //------------------------------------------------------------------------------
    // timeloop
    time_loop_boris_fd<vdim, numspec, degx, degy, degz, degmw, true, false>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_by", strang_order);

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_vlasov_maxwell_by.tmp.0");

    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output file contains all outputs.
    For each dimension, apart from running the main_main for the dimension, the output for the other dimensions needs to be
    outputted, so that the comparison to the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1, GEMPIC_SPACEDIM=2, GEMPIC_SPACEDIM=3 */

#if (GEMPIC_SPACEDIM == 1)
    // Output for GEMPIC_SPACEDIM=1
    main_main<1, 1, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=2
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "0 3.69818e-05 2.9765e-05 0 0.0321487 0.293507 0.967771" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "1 3.69571e-05 2.97344e-05 2.19122e-39 0.0321494 0.293512 0.967785" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "2 3.69133e-05 2.96868e-05 4.9016e-11 0.0321506 0.293524 0.967802" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "3 3.685e-05 2.96234e-05 4.31676e-10 0.0321522 0.293547 0.967821" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "4 3.67666e-05 2.95444e-05 1.67345e-09 0.0321543 0.29358 0.967844" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "5 3.66624e-05 2.94457e-05 4.45131e-09 0.0321568 0.293622 0.967872" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "6 3.65376e-05 2.93262e-05 9.48196e-09 0.0321599 0.293674 0.967908" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "7 3.63924e-05 2.91889e-05 1.74089e-08 0.0321634 0.293739 0.967948" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "8 3.62273e-05 2.9031e-05 2.87007e-08 0.0321673 0.29382 0.967992" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "9 3.6043e-05 2.88571e-05 4.35711e-08 0.0321717 0.293915 0.968039" << std::endl;

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "0 2.82775e-05 1.74142e-05 1.72514e-05 0 0 5e-07 0.315393 1.43412 4.90194 4.95606" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "1 2.82739e-05 1.74401e-05 1.72768e-05 3.03609e-40 1.20022e-39 5e-07 0.31539 1.43414 4.90193 4.95602" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "2 2.82582e-05 1.74626e-05 1.7296e-05 1.07538e-11 5.42619e-11 4.99432e-07 0.315389 1.4342 4.90191 4.956" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "3 2.82305e-05 1.74813e-05 1.73091e-05 9.53964e-11 4.79157e-10 4.98639e-07 0.315389 1.43431 4.9019 4.95597" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "4 2.8191e-05 1.74958e-05 1.73154e-05 3.73233e-10 1.86231e-09 4.98256e-07 0.31539 1.43446 4.90189 4.95596" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "5 2.81396e-05 1.75037e-05 1.73136e-05 1.00717e-09 4.97797e-09 4.99121e-07 0.315393 1.43465 4.90189 4.95594" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "6 2.80766e-05 1.75027e-05 1.73019e-05 2.18734e-09 1.06708e-08 5.02156e-07 0.315397 1.43488 4.90188 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "7 2.80017e-05 1.74911e-05 1.7279e-05 4.11194e-09 1.97274e-08 5.08226e-07 0.315403 1.43515 4.90189 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "8 2.79147e-05 1.74662e-05 1.72456e-05 6.96811e-09 3.27518e-08 5.18012e-07 0.315411 1.43547 4.9019 4.95594" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "9 2.78164e-05 1.74287e-05 1.72001e-05 1.09132e-08 5.00539e-08 5.31905e-07 0.31542 1.43582 4.90192 4.95595" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "0 0.497151 5e-07 6.26259 9.99417" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "1 0.497055 5e-07 6.2635 9.99486" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "2 0.496664 5e-07 6.26629 9.99712" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "3 0.49598 5e-07 6.27091 10.0009" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "4 0.495008 5e-07 6.27737 10.0062" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "5 0.493745 5e-07 6.28562 10.013" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "6 0.492189 5e-07 6.29572 10.0213" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "7 0.490343 5e-07 6.30764 10.0312" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "8 0.488214 5e-07 6.32138 10.0424" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "9 0.485808 5e-07 6.33685 10.0551" << std::endl;

    // Output for GEMPIC_SPACEDIM=2
    main_main<2, 1, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "0 2.82775e-05 1.74142e-05 1.72514e-05 0 0 5e-07 0.315393 1.43412 4.90194 4.95606" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "1 2.82739e-05 1.74401e-05 1.72768e-05 3.03609e-40 1.20022e-39 5e-07 0.31539 1.43414 4.90193 4.95602" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "2 2.82582e-05 1.74626e-05 1.7296e-05 1.07538e-11 5.42619e-11 4.99432e-07 0.315389 1.4342 4.90191 4.956" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "3 2.82305e-05 1.74813e-05 1.73091e-05 9.53964e-11 4.79157e-10 4.98639e-07 0.315389 1.43431 4.9019 4.95597" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "4 2.8191e-05 1.74958e-05 1.73154e-05 3.73233e-10 1.86231e-09 4.98256e-07 0.31539 1.43446 4.90189 4.95596" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "5 2.81396e-05 1.75037e-05 1.73136e-05 1.00717e-09 4.97797e-09 4.99121e-07 0.315393 1.43465 4.90189 4.95594" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "6 2.80766e-05 1.75027e-05 1.73019e-05 2.18734e-09 1.06708e-08 5.02156e-07 0.315397 1.43488 4.90188 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "7 2.80017e-05 1.74911e-05 1.7279e-05 4.11194e-09 1.97274e-08 5.08226e-07 0.315403 1.43515 4.90189 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "8 2.79147e-05 1.74662e-05 1.72456e-05 6.96811e-09 3.27518e-08 5.18012e-07 0.315411 1.43547 4.9019 4.95594" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "9 2.78164e-05 1.74287e-05 1.72001e-05 1.09132e-08 5.00539e-08 5.31905e-07 0.31542 1.43582 4.90192 4.95595" << std::endl;

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1
#if  !(GEMPIC_GPU)
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << std::endl;
#endif

    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "0 0.497151 5e-07 6.26259 9.99417" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "1 0.497055 5e-07 6.2635 9.99486" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "2 0.496664 5e-07 6.26629 9.99712" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "3 0.49598 5e-07 6.27091 10.0009" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "4 0.495008 5e-07 6.27737 10.0062" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "5 0.493745 5e-07 6.28562 10.013" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "6 0.492189 5e-07 6.29572 10.0213" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "7 0.490343 5e-07 6.30764 10.0312" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "8 0.488214 5e-07 6.32138 10.0424" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "9 0.485808 5e-07 6.33685 10.0551" << std::endl;

    // Output for GEMPIC_SPACEDIM=2
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "0 3.69818e-05 2.9765e-05 0 0.0321487 0.293507 0.967771" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "1 3.69571e-05 2.97344e-05 2.19122e-39 0.0321494 0.293512 0.967785" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "2 3.69133e-05 2.96868e-05 4.9016e-11 0.0321506 0.293524 0.967802" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "3 3.685e-05 2.96234e-05 4.31676e-10 0.0321522 0.293547 0.967821" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "4 3.67666e-05 2.95444e-05 1.67345e-09 0.0321543 0.29358 0.967844" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "5 3.66624e-05 2.94457e-05 4.45131e-09 0.0321568 0.293622 0.967872" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "6 3.65376e-05 2.93262e-05 9.48196e-09 0.0321599 0.293674 0.967908" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "7 3.63924e-05 2.91889e-05 1.74089e-08 0.0321634 0.293739 0.967948" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "8 3.62273e-05 2.9031e-05 2.87007e-08 0.0321673 0.29382 0.967992" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by.tmp") << "9 3.6043e-05 2.88571e-05 4.35711e-08 0.0321717 0.293915 0.968039" << std::endl;

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 1, 1, 1, 1, 2>();
#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_by.tmp.0", "test_vlasov_maxwell_by.output");

    amrex::Finalize();
}




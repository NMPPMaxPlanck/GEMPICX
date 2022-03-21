#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hsall_fem.H>

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

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{
    const int degmw = 2;
    const int strang_order = 2;
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params();
    VlMa.propagator = 2;
    VlMa.set_prop_related();
    if (int(vdim / 2.5) * 2 + 1 < 3)
    {
        VlMa.Bx = VlMa.Bz;
        VlMa.Bz = "0.0";
    }
    amrex::GpuArray<std::string, int(vdim / 2.5) * 2 + 1> fields_B;
    fields_B[0] = VlMa.Bx;
    if (int(vdim / 2.5) * 2 + 1 > 1)
    {
        fields_B[1] = VlMa.By;
    }
    if (int(vdim / 2.5) * 2 + 1 > 1)
    {
        fields_B[2] = VlMa.Bz;
    }
    if (GEMPIC_SPACEDIM == 1 & vdim == 1)
    {
        // For 1D1V change parameters to make a Landau damping
        VlMa.sim_name = "Landau";
        VlMa.n_part_per_cell = {10000};
        VlMa.k = {0.5, 0.5, 0.5};
        VlMa.density = "1.0 + 0.5 * cos(kvarx * x)";
        VlMa.Bz = "0.0";
    }
    VlMa.n_steps = 10;
    if (GEMPIC_SPACEDIM == 1)
    {
        VlMa.n_steps = 1;
    }
    VlMa.set_computed_params();

    // Weibel parameters
    std::vector<std::vector<std::vector<amrex::Real>>> meanVelocity{
        {{0.0, 0.0, 0.0}}};  // species, gaussian, vdim
    std::vector<std::vector<std::vector<amrex::Real>>> vThermal{
        {{0.014142135623730949, 0.04898979485566356, 0.04898979485566356}}};
    ;
    std::vector<std::vector<amrex::Real>> vWeight{{1.0}};
    VlMa.meanVelocity = meanVelocity;
    VlMa.vThermal = vThermal;
    VlMa.vWeight = vWeight;

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    // infrastructure
    computational_domain infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    amrex::GpuArray<std::string, 2> fields = {VlMa.rho, VlMa.phi};
    mw_yee.template init_rho_phi<degmw>(fields, VlMa.k_gpu, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    amrex::Real vol = (infra.geom.ProbHi(0) - infra.geom.ProbLo(0)) *
                      (infra.geom.ProbHi(1) - infra.geom.ProbLo(1)) *
                      (infra.geom.ProbHi(2) - infra.geom.ProbLo(2));
    diagnostics<vdim, numspec, degx, degy, degz, degmw> diagn(
        mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim, numspec>(infra, part_gr, VlMa, VlMa.meanVelocity,
                                              VlMa.vThermal, VlMa.vWeight, 0);
    loop_preparation<vdim, numspec, degx, degy, degz, degmw, false>(
        VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, fields_B);

    //------------------------------------------------------------------------------
    // timeloop
    time_loop_hsall_fem<vdim, numspec, degx, degy, degz, degmw, true>(
        infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_all_lower", strang_order);
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output
    file contains all outputs. For each dimension, apart from running the main_main for the
    dimension, the output for the other dimensions needs to be outputted, so that the comparison to
    the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1 vdim=2, GEMPIC_SPACEDIM=2 vdim=3 */

#if (GEMPIC_SPACEDIM == 1)
    // Output for GEMPIC_SPACEDIM=1 vdim=2
    main_main<2, 1, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "0 3.11474e-05 2.66688e-05 0 0 0 5e-07 0.0629525 0.285611 0.990676 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "1 3.11208e-05 2.66766e-05 2.6637e-13 1.22521e-16 8.49692e-16 4.9971e-07 0.0629527 "
           "0.285628 0.990641 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "2 3.10768e-05 2.66765e-05 1.04982e-12 1.94511e-15 1.34181e-14 4.99157e-07 0.0629533 "
           "0.285656 0.990604 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "3 3.10147e-05 2.66612e-05 2.3054e-12 9.72187e-15 6.64634e-14 4.99244e-07 0.0629543 "
           "0.285697 0.990567 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "4 3.0934e-05 2.66326e-05 3.96341e-12 3.01873e-14 2.03742e-13 5.01357e-07 0.0629556 "
           "0.285749 0.990529 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "5 3.08359e-05 2.65986e-05 5.936e-12 7.20631e-14 4.78283e-13 5.07176e-07 0.0629572 "
           "0.285813 0.990492 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "6 3.07198e-05 2.65483e-05 8.12565e-12 1.4544e-13 9.45374e-13 5.18443e-07 0.0629591 "
           "0.285887 0.990458 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "7 3.05846e-05 2.64855e-05 1.04338e-11 2.61095e-13 1.6551e-12 5.3672e-07 0.0629614 "
           "0.285972 0.990427 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "8 3.04298e-05 2.64068e-05 1.27694e-11 4.29798e-13 2.6453e-12 5.6315e-07 0.062964 "
           "0.286068 0.990398 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "9 3.0256e-05 2.63047e-05 1.50563e-11 6.61687e-13 3.93582e-12 5.98254e-07 0.062967 "
           "0.286177 0.990371 0.981747"
        << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "0 0.594233 0 5e-07 0.00623372 0.0609654 0.185672" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    main_main<3, 1, 1, 1, 1>();

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "0 0.594233 0 5e-07 0.00623372 0.0609654 0.185672" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "0 3.11474e-05 2.66688e-05 0 0 0 5e-07 0.0629525 0.285611 0.990676 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "1 3.11208e-05 2.66766e-05 2.6637e-13 1.22521e-16 8.49692e-16 4.9971e-07 0.0629527 "
           "0.285628 0.990641 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "2 3.10768e-05 2.66765e-05 1.04982e-12 1.94511e-15 1.34181e-14 4.99157e-07 0.0629533 "
           "0.285656 0.990604 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "3 3.10147e-05 2.66612e-05 2.3054e-12 9.72187e-15 6.64634e-14 4.99244e-07 0.0629543 "
           "0.285697 0.990567 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "4 3.0934e-05 2.66326e-05 3.96341e-12 3.01873e-14 2.03742e-13 5.01357e-07 0.0629556 "
           "0.285749 0.990529 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "5 3.08359e-05 2.65986e-05 5.936e-12 7.20631e-14 4.78283e-13 5.07176e-07 0.0629572 "
           "0.285813 0.990492 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "6 3.07198e-05 2.65483e-05 8.12565e-12 1.4544e-13 9.45374e-13 5.18443e-07 0.0629591 "
           "0.285887 0.990458 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "7 3.05846e-05 2.64855e-05 1.04338e-11 2.61095e-13 1.6551e-12 5.3672e-07 0.0629614 "
           "0.285972 0.990427 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "8 3.04298e-05 2.64068e-05 1.27694e-11 4.29798e-13 2.6453e-12 5.6315e-07 0.062964 "
           "0.286068 0.990398 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_all_lower.tmp")
        << "9 3.0256e-05 2.63047e-05 1.50563e-11 6.61687e-13 3.93582e-12 5.98254e-07 0.062967 "
           "0.286177 0.990371 0.981747"
        << std::endl;

#endif

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_vlasov_maxwell_hs_all_lower.output.0",
                    "test_vlasov_maxwell_hs_all_lower.output");

    amrex::Finalize();
}

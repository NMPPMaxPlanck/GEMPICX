´ #include<AMReX.H>
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
    VlMa.propagator = 1;
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
    VlMa.set_computed_params();

    // Weibel parameters
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> meanVelocity{
        {{0.0, 0.0, 0.0}}};  // species, gaussian, vdim
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> vThermal{
        {{0.014142135623730949, 0.04898979485566356, 0.04898979485566356}}};
    ;
    amrex::Vector<amrex::Vector<amrex::Real>> vWeight{{1.0}};
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
    mw_yee.template init_rho_phi<degmw>(fields, VlMa.k, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    amrex::Real vol = (infra.geom.ProbHi(0) - infra.geom.ProbLo(0)) *
                      (infra.geom.ProbHi(1) - infra.geom.ProbLo(1)) *
                      (infra.geom.ProbHi(2) - infra.geom.ProbLo(2));
    diagnostics<vdim, numspec, degx, degy, degz, degmw> diagn(
        mw_yee.nsteps, VlMa.save_fields, VlMa.save_particles, VlMa.save_checkpoint, VlMa.sim_name,
        vol);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim, numspec>(infra, part_gr, VlMa, VlMa.meanVelocity,
                                              VlMa.vThermal, VlMa.vWeight, 0);
    loop_preparation<vdim, numspec, degx, degy, degz, degmw>(VlMa, infra, &mw_yee, &part_gr, &diagn,
                                                             VlMa.time_staggered, fields_B);

    //------------------------------------------------------------------------------
    // timeloop
    time_loop_hs_fem<vdim, numspec, degx, degy, degz, degmw, true>(
        infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_lower", strang_order);
}

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      overwrite_amrex_parser_defaults);
    const int vdim = 3, numspec = 1, degx = 1, degy = 1, degz = 1;

    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output
    file contains all outputs. For each dimension, apart from running the main_main for the
    dimension, the output for the other dimensions needs to be outputted, so that the comparison to
    the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1 vdim=2, GEMPIC_SPACEDIM=2 vdim=3 */

#if (GEMPIC_SPACEDIM == 1)
    const int vdim2 = 2;
    // Output for GEMPIC_SPACEDIM=1 vdim=2
    main_main<vdim2, numspec, degx, degy, degz>();

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "0 3.11474e-05 2.66688e-05 0 0 0 5e-07 0.0629525 0.285611 0.990676 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "1 3.11209e-05 2.66766e-05 2.66273e-09 1.22445e-12 8.49627e-12 4.9971e-07 0.0629527 "
           "0.285628 0.990641 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "2 3.10769e-05 2.66765e-05 1.04942e-08 1.94391e-11 1.34171e-10 4.99157e-07 0.0629533 "
           "0.285656 0.990604 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "3 3.10148e-05 2.66611e-05 2.30448e-08 9.71578e-11 6.6458e-10 4.99245e-07 0.0629543 "
           "0.285697 0.990567 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "4 3.09342e-05 2.66326e-05 3.96175e-08 3.0168e-10 2.03725e-09 5.01362e-07 0.0629556 "
           "0.285749 0.990529 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "5 3.0836e-05 2.65985e-05 5.93347e-08 7.20164e-10 4.78244e-09 5.07189e-07 0.0629572 "
           "0.285813 0.990492 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "6 3.072e-05 2.65482e-05 8.12207e-08 1.45345e-09 9.45302e-09 5.18468e-07 0.0629591 "
           "0.285887 0.990458 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "7 3.05848e-05 2.64855e-05 1.0429e-07 2.60923e-09 1.65499e-08 5.36765e-07 0.0629614 "
           "0.285972 0.990426 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "8 3.043e-05 2.64067e-05 1.27632e-07 4.29514e-09 2.64516e-08 5.63221e-07 0.062964 "
           "0.286068 0.990397 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "9 3.02562e-05 2.63047e-05 1.50486e-07 6.61246e-09 3.93568e-08 5.98358e-07 0.062967 "
           "0.286177 0.990371 0.981747"
        << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "0 0.594233 0 5e-07 0.00623372 0.0609654 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "1 0.594006 3.02204e-09 4.99713e-07 0.00680383 0.0878472 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "2 0.593304 1.18655e-08 4.99011e-07 0.0085674 0.149009 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "3 0.592129 2.59082e-08 4.98345e-07 0.0115213 0.218736 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "4 0.590482 4.41996e-08 4.98398e-07 0.0156609 0.289929 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "5 0.588365 6.55708e-08 4.99986e-07 0.0209795 0.361362 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "6 0.585784 8.87651e-08 5.03941e-07 0.0274678 0.43284 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "7 0.582741 1.12546e-07 5.10992e-07 0.0351146 0.504222 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "8 0.579242 1.35791e-07 5.21641e-07 0.0439074 0.575492 0.185673" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "9 0.575293 1.57649e-07 5.36063e-07 0.0538319 0.646583 0.185673" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    main_main<vdim, numspec, degx, degy, degz>();

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "0 0.594233 0 5e-07 0.00623372 0.0609654 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "1 0.594006 3.02204e-09 4.99713e-07 0.00680383 0.0878472 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "2 0.593304 1.18655e-08 4.99011e-07 0.0085674 0.149009 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "3 0.592129 2.59082e-08 4.98345e-07 0.0115213 0.218736 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "4 0.590482 4.41996e-08 4.98398e-07 0.0156609 0.289929 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "5 0.588365 6.55708e-08 4.99986e-07 0.0209795 0.361362 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "6 0.585784 8.87651e-08 5.03941e-07 0.0274678 0.43284 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "7 0.582741 1.12546e-07 5.10992e-07 0.0351146 0.504222 0.185672" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "8 0.579242 1.35791e-07 5.21641e-07 0.0439074 0.575492 0.185673" << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "9 0.575293 1.57649e-07 5.36063e-07 0.0538319 0.646583 0.185673" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp") << std::endl;

    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "0 3.11474e-05 2.66688e-05 0 0 0 5e-07 0.0629525 0.285611 0.990676 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "1 3.11209e-05 2.66766e-05 2.66273e-09 1.22445e-12 8.49627e-12 4.9971e-07 0.0629527 "
           "0.285628 0.990641 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "2 3.10769e-05 2.66765e-05 1.04942e-08 1.94391e-11 1.34171e-10 4.99157e-07 0.0629533 "
           "0.285656 0.990604 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "3 3.10148e-05 2.66611e-05 2.30448e-08 9.71578e-11 6.6458e-10 4.99245e-07 0.0629543 "
           "0.285697 0.990567 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "4 3.09342e-05 2.66326e-05 3.96175e-08 3.0168e-10 2.03725e-09 5.01362e-07 0.0629556 "
           "0.285749 0.990529 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "5 3.0836e-05 2.65985e-05 5.93347e-08 7.20164e-10 4.78244e-09 5.07189e-07 0.0629572 "
           "0.285813 0.990492 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "6 3.072e-05 2.65482e-05 8.12207e-08 1.45345e-09 9.45302e-09 5.18468e-07 0.0629591 "
           "0.285887 0.990458 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "7 3.05848e-05 2.64855e-05 1.0429e-07 2.60923e-09 1.65499e-08 5.36765e-07 0.0629614 "
           "0.285972 0.990426 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "8 3.043e-05 2.64067e-05 1.27632e-07 4.29514e-09 2.64516e-08 5.63221e-07 0.062964 "
           "0.286068 0.990397 0.981747"
        << std::endl;
    PrintToFile("test_vlasov_maxwell_hs_lower.tmp")
        << "9 3.02562e-05 2.63047e-05 1.50486e-07 6.61246e-09 3.93568e-08 5.98358e-07 0.062967 "
           "0.286177 0.990371 0.981747"
        << std::endl;

#endif

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_vlasov_maxwell_hs_lower.output.0", "test_vlasov_maxwell_hs_lower.output");

    amrex::Finalize();
}

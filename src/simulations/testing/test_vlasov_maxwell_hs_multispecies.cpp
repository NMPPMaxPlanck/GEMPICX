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
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_profiling.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;
using namespace Profiling;

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, int propagator>
void main_main ()
{
    const int strang_order = 2;
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("ctest_Ion_Acoustic_Wave", // sim_name
    {AMREX_D_DECL(16, 2, 2)}, // n_cell_vector
    {4000, 4000}, // n_part_per_cell
                    10, // n_steps
                    10000, // output freq
                    10000, // output freq
                    10000, // output freq
    {AMREX_D_DECL(1, 1, 1)}, // periodicity
                    {2,2,2}, // max_grid_size
                    0.05, // dt
    {-1.0, 1.0}, // charge
    {1.0, 200.0}, // mass
                    0.6283185, // k
                    "1.0", // WF (overwritten later)
                    "0.0", // Bx
                    "0.0", // By
                    "0.0", // Bz
                    "4 * 0.5 * cos(0.5 * x)", // phi
                    1, // num_gaussians
                    1); // propagator
    VlMa.n_steps = 5;
    VlMa.set_computed_params();

    amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
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
    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic, VlMa.real_box);
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);
    amrex::GpuArray<std::string, 2> fields = {VlMa.rho, VlMa.phi};
    mw_yee.template init_rho_phi<degmw>(fields, VlMa.k, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa.charge, VlMa.mass, infra);

    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec,degx,degy,degz,degmw> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    // FIRST SPECIES
    std::array<std::vector<amrex::Real>, vdim> VM{}, VD{}, VW{};
    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VD[j].push_back(1.0);
        VW[j].push_back(1.0);
    }
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0);

    // SECOND SPECIES
    std::array<std::vector<amrex::Real>, vdim> VM2{}, VD2{}, VW2{};
    for (int j=0; j<vdim; j++) {
        VM2[j].push_back(0.0);
        VD2[j].push_back(0.00070710678118654751);
        VW2[j].push_back(1.0);
    }
    VlMa.VM = VM2;
    VlMa.VD = VD2;
    VlMa.VW = VW2;
    VlMa.WF = "1.0 + 0.2 * cos(kvarx * x)";
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 1);

    loop_preparation<vdim, numspec, degx, degy, degz, degmw, true>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, fields_B);


    //------------------------------------------------------------------------------
    // timeloop
    switch (propagator) {
    case 1:
        time_loop_hs_fem<vdim, numspec, degx, degy, degz, degmw, true>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_multispecies", strang_order);
        break;
    case 3:
        time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degmw, true, false, true>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_multispecies", strang_order);
        break;
    }

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_vlasov_maxwell_hs_multispecies.tmp.0");

    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output file contains all outputs.
    For each dimension, apart from running the main_main for the dimension, the output for the other dimensions needs to be
    outputted, so that the comparison to the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1, GEMPIC_SPACEDIM=2, GEMPIC_SPACEDIM=3 */

#if (GEMPIC_SPACEDIM == 1)
    // Output for GEMPIC_SPACEDIM=1
    main_main<1, 2, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=2
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "0 0.0506265 4.23364e-05 0 99.7186 79.4468 79.5905" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "1 0.0504997 4.23808e-05 1.42686e-09 99.7247 79.4522 79.5906" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "2 0.0501258 4.21012e-05 2.17887e-08 99.7434 79.4681 79.5906" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "3 0.0494968 4.13694e-05 1.09428e-07 99.7747 79.4946 79.5906" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "4 0.0486315 4.03803e-05 3.3164e-07 99.8177 79.5309 79.5907" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "5 0.0475393 3.95e-05 7.25928e-07 99.872 79.5766 79.5908" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "6 0.0462339 3.82532e-05 1.25118e-06 99.9371 79.6318 79.5909" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "7 0.0447294 3.69079e-05 1.79882e-06 100.012 79.6956 79.591" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "8 0.0430483 3.61116e-05 2.29378e-06 100.095 79.7665 79.591" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "9 0.0412097 3.56089e-05 2.78074e-06 100.187 79.8439 79.5911" << std::endl;

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "0 0.0505593 1.41654e-06 4.02702e-06 0 0 0 1492.06 796.104 791.187 797.504" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "1 0.0504335 1.21928e-06 4.73958e-06 1.26102e-11 1.12859e-09 1.74713e-09 1492.12 796.156 791.187 797.503" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "2 0.0500578 1.55051e-06 5.99684e-06 1.94684e-10 1.74762e-08 2.88881e-08 1492.31 796.313 791.187 797.502" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "3 0.0494331 2.3849e-06 7.89144e-06 9.35913e-10 8.27629e-08 1.49491e-07 1492.62 796.576 791.187 797.501" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "4 0.0485704 3.42868e-06 1.02786e-05 2.78406e-09 2.31218e-07 4.56042e-07 1493.04 796.938 791.186 797.499" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "5 0.0474831 4.46523e-06 1.31909e-05 6.36678e-09 4.72266e-07 9.9887e-07 1493.58 797.395 791.186 797.498" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "6 0.046181 5.50248e-06 1.66223e-05 1.22832e-08 7.85988e-07 1.74595e-06 1494.22 797.942 791.185 797.495" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "7 0.0446791 6.61992e-06 2.04839e-05 2.10408e-08 1.1508e-06 2.59629e-06 1494.97 798.575 791.185 797.493" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "8 0.0430007 7.83768e-06 2.46452e-05 3.31693e-08 1.57038e-06 3.38028e-06 1495.8 799.283 791.185 797.49" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "9 0.0411656 9.45255e-06 2.92716e-05 4.92708e-08 2.07444e-06 3.9005e-06 1496.71 800.061 791.184 797.488" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "0 0.52624 0 4.94354 7.93481" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "1 0.524937 0 4.95001 7.9408" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "2 0.521043 0 4.96937 7.95747" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "3 0.514625 0 5.00122 7.9847" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "4 0.505788 0 5.04517 8.02223" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "5 0.494587 0 5.10077 8.06939" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "6 0.481251 0 5.16706 8.12531" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "7 0.465967 0 5.24301 8.18877" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "8 0.448927 0 5.32772 8.25923" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "9 0.43032 0 5.42023 8.33567" << std::endl;

    // Output for GEMPIC_SPACEDIM=2
    main_main<2, 2, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "0 0.0505593 1.41654e-06 4.02702e-06 0 0 0 1492.06 796.104 791.187 797.504" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "1 0.0504335 1.21928e-06 4.73958e-06 1.26102e-11 1.12859e-09 1.74713e-09 1492.12 796.156 791.187 797.503" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "2 0.0500578 1.55051e-06 5.99684e-06 1.94684e-10 1.74762e-08 2.88881e-08 1492.31 796.313 791.187 797.502" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "3 0.0494331 2.3849e-06 7.89144e-06 9.35913e-10 8.27629e-08 1.49491e-07 1492.62 796.576 791.187 797.501" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "4 0.0485704 3.42868e-06 1.02786e-05 2.78406e-09 2.31218e-07 4.56042e-07 1493.04 796.938 791.186 797.499" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "5 0.0474831 4.46523e-06 1.31909e-05 6.36678e-09 4.72266e-07 9.9887e-07 1493.58 797.395 791.186 797.498" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "6 0.046181 5.50248e-06 1.66223e-05 1.22832e-08 7.85988e-07 1.74595e-06 1494.22 797.942 791.185 797.495" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "7 0.0446791 6.61992e-06 2.04839e-05 2.10408e-08 1.1508e-06 2.59629e-06 1494.97 798.575 791.185 797.493" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "8 0.0430007 7.83768e-06 2.46452e-05 3.31693e-08 1.57038e-06 3.38028e-06 1495.8 799.283 791.185 797.49" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "9 0.0411656 9.45255e-06 2.92716e-05 4.92708e-08 2.07444e-06 3.9005e-06 1496.71 800.061 791.184 797.488" << std::endl;

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "0 0.52624 0 4.94354 7.93481" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "1 0.524937 0 4.95001 7.9408" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "2 0.521043 0 4.96937 7.95747" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "3 0.514625 0 5.00122 7.9847" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "4 0.505788 0 5.04517 8.02223" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "5 0.494587 0 5.10077 8.06939" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "6 0.481251 0 5.16706 8.12531" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "7 0.465967 0 5.24301 8.18877" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "8 0.448927 0 5.32772 8.25923" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "9 0.43032 0 5.42023 8.33567" << std::endl;

    // Output for GEMPIC_SPACEDIM=2
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "0 0.0506265 4.23364e-05 0 99.7186 79.4468 79.5905" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "1 0.0504997 4.23808e-05 1.42686e-09 99.7247 79.4522 79.5906" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "2 0.0501258 4.21012e-05 2.17887e-08 99.7434 79.4681 79.5906" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "3 0.0494968 4.13694e-05 1.09428e-07 99.7747 79.4946 79.5906" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "4 0.0486315 4.03803e-05 3.3164e-07 99.8177 79.5309 79.5907" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "5 0.0475393 3.95e-05 7.25928e-07 99.872 79.5766 79.5908" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "6 0.0462339 3.82532e-05 1.25118e-06 99.9371 79.6318 79.5909" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "7 0.0447294 3.69079e-05 1.79882e-06 100.012 79.6956 79.591" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "8 0.0430483 3.61116e-05 2.29378e-06 100.095 79.7665 79.591" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_multispecies.tmp") << "9 0.0412097 3.56089e-05 2.78074e-06 100.187 79.8439 79.5911" << std::endl;

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 2, 3, 1, 2, 2, 3>(); // hs_zigzag
#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_hs_multispecies.tmp.0", "test_vlasov_maxwell_hs_multispecies.output");

    amrex::Finalize();
}




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
#include <GEMPIC_parameters.H>
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

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    const int degmw = 2;
    const int strang_order = 2;
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params();
    if (int(vdim/2.5)*2+1 < 3) {
        VlMa.Bx = VlMa.Bz;
        VlMa.Bz = "0.0";
    }
    amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_B[0] = VlMa.Bx;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[1] = VlMa.By;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[2] = VlMa.Bz;
    }
    if (GEMPIC_SPACEDIM==1 & vdim==1) {
        // For 1D1V change parameters to make a Landau damping
        VlMa.sim_name = "Landau";
        VlMa.n_part_per_cell = {10000};
        VlMa.k = {0.5,0.5,0.5};
        VlMa.WF = "1.0 + 0.5 * cos(kvarx * x)";
        VlMa.Bz = "0.0";
    }
    VlMa.n_steps = 10;
    VlMa.set_computed_params();

  // Weibel parameters
    std::vector<std::vector<std::vector<amrex::Real>>> VM {{{0.0,0.0,0.0}}};  // species, gaussian, vdim
    std::vector<std::vector<std::vector<amrex::Real>>> VD {{{0.014142135623730949, 0.04898979485566356, 0.04898979485566356}}};;
    std::vector<std::vector<amrex::Real>> VW {{1.0}};
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
    mw_yee.template init_rho_phi<degmw>(fields, VlMa.k_gpu, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec,degx,degy,degz,degmw> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);
    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0);
    loop_preparation<vdim, numspec, degx, degy, degz, degmw, false>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, fields_B);

    //------------------------------------------------------------------------------
    // timeloop
    time_loop_boris_fd<vdim, numspec, degx, degy, degz, degmw, true, false>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_by_lower", strang_order);

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_vlasov_maxwell_by_lower.tmp.0");

    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output file contains all outputs.
    For each dimension, apart from running the main_main for the dimension, the output for the other dimensions needs to be
    outputted, so that the comparison to the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1 vdim=2, GEMPIC_SPACEDIM=2 vdim=3 */

#if (GEMPIC_SPACEDIM == 1)
    // Output for GEMPIC_SPACEDIM=1 vdim=2
    main_main<2, 1, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "0 3.11474e-05 2.66689e-05 0 0 0 5e-07 0.0629525 0.285611 0.990676 0.981747" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "1 3.11208e-05 2.66775e-05 2.68111e-09 0 0 4.99689e-07 0.0629528 0.285628 0.990642 0.981747" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "2 3.10766e-05 2.66739e-05 1.05645e-08 4.89478e-12 3.39823e-11 4.98853e-07 0.0629534 0.285657 0.990607 0.981746" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "3 3.10144e-05 2.6656e-05 2.31901e-08 4.35944e-11 3.00522e-10 4.98118e-07 0.0629543 0.285698 0.990572 0.981743" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "4 3.09342e-05 2.66302e-05 3.98433e-08 1.71704e-10 1.17087e-09 4.98662e-07 0.0629555 0.285751 0.990535 0.98174" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "5 3.08365e-05 2.65948e-05 5.96243e-08 4.67356e-10 3.1403e-09 5.02065e-07 0.0629569 0.285815 0.990499 0.981735" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "6 3.07205e-05 2.65447e-05 8.15317e-08 1.0255e-09 6.76271e-09 5.10082e-07 0.0629587 0.28589 0.990467 0.981729" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "7 3.05852e-05 2.64832e-05 1.04553e-07 1.95122e-09 1.25763e-08 5.24417e-07 0.0629606 0.285975 0.990437 0.981722" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "8 3.04306e-05 2.64042e-05 1.27752e-07 3.35266e-09 2.10291e-08 5.46464e-07 0.0629629 0.286073 0.990409 0.981715" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "9 3.02561e-05 2.62975e-05 1.50345e-07 5.33428e-09 3.24135e-08 5.7708e-07 0.0629655 0.286182 0.990383 0.981706" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "0 0.594233 7.76798e-11 5e-07 0.00623372 0.0609654 0.185672" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "1 0.594006 3.42252e-09 4.99689e-07 0.00680395 0.0878519 0.185672" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "2 0.593304 1.26299e-08 4.98833e-07 0.00856848 0.14904 0.185671" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "3 0.592129 2.70608e-08 4.97743e-07 0.0115248 0.218805 0.185671" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "4 0.590481 4.574e-08 4.97004e-07 0.0156687 0.290048 0.185669" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "5 0.588364 6.747e-08 4.97382e-07 0.0209946 0.361546 0.185668" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "6 0.585781 9.09519e-08 4.99723e-07 0.0274931 0.433097 0.185666" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "7 0.582737 1.14885e-07 5.04826e-07 0.0351548 0.504568 0.185664" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "8 0.579236 1.38105e-07 5.13315e-07 0.0439667 0.575943 0.185661" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "9 0.575284 1.59745e-07 5.25529e-07 0.0539171 0.647159 0.185658" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    main_main<3, 1, 1, 1, 1>();

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "0 0.594233 7.76798e-11 5e-07 0.00623372 0.0609654 0.185672" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "1 0.594006 3.42252e-09 4.99689e-07 0.00680395 0.0878519 0.185672" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "2 0.593304 1.26299e-08 4.98833e-07 0.00856848 0.14904 0.185671" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "3 0.592129 2.70608e-08 4.97743e-07 0.0115248 0.218805 0.185671" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "4 0.590481 4.574e-08 4.97004e-07 0.0156687 0.290048 0.185669" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "5 0.588364 6.747e-08 4.97382e-07 0.0209946 0.361546 0.185668" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "6 0.585781 9.09519e-08 4.99723e-07 0.0274931 0.433097 0.185666" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "7 0.582737 1.14885e-07 5.04826e-07 0.0351548 0.504568 0.185664" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "8 0.579236 1.38105e-07 5.13315e-07 0.0439667 0.575943 0.185661" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "9 0.575284 1.59745e-07 5.25529e-07 0.0539171 0.647159 0.185658" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "0 3.11474e-05 2.66689e-05 0 0 0 5e-07 0.0629525 0.285611 0.990676 0.981747" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "1 3.11208e-05 2.66775e-05 2.68111e-09 0 0 4.99689e-07 0.0629528 0.285628 0.990642 0.981747" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "2 3.10766e-05 2.66739e-05 1.05645e-08 4.89478e-12 3.39823e-11 4.98853e-07 0.0629534 0.285657 0.990607 0.981746" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "3 3.10144e-05 2.6656e-05 2.31901e-08 4.35944e-11 3.00522e-10 4.98118e-07 0.0629543 0.285698 0.990572 0.981743" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "4 3.09342e-05 2.66302e-05 3.98433e-08 1.71704e-10 1.17087e-09 4.98662e-07 0.0629555 0.285751 0.990535 0.98174" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "5 3.08365e-05 2.65948e-05 5.96243e-08 4.67356e-10 3.1403e-09 5.02065e-07 0.0629569 0.285815 0.990499 0.981735" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "6 3.07205e-05 2.65447e-05 8.15317e-08 1.0255e-09 6.76271e-09 5.10082e-07 0.0629587 0.28589 0.990467 0.981729" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "7 3.05852e-05 2.64832e-05 1.04553e-07 1.95122e-09 1.25763e-08 5.24417e-07 0.0629606 0.285975 0.990437 0.981722" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "8 3.04306e-05 2.64042e-05 1.27752e-07 3.35266e-09 2.10291e-08 5.46464e-07 0.0629629 0.286073 0.990409 0.981715" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_by_lower.tmp") << "9 3.02561e-05 2.62975e-05 1.50345e-07 5.33428e-09 3.24135e-08 5.7708e-07 0.0629655 0.286182 0.990383 0.981706" << std::endl;

#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_by_lower.tmp.0", "test_vlasov_maxwell_by_lower.output");

    amrex::Finalize();
}




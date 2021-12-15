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
#include <GEMPIC_parameters.H>
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
    gempic_parameters<vdim, numspec> VlMa;
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
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa.n_part_per_cell, VlMa.k, VlMa.WF, VlMa.VM, VlMa.VD, VlMa.VW, 0);

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
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa.n_part_per_cell, VlMa.k, VlMa.WF, VlMa.VM, VlMa.VD, VlMa.VW, 1);

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

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 2, 3, 1, 2, 2, 3>(); // hs_zigzag

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_hs_multispecies.tmp.0", "test_vlasov_maxwell_hs_multispecies.output");

    amrex::Finalize();
}




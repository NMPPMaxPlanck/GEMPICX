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
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
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
    VlMa.set_params("test_vlasov_maxwell_hs_all", {12,8,8});
    VlMa.propagator = 3;
    VlMa.set_prop_related();

    VlMa.n_steps = 5;
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
    computational_domain infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    mw_yee.template init_rho_phi<degmw>(zero, zero, VlMa.k_gpu, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec,degx,degy,degz,degmw> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0, wave_function);
    loop_preparation<vdim, numspec, degx, degy, degz, degmw, true>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, zero, zero, zero);


    //------------------------------------------------------------------------------
    // timeloop
    time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degmw, true, false, true>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_zigzag_C2", strang_order);

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (ParallelDescriptor::MyProc()==0) remove("test_vlasov_maxwell_hs_zigzag_C2.tmp.0");

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 1, 6, 5, 4, 4>();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_hs_zigzag_C2.tmp.0", "test_vlasov_maxwell_hs_zigzag_C2.output");
    amrex::Finalize();
}




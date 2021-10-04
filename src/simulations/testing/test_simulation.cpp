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

//#include <cudaProfiler.h>
//#include <nvToolsExt.h>

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

#define VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO 0
#define VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION 1

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t, int funcSelect)
{
    switch(funcSelect){
    case VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION :
      return 1.0 ;//+ 0.5 * std::cos(0.5 * x);
    case VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO :
      return 0.0 ;
    }
    return 0.0;
}

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
    VlMa.set_params("test_simulation",

            {12,8,8});//{40,40,40}); // Number of cells:
    VlMa.n_part_per_cell = {2000};//{20};
    VlMa.n_steps = 10;

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
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic, VlMa.real_box);
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);
    amrex::GpuArray<int, 2> funcSelect;
    funcSelect[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelect[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    mw_yee.template init_rho_phi<degmw>(infra, funcSelect);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa.charge, VlMa.mass, infra);

    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec,degx,degy,degz,degmw> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);

    const bool output = false;
    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0, wave_function);

    amrex::GpuArray<int, int(vdim/2.5)*2+1> funcSelectB;
    funcSelectB[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[2] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    loop_preparation<vdim, numspec, degx, degy, degz, degmw, output>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, funcSelectB);

  //  cuProfilerStart();
  //  nvtxRangePush("time_loop");
    //------------------------------------------------------------------------------
    // timeloop
    time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degmw, true,
            false, // bool to activate profiling
            output>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_zigzag_C2", strang_order);
  //  nvtxRangePop();
  //  cuProfilerStop();

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (ParallelDescriptor::MyProc()==0) remove("test_vlasov_maxwell_hs_zigzag_C2.tmp.0");

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 1, 2, 2, 2, 2>();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_hs_zigzag_C2.tmp.0", "test_vlasov_maxwell_hs_zigzag_C2.output");
    amrex::Finalize();
}




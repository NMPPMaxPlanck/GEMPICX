#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
#include <GEMPIC_vlasov_maxwell.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Profiling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real cosine(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = 1e-3 * std::cos(1.25 * x);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real zero(amrex::Real , amrex::Real , amrex::Real , amrex::Real )
{
    amrex::Real val = 0.0;
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func_phi(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = 2.0 * std::cos(0.5 * x);
    return val;
}


template< int vdim, int numspec, int degx, int degy, int degz, int degmw, int propagator>
void main_main (bool ctest)
{
    int const strang_order = 2;
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // initialize parameters
    std::string sim_name = "One_Particle";
    amrex::IntVect n_cell = {AMREX_D_DECL(4,4,4)};
    std::array<int, numspec> n_part_per_cell = {1};
    int n_steps = 1;
    int freq_x = 2;
    int freq_v = 2;
    int freq_slice = 1;
    amrex::IntVect is_periodic = {AMREX_D_DECL(1,1,1)};
    amrex::IntVect max_grid_size = {4,4,4};
    amrex::Real dt = 0.02;
    amrex::GpuArray<amrex::Real, numspec> charge = {-1.0};
    amrex::GpuArray<amrex::Real, numspec> mass = {1.0};
    //std::array<amrex::Real,GEMPIC_SPACEDIM> k = {AMREX_D_DECL(1.25,1.25,1.25)};
    amrex::Real k = 1.25;
    std::string WF = "1.0";
    std::string Bx = "0.0";
    std::string By = "0.0";
    std::string Bz = "1e-3 * cos(kvarx * x)";
    amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_B[0] = Bx;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[1] = By;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[2] = Bz;
    }
    std::string phi = "4 * 0.5 * cos(0.5 * x)";
    std::string rho = "0.0";
    bool time_staggered = false;
    amrex::Real tolerance_particles = 1.e-10;


    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params(sim_name, n_cell, n_part_per_cell, n_steps, freq_x, freq_v,
                    freq_slice, is_periodic, max_grid_size, dt, charge, mass, k,
                    WF, Bx, By, Bz, phi, 1, propagator, tolerance_particles);
    VlMa.set_computed_params();

    // infrastructure
    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic, VlMa.real_box);
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);
    mw_yee.template init_rho_phi<degmw>(zero, func_phi, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa.charge, VlMa.mass, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0;
    for(amrex::MFIter mfi=(*(part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {
        if(mfi.index() == 0) {
            amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            amrex::GpuArray<amrex::Real,vdim> velocity;
            for (int comp = 0; comp < vdim; comp++) {
                velocity[comp] = 0.1;
            }
            (part_gr).add_particle({AMREX_D_DECL(2.512, 2.2, 2.3)}, velocity, 1.0, particles);
        }
    }

    timers profiling_timers(true);

    //------------------------------------------------------------------------------
    // solve:
    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec, degx, degy, degz,degmw> diagn(mw_yee.nsteps, freq_x, freq_v, freq_slice, sim_name, vol);
    loop_preparation<vdim,numspec,degx,degy,degz,degmw, true>(VlMa, infra, &mw_yee, &part_gr, &diagn, time_staggered, zero, zero, cosine);
    std::ofstream ofs("PIC.output", std::ofstream::out);
    amrex::Print(ofs) << endl;
    switch (propagator) {
    case 0:
        time_loop_boris_fd<vdim, numspec, degx, degy, degz, degmw, true, false>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_one_part", strang_order);
        break;
    case 1:
        time_loop_hs_fem<vdim, numspec, degx, degy, degz, degmw, true>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_one_part", strang_order);
        break;
    case 3:
        time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degmw, true>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_one_part", strang_order);
    default:
        break;
    }

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_one_part.tmp.0");

    /* This ctest has a different output for each GEMPIC_SPACEDIM and vdim. Therefore, the expected_output file contains all outputs.
    For each dimension, apart from running the main_main for the dimension, the output for the other dimensions needs to be
    outputted, so that the comparison to the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1 vdim=2, GEMPIC_SPACEDIM=2 vdim=2, GEMPIC_SPACEDIM=2 vdim=3, GEMPIC_SPACEDIM=3 vdim=3 */

#if (GEMPIC_SPACEDIM == 1)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_one_part.tmp") << std::endl;
    main_main<2, 1, 1, 1, 1>(argc==1);

    // Output for GEMPIC_SPACEDIM=2 vdim=2
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.00430711 0.0028235 0 0.51 0.1 0.1" << std::endl;
    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.00430711 0.0028235 0 0 0 5e-07 0.015 0.1 0.1 0.1" << std::endl;

    // Output for GEMPIC_SPACEDIM=3 vdim=3
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.000268783 0.00017768 0.000204417 0 0 5e-07 0.015 0.1 0.1 0.1" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.581609 0 0 0.51 0.1 0.1" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=2
    AllPrintToFile("test_one_part.tmp") << std::endl;
    main_main<2, 1, 1, 1, 1>(argc==1);
    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_one_part.tmp") << std::endl;
    main_main<3, 1, 1, 1, 1>(argc==1);

    // Output for GEMPIC_SPACEDIM=3 vdim=3
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.000268783 0.00017768 0.000204417 0 0 5e-07 0.015 0.1 0.1 0.1" << std::endl;
#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.581609 0 0 0.51 0.1 0.1" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=2
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.00430711 0.0028235 0 0.51 0.1 0.1" << std::endl;
    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "0 0.00430711 0.0028235 0 0 0 5e-07 0.015 0.1 0.1 0.1" << std::endl;

    // Output for GEMPIC_SPACEDIM=3 vdim=3
    //AllPrintToFile("test_one_part.tmp") << std::endl;
    main_main<3, 1, 1, 1, 1, 2, 0>(argc==1);
    main_main<3, 1, 3, 2, 1, 4, 1>(argc==1);
    main_main<3, 1, 2, 4, 2, 4, 3>(argc==1);
#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_one_part.tmp.0", "test_one_part.output");
    amrex::Finalize();
}




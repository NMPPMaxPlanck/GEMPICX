#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_sampler.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Vlasov_Maxwell;

AMREX_GPU_HOST_DEVICE amrex::Real initial_bfield(amrex::Real x, amrex::Real y, amrex::Real z,
                                                 amrex::Real t)
{
    amrex::Real val = 1e-3 * std::cos(1.25 * x);
    return val;
}

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{
    const int degmw = 2;
    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params();
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
    if (GEMPIC_SPACEDIM == 1 && vdim == 1)
    {
        // For 1D1V change parameters to make a Landau damping
        VlMa.sim_name = "Landau";
        VlMa.n_part_per_cell = {10000};
        VlMa.k = {0.5, 0.5, 0.5};
        VlMa.density[0] = "1.0 + 0.5 * cos(kvarx * x)";
        VlMa.Bz = "0.0";
    }
    VlMa.n_steps = 10;
    VlMa.set_computed_params();

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------
    // infrastructure
    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);
    (mw_yee).template initB<degmw>(initial_bfield, initial_bfield, initial_bfield, infra);
    amrex::Real ScalarProd = amrex::MultiFab::Dot(*mw_yee.B_Masks[2], *mw_yee.B_Array[2], 0,
                                                  *mw_yee.B_Array[2], 0, 1, VlMa.Nghost);
    amrex::Real NormSquared = pow((*mw_yee.B_Array[2]).norm2(0, infra.geom.periodicity()), 2.);
    amrex::Real Norm = (*mw_yee.B_Array[2]).norm2(0, infra.geom.periodicity());
    amrex::PrintToFile("test_AMReX_Dot_additional.tmp") << std::endl;
    amrex::PrintToFile("test_AMReX_Dot_additional.tmp") << "<x,x> = " << ScalarProd << std::endl;
    amrex::PrintToFile("test_AMReX_Dot_additional.tmp") << "||x||2 = " << NormSquared << std::endl;
    amrex::PrintToFile("test_AMReX_Dot_additional.tmp") << "||x|| = " << Norm << std::endl;
    amrex::PrintToFile("test_AMReX_Dot_additional.tmp")
        << "<x,x>-||x||2 = " << ScalarProd - NormSquared << std::endl;

    bool passed = (std::abs(ScalarProd - NormSquared) < 1e-12);
    amrex::PrintToFile("test_AMReX_Dot.tmp") << std::endl;
    amrex::PrintToFile("test_AMReX_Dot.tmp") << passed << std::endl;
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_Dot.tmp.0");
    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_Dot.tmp.0");

    main_main<3, 1, 1, 1, 1>();

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_AMReX_Dot.tmp.0", "test_AMReX_Dot.output");
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_AMReX_Dot_additional.tmp.0", "test_AMReX_Dot_additional.output");

    amrex::Finalize();
}

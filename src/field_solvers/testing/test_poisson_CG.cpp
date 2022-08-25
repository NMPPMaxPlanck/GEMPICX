#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_assertion.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>

using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;

#define POISSON_CG_PHI 0
#define POISSON_CG_RHO 1

AMREX_GPU_HOST_DEVICE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z,
                                                      amrex::Real t, int funcSelect)
{
    switch (funcSelect)
    {
        case POISSON_CG_PHI:
            return std::cos(x) - std::cos(x) * std::cos(y) * std::cos(z) -
                   1.0 / 4.0 * std::cos(2 * x) * std::cos(2 * y) * std::cos(2 * z);
            break;
        case POISSON_CG_RHO:
            return -3.0 * (std::cos(x) * std::cos(y) * std::cos(z) +
                           std::cos(2 * x) * std::cos(2 * y) * std::cos(2 * z));
            break;
    }
    return 0.0;
}

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{
    const int degree = 4;

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    amrex::IntVect is_periodic = {AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect n_cell = {AMREX_D_DECL(32, 32, 32)};
    amrex::IntVect mx_grid = {AMREX_D_DECL(32, 32, 32)};

    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic, mx_grid, 0.01,
                    {1.0}, {1.0}, 0.5);
    VlMa.set_computed_params();
    VlMa.Nghost = 3;

    CompDom::computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    amrex::MultiFab kx(convert(infra.grid, *mw_yee.E_Index[0]), infra.distriMap, 1, mw_yee.Nghost);
    kx.setVal(1.0);
    amrex::MultiFab ky(convert(infra.grid, *mw_yee.E_Index[1]), infra.distriMap, 1, mw_yee.Nghost);
    ky.setVal(1.0);
    amrex::MultiFab kz(convert(infra.grid, *mw_yee.E_Index[2]), infra.distriMap, 1, mw_yee.Nghost);
    kz.setVal(1.0);

    amrex::MLNodeLaplacian linop({infra.geom}, {infra.grid},
                                 {infra.distriMap});  // linear operator class
    amrex::MultiFab sigma;
    sigma.define(infra.grid, infra.distriMap, 1, 0);
    sigma.setVal(-1.0);  // sigma is the identity (lapl = nabla dot ID nabla)
    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic,  // for lower ends
                                    amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)},
                      {AMREX_D_DECL(amrex::LinOpBCType::Periodic,  // for higher ends
                                    amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)});
    linop.setSigma(0, sigma);  // first argument: level-nr

    amrex::MLMG mlmg(linop);  // solver class
    mlmg.setMaxIter(100);
    mlmg.setMaxFmgIter(0);
    mlmg.setVerbose(0);
    mlmg.setBottomVerbose(0);

    amrex::GpuArray<int, 2> funcSelect;
    funcSelect[0] = POISSON_CG_PHI;
    funcSelect[1] = POISSON_CG_RHO;
    mw_yee.template init_rho_phi<degree>(infra, funcSelect);

    // ----------------------------------------------------------------------------------

    // Poisson solver

    amrex::MultiFab rho_copy(mw_yee.rho, amrex::make_alias, 0, 1);
    mw_yee.template solve_poisson_CG<degree>(mw_yee.rho, mw_yee.phi, kx, ky, kz, infra, 2, 5.e-11,
                                             300);

    // Poisson operator
    mw_yee.template poisson_operator<degree>(mw_yee.phi, mw_yee.rho, kx, ky, kz, infra, 0, 1e-16,
                                             100);

    amrex::PrintToFile("test_poisson_CG.output") << std::endl;
    // comparing poisson(ipoisson(rho)) to rho
    bool passed = true;
    rho_copy.minus(mw_yee.rho, 0, 1, 0);
    amrex::Real err_norm = Utils::gempic_norm(rho_copy, infra, 2);
    amrex::PrintToFile("test_poisson_CG.output") << "error is: " << err_norm << std::endl;
    amrex::Real rho_norm = Utils::gempic_norm(mw_yee.rho, infra, 2);
    gempic_assert_err(passed, rho_norm, err_norm * err_norm);
}

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(
        argc,
        argv,
        build_parm_parse,
        MPI_COMM_WORLD,
        overwrite_amrex_parser_defaults
    );
    const int vdim1=1, vdim2=2, vdim=3, numspec=1, degx=1, degy=1, degz=1;

#if (GEMPIC_SPACEDIM == 1)
    main_main<vdim1, numspec, degx, degy, degz>();
    main_main<vdim2, numspec, degx, degy, degz>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<vdim2, numspec, degx, degy, degz>();
    main_main<vdim, numspec, degx, degy, degz>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<vdim, numspec, degx, degy, degz>();
#endif
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_poisson_CG.output.0", "test_poisson_CG.output");
    amrex::Finalize();
}

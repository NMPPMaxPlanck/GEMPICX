#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_assertion.H>

using namespace std;
using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{  //------------------------------------------------------------------------------
    // Analytical solutions -- Maxwell
    amrex::GpuArray<std::string, vdim> fields_E;
    amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_E[0] = "cos(x+y+z-sqrt(3.0)*t)";
    fields_E[1] = "-2*cos(x+y+z-sqrt(3.0)*t)";
    fields_E[2] = "cos(x+y+z-sqrt(3.0)*t)";
    fields_B[0] = "sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";
    fields_B[1] = "0.0";
    fields_B[2] = "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";

    const int degree = 4;

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(32,32,32)};
    std::array<int,GEMPIC_SPACEDIM> mx_grid = {AMREX_D_DECL(32,32,32)};

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic, mx_grid, 0.01, {1.0}, {1.0}, 0.5);
    VlMa.set_computed_params();
    VlMa.Nghost = 3;

    CompDom::computational_domain infra;
    VlMa.initialize_infrastructure(&infra);

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(VlMa, infra);

    std::string phi = "-cos(x)*cos(y)*cos(z) - 1.0/4.0*cos(2*x)*cos(2*y)*cos(2*z)";
    std::string rho = "-3*(cos(x)*cos(y)*cos(z)+cos(2*x)*cos(2*y)*cos(2*z))";

    amrex::MultiFab kx(convert(infra.grid, *mw_yee.E_Index[0]),infra.distriMap,1,mw_yee.Nghost);
    kx.setVal(1.0, 0);
    kx.FillBoundary(infra.geom.periodicity());
    amrex::MultiFab ky(convert(infra.grid, *mw_yee.E_Index[1]),infra.distriMap,1,mw_yee.Nghost);
    ky.setVal(1.0, 0);
    ky.FillBoundary(infra.geom.periodicity());
    amrex::MultiFab kz(convert(infra.grid, *mw_yee.E_Index[2]),infra.distriMap,1,mw_yee.Nghost);
    kz.setVal(1.0, 0);
    kz.FillBoundary(infra.geom.periodicity());


    amrex::MLNodeLaplacian_FD linop({infra.geom}, {infra.grid}, {infra.distriMap}); // linear operator class
    amrex::MultiFab sigma;
    sigma.define(infra.grid, infra.distriMap, 1, 0);
    sigma.setVal(-1.0); // sigma is the identity (lapl = nabla dot ID nabla)
    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, // for lower ends
                       amrex::LinOpBCType::Periodic,
                       amrex::LinOpBCType::Periodic)},
    {AMREX_D_DECL(amrex::LinOpBCType::Periodic, // for higher ends
     amrex::LinOpBCType::Periodic,
     amrex::LinOpBCType::Periodic)});
    linop.setSigma(0, sigma); // first argument: level-nr

    amrex::MLMG mlmg(linop); // solver class
    mlmg.setMaxIter(100);
    mlmg.setMaxFmgIter(0);
    mlmg.setVerbose(0);
    mlmg.setBottomVerbose(0);

    amrex::GpuArray<std::string, 2> fields = {rho, phi};
    mw_yee.template init_rho_phi<degree>(fields, VlMa.k_gpu, infra);
    const int stencil_length = 3;

    // ----------------------------------------------------------------------------------

    // Poisson solver

    amrex::MultiFab rho_copy(mw_yee.rho, amrex::make_alias, 0, 1);
    mw_yee.template solve_poisson_CG<degree>(&mw_yee.rho, &mw_yee.phi, &kx, &ky, &kz, infra, 2, 5.e-11,300);

    // Poisson operator
    mw_yee.template poisson_operator<degree>(&mw_yee.phi, &mw_yee.rho, &kx, &ky, &kz, infra, 0, 1e-16, 100);

    amrex::AllPrintToFile("test_poisson_CG_additional.tmp") << std::endl;
    // comparing poisson(ipoisson(rho)) to rho
    bool passed = true;
    rho_copy.minus(mw_yee.rho, 0, 1, 0);
    amrex::Real err_norm = Utils::gempic_norm(&rho_copy, infra, 2);
    amrex::AllPrintToFile("test_poisson_CG_additional.tmp") << "error is: " << err_norm << std::endl;
    amrex::Real rho_norm = Utils::gempic_norm(&(mw_yee.rho), infra, 2);
    gempic_assert_err(&passed, rho_norm, err_norm*err_norm);

    amrex::AllPrintToFile("test_poisson_CG.tmp") << std::endl;
    amrex::AllPrintToFile("test_poisson_CG.tmp") << passed << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_poisson_CG.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_poisson_CG_additional.tmp.0");

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1>();
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1>();
    main_main<3, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_poisson_CG.tmp.0", "test_poisson_CG.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_poisson_CG_additional.tmp.0", "test_poisson_CG_additional.output");
    amrex::Finalize();
}




/*------------------------------------------------------------------------------
 Test 3D Maxwell Yee Solver (finite differences) on periodic grid

  For the Maxwell-equations we use the solution
  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\
                          -2\cos(x_1+x_2+x_3 - \sqrt(3) t) \\
                            \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
  B(x,t) = \begin{pmatrix} \sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ 0 \\ -\sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}

 For the Poisson equation we use:
 E(x,t) = \begin{pmatrix} -\sin(x)\cos(y)\cos(z)-0.5\sin(2x)cos(2y)cos(2z)\\
                          -\cos(x)\sin(y)\cos(z)-0.5\cos(2x)sin(2y)cos(2z)\\
                          -\cos(x)\cos(y)\sin(z)-0.5\cos(2x)cos(2y)sin(2z) \end{pmatrix}
------------------------------------------------------------------------------*/

#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_vlasov_maxwell.H>

using namespace std;
using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{  //------------------------------------------------------------------------------
    // Analytical solutions -- Maxwell
    std::array<std::string, vdim> fields_E;
    std::array<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_E[0] = "cos(x+y+z-sqrt(3.0)*t)";
    fields_E[1] = "-2*cos(x+y+z-sqrt(3.0)*t)";
    fields_E[2] = "cos(x+y+z-sqrt(3.0)*t)";
    fields_B[0] = "sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";
    fields_B[1] = "0.0";
    fields_B[2] = "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";

    const int degree = 4;
    int bdim = int(vdim/2.5)*2+1;

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(32,32,32)};

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic,
                    32, 0.01, {1.0}, {1.0}, 0.5);
    VlMa.set_computed_params();
    VlMa.Nghost = 3;

    Infra::infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(VlMa, infra);

    std::string phi = "-cos(x)*cos(y)*cos(z) - 1.0/4.0*cos(2*x)*cos(2*y)*cos(2*z)";
    std::string rho = "sin(x)"; //"-3*(cos(x)*cos(y)*cos(z)+cos(2*x)*cos(2*y)*cos(2*z))";
    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}};
    int varcount = 3;
    te_expr *rho_parse = te_compile(rho.c_str(), read_vars, varcount, &err);
    te_expr *phi_parse = te_compile(phi.c_str(), read_vars, varcount, &err);

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

    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);
    const int stencil_length = 3;
    std::array<std::array<amrex::Real, stencil_length>, GEMPIC_SPACEDIM> stencil_x;
    //stencil_x[0] = {-1.0/infra.dx[0], 1.0/infra.dx[0], 0.0};
    stencil_x[0] = {0.0, 1.0/infra.dx[0], -1.0/infra.dx[0]};
    stencil_x[1] = {0.0, 0.0, 0.0};
    stencil_x[2] = {0.0, 0.0, 0.0};

    /*
    // 1) Apply D
    mw_yee.template matrix_mult<stencil_length>(infra, stencil_x, &mw_yee.rho, &mw_yee.phi, amrex::IndexType(IntVect::TheNodeVector()));
    amrex::Vector<std::string> varnames = {"rho"};
    WriteSingleLevelPlotfile("rho", mw_yee.rho, varnames, infra.geom, 0, 0);
    varnames = {"drho"};
    WriteSingleLevelPlotfile("drho", mw_yee.phi, varnames, infra.geom, 0, 0);
*/
    // ----------------------------------------------------------------------------------


    amrex::Vector<std::string> varnames = {"rho"};
    WriteSingleLevelPlotfile("rho_init", mw_yee.rho, varnames, infra.geom, 0, 0);

    // Poisson solver
    //mlmg.solve({&mw_yee.phi}, {&mw_yee.rho}, 1e-12, 0.0);
    //mw_yee.phi.FillBoundary(infra.geom.periodicity());

    varnames = {"phi"};
    WriteSingleLevelPlotfile("phi", mw_yee.phi, varnames, infra.geom, 0, 0);


    mw_yee.template solve_poisson_CG<degree>(&mw_yee.rho, &mw_yee.phi, &kx, &ky, &kz, infra, 2, 5.e-11,300);

    // Poisson operator
    mw_yee.template poisson_operator<degree>(&mw_yee.phi, &mw_yee.rho, &kx, &ky, &kz, infra, 0, 1e-16, 100);

    //WriteSingleLevelPlotfile("phi_CG", mw_yee.phi, varnames, infra.geom, 0, 0);

    varnames = {"rho"};
    WriteSingleLevelPlotfile("rho_end", mw_yee.rho, varnames, infra.geom, 0, 0);



}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1>();
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1>();
    main_main<3, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_maxwell_yee.tmp.0", "test_maxwell_yee.output");
    amrex::Finalize();
}




/*------------------------------------------------------------------------------
 Test 3D Maxwell Yee Solver (finite differences) on periodic grid

  For the Maxwell-equations we use the solution
  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\
                          -2\cos(x_1+x_2+x_3 - \sqrt(3) t) \\
                            \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
  B(x,t) = \begin{pmatrix} \sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ 0 \\ -\sqrt{3}
\cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}

 For the Poisson equation we use:
 E(x,t) = \begin{pmatrix} -\sin(x)\cos(y)\cos(z)-0.5\sin(2x)cos(2y)cos(2z)\\
                          -\cos(x)\sin(y)\cos(z)-0.5\cos(2x)sin(2y)cos(2z)\\
                          -\cos(x)\cos(y)\sin(z)-0.5\cos(2x)cos(2y)sin(2z) \end{pmatrix}
------------------------------------------------------------------------------*/

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_assertion.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>

using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;

#define IHODGE_CG_ZERO 0
#define IHODGE_CG_E0 1
#define IHODGE_CG_E1 2
#define IHODGE_CG_E2 3
#define IHODGE_CG_B0 4
#define IHODGE_CG_B2 5

AMREX_GPU_HOST_DEVICE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z,
                                                      amrex::Real t, int funcSelect)
{
    switch (funcSelect)
    {
        case IHODGE_CG_E0:
            return std::cos(x);
            break;
        case IHODGE_CG_E1:
            return -2.0 * std::cos(x + y + z - std::sqrt(3.0) * t);
            break;
        case IHODGE_CG_E2:
            return std::cos(x + y + z - std::sqrt(3.0) * t);
            break;
        case IHODGE_CG_B0:
            return std::sqrt(3.) * std::cos(x + y + z - std::sqrt(3.0) * t);
            break;
        case IHODGE_CG_B2:
            return -std::sqrt(3.) * std::cos(x + y + z - std::sqrt(3.0) * t);
            break;
        case IHODGE_CG_ZERO:
            return 0.0;
            break;
    }
    return 0.0;
}
template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{  //------------------------------------------------------------------------------
    // Analytical solutions -- Maxwell
    /* amrex::GpuArray<std::string, vdim> fields_E;
     amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
     fields_E[0] = "cos(x+y+z-sqrt(3.0)*t)";
     fields_E[0] = "cos(x)";
     fields_E[1] = "-2*cos(x+y+z-sqrt(3.0)*t)";
     fields_E[2] = "cos(x+y+z-sqrt(3.0)*t)";
     fields_B[0] = "sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";
     fields_B[1] = "0.0";
     fields_B[2] = "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";
     */

    const int degree = 4;

    double twopi = 4 * asin(1.0);  // 2.0*3.14159265359;
    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    amrex::IntVect is_periodic = {AMREX_D_DECL(1, 1, 1)};
    // std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(32,32,32)};
    amrex::IntVect n_cell = {AMREX_D_DECL(32, 32, 32)};
    amrex::IntVect mx_grid = {AMREX_D_DECL(32, 32, 32)};

    amrex::Real boxLo[GEMPIC_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
    amrex::Real boxHi[GEMPIC_SPACEDIM] = {AMREX_D_DECL(twopi / 0.5, twopi / 0.5, twopi / 0.5)};
    amrex::RealBox real_box;
    real_box.setLo(boxLo);
    real_box.setHi(boxHi);

    CompDom::computational_domain infra;
    infra.initialize_computational_domain(n_cell, mx_grid, is_periodic, real_box);

    //------------------------------------------------------------------------------
    // Solve
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
    int Nghost = *(std::max_element(degs.begin(), degs.end()));
    
    const int Nsteps=5;
    const amrex::Real dt = 0.01;
    maxwell_yee<vdim> mw_yee(infra, dt, Nsteps, Nghost);

    amrex::GpuArray<int, int(vdim / 2.5) * 2 + 1> funcSelectB;
    funcSelectB[0] = IHODGE_CG_B0;
    funcSelectB[1] = IHODGE_CG_ZERO;
    funcSelectB[2] = IHODGE_CG_B2;
    mw_yee.template initB<degree>(infra, funcSelectB);

    amrex::GpuArray<int, vdim> funcSelectE;
    funcSelectE[0] = IHODGE_CG_E0;
    funcSelectE[1] = IHODGE_CG_E1;
    funcSelectE[2] = IHODGE_CG_E2;
    mw_yee.template initE<degree>(infra, funcSelectE);

    mw_yee.template hodge_full<degree>(infra, mw_yee.E_Array, mw_yee.HE_Array, true);

    for (int dim = 0; dim < vdim; dim++)
    {
        amrex::MultiFab k(convert(infra.grid, *mw_yee.E_Index[dim]), infra.distriMap, 1,
                          mw_yee.Nghost);
        
        k.setVal(-2.0, Nghost);
        k.FillBoundary(infra.geom.periodicity());

        (mw_yee.E_sol_Array[dim])->setVal(1.0, 0);
        (mw_yee.E_sol_Array[dim])->FillBoundary(infra.geom.periodicity());

        mw_yee.template solve_hodge_CG<degree>(*mw_yee.HE_Array[dim], *mw_yee.E_sol_Array[dim], k,
                                               infra, dim, 2, 1.e-16);
    }

    amrex::PrintToFile("test_ihodge_CG.output") << std::endl;
    // comparing ihodge(hoge(E)) to E
    bool passed = true;
    for (int dim = 0; dim < vdim; dim++)
    {
        (mw_yee.E_sol_Array[dim])->minus(*(mw_yee.E_Array[dim]), 0, 1, 0);
        amrex::Real err_norm = Utils::gempic_norm(*(mw_yee.E_sol_Array[dim]), infra, 2);
        amrex::PrintToFile("test_ihodge_CG.output")
            << "For component " << dim << " the error is: " << err_norm << std::endl;
        amrex::Real E_norm = Utils::gempic_norm(*(mw_yee.E_Array[dim]), infra, 2);
        gempic_assert_err(passed, E_norm, err_norm * err_norm);
    }
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
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
        std::rename("test_ihodge_CG.output.0", "test_ihodge_CG.output");
    amrex::Finalize();
}

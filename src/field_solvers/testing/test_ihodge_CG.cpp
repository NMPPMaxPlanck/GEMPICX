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
#include <GEMPIC_assertion.H>

using namespace std;
using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_e0(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = std::cos(x);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_e1(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = -2.0 * std::cos(x+y+z-std::sqrt(3.0)*t);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_e2(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = std::cos(x+y+z-std::sqrt(3.0)*t);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_b0(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = std::sqrt(3.)*std::cos(x+y+z-std::sqrt(3.0)*t);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_b2(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = -std::sqrt(3.)*std::cos(x+y+z-std::sqrt(3.0)*t);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real zero(amrex::Real , amrex::Real , amrex::Real , amrex::Real )
{
    amrex::Real val = 0.0;
    return val;
}

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
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
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic,
                    mx_grid, 0.01, {1.0}, {1.0}, 0.5);
    VlMa.set_computed_params();

    CompDom::computational_domain infra;
    VlMa.initialize_infrastructure(&infra);

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    mw_yee.template initB<degree>(funct_b0, zero, funct_b2, VlMa.k_gpu, infra);
    mw_yee.template initE<degree>(funct_e0, funct_e1, funct_e2, VlMa.k_gpu, infra);

    mw_yee.template hodge_full<degree>(infra, &(mw_yee.E_Array), &(mw_yee.HE_Array), true);

    for (int dim = 0; dim < vdim; dim++) {
        amrex::MultiFab k(convert(infra.grid, *mw_yee.E_Index[dim]),infra.distriMap,1,mw_yee.Nghost);
        k.setVal(-2.0, 0);
        k.FillBoundary(infra.geom.periodicity());

        (mw_yee.E_sol_Array[dim])->setVal(1.0, 0);
        (mw_yee.E_sol_Array[dim])->FillBoundary(infra.geom.periodicity());

        mw_yee.template solve_hodge_CG<degree>(&(*mw_yee.HE_Array[dim]), &(*mw_yee.E_sol_Array[dim]), &k, infra, dim, 2, 1.e-16);

    }

    amrex::AllPrintToFile("test_ihodge_CG_additional.tmp") << std::endl;
    // comparing ihodge(hoge(E)) to E
    bool passed = true;
    for (int dim = 0; dim < vdim; dim++) {
        (mw_yee.E_sol_Array[dim])->minus(*(mw_yee.E_Array[dim]), 0, 1, 0);
        amrex::Real err_norm = Utils::gempic_norm(&(*(mw_yee.E_sol_Array[dim])), infra, 2);
        amrex::AllPrintToFile("test_ihodge_CG_additional.tmp") << "For component " << dim << " the error is: " << err_norm << std::endl;
        amrex::Real E_norm = Utils::gempic_norm(&(*(mw_yee.E_Array[dim])), infra, 2);
        gempic_assert_err(&passed, E_norm, err_norm*err_norm);
    }
    amrex::AllPrintToFile("test_ihodge_CG.tmp") << std::endl;
    amrex::AllPrintToFile("test_ihodge_CG.tmp") << passed << std::endl;
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_ihodge_CG.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_ihodge_CG_additional.tmp.0");

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1>();
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1>();
    main_main<3, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_ihodge_CG.tmp.0", "test_ihodge_CG.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_ihodge_CG_additional.tmp.0", "test_ihodge_CG_additional.output");
    amrex::Finalize();
}




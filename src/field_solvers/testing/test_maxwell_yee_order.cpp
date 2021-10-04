/*------------------------------------------------------------------------------
 Test 3D Maxwell Solver (finite differences) for 4th order on periodic grid

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

//------------------------------------------------------------------------------
// Solutions
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

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func_phi(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = std::cos(x)-std::cos(x)*std::cos(y)*std::cos(z) - 1.0/4.0*std::cos(2*x)*std::cos(2*y)*std::cos(2*z);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func_rho(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = -3.0*(std::cos(x)*std::cos(y)*std::cos(z)+std::cos(2*x)*std::cos(2*y)*std::cos(2*z));
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func_e0(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = -sin(x)*cos(y)*cos(z)-0.5*sin(2*x)*cos(2*y)*cos(2*z);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func_e1(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = -cos(x)*sin(y)*cos(z)-0.5*cos(2*x)*sin(2*y)*cos(2*z);
    return val;
}

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func_e2(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = -cos(x)*cos(y)*sin(z)-0.5*cos(2*x)*cos(2*y)*sin(2*z);
    return val;
}


template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    //------------------------------------------------------------------------------
    // Analytical solutions
    amrex::GpuArray<std::string, vdim> fields_E;
    amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
    if (GEMPIC_SPACEDIM == 1 && vdim == 1) {
        fields_E[0] = "cos(x+y+z)";
        fields_B[0] = "0.0";
    } else if (GEMPIC_SPACEDIM == 1 && vdim == 2) {
        fields_E[0] = "cos(x)";
        fields_E[1] = "cos(x)*cos(t)";
        fields_B[0] = "sin(x)*sin(t)";
    } else if (GEMPIC_SPACEDIM == 2 && vdim == 2) {
        fields_E[0] = "cos(x)*sin(y)*sin(sqrt(2.0)*t)/sqrt(2.0)";
        fields_E[1] = "-sin(x)*cos(y)*sin(sqrt(2)*t)/sqrt(2)";
        fields_B[0] = "-cos(x[0])*cos(x[1])*cos(sqrt(2)*t)";
    } else if (GEMPIC_SPACEDIM == 2 && vdim == 3) {
        fields_E[0] = "cos(x+y-sqrt(2.0)*t";
        fields_E[1] = "-cos(x+y-sqrt(2.0)*t)";
        fields_E[2] = "-sqrt(2.0)*cos(x+y-sqrt(2.0)*t)";
        fields_B[0] = "-cos(x+y-sqrt(2.0)*t)";
        fields_B[1] = "cos(x+y-sqrt(2.0)*t)";
        fields_B[2] = "-sqrt(2)*cos(x+y-sqrt(2.0)*t)";
    } else if (GEMPIC_SPACEDIM == 3 && vdim == 3) {
        fields_E[0] = "cos(x+y+z-sqrt(3.0)*t)";
        fields_E[1] = "-2*cos(x+y+z-sqrt(3.0)*t)";
        fields_E[2] = "cos(x+y+z-sqrt(3.0)*t)";
        fields_B[0] = "sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";
        fields_B[1] = "0.0";
        fields_B[2] = "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)";

    }
    //------------------------------------------------------------------------------

    const int degree = 4;

    int bdim = int(vdim/2.5)*2+1;
    std::cout << "x DIM: " << GEMPIC_SPACEDIM << ", v&E DIM: " << vdim << ", B DIM: " << bdim << std::endl;


    //------------------------------------------------------------------------------
    amrex::GpuArray<Real,vdim+int(vdim/2.5)*2+1> E_B_error; //array for storing errors

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    amrex::IntVect is_periodic = {AMREX_D_DECL(1,1,1)};
    amrex::IntVect n_cell = {AMREX_D_DECL(64,64,64)};

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.Nghost++;
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic,
                    {32,32,32}, 0.01, {1.0}, {1.0}, 0.5);
    VlMa.dt = 0.01;
    VlMa.set_computed_params();

    CompDom::computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic, VlMa.real_box);

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);


    for (int i=0; i<vdim; i++) {
        (*(mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
        (*(mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
    }

    mw_yee.template initB<degree>(funct_b0, zero, funct_b2, infra);
    mw_yee.template initE<degree>(funct_e2, funct_e1, funct_e2, infra);

    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee.template computeError<degree>(funct_e2, funct_e1, funct_e2, funct_b0, zero, funct_b2, true, infra);
    AllPrintToFile("test_maxwell_yee_order_additional.tmp") << endl;
    AllPrintToFile("test_maxwell_yee_order_additional.tmp") << "Maxwell" << endl;
    AllPrintToFile("test_maxwell_yee_order_additional.tmp") << "step " << 0 << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }

    switch (bdim) {
    case 1:
        amrex::AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Bx error: " << E_B_error[vdim] << std::endl;
        break;
    case 3:
        amrex::AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
        break;
    }


    for (int n=1;n<=mw_yee.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee.template hodge_full<degree>(infra, &(mw_yee.B_Array), &(mw_yee.HB_Array), false);
        mw_yee.advance_E(infra, VlMa.dt, true, false, &(mw_yee.HB_Array), &(mw_yee.E_Array));
        mw_yee.template hodge_full<degree>(infra, &(mw_yee.E_Array), &(mw_yee.HE_Array), true);
        mw_yee.advance_B(infra, VlMa.dt, &(mw_yee.HE_Array), &(mw_yee.B_Array));
        mw_yee.advance_time();
        E_B_error = mw_yee.template computeError<degree>(funct_e2, funct_e1, funct_e2, funct_b0, zero, funct_b2, true, infra);

        AllPrintToFile("test_maxwell_yee_order_additional.tmp") << "step " << n << endl;
        switch (vdim) {
        case 1:
            AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << std::endl;
            break;
        case 2:
            AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
            break;
        case 3:
            AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
            break;
        }

        switch (bdim) {
        case 1:
            amrex::AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Bx error: " << E_B_error[vdim] << std::endl;
            break;
        case 3:
            amrex::AllPrintToFile("test_maxwell_yee_order_additional.tmp").SetPrecision(20) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
            break;
        }
    }
    bool passed = true;
    gempic_assert_err(&passed, 1, E_B_error[0]);
    gempic_assert_err(&passed, 1, E_B_error[1]);
    gempic_assert_err(&passed, 1, E_B_error[2]);
    gempic_assert_err(&passed, 1, E_B_error[3]);
    gempic_assert_err(&passed, 1, E_B_error[4]);
    gempic_assert_err(&passed, 1, E_B_error[5]);

    amrex::AllPrintToFile("test_maxwell_yee_order.tmp") << std::endl;
    amrex::AllPrintToFile("test_maxwell_yee_order.tmp") << passed << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_maxwell_yee_order.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_maxwell_yee_order_additional.tmp.0");

#if (GEMPIC_SPACEDIM == 1)
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_maxwell_yee_order.tmp.0", "test_maxwell_yee_order.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_maxwell_yee_order_additional.tmp.0", "test_maxwell_yee_order_additional.output");
    amrex::Finalize();
}




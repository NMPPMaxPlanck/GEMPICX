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

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_assertion.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_parameters.H>

using namespace std;
using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;

#define MAXWELL_YEE_ZERO 0
#define MAXWELL_YEE_E1   1
#define MAXWELL_YEE_E2   2
#define MAXWELL_YEE_B0   3
#define MAXWELL_YEE_B2   4
#define MAXWELL_YEE_PHI  5
#define MAXWELL_YEE_RHO  6
#define MAXWELL_YEE_E0_1 7
#define MAXWELL_YEE_E1_1 8
#define MAXWELL_YEE_E2_1 9

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t, int funcSelect)
{
  switch(funcSelect){
  case MAXWELL_YEE_E1 :
    return -2.0 * std::cos(x+y+z-std::sqrt(3.0)*t);
  case MAXWELL_YEE_E2 :
    return std::cos(x+y+z-std::sqrt(3.0)*t);
  case MAXWELL_YEE_B0 :
    return std::sqrt(3.)*std::cos(x+y+z-std::sqrt(3.0)*t);
  case MAXWELL_YEE_B2 :
    return -std::sqrt(3.)*std::cos(x+y+z-std::sqrt(3.0)*t);
  case MAXWELL_YEE_PHI :
    return std::cos(x)-std::cos(x)*std::cos(y)*std::cos(z) - 1.0/4.0*std::cos(2*x)*std::cos(2*y)*std::cos(2*z);
  case MAXWELL_YEE_RHO :
    return -3.0*(std::cos(x)*std::cos(y)*std::cos(z)+std::cos(2*x)*std::cos(2*y)*std::cos(2*z));
  case MAXWELL_YEE_E0_1:
    return -sin(x)*cos(y)*cos(z)-0.5*sin(2*x)*cos(2*y)*cos(2*z); 
  case MAXWELL_YEE_E1_1 :
    return -cos(x)*sin(y)*cos(z)-0.5*cos(2*x)*sin(2*y)*cos(2*z);
  case MAXWELL_YEE_E2_1 :
    return -cos(x)*cos(y)*sin(z)-0.5*cos(2*x)*cos(2*y)*sin(2*z);
  case MAXWELL_YEE_ZERO :
    return 0.0;
  }
  return 0.0;
}

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    bool passed = true;
    //------------------------------------------------------------------------------
    // Analytical solutions -- Maxwell
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
    // Analytical solutions -- Poisson
    amrex::GpuArray<std::string, vdim> fields_EP;
    if (GEMPIC_SPACEDIM == 1 && vdim == 1) {
        fields_EP[0] = "-sin(x)-0.5*sin(2*x)";
    } else if (GEMPIC_SPACEDIM == 1 && vdim == 2) {
        fields_EP[0] = "-sin(x)-0.5*sin(2*x)";
        fields_EP[1] = "0.0";
    } else if (GEMPIC_SPACEDIM == 2 && vdim == 2) {
        fields_EP[0] = "-sin(x)*cos(y)-0.5*sin(2*x)*cos(2*y)";
        fields_EP[1] = "-cos(x)*sin(y)-0.5*cos(2*x)*sin(2*y)";
    } else if (GEMPIC_SPACEDIM == 2 && vdim == 3) {
        fields_EP[0] = "-sin(x)*cos(y)-0.5*sin(2*x)*cos(2*y)";
        fields_EP[1] = "-cos(x)*sin(y)-0.5*cos(2*x)*sin(2*y)";
        fields_EP[2] = "0.0";
    } else if (GEMPIC_SPACEDIM == 3 && vdim == 3) {
        fields_EP[0] = "-sin(x)*cos(y)*cos(z)-0.5*sin(2*x)*cos(2*y)*cos(2*z)";
        fields_EP[1] = "-cos(x)*sin(y)*cos(z)-0.5*cos(2*x)*sin(2*y)*cos(2*z)";
        fields_EP[2] = "-cos(x)*cos(y)*sin(z)-0.5*cos(2*x)*cos(2*y)*sin(2*z)";
    }
    //------------------------------------------------------------------------------

    const int degree = 2;
    int bdim = int(vdim/2.5)*2+1;
    std::cout << "x DIM: " << GEMPIC_SPACEDIM << ", v&E DIM: " << vdim << ", B DIM: " << bdim << std::endl;

    //------------------------------------------------------------------------------
    amrex::GpuArray<Real,vdim+int(vdim/2.5)*2+1> E_B_error; //array for storing errors

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    amrex::IntVect is_periodic = {AMREX_D_DECL(1,1,1)};
    amrex::IntVect n_cell = {AMREX_D_DECL(32,32,32)};


    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic,
    {32, 32, 32}, 0.01, {1.0}, {1.0}, 0.5);
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

    amrex::GpuArray<int, int(vdim/2.5)*2+1> funcSelectB;
    funcSelectB[0] = MAXWELL_YEE_B0;
    funcSelectB[1] = MAXWELL_YEE_ZERO;
    funcSelectB[2] = MAXWELL_YEE_B2;
    mw_yee.template initB<degree>(infra, funcSelectB);
    amrex::GpuArray<int, vdim> funcSelectE;
    funcSelectE[0] = MAXWELL_YEE_E2;
    funcSelectE[1] = MAXWELL_YEE_E1;
    funcSelectE[2] = MAXWELL_YEE_E2;
    mw_yee.template initE<degree>(infra, funcSelectE);


    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee.template computeError<degree>(true, infra, funcSelectE, funcSelectB);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[0]), infra, 2), E_B_error[0]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[1]), infra, 2), E_B_error[1]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[2]), infra, 2), E_B_error[2]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[0]), infra, 2), E_B_error[3]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[1]), infra, 2), E_B_error[4]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[2]), infra, 2), E_B_error[5]);

    AllPrintToFile("test_maxwell_yee.tmp") << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp") << "Maxwell" << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp") << "step " << 0 << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Bx error: " << E_B_error[3] << " |By error: " << E_B_error[4] << " |Bz error: " << E_B_error[5] << std::endl;


    for (int n=1;n<=mw_yee.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee.template hodge_full<degree>(infra, &(mw_yee.B_Array), &(mw_yee.HB_Array), false);
        mw_yee.advance_E(infra, VlMa.dt, true, true, &(mw_yee.HB_Array), &(mw_yee.E_Array));
        mw_yee.template hodge_full<degree>(infra, &(mw_yee.E_Array), &(mw_yee.HE_Array), true);
        mw_yee.advance_B(infra, VlMa.dt, &(mw_yee.HE_Array), &(mw_yee.B_Array));
        mw_yee.advance_time();
        E_B_error = mw_yee.template computeError<degree>(true, infra, funcSelectE, funcSelectB);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[0]), infra, 2), E_B_error[0]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[1]), infra, 2), E_B_error[1]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[2]), infra, 2), E_B_error[2]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[0]), infra, 2), E_B_error[3]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[1]), infra, 2), E_B_error[4]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[2]), infra, 2), E_B_error[5]);


        AllPrintToFile("test_maxwell_yee_additional.tmp") << "step " << n << endl;
        AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Bx error: " << E_B_error[3] << " |By error: " << E_B_error[4] << " |Bz error: " << E_B_error[5] << std::endl;


    }

    //------------------------------------------------------------------------------
    // Second maxwell test
    maxwell_yee<vdim> mw_yee_2(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    for (int i=0; i<vdim; i++) {
        (*(mw_yee_2).J_Array[i]).setVal(0.0, 0); // value and component
        (*(mw_yee_2).J_Array[i]).FillBoundary(infra.geom.periodicity());
    }

    funcSelectB[0] = MAXWELL_YEE_B0;
    funcSelectB[1] = MAXWELL_YEE_ZERO;
    funcSelectB[2] = MAXWELL_YEE_B2;
    mw_yee_2.template initB<degree>(infra, funcSelectB);
    funcSelectE[0] = MAXWELL_YEE_E2;
    funcSelectE[1] = MAXWELL_YEE_E1;
    funcSelectE[2] = MAXWELL_YEE_E2;
    mw_yee_2.template initE<degree>(infra, funcSelectE);

    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee_2.template computeError<degree>(true, infra, funcSelectE, funcSelectB);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[0]), infra, 2), E_B_error[0]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[1]), infra, 2), E_B_error[1]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[2]), infra, 2), E_B_error[2]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[0]), infra, 2), E_B_error[3]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[1]), infra, 2), E_B_error[4]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[2]), infra, 2), E_B_error[5]);

    AllPrintToFile("test_maxwell_yee_additional.tmp") << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp") << "Maxwell" << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp") << "step " << 0 << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Bx error: " << E_B_error[3] << " |By error: " << E_B_error[4] << " |Bz error: " << E_B_error[5] << std::endl;


    for (int n=1;n<=mw_yee_2.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee_2.advance_time();
        mw_yee_2.template hodge_full<degree>(infra, &(mw_yee_2.B_Array), &(mw_yee_2.HB_Array), false);
        mw_yee_2.advance_E(infra, mw_yee_2.dt, true, false, &(mw_yee_2.HB_Array), &(mw_yee_2.E_Array));
        mw_yee_2.advance_E(infra, mw_yee_2.dt, false, true, &(mw_yee_2.HB_Array), &(mw_yee_2.E_Array));
        mw_yee_2.template hodge_full<degree>(infra, &(mw_yee_2.E_Array), &(mw_yee_2.HE_Array), true);
        mw_yee_2.advance_B(infra, mw_yee_2.dt, &(mw_yee_2.HE_Array), &(mw_yee_2.B_Array));
        E_B_error = mw_yee_2.template computeError<degree>(true, infra, funcSelectE, funcSelectB);

        gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[0]), infra, 2), E_B_error[0]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[1]), infra, 2), E_B_error[1]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[2]), infra, 2), E_B_error[2]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[0]), infra, 2), E_B_error[3]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[1]), infra, 2), E_B_error[4]);
        gempic_assert_err(passed, gempic_norm(&(*mw_yee.B_Array[2]), infra, 2), E_B_error[5]);


        AllPrintToFile("test_maxwell_yee_additional.tmp") << "step " << n << endl;
        AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        amrex::AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Bx error: " << E_B_error[3] << " |By error: " << E_B_error[4] << " |Bz error: " << E_B_error[5] << std::endl;

    }

    //------------------------------------------------------------------------------
    // Poisson
  
    amrex::GpuArray<int,2> funcSelectRhoPhi;
    funcSelectRhoPhi[0] = MAXWELL_YEE_RHO;
    funcSelectRhoPhi[1] = MAXWELL_YEE_PHI;
    mw_yee.template init_rho_phi<degree>(infra, funcSelectRhoPhi);
    mw_yee.solve_poisson(infra);
    funcSelectE[0]=MAXWELL_YEE_E0_1;
    funcSelectE[1]=MAXWELL_YEE_E1_1;
    funcSelectE[2]=MAXWELL_YEE_E2_1;
    funcSelectB[0]=MAXWELL_YEE_B0;
    funcSelectB[1]=MAXWELL_YEE_ZERO;
    funcSelectB[2]=MAXWELL_YEE_B2;
    E_B_error = mw_yee.template computeError<degree>(false, infra, funcSelectE, funcSelectB);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[0]), infra, 2), E_B_error[0]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[1]), infra, 2), E_B_error[1]);
    gempic_assert_err(passed, gempic_norm(&(*mw_yee.E_Array[2]), infra, 2), E_B_error[2]);

    AllPrintToFile("test_maxwell_yee_additional.tmp") << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp") << "Poisson" << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;



    //------------------------------------------------------------------------------
    // Rho from E

    AllPrintToFile("test_maxwell_yee_additional.tmp") << endl;
    AllPrintToFile("test_maxwell_yee_additional.tmp") << "rho_from_E" << endl;

    mw_yee.rho_from_E(infra); // fills rho_gauss_law
    mw_yee.rho_gauss_law.minus(mw_yee.rho, 0, 1, 0);
    amrex::Real rho_norm = Utils::gempic_norm(&(mw_yee.rho_gauss_law), infra, 2);
    gempic_assert_err(passed, gempic_norm(&(mw_yee.rho), infra, 2), rho_norm*rho_norm);

    AllPrintToFile("test_maxwell_yee_additional.tmp").SetPrecision(5) << "rho Error: " << rho_norm*rho_norm << std::endl;

    AllPrintToFile("test_maxwell_yee.tmp") << passed << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_maxwell_yee.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_maxwell_yee_additional.tmp.0");


#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1, false>();
    main_main<2, 1, 1, 1, 1, false>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1, false>();
    main_main<3, 1, 1, 1, 1, false>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_maxwell_yee.tmp.0", "test_maxwell_yee.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_maxwell_yee_additional.tmp.0", "test_maxwell_yee_additional.output");

    amrex::Finalize();
}




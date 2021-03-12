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
    std::array<std::string, vdim> fields_EP;
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
    array<Real,vdim+int(vdim/2.5)*2+1> E_B_error; //array for storing errors

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(128,128,128)};

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
    if (vdim > 1) {
        VM[1].push_back(0.0);
        VD[1].push_back(1.0);
        VW[1].push_back(1.0);
    }
    if (vdim > 2) {
        VM[2].push_back(0.0);
        VD[2].push_back(1.0);
        VW[2].push_back(1.0);
    }

    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(degx, degy, degz)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic,
                    {32, 32, 32}, 0.01, {1.0}, {1.0}, 0.5);
    VlMa.set_computed_params();

    Infra::infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    //------------------------------------------------------------------------------
    // Solve
    maxwell_yee<vdim> mw_yee(VlMa, infra);


    for (int i=0; i<vdim; i++) {
        (*(mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
        (*(mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
    }

    mw_yee.template init_E_B<degree>(fields_E, fields_B, VlMa.k, infra);

    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee.template computeError<degree>(fields_E, fields_B, VlMa.k, true, infra);
    AllPrintToFile("test_maxwell_yee.tmp") << endl;
    AllPrintToFile("test_maxwell_yee.tmp") << "Maxwell" << endl;
    AllPrintToFile("test_maxwell_yee.tmp") << "step " << 0 << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }

    switch (bdim) {
    case 1:
        amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
        break;
    case 3:
        amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
        break;
    }

    for (int n=1;n<=mw_yee.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee.template hodge_full<degree>(infra, &(mw_yee.B_Array), &(mw_yee.HB_Array), false);
        mw_yee.advance_E(infra, VlMa.dt, true, true, &(mw_yee.HB_Array), &(mw_yee.E_Array));
        mw_yee.template hodge_full<degree>(infra, &(mw_yee.E_Array), &(mw_yee.HE_Array), true);
        mw_yee.advance_B(infra, VlMa.dt, &(mw_yee.HE_Array), &(mw_yee.B_Array));
        mw_yee.advance_time();
        E_B_error = mw_yee.template computeError<degree>(fields_E, fields_B, VlMa.k, true, infra);

        AllPrintToFile("test_maxwell_yee.tmp") << "step " << n << endl;
        switch (vdim) {
        case 1:
            AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
            break;
        case 2:
            AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
            break;
        case 3:
            AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
            break;
        }
        switch (bdim) {
        case 1:
            amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
            break;
        case 3:
            amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
            break;
        }

    }

    //------------------------------------------------------------------------------
    // Second maxwell test
    maxwell_yee<vdim> mw_yee_2(VlMa, infra);

    for (int i=0; i<vdim; i++) {
        (*(mw_yee_2).J_Array[i]).setVal(0.0, 0); // value and component
        (*(mw_yee_2).J_Array[i]).FillBoundary(infra.geom.periodicity());
    }

    mw_yee_2.template init_E_B<2>(fields_E, fields_B, VlMa.k, infra);

    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee_2.template computeError<degree>(fields_E, fields_B, VlMa.k, true, infra);
    AllPrintToFile("test_maxwell_yee.tmp") << endl;
    AllPrintToFile("test_maxwell_yee.tmp") << "Maxwell" << endl;
    AllPrintToFile("test_maxwell_yee.tmp") << "step " << 0 << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }
    switch (bdim) {
    case 1:
        amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
        break;
    case 3:
        amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
        break;
    }

    for (int n=1;n<=mw_yee_2.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee_2.advance_time();
        mw_yee_2.template hodge_full<degree>(infra, &(mw_yee_2.B_Array), &(mw_yee_2.HB_Array), false);
        mw_yee_2.advance_E(infra, mw_yee_2.dt, true, false, &(mw_yee_2.HB_Array), &(mw_yee_2.E_Array));
        mw_yee_2.advance_E(infra, mw_yee_2.dt, false, true, &(mw_yee_2.HB_Array), &(mw_yee_2.E_Array));
        mw_yee_2.template hodge_full<degree>(infra, &(mw_yee_2.E_Array), &(mw_yee_2.HE_Array), true);
        mw_yee_2.advance_B(infra, mw_yee_2.dt, &(mw_yee_2.HE_Array), &(mw_yee_2.B_Array));
        E_B_error = mw_yee_2.template computeError<degree>(fields_E, fields_B, VlMa.k, true, infra);

        AllPrintToFile("test_maxwell_yee.tmp") << "step " << n << endl;
        switch (vdim) {
        case 1:
            AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
            break;
        case 2:
            AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
            break;
        case 3:
            AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
            break;
        }
        switch (bdim) {
        case 1:
            amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << std::endl;
            break;
        case 3:
            amrex::AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim+1] << " |Bz error: " << E_B_error[vdim+2] << std::endl;
            break;
        }
    }

    //------------------------------------------------------------------------------
    // Poisson

#if (GEMPIC_SPACEDIM == 1)
    std::string phi = "-cos(x) - 1.0/4.0*cos(2*x)";
    std::string rho = "-cos(x) - cos(2*x)";
#elif (GEMPIC_SPACEDIM == 2)
    std::string phi = "cos(x)*cos(y) + 1.0/4.0*cos(2*x)*cos(2*y)";
    std::string rho = "-2*(cos(x)*cos(y)+cos(2*x)*cos(2*y))";
#else
    std::string phi = "-cos(x)*cos(y)*cos(z) - 1.0/4.0*cos(2*x)*cos(2*y)*cos(2*z)";
    std::string rho = "-3*(cos(x)*cos(y)*cos(z)+cos(2*x)*cos(2*y)*cos(2*z))";
#endif

    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}};
    int varcount = 3;

    te_expr *rho_parse = te_compile(rho.c_str(), read_vars, varcount, &err);
    te_expr *phi_parse = te_compile(phi.c_str(), read_vars, varcount, &err);

    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);
    mw_yee.solve_poisson(infra);
    E_B_error = mw_yee.template computeError<degree>(fields_EP, fields_B, VlMa.k, false, infra);

    AllPrintToFile("test_maxwell_yee.tmp") << endl;
    AllPrintToFile("test_maxwell_yee.tmp") << "Poisson" << endl;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << " |Ez error: " << E_B_error[2] << std::endl;
        break;
    }


    //------------------------------------------------------------------------------
    // Rho from E

    AllPrintToFile("test_maxwell_yee.tmp") << endl;
    AllPrintToFile("test_maxwell_yee.tmp") << "rho_from_E" << endl;
    mw_yee.rho_from_E(infra); // fills rho_gauss_law
    mw_yee.rho_gauss_law.minus(mw_yee.rho, 0, 1, 0);
    AllPrintToFile("test_maxwell_yee.tmp").SetPrecision(5) << "rho Error: " << Utils::gempic_norm(&(mw_yee.rho_gauss_law), infra, 2) << std::endl;

    //ofs.close();
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_maxwell_yee.tmp.0");

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




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

//------------------------------------------------------------------------------
// Solutions

#if (GEMPIC_SPACEDIM == 1)
template<int vdim>
double E_x(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 1)
        return(cos(std::accumulate(x.begin(), x.end(), 0.)));
    else if (vdim == 2)
        return(cos(x[0]));
    else if (vdim == 3)
        return(0.);
}
template<int vdim>
double E_y(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 2)
        return(cos(x[0])*cos(t));
    else
        return(0.);
}
template<int vdim>
double B_x(std::array<double,GEMPIC_SPACEDIM> x, double t)
{
    if (vdim == 2)
        return(sin(x[0])*sin(t));
    else
        return(0.);
}
#endif

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{  
#if (GEMPIC_SPACEDIM == 1)
    int bdim = int(vdim/2.5)*2+1;
    std::cout << "x DIM: " << GEMPIC_SPACEDIM << ", v&E DIM: " << vdim << ", B DIM: " << bdim << std::endl;

    // make pointer-array for functions
    double (*fields[vdim+bdim]) (std::array<double,GEMPIC_SPACEDIM> x, double t);
    fields[0] = E_x<vdim>;
    if (vdim > 1){
        fields[1] = E_y<vdim>;
    }

    fields[vdim] = B_x<vdim>;

    //------------------------------------------------------------------------------
    array<Real,vdim+int(vdim/2.5)*2+1> E_B_error; //array for storing errors

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(64,64,64)};

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
    VlMa.Nghost++;
    VlMa.set_params("maxwell_yee_ctest", n_cell, {1}, 5, 10, 10, 10, is_periodic,
                    32, 0.01, {1.0}, {1.0}, 0.5);
    VlMa.dt = 10;
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

    mw_yee.init_E_B(fields, infra);
    mw_yee.init_HE_HB(fields, infra);
    mw_yee.hodge01_B(infra, 0, 2);
    mw_yee.hodge01_E(infra, 1, 2);

    std::cout <<  "step: " << 0 << std::endl;
    E_B_error = mw_yee.computeError(fields, true, infra);
    AllPrintToFile("test_maxwell_yee_order.tmp") << endl;
    AllPrintToFile("test_maxwell_yee_order.tmp") << "Maxwell" << endl;
    AllPrintToFile("test_maxwell_yee_order.tmp") << "step " << 0 << endl;
    AllPrintToFile("test_maxwell_yee_order.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
    AllPrintToFile("test_maxwell_yee_order.tmp").SetPrecision(20) << "Bx error: " << E_B_error[vdim] << std::endl;


    for (int n=1;n<=mw_yee.nsteps;n++){
        std::cout << "step: " << n << std::endl;
        mw_yee.hodge10_B(infra, 0, 2);
        mw_yee.advance_E_from_B_hodge(infra, VlMa.dt);
        mw_yee.hodge10_E(infra, 1, 2);
        mw_yee.advance_B_hodge(infra, VlMa.dt);
        E_B_error = mw_yee.computeError(fields, true, infra);

        AllPrintToFile("test_maxwell_yee_order.tmp") << "step " << n << endl;
        AllPrintToFile("test_maxwell_yee_order.tmp").SetPrecision(20) << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1] << std::endl;
        AllPrintToFile("test_maxwell_yee_order.tmp").SetPrecision(20) << "Bx error: " << E_B_error[vdim] << std::endl;
    }
#endif

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
#elif (GEMPIC_SPACEDIM == 3)
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_maxwell_yee_order.tmp.0", "test_maxwell_yee_order.output");
    amrex::Finalize();
}





/*------------------------------------------------------------------------------
 Test Gempic Norm

  Constant function
  g(x,y,z) = C
  norm(g,0) = norm(g,1) = norm(g,2) = C

  linear function
  f(x,y,z) = ax + by + cz
  norm(f,0) = aLx + bLy + cLz
  1D
  norm(f,1) = 1/2*Lx*a
  norm(f,2) = 1/3*Lx^2a^2
  2D
  norm(f,1) = 1/2*(a*Lx + b*Ly)
  norm(f,2) = 1/6*(2*a^2*Lx^2 + 3*a*b*Lx*Ly + 2*b^2*Ly^2)
  3D
  norm(f,1) = 1/2*(a*Lx + b*Ly + c*Lz)
  norm(f,2) = 1/6*(2*a^2*Lx^2 + 3*a*c*Lx*Lz + 3*b*c*Ly*Lz + 3*a*b*Lx*Ly + 2*b^2*Ly^2 + 2*c^2*Lz^2)
------------------------------------------------------------------------------*/

#include <cmath>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_assertion.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_vlasov_maxwell.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Utils;

//------------------------------------------------------------------------------
// function
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real func(amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> x, amrex::Real a, amrex::Real b, amrex::Real c){return(a*x[0]
        #if (GEMPIC_SPACEDIM > 0)
            +b*x[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
            +c*x[2]
        #endif
            );}

template<int vdim, int numspec>
void main_main ()
{
    amrex::Real C = 2.;
    amrex::Real a = 1.;
    amrex::Real b = 1.;
    amrex::Real c = 1.;
    //------------------------------------------------------------------------------
    // Initialize Infrastructure

    amrex::IntVect is_periodic = {AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect n_cell = {AMREX_D_DECL(128, 128, 128)};

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(1, 1, 1);
    VlMa.set_params("norm_ctest", n_cell, {1}, 1, 3, 3, 3, is_periodic, {64,64,64});
    VlMa.set_computed_params();

    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic, VlMa.real_box);

    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    // Constant case
    mw_yee.rho.setVal(C, 0);
    AllPrintToFile("test_gempic_norm_additional.tmp") << endl << "Constant case: " << endl;
    AllPrintToFile("test_gempic_norm_additional.tmp") << "0-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - C) << endl;
    AllPrintToFile("test_gempic_norm_additional.tmp").SetPrecision(5) << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - C) << endl;
    AllPrintToFile("test_gempic_norm_additional.tmp").SetPrecision(5) << "2-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 2) - C) << endl;
    AllPrintToFile("test_gempic_norm_additional.tmp") << endl;

    bool passed = true;
    gempic_assert(&passed, C, gempic_norm(&mw_yee.rho, infra, 0));
    gempic_assert(&passed, C, gempic_norm(&mw_yee.rho, infra, 1));
    gempic_assert(&passed, C, gempic_norm(&mw_yee.rho, infra, 2));


    // Linear case
    for ( amrex::MFIter mfi(mw_yee.rho); mfi.isValid(); ++mfi ){

        const amrex::Box& bx = mfi.validbox();

	amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo = infra.plo;
	amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx = infra.dx;
        amrex::Array4<amrex::Real> const& rho_arr = mw_yee.rho[mfi].array();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> x;
            x[0] = plo[0] + ((amrex::Real)i)*dx[0];
            x[1] = plo[1] + ((amrex::Real)j)*dx[1];
            x[2] = plo[2] + ((amrex::Real)k)*dx[2];
            rho_arr(i,j,k) = func(x,a,b,c);
        });
    }
    AllPrintToFile("test_gempic_norm_additional.tmp") << "Linear case: " << endl;
#if(GEMPIC_SPACEDIM == 1)
    AllPrintToFile("test_gempic_norm.tmp").SetPrecision(5) << "0-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - a*infra.Length[0]) << endl;
    double Lx = infra.Length[0];
    double norm1 = 1./2.*Lx*a;
    double norm2 = sqrt(1./3.*pow(Lx,2.)*pow(a,2.));
    AllPrintToFile("test_gempic_norm.tmp") << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - norm1) << endl;
    AllPrintToFile("test_gempic_norm.tmp").SetPrecision(3) << "2-norm error: " << floor(fabs(gempic_norm(&mw_yee.rho, infra, 2) - norm2)*1000) << endl;
#endif
#if (GEMPIC_SPACEDIM == 2)
    AllPrintToFile("test_gempic_norm.tmp").SetPrecision(5) << "0-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - (a*infra.Length[0] + b*infra.Length[1])) << endl;
    double Lx = infra.Length[0];
    double Ly = infra.Length[1];
    double norm1 = 1./2.*(a*Lx + b*Ly);
    double norm2 = sqrt(1./6.*(2.*pow(a,2.)*pow(Lx,2.) + 3.*a*b*Lx*Ly + 2.*pow(b,2.)*pow(Ly,2.)));
    AllPrintToFile("test_gempic_norm.tmp") << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - norm1) << endl;
    AllPrintToFile("test_gempic_norm.tmp").SetPrecision(1) << "2-norm error: " << floor(fabs(gempic_norm(&mw_yee.rho, infra, 2) - norm2)*1000) << endl;
#endif
#if (GEMPIC_SPACEDIM == 3)
    AllPrintToFile("test_gempic_norm_additional.tmp").SetPrecision(5) << "0-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - (a*infra.Length[0] + b*infra.Length[1] + c*infra.Length[2])) << endl;
    amrex::Real Lx = infra.Length[0];
    amrex::Real Ly = infra.Length[1];
    amrex::Real Lz = infra.Length[2];
    amrex::Real norm0 = a*Lx + b*Ly + c*Lz;
    amrex::Real norm1 = 1./2.*(a*Lx + b*Ly + c*Lz);
    amrex::Real norm2 = sqrt(1./6.*(2.*pow(a,2.)*pow(Lx,2.) + 3.*a*c*Lx*Lz + 3.*b*c*Ly*Lz + 3.*a*b*Lx*Ly + 2.*pow(b,2.)*pow(Ly,2.) + 2.*pow(c,2.)*pow(Lz,2.)));
    AllPrintToFile("test_gempic_norm_additional.tmp") << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - norm1) << endl;
    AllPrintToFile("test_gempic_norm_additional.tmp").SetPrecision(1) << "2-norm error: " << floor(fabs(gempic_norm(&mw_yee.rho, infra, 2) - norm2)*1000) << endl;

    gempic_assert(&passed, norm0, gempic_norm(&mw_yee.rho, infra, 0));
    gempic_assert(&passed, norm1, gempic_norm(&mw_yee.rho, infra, 1));
    gempic_assert(&passed, norm2, gempic_norm(&mw_yee.rho, infra, 2));

#endif
    AllPrintToFile("test_gempic_norm.tmp") << endl;
    AllPrintToFile("test_gempic_norm.tmp") << passed << endl;


}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_gempic_norm.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_gempic_norm_additional.tmp.0");

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1>();
    main_main<2, 1>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1>();
    main_main<3, 1>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_gempic_norm.tmp.0", "test_gempic_norm.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_gempic_norm_additional.tmp.0", "test_gempic_norm_additional.output");

    amrex::Finalize();
}




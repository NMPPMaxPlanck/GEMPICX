
//------------------------------------------------------------------------------
// Test Gempic Norm
//
//  Constant function
//  g(x,y,z) = C
//  norm(g,0) = norm(g,1) = norm(g,2) = C
//
//  linear function
//  f(x,y,z) = ax + by + cz
//  norm(f,0) = aLx + bLy + cLz
//  1D
//  norm(f,1) = 1/2*Lx*a
//  norm(f,2) = 1/3*Lx^2a^2
//  2D
//  norm(f,1) = 1/2*(a*Lx + b*Ly)
//  norm(f,2) = 1/6*(2*a^2*Lx^2 + 3*a*b*Lx*Ly + 2*b^2*Ly^2)
//  3D
//  norm(f,1) = 1/2*(a*Lx + b*Ly + c*Lz)
//  norm(f,2) = 1/6*(2*a^2*Lx^2 + 3*a*c*Lx*Lz + 3*b*c*Ly*Lz + 3*a*b*Lx*Ly + 2*b^2*Ly^2 + 2*c^2*Lz^2)
//------------------------------------------------------------------------------

#include <cmath>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Utils;

//------------------------------------------------------------------------------
// function
double f(std::array<double,GEMPIC_SPACEDIM> x, double a, double b, double c){return(a*x[0]
        #if (GEMPIC_SPACEDIM > 0)
            +b*x[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
            +c*x[2]
        #endif
            );}

void main_main ()
{
    double C = 2.;
    double a = 1.;
    double b = 1.;
    double c = 1.;
    //------------------------------------------------------------------------------
    // Initialize Infrastructure

    initializer init;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
    amrex::IntVect n_cell(AMREX_D_DECL(512,512,512));

    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VM{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VD{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
#if (GEMPIC_VDIM > 1)
    VM[1].push_back(0.0);
    VD[1].push_back(1.0);
    VW[1].push_back(1.0);
#endif
#if (GEMPIC_VDIM > 2)
    VM[2].push_back(0.0);
    VD[2].push_back(1.0);
    VW[2].push_back(1.0);
#endif

    init.initialize_from_parameters(n_cell,64,is_periodic,1,0.01,5,{1.0},{1.0},1,1,VM,VD,VW);
    infrastructure infra(init);

    maxwell_yee mw_yee(init, infra, init.Nghost);
    std::ofstream ofs("test_gempic_norm.output", std::ofstream::out);

    // Constant case
    mw_yee.rho.setVal(C, 0);
    Print(ofs) << endl << "Constant case: " << endl;
    Print(ofs).SetPrecision(5) << "0-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - C) << endl;
    Print(ofs).SetPrecision(5) << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - C) << endl;
    Print(ofs).SetPrecision(5) << "2-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - C) << endl;
    Print(ofs) << endl;

    // Linear case
    std::array<double,GEMPIC_SPACEDIM> x;
    for ( amrex::MFIter mfi(mw_yee.rho); mfi.isValid(); ++mfi ){

        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
        for(int k=lo[2]; k<=hi[2]; k++){
            x[2] = infra.geom.ProbLo()[2] + ((double)k)*infra.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
            for(int j=lo[1]; j<=hi[1]; j++){
                x[1] = infra.geom.ProbLo()[1] + ((double)j)*infra.dx[1];
#endif
                for(int l=lo[0]; l<=hi[0]; l++){
                    x[0] = infra.geom.ProbLo()[0] + ((double)l)*infra.dx[0];
                    // the box for these values:
                    amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect::TheNodeVector());
                    (mw_yee.rho)[mfi].setVal(f(x,a,b,c), cc, 0, 1);
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }
    Print(ofs) << "Linear case: " << endl;
    Print(ofs).SetPrecision(5) << "0-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 0) - (a*infra.Length[0] + b*infra.Length[1] + c*infra.Length[2])) << endl;
#if(GEMPIC_SPACEDIM == 1)
    double Lx = infra.Length[0];
    double norm1 = 1/2*Lx*a;
    double norm2 = 1/3*pow(Lx,2)*pow(a,2);
    Print(ofs).SetPrecision(5) << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - norm1) << endl;
    Print(ofs).SetPrecision(5) << "2-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 2) - norm2) << endl;
#endif
#if (GEMPIC_SPACEDIM == 2)
    double Lx = infra.Length[0];
    double Ly = infra.Length[1];
    double norm1 = 1/2*(a*Lx + b*Ly);
    double norm2 = 1/6*(2*pow(a,2)*pow(Lx,2) + 3*a*b*Lx*Ly + 2*pow(b,2)*pow(Ly,2));
    Print(ofs).SetPrecision(5) << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - norm1) << endl;
    Print(ofs).SetPrecision(5) << "2-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 2) - norm2) << endl;
#endif
#if (GEMPIC_SPACEDIM == 3)
    double Lx = infra.Length[0];
    double Ly = infra.Length[1];
    double Lz = infra.Length[2];
    double norm1 = 1./2.*(a*Lx + b*Ly + c*Lz);
    double norm2 = 1./6.*(2.*pow(a,2.)*pow(Lx,2.) + 3.*a*c*Lx*Lz + 3.*b*c*Ly*Lz + 3.*a*b*Lx*Ly + 2.*pow(b,2.)*pow(Ly,2.) + 2.*pow(c,2.)*pow(Lz,2.));
    Print(ofs).SetPrecision(5) << "1-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 1) - norm1) << endl;
    Print(ofs).SetPrecision(5) << "2-norm error: " << fabs(gempic_norm(&mw_yee.rho, infra, 2) - norm2) << endl;
#endif

    ofs.close();
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}




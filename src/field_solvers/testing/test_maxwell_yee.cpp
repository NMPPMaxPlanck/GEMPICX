//------------------------------------------------------------------------------
// Test 3D Maxwell Yee Solver (finite differences) on periodic grid
//
//  For the Maxwell-equations we use the solution
//  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\
//                          -2\cos(x_1+x_2+x_3 - \sqrt(3) t) \\
//                            \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
//  B(x,t) = \begin{pmatrix} \sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ 0 \\ -\sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
//
// For the Poisson equation we use:
// E(x,t) = \begin{pmatrix} -\sin(x)\cos(y)\cos(z)-0.5\sin(2x)cos(2y)cos(2z)\\
//                          -\cos(x)\sin(y)\cos(z)-0.5\cos(2x)sin(2y)cos(2z)\\
//                          -\cos(x)\cos(y)\sin(z)-0.5\cos(2x)cos(2y)sin(2z) \end{pmatrix}
//------------------------------------------------------------------------------


#include <AMReX.H>
#include <AMReX_Print.H>
#include <maxwell_yee.H>
#include <gempic_Config.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace std;
using namespace amrex;

//------------------------------------------------------------------------------
  // Solutions and RHS

double E_x(std::array<double,GEMPIC_SPACEDIM> x, double t){return(cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}
double E_y(std::array<double,GEMPIC_SPACEDIM> x, double t){return(-2*cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}
double E_z(std::array<double,GEMPIC_SPACEDIM> x, double t){return(cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}

double B_x(std::array<double,GEMPIC_SPACEDIM> x, double t){return(sqrt(3)*cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}
double B_y(std::array<double,GEMPIC_SPACEDIM> x, double t){return(0);}
double B_z(std::array<double,GEMPIC_SPACEDIM> x, double t){return(-sqrt(3)*cos(x[0]+x[1]+x[2]-sqrt(3.0)*t));}

double phi_fun(std::array<double,GEMPIC_SPACEDIM> x){return(cos(x[0])*cos(x[1])*cos(x[2])+1/4.0*cos(2.0*x[0])*cos(2.0*x[1])*cos(2.0*x[2]));}
double rho_fun(std::array<double,GEMPIC_SPACEDIM> x){return(-3.0*(cos(x[0])*cos(x[1])*cos(x[2])+cos(2.0*x[0])*cos(2.0*x[1])*cos(2.0*x[2])));}
double Ep_x(std::array<double,GEMPIC_SPACEDIM> x,double t){return(-sin(x[0])*cos(x[1])*cos(x[2])-0.5*sin(2.0*x[0])*cos(2.0*x[1])*cos(2.0*x[2]));}
double Ep_y(std::array<double,GEMPIC_SPACEDIM> x,double t){return(-cos(x[0])*sin(x[1])*cos(x[2])-0.5*cos(2.0*x[0])*sin(2.0*x[1])*cos(2.0*x[2]));}
double Ep_z(std::array<double,GEMPIC_SPACEDIM> x,double t){return(-cos(x[0])*cos(x[1])*sin(x[2])-0.5*cos(2.0*x[0])*cos(2.0*x[1])*sin(2.0*x[2]));}

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_SPACEDIM> v,int Np,double k) {
  return(0.0);
}

void main_main ()
{

  // make pointer-array for functions
  double (*fields[2*GEMPIC_SPACEDIM]) (std::array<double,GEMPIC_SPACEDIM> x, double t);
  fields[0] = E_x;
  fields[GEMPIC_SPACEDIM] = B_x;
#if (GEMPIC_SPACEDIM > 1)
  fields[1] = E_y;
  fields[GEMPIC_SPACEDIM+1] = B_y;
#endif
#if (GEMPIC_SPACEDIM > 2)
  fields[2] = E_z;
  fields[GEMPIC_SPACEDIM+2] = B_z;
#endif

  double (*fields_poisson[GEMPIC_SPACEDIM]) (std::array<double,GEMPIC_SPACEDIM> x,double t);
  fields_poisson[0] = Ep_x;
#if (GEMPIC_SPACEDIM > 1)
  fields_poisson[1] = Ep_y;
#endif
#if (GEMPIC_SPACEDIM > 2)
  fields_poisson[2] = Ep_z;
#endif
  
//------------------------------------------------------------------------------
  array<Real,2*GEMPIC_SPACEDIM> E_B_error; //array for storing errors

//------------------------------------------------------------------------------
  // Initialize Infrastructure
  initializer init;
  amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
  amrex::IntVect n_cell(AMREX_D_DECL(64,64,64));
  init.initialize_from_parameters(n_cell,32,is_periodic,0.01,5,{1.0},{1.0},1000,0.5,
                  {0.0},{1.0},{1.0},WF);
  infrastructure infra(init);
  
//------------------------------------------------------------------------------
  // Solve
  
  maxwell_yee mw_yee(init, infra);

  mw_yee.init_E_B(fields, infra);

  std::ofstream ofs("test_maxwell_yee.output", std::ofstream::out);
  Print(ofs) << "Maxwell" << endl;
  for (int n=1;n<=mw_yee.nsteps;n++){
    mw_yee.advance(infra);
    E_B_error = mw_yee.computeError(fields, true, infra);
    
    Print(ofs) << "step " << n << endl;
    Print(ofs).SetPrecision(5) << "Ex error: " << E_B_error[0] <<
                                  " |Ey error: " << E_B_error[1] <<
                                  " |Ez error: " << E_B_error[2] << endl;
    Print(ofs).SetPrecision(5) << "Bx error: " << E_B_error[3] <<
                                  " |By error: " << E_B_error[4] <<
                                  " |Bz error: " << E_B_error[5] << endl; 
  }

  //------------------------------------------------------------------------------
  // Poisson

  mw_yee.init_rho_phi(phi_fun, rho_fun, infra);
  mw_yee.solve_poisson(infra);
  mw_yee.computeError(fields_poisson, false, infra);

  Print(ofs) << endl;
  Print(ofs) << "Poisson" << endl;
  Print(ofs).SetPrecision(5) << "Ex error: " << E_B_error[0] <<
                                " |Ey error: " << E_B_error[1] <<
                                " |Ez error: " << E_B_error[2] << endl;  

  ofs.close();
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




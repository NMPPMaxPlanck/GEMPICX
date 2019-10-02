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

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace std;
using namespace amrex;

//------------------------------------------------------------------------------
  // Solution

double E_x(double x,double y,double z, double t){return(cos(x+y+z-sqrt(3.0)*t));}
double E_y(double x,double y,double z, double t){return(-2*cos(x+y+z-sqrt(3.0)*t));}
double E_z(double x,double y,double z, double t){return(cos(x+y+z-sqrt(3.0)*t));}

double B_x(double x,double y,double z, double t){return(sqrt(3)*cos(x+y+z-sqrt(3.0)*t));}
double B_y(double x,double y,double z, double t){return(0);}
double B_z(double x,double y,double z, double t){return(-sqrt(3)*cos(x+y+z-sqrt(3.0)*t));}

double phi_fun(double x,double y,double z){return(cos(x)*cos(y)*cos(z)+1/4.0*cos(2.0*x)*cos(2.0*y)*cos(2.0*z));}
double rho_fun(double x,double y,double z){return(-3.0*(cos(x)*cos(y)*cos(z)+cos(2.0*x)*cos(2.0*y)*cos(2.0*z)));}
double Ep_x(double x,double y,double z,double t){return(-sin(x)*cos(y)*cos(z)-0.5*sin(2.0*x)*cos(2.0*y)*cos(2.0*z));}
double Ep_y(double x,double y,double z,double t){return(-cos(x)*sin(y)*cos(z)-0.5*cos(2.0*x)*sin(2.0*y)*cos(2.0*z));}
double Ep_z(double x,double y,double z,double t){return(-cos(x)*cos(y)*sin(z)-0.5*cos(2.0*x)*cos(2.0*y)*sin(2.0*z));}

void main_main ()
{

  // make pointer-array for functions
  double (*fields[6]) (double x,double y,double z, double t);
  fields[0] = E_x;
  fields[1] = E_y;
  fields[2] = E_z;
  fields[3] = B_x;
  fields[4] = B_y;
  fields[5] = B_z;

  double (*fields_poisson[3]) (double x,double y,double z,double t);
  fields_poisson[0] = Ep_x;
  fields_poisson[1] = Ep_y;
  fields_poisson[2] = Ep_z;
  
//------------------------------------------------------------------------------
  // Parameters
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int nsteps; // number of simulation steps
  int is_periodic[3]; // periodicity: 1 -> periodic

  n_cell = 64;
  max_grid_size = 32;
  nsteps = 5;
  // periodic in all directions:
  is_periodic[0] = 1;
  is_periodic[1] = 1;
  is_periodic[2] = 1;

  // physical box (geometry)
  double twopi = 2.0*3.14159265359;
  RealBox real_box({0.0, 0.0, 0.0},
		   {twopi, twopi, twopi});
  array<Real,6> E_B_error; //array for storing errors
    
//------------------------------------------------------------------------------
  // Solve
  
  maxwell_yee mw_yee(n_cell, max_grid_size, nsteps, is_periodic,
		       real_box);

  mw_yee.init_E_B(fields);

  std::ofstream ofs("test_maxwell_yee.output", std::ofstream::out);
  Print(ofs) << "Maxwell" << endl;
  for (int n=1;n<=mw_yee.nsteps;n++){
    mw_yee.advance();
    E_B_error = mw_yee.computeError(fields, true);

    
    
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

  mw_yee.init_rho_phi(phi_fun, rho_fun);
  mw_yee.solve_poisson();
  mw_yee.computeError(fields_poisson, false);

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




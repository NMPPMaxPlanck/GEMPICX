
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
  
//------------------------------------------------------------------------------
  // Parameters
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int nsteps; // number of simulation steps
  int is_periodic[3]; // periodicity: 1 -> periodic

  n_cell = 128;
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
    

  maxwell_yee mw_yee(n_cell, max_grid_size, nsteps, is_periodic,
		       real_box);

  mw_yee.init_E_B(fields);

  for (int n=1;n<=mw_yee.nsteps;n++){
    mw_yee.advance();
    E_B_error = mw_yee.computeError(fields);
    
    Print() << "step " << n << endl;
    Print().SetPrecision(5) << "Ex error: " << E_B_error[0] <<
                               " |Ey error: " << E_B_error[1] <<
                               " |Ez error: " << E_B_error[2] << endl;
    Print().SetPrecision(5) << "Bx error: " << E_B_error[3] <<
                               " |By error: " << E_B_error[4] <<
                               " |Bz error: " << E_B_error[5] << endl;
  }
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




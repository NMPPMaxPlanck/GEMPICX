
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <gempic_Particle_mod_K.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double E_x(double x,double y,double z, double t){return(cos(x+y+z-sqrt(3.0)*t));}
double E_y(double x,double y,double z, double t){return(-2*cos(x+y+z-sqrt(3.0)*t));}
double E_z(double x,double y,double z, double t){return(cos(x+y+z-sqrt(3.0)*t));}

double B_x(double x,double y,double z, double t){return(sqrt(3)*cos(x+y+z-sqrt(3.0)*t));}
double B_y(double x,double y,double z, double t){return(0);}
double B_z(double x,double y,double z, double t){return(-sqrt(3)*cos(x+y+z-sqrt(3.0)*t));}

void main_main ()
{
  double pi = 3.14159265359;

  // make pointer-array for functions
  double (*fields[6]) (double x,double y,double z, double t);
  fields[0] = E_x;
  fields[1] = E_y;
  fields[2] = E_z;
  fields[3] = B_x;
  fields[4] = B_y;
  fields[5] = B_z;

//------------------------------------------------------------------------------
  // Initialize Infrastructure
  
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int is_periodic[3]; // periodicity: 1 -> periodic

  n_cell = 64;
  max_grid_size = 32;
  // periodic in all directions:
  is_periodic[0] = 1;
  is_periodic[1] = 1;
  is_periodic[2] = 1;

  // physical box (geometry)
  double twopi = 2.0*3.14159265359;
  RealBox real_box({0.0, 0.0, 0.0},
		   {twopi, twopi, twopi});
  // build infrastructure
  infrastructure infra(n_cell, max_grid_size, is_periodic, real_box);

//------------------------------------------------------------------------------
  //Initialize Particle Groups

  const int n_species = 1;
  array<Real, n_species> charge = {1.0};
  array<Real, n_species> mass = {1.0};
  particle_groups part_gr(infra, n_species, charge, mass);

  //Add particles one by one
  int N_parts = 3;
  int species = 0;
  array<Real,3> position;
  array<Real,3> velocity;
  Real weight;

  for (int pp=0;pp<N_parts;pp++) {
    position = {(double)pp, (double)(pp%2)*pi, 0.0};
    velocity = {0.0, 0.0, 0.0};
    weight = 1.0;
    part_gr.add_particle(species, position, velocity, weight);
  }

//------------------------------------------------------------------------------
  //Initialize Maxwell Yee
  int n_steps = 5;
  maxwell_yee mw_yee(n_steps, real_box, infra);
  mw_yee.init_E_B(fields, infra);

//------------------------------------------------------------------------------
  //Test CIC

  std::ofstream ofs("test_particle_groups.output", std::ofstream::out);
  Print(ofs) << "field_interpolation" << endl;
  
  Real tol = 0.01;
    GpuArray<amrex::Real,3> plo;
    plo[0] = infra.geom.ProbLo()[0];
    plo[1] = infra.geom.ProbLo()[1];
    plo[2] = infra.geom.ProbLo()[2];
    GpuArray<amrex::Real,3> dxi;
    dxi[0] = 1.0/infra.dx[0];
    dxi[1] = 1.0/infra.dx[1];
    dxi[2] = 1.0/infra.dx[2];

    // fill ghost cells
    (*mw_yee.E_Array[0]).FillBoundary(infra.geom.periodicity());
    (*mw_yee.E_Array[1]).FillBoundary(infra.geom.periodicity());
    (*mw_yee.E_Array[2]).FillBoundary(infra.geom.periodicity());

    (*mw_yee.B_Array[0]).FillBoundary(infra.geom.periodicity());
    (*mw_yee.B_Array[1]).FillBoundary(infra.geom.periodicity());
    (*mw_yee.B_Array[2]).FillBoundary(infra.geom.periodicity());

  for (ParIter<4,0,0,0> pti(*part_gr.mypc[0], 0); pti.isValid(); ++pti) {

    const Box& box = pti.validbox();
    auto& particles = pti.GetArrayOfStructs();
    const long np  = pti.numParticles();
    int nc = 3; // number of components

    // interpolate one field at one particle position
    Real eres[nc], bres[nc];
    double esol, bsol;
    for (int pp = 0; pp<np; pp++) {
      for (int cc = 0; cc < nc; cc++){
	//E-field
	Array4<Real> const& earr = (*mw_yee.E_Array[cc])[pti].array();
	eres[cc] = gempic_interpolate_cic(particles[pp], earr, plo, dxi,
				       *mw_yee.E_Index[cc]);
        esol = (*fields[cc])(particles[pp].pos(0),particles[pp].pos(1),particles[pp].pos(2),0.0);
	if (abs(eres[cc]-esol) > tol) {
	  Print(ofs) << "check results at particle " << pp << ", E-component " << cc <<  endl;
	}

	//B-field
	Array4<Real> const& barr = (*mw_yee.B_Array[cc])[pti].array();
	bres[cc] = gempic_interpolate_cic(particles[pp], barr, plo, dxi,
				       *mw_yee.B_Index[cc]);
        bsol = (*fields[cc+3])(particles[pp].pos(0),particles[pp].pos(1),particles[pp].pos(2),0.0);
	if (abs(bres[cc]-bsol) > tol) {
	  Print(ofs) << "check results at particle " << pp << ", B-component " << cc <<  endl;
	}
      }
    }
  }
  ofs.close();
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




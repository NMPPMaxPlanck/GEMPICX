#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <infrastructure.H>
#include <maxwell_yee.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#ifndef INITIALIZER_H_
#define INITIALIZER_H_

//------------------------------------------------------------------------------
// user defined parameters

// parameters for infrastructure:
int n_cell = 8; // number of cells
int max_grid_size = 4; // maximum number of cells in each direction
int is_periodic[3] = {1,1,1}; // periodicity: 1 -> periodic

// parameters for time evolution:
amrex::Real dt = 0.01;
int n_steps = 5;

// parameters for particles:
const int n_species = 1;
std::array<amrex::Real, n_species> charge = {1.0};
std::array<amrex::Real, n_species> mass = {1.0};
int Np_cell = 1000; //number of particles per cell

// parameters/functions for particle distribution:
double k = 0.5;
double alpha = 0.5;
double weight_fun(double x,double y,double z,double v_x,double v_y,double v_z,int Np){
  return((1.0 + alpha*cos(k*x))/Np);
}

//------------------------------------------------------------------------------
// Data types necessary to build objects

// infrastructure: physical box (geometry)
double twopi = 2.0*3.14159265359;
amrex::RealBox real_box({0.0, 0.0, 0.0},
			{twopi/k, twopi/k, twopi/k});

#endif

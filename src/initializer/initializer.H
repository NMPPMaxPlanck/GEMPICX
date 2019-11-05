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

// parameters/functions for particle distribution:
double weight_fun(double x,double y,double z,double v_x,double v_y,double v_z,int Np){
  double k = 0.5;
double alpha = 0.5;
  return((1.0 + alpha*cos(k*x))/Np);
}

infrastructure initialize_ifr(){

  double k = 0.5;
  
  //------------------------------------------------------------------------------
  // user defined parameters

  // parameters for infrastructure:
  int n_cell = 8; // number of cells
  int max_grid_size = 4; // maximum number of cells in each direction
  int is_periodic[3] = {1,1,1}; // periodicity: 1 -> periodic

  // parameters for time evolution:
  amrex::Real dt = 0.01;
  int n_steps = 5;

  //------------------------------------------------------------------------------
  // Initialize Infrastructure

  // physical box (geometry)
  double twopi = 2.0*3.14159265359;
  amrex::RealBox real_box({0.0, 0.0, 0.0},
			  {twopi/k, twopi/k, twopi/k});

  infrastructure infra(n_cell, max_grid_size, is_periodic, real_box);
  return(infra);
}

#endif

#include <AMReX.H>
#include <AMReX_Print.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#ifndef INFRASTRUCTURE_H_
#define INFRASTRUCTURE_H_

class infrastructure
{
  public:

  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int is_periodic[3];
  amrex::RealBox real_box; // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}
  
  amrex::BoxArray grid;
  amrex::Geometry geom;
  amrex::Box domain;
  amrex::DistributionMapping distriMap;
  const amrex::Real* dx;

  // The default constructor.
  infrastructure ();

  // Explicit const
  infrastructure (int nc, int max_s, int is_per[], amrex::RealBox rb){
    n_cell = nc;
    max_grid_size = max_s;

    is_periodic[0] = is_per[0];
    is_periodic[1] = is_per[1];
    is_periodic[2] = is_per[2];
    real_box = rb;

    amrex::IntVect dom_lo = {0, 0, 0};
    amrex::IntVect dom_hi = {n_cell-1, n_cell-1, n_cell-1};

    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);
    grid.define(domain);
    grid.maxSize(max_grid_size);
    geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic);
    dx = geom.CellSize();

    distriMap.define(grid);	

  }
};

#endif

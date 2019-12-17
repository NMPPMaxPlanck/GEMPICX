#include <AMReX.H>
#include <AMReX_Print.H>

#include <initializer.H>

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

  amrex::IntVect n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int is_periodic[GEMPIC_SPACEDIM];
  amrex::RealBox real_box; // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}

  amrex::RealVect Length; // Length of domain in x,y,z directions
  amrex::BoxArray grid;
  amrex::Geometry geom;
  amrex::Box domain;
  amrex::DistributionMapping distriMap;
  const amrex::Real* dx;
  std::array<std::array<amrex::Real, GEMPIC_SPACEDIM>,GEMPIC_SPACEDIM> ploE;
  std::array<std::array<amrex::Real,GEMPIC_SPACEDIM>,GEMPIC_SPACEDIM> ploB;

  // The default constructor.
  infrastructure ();

  // Explicit const
  infrastructure (initializer init){
    n_cell = init.n_cell;
    max_grid_size = init.max_grid_size;

    for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
        is_periodic[cc] = init.is_periodic[cc];
    }
    real_box = init.real_box;

    for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
      Length[cc] = (real_box.hi(cc)-real_box.lo(cc));
    }

    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);
    grid.define(domain);
    grid.maxSize(max_grid_size);
    geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic);
    dx = geom.CellSize();

    distriMap.define(grid);

    // Ex
    ploE[0][0] = geom.ProbLo()[0]*1.0/dx[0]+0.5;
    // Bx
    ploB[0][0] = geom.ProbLo()[0]*1.0/dx[0];
#if (GEMPIC_SPACEDIM > 1)
    // Ex
    ploE[0][1] = geom.ProbLo()[1]*1.0/dx[1];
    // Bx
    ploB[0][1] = geom.ProbLo()[1]*1.0/dx[1]+0.5;
    //Ey
    ploE[1][0] = geom.ProbLo()[0]*1.0/dx[0];
    ploE[1][1] = geom.ProbLo()[1]*1.0/dx[1]+0.5;
    //By
    ploB[1][0] = geom.ProbLo()[0]*1.0/dx[0]+0.5;
    ploB[1][1] = geom.ProbLo()[1]*1.0/dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
    //Ex
    ploE[0][2] = geom.ProbLo()[2]*1.0/dx[2];
    //Bx
    ploB[0][2] = geom.ProbLo()[2]*1.0/dx[2]+0.5;
    //Ey
    ploE[1][2] = geom.ProbLo()[2]*1.0/dx[2];
    //By
    ploB[1][2] = geom.ProbLo()[2]*1.0/dx[2]+0.5;
    //Ez
    ploE[2][0] = geom.ProbLo()[0]*1.0/dx[0];
    ploE[2][1] = geom.ProbLo()[1]*1.0/dx[1];
    ploE[2][2] = geom.ProbLo()[2]*1.0/dx[2]+0.5;
    //Bz
    ploB[2][0] = geom.ProbLo()[0]*1.0/dx[0]+0.5;
    ploB[2][1] = geom.ProbLo()[1]*1.0/dx[1]+0.5;
    ploB[2][2] = geom.ProbLo()[2]*1.0/dx[2];
#endif

  }
};

#endif

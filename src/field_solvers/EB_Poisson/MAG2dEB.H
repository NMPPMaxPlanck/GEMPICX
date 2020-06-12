#ifndef MAG2dEB_H_
#define MAG2dEB_H_

#include <AMReX_EBFabFactory.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

class MAG2dEB
{
public:

    MAG2dEB();
    ~MAG2dEB() {
      if(ebf) delete ebf;
    };

    void solve ();
    void writeEBPlotfileS () const;
  //    void writeEBPlotfileM () const;

private:

    void readParameters ();
    void initData ();
    void initEBPoisson ();
    void initEBPolar ();
    void solveEBPoisson ();

    int max_level = 1;
    int ref_ratio = 2;
    int n_cell = 512;
    int max_grid_size = 32;
    
    bool composite_solve = true;
  bool polprint = true;

    int prob_type = 2;  // 1. EBPoisson,  2. EBPolar
    Real rmin = 0.5;
    Real rmax = 4.0;

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 100;
    int max_fmg_iter = 0;
    int linop_maxorder = 2;
    bool agglomeration = true;
    bool consolidation = true;
    int max_coarsening_level = 10;
    int normtype = 0;
        // relative and absolute tolerances for linear solve
    Real tol_rel = 1.e-10;
    Real tol_abs = 0.0;

    bool use_hypre = false;
    bool use_petsc = false;

  // No adaptive mesh refinement
    Geometry geom;
    BoxArray grids;
    DistributionMapping dmap;
    EBFArrayBoxFactory *ebf;
  
    MultiFab solution;
    MultiFab rhs;
    MultiFab exact_solution;
    MultiFab acoef;
    MultiFab bcoef;

    Real ascalar = 0.0;
    Real bscalar = 1.0;
};

#endif

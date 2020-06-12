#include "MAG2dEB.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB2_IF_Sphere.H>
#include <AMReX_MLEBABecLap.H>
// #include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MLMG.H>

using namespace amrex;

MAG2dEB::MAG2dEB()
{
    readParameters();
    initData();
}

void
MAG2dEB::solve ()
{
    if (prob_type == 1) {
        solveEBPoisson();
    }
    else {
        solveEBPoisson();
    }
}

void
MAG2dEB::solveEBPoisson ()
{
  LPInfo info;

  MLEBABecLap mlebabec({geom},{grids},{dmap},info,{ebf});

        // define array of LinOpBCType for domain boundary conditions
  std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
  std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bc_lo[idim] = LinOpBCType::Periodic;
            bc_hi[idim] = LinOpBCType::Periodic;
  }

        // Boundary of the whole domain. This functions must be called,
        // and must be called before other bc functions.
  mlebabec.setDomainBC(bc_lo,bc_hi);

        // see AMReX_MLLinOp.H for an explanation
  mlebabec.setLevelBC(0, nullptr);
  Array<MultiFab,AMREX_SPACEDIM> bbcoef;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    bbcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)),
		       dmap, 1, 0, MFInfo(), *ebf);
    bbcoef[idim].setVal(1.0);
  }
        // think of this beta as the "BCoef" associated with an EB face
  MultiFab beta(grids, dmap, 1, 0, MFInfo(), *ebf);
  beta.setVal(1.);
    
        // operator looks like (ACoef - div BCoef grad) phi = rhs
  mlebabec.setACoeffs(0, acoef);
  mlebabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bbcoef));

        // scaling factors; these multiply ACoef and BCoef
  mlebabec.setScalars(ascalar, bscalar);

        // set homogeneous Dirichlet BC for EB
  mlebabec.setEBHomogDirichlet(0,beta);

  MLMG mlmg(mlebabec);
  mlmg.setVerbose(verbose);
  mlmg.setNormType(normtype);
        
        // Solve linear system
  mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);
}


void
MAG2dEB::readParameters ()
{
    ParmParse pp;
    pp.query("max_level", max_level);
    pp.query("ref_ratio", ref_ratio);
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    pp.query("composite_solve", composite_solve);
    pp.query("polprint", polprint);

    pp.query("prob_type", prob_type);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("linop_maxorder", linop_maxorder);
    pp.query("agglomeration", agglomeration);
    pp.query("consolidation", consolidation);
    pp.query("max_coarsening_level", max_coarsening_level);
    pp.query("norm_type", normtype);
    pp.query("Stop_error", tol_rel);
}

void
MAG2dEB::initData ()
{
    RealBox rb({AMREX_D_DECL(-4.2,-4.2,-4.2)}, {AMREX_D_DECL(4.2,4.2,4.2)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};
    Box domain(IntVect{AMREX_D_DECL(0,0,0)},
                       IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    geom.define(domain, rb, CoordSys::cartesian, is_periodic);
    grids.define(domain); // define the BoxArray to be a single grid
    grids.maxSize(max_grid_size); 
    dmap.define(grids); 

    if (prob_type == 1) {
      int required_coarsening_level = 0;
                // typically the same as the max AMR level index
      // typically a huge number so MG coarsens as much as possible
      // build a simple geometry using the "eb2." parameters in the inputs file
      RealArray center{0.0,0.0};
      Real radius = 4.0;
      bool has_fluid_inside = 1;
      int ngrow = 4;
      EB2::SphereIF sf(radius, center, has_fluid_inside);
      EB2::GeometryShop<EB2::SphereIF> gshop(sf);
      EB2::Build(gshop, geom, required_coarsening_level,
                   max_coarsening_level, ngrow);
      // EB2::Build(geom, required_coarsening_level, max_coarsening_level);
    }
    else if(prob_type == 3) {
      int required_coarsening_level = 0;
                // typically the same as the max AMR level index
      // typically a huge number so MG coarsens as much as possible
      // build a simple geometry using the "eb2." parameters in the inputs file
      RealArray center{0.0,0.0};
      Real radius = 0.5;
      bool has_fluid_inside = 0;
      int ngrow = 4;
      EB2::SphereIF sf(radius, center, has_fluid_inside);
      //      EB2::SphereIF sf0(0.5, center, 0);
      //      auto twospheres = EB2::makeIntersection(sf,sf0);
      // auto twospheres = EB2::makeUnion(sf,sf0);
      // auto gshop = makeShop(twospheres);
      EB2::GeometryShop<EB2::SphereIF> gshop(sf);
      EB2::Build(gshop, geom, required_coarsening_level,
                   max_coarsening_level, ngrow);
    }
    else {
      int required_coarsening_level = 0;
                // typically the same as the max AMR level index
      // typically a huge number so MG coarsens as much as possible
      // build a simple geometry using the "eb2." parameters in the inputs file
      RealArray center{0.0,0.0};
      Real radius = 4.0;
      bool has_fluid_inside = 1;
      int ngrow = 4;
      EB2::SphereIF sf(radius, center, has_fluid_inside);
      EB2::SphereIF sf0(0.5, center, 0);
      //      auto twospheres = EB2::makeIntersection(sf,sf0);
      auto twospheres = EB2::makeUnion(sf,sf0);
      auto gshop = makeShop(twospheres);
      EB2::Build(gshop, geom, required_coarsening_level,
                   max_coarsening_level, ngrow);
    }
    const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
    const EB2::Level& eb_level = eb_is.getLevel(geom);
    // options are basic, volume, or full
    EBSupport ebs = EBSupport::full;
    // number of ghost cells for each of the 3 EBSupport types
    Vector<int> ng_ebs = {2,2,2};

    EBFArrayBoxFactory ebfac(eb_level, geom,
				     grids, dmap, ng_ebs, ebs);
    solution.define(grids, dmap, 1, 0, MFInfo(), ebfac);
    rhs.define(grids, dmap, 1, 0, MFInfo(), ebfac);
    exact_solution.define(grids, dmap, 1, 0, MFInfo(), ebfac);
    acoef.define(grids, dmap, 1, 0, MFInfo(), ebfac);
    bcoef.define(grids, dmap, 1, 0, MFInfo(), ebfac);
    if (prob_type == 1) {
      initEBPoisson();
    }
    else {
      initEBPolar();
    }
    ebf = ebfac.clone();
}


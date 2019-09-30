//------------------------------------------------------------------------------
// This example is a copy of the demo "HeatEquation_EX1_C" from AmreX
// some changes have been made: rewriting fortran-functions into c++
//------------------------------------------------------------------------------

#include <AMReX.H>
#include <AMReX_Print.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace std;
using namespace amrex;
//------------------------------------------------------------------------------
// init phi
void init_phi(amrex::IntVect  lo,
	      amrex::IntVect  hi,
	      FArrayBox &phi,
	      //amrex::MultiFab phi,
	      const double  dx[3],
	      const double  prob_lo[3],
	      const double  prob_hi[3]) {
  
  int k,j,i;
  double x,y,z,r2;

  
  for(int k=lo[2]; k<=hi[2]; k++){
    z = prob_lo[2] + ((double)k+0.5) * dx[2];
    for(int j=lo[1]; j<=hi[1]; j++){
      y = prob_lo[1] + ((double)j+0.5) * dx[1];
      for(int i=lo[0]; i<=hi[0]; i++){
        x = prob_lo[0] + ((double)i+0.5) * dx[0];
	r2 = pow((x-0.25),2.0) + pow((y-0.25),2.0) + pow((z-0.25),2.0) / 0.01;
	// the box for this value of phi:
	Box cc({i,j,k}, {i+1,j+1,k+1});
        phi.setVal(1.0 + exp(-r2), cc);
      }
    }
  }
  
 }

//------------------------------------------------------------------------------
// update phi
void update_phi(amrex::IntVect lo,
		amrex::IntVect hi,
		FArrayBox &phiold,
		FArrayBox &phinew,
		FArrayBox &fluxx,
		FArrayBox &fluxy,
		FArrayBox &fluxz,
		const double dx[3],
		Real dt) {
  int i,j,k;
  Real dtdx[3];
  Real addphi;
  Real valx0[1];
  Real valx1[1];
  Real valy0[1];
  Real valy1[1];
  Real valz0[1];
  Real valz1[1];

  dtdx[0] = dt/ dx[0];
  dtdx[1] = dt/ dx[1];
  dtdx[2] = dt/ dx[2];

  phinew.copy(phiold);

  for(int k=lo[2]; k<=hi[2]; k++){
    for(int j=lo[1]; j<=hi[1]; j++){
      for(int i=lo[0]; i<=hi[0]; i++){

	fluxx.getVal(valx0, {  i,  j,  k});
	fluxx.getVal(valx1, {i+1,  j,  k});
	
	fluxy.getVal(valy0, {  i,  j,  k});
	fluxy.getVal(valy1, {  1,j+1,  k});
	
	fluxz.getVal(valz0, {  i,  j,  k});
	fluxz.getVal(valz1, {  1,  j,k+1});


	addphi = dtdx[0] * (valx1 - valx0) +
                 dtdx[1] * (valy1 - valy0) +
	         dtdx[2] * (valz1 - valz0);

	Box cc( {i,j,k}, {i+1,j+1,k+1});
	phinew.plus(addphi, cc);
      }
    }
  }
  
}
//------------------------------------------------------------------------------
// compute_flux

void compute_flux(amrex::IntVect lo,
		  amrex::IntVect hi,
		  FArrayBox &phi,
		  FArrayBox &fluxx,
		  FArrayBox &fluxy,
		  FArrayBox &fluxz,
		  const double dx[3]) {
  int i,j,k;
  Real valx0[1];
  Real valx1[1];
  Real valy0[1];
  Real valy1[1];
  Real valz0[1];
  Real valz1[1];
  
  // x-flux
  for(int k=lo[2]; k<hi[2]; k++){
    for(int j=lo[1]; j<hi[1]; j++){
      for(int i=lo[0]; i<hi[0]+1; i++){
	phi.getVal(valx0, {i,j,k});
	phi.getVal(valx1, {i-1,j,k});
	
	Box cc( {i,j,k}, {i+1,j+1,k+1});
	fluxx.setVal((*valx0 - *valx1)/dx[0], cc);
      }
    }
  }
  
  // y-flux
  for(int k=lo[2]; k<hi[2]; k++){
    for(int j=lo[1]; j<hi[1]+1; j++){
      for(int i=lo[0]; i<hi[0]; i++){
	phi.getVal(valy0, {i,j,k});
	phi.getVal(valy1, {i,j-1,k});
	
	Box cc( {i,j,k}, {i+1,j+1,k+1});
	fluxy.setVal((*valy0 - *valy1)/dx[1],cc);
      }
    }
  }

  // z-flux
  for(int k=lo[2]; k<hi[2]+1; k++){
    for(int j=lo[1]; j<hi[1]; j++){
      for(int i=lo[0]; i<hi[0]; i++){
	phi.getVal(valz0, {i,j,k});
	phi.getVal(valz1, {i,j,k-1});
	
	Box cc( {i,j,k}, {i+1,j+1,k+1});
	fluxz.setVal((*valz0 - *valz1)/dx[2],cc);
      }
    }
  }
}

//------------------------------------------------------------------------------
// advance
void advance (MultiFab& phi_old,
	      MultiFab& phi_new,
	      Array<MultiFab,3>& flux,
	      Real dt,
	      const Geometry& geom)
{
  // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    phi_old.FillBoundary(geom.periodicity());

    int Ncomp = phi_old.nComp();
    int ng_p = phi_old.nGrow();
    int ng_f = flux[0].nGrow();

    const Real* dx = geom.CellSize(); 

    const Box& domain_bx = geom.Domain();

    // Compute fluxes one grid at a time
    for (MFIter mfi(phi_old); mfi.isValid(); ++mfi ) {
      const Box& bx = mfi.validbox();

      amrex::IntVect lo = {bx.smallEnd()};
      amrex::IntVect hi = {bx.bigEnd()};
      compute_flux(lo, hi, phi_old[mfi], flux[0][mfi], flux[1][mfi],
		   flux[2][mfi], geom.CellSize());
    }

    // Advance the solution one grid at a time
    for (MFIter mfi(phi_old); mfi.isValid(); ++mfi ) {
      const Box& bx = mfi.validbox();

      amrex::IntVect lo = {bx.smallEnd()};
      amrex::IntVect hi = {bx.bigEnd()};

      update_phi(lo, hi, phi_old[mfi], phi_new[mfi], flux[0][mfi], flux[1][mfi],
		 flux[2][mfi], geom.CellSize(), dt);
    }
}

void main_main ()
{
//------------------------------------------------------------------------------
  // Parameters
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int nsteps; // number of simulation steps
  int plot_int; //
  int is_periodic[3]; // periodicity: 1 -> periodic

  {
    n_cell = 128;
    max_grid_size = 32;
    plot_int = 100;
    nsteps = 5; //1000;
    // periodic in all directions:
    is_periodic[0] = 1;
    is_periodic[1] = 1;
    is_periodic[2] = 1;
  }

//------------------------------------------------------------------------------
  // Box Array and Geometry
  BoxArray ba;
  Geometry geom;
  {
    // grid box
    IntVect dom_lo = {       0,        0,        0};
    IntVect dom_hi = {n_cell-1, n_cell-1, n_cell-1};
    Box domain(dom_lo, dom_hi);

    // individual boxes (boxarray)
    ba.define(domain);
    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // physical box (geometry)
    RealBox real_box({-1.0,-1.0,-1.0},
		     { 1.0, 1.0, 1.0});
    geom.define(domain,&real_box,CoordSys::cartesian,is_periodic);
  }
  int Nghost = 1; // number of ghost cells for each array
  int Ncomp = 1;  // number of components for each array
  DistributionMapping dm(ba); // how Boxes are distrubuted among MPI processes

//------------------------------------------------------------------------------
  // Initializing phi
  
  // we allocate two phi multifabs; one will store the old state, the other the new.
  MultiFab phi_old(ba, dm, Ncomp, Nghost);
  MultiFab phi_new(ba, dm, Ncomp, Nghost);

  // MFIter = MultiFab Iterator
    for ( MFIter mfi(phi_new); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();

        amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};
	const double * dx[3] = {geom.CellSize()};
	const double * prob_lo[3] = {geom.ProbLo()};
	const double * prob_hi[3] = {geom.ProbHi()};

        init_phi(lo, hi, phi_new[mfi],
		 geom.CellSize(), geom.ProbLo(), geom.ProbHi());

    }


//------------------------------------------------------------------------------
    // time step
    const Real* dx = geom.CellSize();
    Real dt = 0.9*dx[0]*dx[0] / (2.0*3);

    // time to start simulation
    Real time = 0.0;

//------------------------------------------------------------------------------

    // build flux multifabs
    Array<MultiFab, 3> flux;
    for (int dir = 0; dir < 3; dir++) {
      // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(dir);
        flux[dir].define(edge_ba, dm, 1, 0);
    }
    
//------------------------------------------------------------------------------
    // time loop
    
    for (int n = 1; n <= nsteps; ++n)
    {
        // old_phi <- new_phi from previous iter
        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);
	// advance phi
	advance(phi_old, phi_new, flux, dt, geom);
	// advance time
	time = time + dt;

    }
  
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
}




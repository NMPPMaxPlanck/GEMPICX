//------------------------------------------------------------------------------
// Test 3D Maxwell Yee Solver (finite differences) on periodic grid
//
//  We use the solution
//  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ -2\cos(x_1+x_2+x_3) \\ \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
//  B(x,t) = \begin{pmatrix} \sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ 0 \\ -\sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
//------------------------------------------------------------------------------

#include <AMReX.H>
#include <AMReX_Print.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace std;
using namespace amrex;
//------------------------------------------------------------------------------
// init E and B
void init_E_B(amrex::IntVect  lo,
	      amrex::IntVect  hi,
	      FArrayBox &E_new,
	      FArrayBox &E_sol,
	      FArrayBox &B_new,
	      FArrayBox &B_sol,
	      const double  dx[3],
	      const double  prob_lo[3]) {
  
  int k,j,i;
  double x,y,z;
  
  for(int k=lo[2]; k<=hi[2]; k++){
    z = prob_lo[2] + ((double)k+0.5) * dx[2];
    for(int j=lo[1]; j<=hi[1]; j++){
      y = prob_lo[1] + ((double)j+0.5) * dx[1];
      for(int i=lo[0]; i<=hi[0]; i++){
        x = prob_lo[0] + ((double)i+0.5) * dx[0];
	
	// the box for these values:
	Box cc({i,j,k}, {i+1,j+1,k+1});
	
        E_new.setVal(cos(x+y+z), cc, 0, 1);
	E_new.setVal(-2*cos(x+y+z), cc, 1, 1);
	E_new.setVal(cos(x+y+z), cc, 2, 1);

	B_new.setVal(sqrt(3)*cos(x+y+z), cc, 0, 1);
	B_new.setVal(0, cc, 1, 1);
	B_new.setVal(-sqrt(3)*cos(x+y+z), cc, 2, 1);
	
      }
    }
  }

  // at time 0 sol = new
  // arguments: Fab to copy, sorce comp, copy comp, number of comps
  B_sol.copy(B_new,0,0,3);
  
  E_sol.copy(E_new,0,0,3);
  
 }

//------------------------------------------------------------------------------
// update E and B
void update_E_B(amrex::IntVect lo,
		amrex::IntVect hi,
		FArrayBox &E_old,
	        FArrayBox &E_new,
	        FArrayBox &E_sol,
		FArrayBox &B_old,
	        FArrayBox &B_new,
	        FArrayBox &B_sol,
		const double dx[3],
		Real dt,
		const double  prob_lo[3],
	        Real time) {
  int i,j,k;
  double x,y,z;
  Real c1[1];
  Real c2[1];
  Real c3[1];
  Real c4[1];
  Real c5[1];

  for(int k=lo[2]; k<=hi[2]; k++){
    z = prob_lo[2] + ((double)k+0.5) * dx[2];
    for(int j=lo[1]; j<=hi[1]; j++){
      y = prob_lo[1] + ((double)j+0.5) * dx[1];
      for(int i=lo[0]; i<=hi[0]; i++){
	x = prob_lo[0] + ((double)i+0.5) * dx[0];

	// the box for these values:
	Box cc({i,j,k}, {i+1,j+1,k+1});

	// E,B
	
	// E_x
	E_old.getVal(c1, {  i,  j,  k},0,1);
	B_old.getVal(c2, {  i,j+1,  k},2,1);
	B_old.getVal(c3, {  i,  j,  k},2,1);
	B_old.getVal(c4, {  i,  j,k+1},1,1);
	B_old.getVal(c5, {  i,  j,  k},1,1);
        E_new.setVal(*c1+dt*((*c2-*c3)/dx[1]-(*c4-*c5)/dx[2]), cc, 0, 1);

	// E_y
	E_old.getVal(c1, {  i,  j,  k},1,1);
	B_old.getVal(c2, {  i,  j,k+1},0,1);
	B_old.getVal(c3, {  i,  j,  k},0,1);
	B_old.getVal(c4, {i+1,  j,  k},2,1);
	B_old.getVal(c5, {  i,  j,  k},2,1);
        E_new.setVal(*c1+dt*((*c2-*c3)/dx[2]-(*c4-*c5)/dx[0]), cc, 1, 1);

	// E_z
	E_old.getVal(c1, {  i,  j,  k},2,1);
	B_old.getVal(c2, {i+1,  j,  k},1,1);
	B_old.getVal(c3, {  i,  j,  k},1,1);
	B_old.getVal(c4, {i  ,j+1,  k},0,1);
	B_old.getVal(c5, {  i,  j,  k},0,1);
        E_new.setVal(*c1+dt*((*c2-*c3)/dx[0]-(*c4-*c5)/dx[1]), cc, 2, 1);

	// B_x
	B_old.getVal(c1, {  i,  j,  k},0,1);
	E_old.getVal(c2, {  i,j+1,  k},2,1);
	E_old.getVal(c3, {  i,  j,  k},2,1);
	E_old.getVal(c4, {  i,  j,k+1},1,1);
	E_old.getVal(c5, {  i,  j,  k},1,1);
        B_new.setVal(*c1-dt*((*c2-*c3)/dx[1]-(*c4-*c5)/dx[2]), cc, 0, 1);

	// B_y
	B_old.getVal(c1, {  i,  j,  k},1,1);
	E_old.getVal(c2, {  i,  j,k+1},0,1);
	E_old.getVal(c3, {  i,  j,  k},0,1);
	E_old.getVal(c4, {i+1,  j,  k},2,1);
	E_old.getVal(c5, {  i,  j,  k},2,1);
        B_new.setVal(*c1-dt*((*c2-*c3)/dx[2]-(*c4-*c5)/dx[0]), cc, 1, 1);

	// B_z
	B_old.getVal(c1, {  i,  j,  k},2,1);
	E_old.getVal(c2, {i+1,  j,  k},1,1);
	E_old.getVal(c3, {  i,  j,  k},1,1);
	E_old.getVal(c4, {i  ,j+1,  k},0,1);
	E_old.getVal(c5, {  i,  j,  k},0,1);
        B_new.setVal(*c1-dt*((*c2-*c3)/dx[0]-(*c4-*c5)/dx[1]), cc, 2, 1);

        // Analytic values
	E_sol.setVal(cos(x+y+z-sqrt(3)*time), cc, 0, 1);
	E_sol.setVal(-2*cos(x+y+z), cc, 1, 1);
	E_sol.setVal(cos(x+y+z-sqrt(3)*time), cc, 2, 1);
	
	B_sol.setVal(sqrt(3)*cos(x+y+z-sqrt(3)*time), cc, 0, 1);
	B_sol.setVal(0, cc, 1, 1);
	B_sol.setVal(-sqrt(3)*cos(x+y+z-sqrt(3)*time), cc, 2, 1);
	
      }
    }
  }
  
}

//------------------------------------------------------------------------------
// advance
void advance (MultiFab& E_old,
	      MultiFab& E_new,
	      MultiFab& E_sol,
	      MultiFab& B_old,
	      MultiFab& B_new,
	      MultiFab& B_sol,
	      Real dt,
	      const Geometry& geom,
	      Real time)
{
  // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    E_old.FillBoundary(geom.periodicity());
    B_old.FillBoundary(geom.periodicity());

    int Ncomp = E_old.nComp();
    int ng_p = E_old.nGrow();

    const Real* dx = geom.CellSize();
    const Box& domain_bx = geom.Domain();

    // Advance the solution one grid at a time
    for (MFIter mfi(E_old); mfi.isValid(); ++mfi ) {
      const Box& bx = mfi.validbox();

      amrex::IntVect lo = {bx.smallEnd()};
      amrex::IntVect hi = {bx.bigEnd()};

      update_E_B(lo, hi, E_old[mfi], E_new[mfi], E_sol[mfi], B_old[mfi],
		 B_new[mfi], B_sol[mfi], geom.CellSize(), dt, geom.ProbLo(),
		 time);
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
  int Ncomp = 3;  // number of components for each array
  DistributionMapping dm(ba); // how Boxes are distrubuted among MPI processes

//------------------------------------------------------------------------------
  // Initializing E & B
  
  // we allocate three E/B multifabs; one will store the old state, one the new state,
  // and one the analytic solution.
  MultiFab E_old(ba, dm, Ncomp, Nghost);
  MultiFab E_new(ba, dm, Ncomp, Nghost);
  MultiFab E_sol(ba, dm, Ncomp, Nghost);

  MultiFab B_old(ba, dm, Ncomp, Nghost);
  MultiFab B_new(ba, dm, Ncomp, Nghost);
  MultiFab B_sol(ba, dm, Ncomp, Nghost);

  // MFIter = MultiFab Iterator
  for ( MFIter mfi(E_new); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();

        amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

        init_E_B(lo, hi, E_new[mfi], E_sol[mfi], B_new[mfi], B_sol[mfi],
		 geom.CellSize(), geom.ProbLo());

    }


//------------------------------------------------------------------------------
    // time step
    const Real* dx = geom.CellSize();
    Real dt = 0.9*dx[0]*dx[0] / (2.0*3);

    // time to start simulation
    Real time = 0.0;
    
//------------------------------------------------------------------------------
    // time loop
    Vector<double> Eerr;
    Vector<double> Berr;
    
    for (int n = 1; n <= nsteps; ++n)
    {
        // old_data <- new_data from previous iter
        MultiFab::Copy(E_old, E_new, 0, 0, 3, 0);
	MultiFab::Copy(B_old, B_new, 0, 0, 3, 0);
	// advance E & B
	advance(E_old, E_new, E_sol, B_old, B_new, B_sol, dt, geom, time+dt);
	// advance time
	time = time + dt;

	// ERROR
	// Subtract new from sol (Fab, start comp, num of comp, nghost):
	E_sol.minus(E_new, 0, 3, 0);
	B_sol.minus(B_new, 0, 3, 0);
	
	// Return max absolute value (components, nghost, local):
	Eerr = E_sol.norm0({0,1,2}, 0, false);
	Berr = B_sol.norm0({0,1,2}, 0, false);
	
	cout << "step " << n << endl;
	cout << "Ex error: " << Eerr[0] << " |Ey error: " << Eerr[1] <<
	  " |Ez error: " << Eerr[2] << endl;
	cout << "Bx error: " << Berr[0] << " |By error: " << Berr[1] <<
	  " |Bz error: " << Berr[2] << endl;
      
    }
  
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
}




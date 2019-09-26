//------------------------------------------------------------------------------
// Test 3D Maxwell Yee Solver (finite differences) on periodic grid
//
//  We use the solution
//  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\ -2\cos(x_1+x_2+x_3 - sqrt(3) t) \\ \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
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
void init_E_B(IntVect  lo,
	      IntVect  hi,
	      FArrayBox &Field,
	      FArrayBox &Field_sol,
	      const double  dx[3],
	      const double  prob_lo[3],
	      int component,
	      IndexType typ,
	      Real dt) {
  
  int k,j,i;
  double x,y,z;
  double xd, yd, zd; // dual
  dt = dt/2.0; // for leap frog initialization
  
  for(int k=lo[2]; k<=hi[2]; k++){
    z = prob_lo[2] + ((double)k+0.5) * dx[2];
    zd = prob_lo[2] + (double)k * dx[2];
    for(int j=lo[1]; j<=hi[1]; j++){
      y = prob_lo[1] + ((double)j+0.5) * dx[1];
      yd = prob_lo[1] + (double)j * dx[1];
      for(int i=lo[0]; i<=hi[0]; i++){
        x = prob_lo[0] + ((double)i+0.5) * dx[0];
	xd = prob_lo[0] + (double)i * dx[0];
	
	// the box for these values:
	Box cc({i,j,k}, {i,j,k}, typ);

	switch (component) {
	case 0: //Ex
	  Field.setVal(cos(x+yd+zd), cc, 0, 1);
	case 1: //Ey
	  Field.setVal(-2*cos(xd+y+zd), cc, 0, 1);
	case 2: //Ez
	  Field.setVal(cos(xd+yd+z), cc, 0, 1);
	case 3: //Bx
	  Field.setVal(sqrt(3)*cos(xd+y+z-sqrt(3)*dt), cc, 0, 1);
	case 4: //By
	  Field.setVal(0, cc, 0, 1);
	case 5: //Bz
	  Field.setVal(-sqrt(3)*cos(x+y+zd-sqrt(3)*dt), cc, 0, 1);
	} 
      }
    }
  }

  // at time 0 sol = new
  // arguments: Fab to copy, sorce comp, copy comp, number of comps
  Field_sol.copy(Field,0,0,1);
  
 }

//------------------------------------------------------------------------------
// update E and B
void update_E_B(IntVect lo,
	        IntVect hi,
		FArrayBox &Field_out,
	        FArrayBox &Field_out_sol,
	        FArrayBox &Field_in_1,
		FArrayBox &Field_in_2,
		const double dx[3],
		Real dt,
		const double  prob_lo[3],
	        Real time,
		int component,
		IndexType typ) {
 
  int i,j,k;
  double x,y,z;
  double xd, yd, zd; //dual
  Real c1[1];
  Real c2[1];
  Real c3[1];
  Real c4[1];
  Real c5[1];
  Real btime = time + dt/2.0;

  for(int k=lo[2]; k<=hi[2]; k++){
    z = prob_lo[2] + ((double)k+0.5) * dx[2];
    zd = prob_lo[2] + (double)k * dx[2];
    for(int j=lo[1]; j<=hi[1]; j++){
      y = prob_lo[1] + ((double)j+0.5) * dx[1];
      yd = prob_lo[1] + (double)j * dx[1];
      for(int i=lo[0]; i<=hi[0]; i++){
	x = prob_lo[0] + ((double)i+0.5) * dx[0];
	xd = prob_lo[0] + (double)i * dx[0];

	// the box for these values:
	Box cc({i,j,k}, {i,j,k}, typ);

	// old field_out value
	Field_out.getVal(c1, { i, j,k},0,1);

	// get field_in values
	switch (component) {
	case 0: //Ex
	  Field_in_1.getVal(c2, {  i,  j,  k},0,1);
	  Field_in_1.getVal(c3, {  i,j-1,  k},0,1);
	  Field_in_2.getVal(c4, {  i,  j,  k},0,1);
	  Field_in_2.getVal(c5, {  i,  j,k-1},0,1);

	  Field_out_sol.setVal(cos(x+yd+zd-sqrt(3)*time), cc, 0, 1);
	case 1: //Ey
	  Field_in_1.getVal(c2, {  i,  j,  k},0,1);
	  Field_in_1.getVal(c3, {  i,  j,k-1},0,1);
	  Field_in_2.getVal(c4, {  i,  j,  k},0,1);
	  Field_in_2.getVal(c5, {i-1,  j,  k},0,1);

	  Field_out_sol.setVal(-2*cos(xd+y+zd-sqrt(3)*time), cc, 0, 1);
	case 2: //Ez
	  Field_in_1.getVal(c2, {  i,  j,  k},0,1);
	  Field_in_1.getVal(c3, {i-1,  j,  k},0,1);
	  Field_in_2.getVal(c4, {i  ,  j,  k},0,1);
	  Field_in_2.getVal(c5, {  i,j-1,  k},0,1);

	  Field_out_sol.setVal(cos(xd+yd+z-sqrt(3)*time), cc, 0, 1);
	case 3: //Bx
	  Field_in_1.getVal(c2, {  i,j+1,  k},0,1);
	  Field_in_1.getVal(c3, {  i,  j,  k},0,1);
	  Field_in_2.getVal(c4, {  i,  j,k+1},0,1);
	  Field_in_2.getVal(c5, {  i,  j,  k},0,1);
	  dt = -dt;

	  Field_out_sol.setVal(sqrt(3)*cos(xd+y+z-sqrt(3)*btime), cc, 0, 1);
	case 4: //By
	  Field_in_1.getVal(c2, {  i,  j,k+1},0,1);
	  Field_in_1.getVal(c3, {  i,  j,  k},0,1);
	  Field_in_2.getVal(c4, {i+1,  j,  k},0,1);
	  Field_in_2.getVal(c5, {  i,  j,  k},0,1);
	  dt = -dt;

	  Field_out_sol.setVal(0, cc, 0, 1);
	case 5: //Bz
	  Field_in_1.getVal(c2, {i+1,  j,  k},0,1);
	  Field_in_1.getVal(c3, {  i,  j,  k},0,1);
	  Field_in_2.getVal(c4, {  i,j+1,  k},0,1);
	  Field_in_2.getVal(c5, {  i,  j,  k},0,1);
	  dt = -dt;

	  Field_out_sol.setVal(-sqrt(3)*cos(xd+y+z-sqrt(3)*btime), cc, 0, 1);
	}

	Field_out.setVal(*c1+dt*((*c2-*c3)/dx[(component+1)%3]-
				 (*c4-*c5)/dx[(component+2)%3]), cc, 0, 1);
	
      }
    }
  }
}

//------------------------------------------------------------------------------
// advance
void advance (array< unique_ptr<MultiFab>, 3> &E_Array,
	      array< unique_ptr<MultiFab>, 3> &E_sol_Array,
	      array< unique_ptr<MultiFab>, 3> &B_Array,
	      array< unique_ptr<MultiFab>, 3> &B_sol_Array,
	      array< unique_ptr<IndexType>, 3> &E_Index,
	      array< unique_ptr<IndexType>, 3> &B_Index,
	      Real dt,
	      const Geometry& geom,
	      Real time)
{
  // Fill the ghost cells of each grid from the other grids
  // includes periodic domain boundaries
  for (int i=0; i<=2; i++) {
    (*E_Array[i]).FillBoundary(geom.periodicity());
    (*B_Array[i]).FillBoundary(geom.periodicity());
  }

  for (int i=0; i<=2; i++) {
    for (MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
      const Box& bx = mfi.validbox();

      IntVect lo = {bx.smallEnd()};
      IntVect hi = {bx.bigEnd()};

      int component = i;
      update_E_B(lo, hi, (*E_Array[i])[mfi], (*E_sol_Array[i])[mfi],
		 (*B_Array[(i+2)%3])[mfi], (*B_Array[(i+1)%3])[mfi], geom.CellSize(),
		 dt, geom.ProbLo(), time, component, *E_Index[i]);
    }
  }

  for (int i=0; i<=2; i++) {
    for (MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi ) {
      const Box& bx = mfi.validbox();

      IntVect lo = {bx.smallEnd()};
      IntVect hi = {bx.bigEnd()};

      int component = i+3;
      update_E_B(lo, hi, (*B_Array[i])[mfi], (*B_sol_Array[i])[mfi],
      		 (*E_Array[(i+2)%3])[mfi], (*E_Array[(i+1)%3])[mfi], geom.CellSize(),
      		 dt, geom.ProbLo(), time, component, *B_Index[i]);
    }
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
    n_cell = 64;
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

  BoxArray grid;
  Geometry geom;
  
    // grid box
    IntVect dom_lo = {       0,        0,        0};
    IntVect dom_hi = {n_cell-1, n_cell-1, n_cell-1};

    // nodal flags for all fields
    array<unique_ptr<IndexType>, 3> E_Index;
    E_Index[0].reset(new IndexType ({0,1,1}));
    E_Index[1].reset(new IndexType ({1,0,1}));
    E_Index[2].reset(new IndexType ({1,1,0}));

    array<unique_ptr<IndexType>, 3> B_Index;
    B_Index[0].reset(new IndexType ({1,0,0}));
    B_Index[1].reset(new IndexType ({0,1,0}));
    B_Index[2].reset(new IndexType ({0,0,1}));

    Box domain(dom_lo, dom_hi); // cell-centered box for base-grid

    // individual boxes (boxarray)
    grid.define(domain);
    
    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    grid.maxSize(max_grid_size);

    // physical box (geometry)
    double twopi = 2.0*3.14159265359;
    
    RealBox real_box({0.0, 0.0, 0.0},
		     {twopi, twopi, twopi});

    // define geoms
    geom.define(domain,&real_box,CoordSys::cartesian,is_periodic);

    int Nghost = 1; // number of ghost cells for each array
    int Ncomp = 1;  // number of components for each array

    // distribution mapping: how Boxes are distrubuted among MPI processes
    DistributionMapping distriMap(grid);

//------------------------------------------------------------------------------
    // time
  const Real* dx = geom.CellSize();
  Real dt = 0.9*dx[0]*dx[0] / (2.0*3);
  Real time = 0.0;
  
//------------------------------------------------------------------------------
  // Initializing E & B
  
  // we allocate E/B multifabs; one with the new state,
  // and one with the analytic solution.
  array< unique_ptr<MultiFab>, 3 > E_Array;
  array< unique_ptr<MultiFab>, 3 > E_sol_Array;
  array< unique_ptr<MultiFab>, 3 > B_Array;
  array< unique_ptr<MultiFab>, 3 > B_sol_Array;

  for (int i = 0; i<=2; i++) {
    E_Array[i].reset(new MultiFab (convert(grid, *E_Index[i]),distriMap,Ncomp,Nghost));
    E_sol_Array[i].reset(new MultiFab (convert(grid, *E_Index[i]),distriMap,Ncomp,Nghost));
    B_Array[i].reset(new MultiFab (convert(grid, *B_Index[i]),distriMap,Ncomp,Nghost));
    B_sol_Array[i].reset(new MultiFab (convert(grid, *B_Index[i]),distriMap,Ncomp,Nghost));
  }
  
  // MFIter = MultiFab Iterator: to initialize we iterate over each of the
  // MultiFabs individually

  // E
  for (int i = 0; i<=2; i++) {
    for ( MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();

        IntVect lo = {bx.smallEnd()};
        IntVect hi = {bx.bigEnd()};

	int component = i;
        init_E_B(lo, hi, (*E_Array[i])[mfi], (*E_sol_Array[i])[mfi], geom.CellSize(), geom.ProbLo(),
		 component, *E_Index[i], dt);
    }
  }

  // B
  for (int i = 0; i<=2; i++) {
    for ( MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();

        IntVect lo = {bx.smallEnd()};
        IntVect hi = {bx.bigEnd()};

	int component = i+3;
        init_E_B(lo, hi, (*B_Array[i])[mfi], (*B_sol_Array[i])[mfi], geom.CellSize(), geom.ProbLo(),
		 component, *B_Index[i], dt);
    }
  }
  
    
//------------------------------------------------------------------------------
    // time loop

    array<Real, 3 > E_error;
    array<Real, 3 > B_error;
    
    for (int n = 1; n <= nsteps; ++n)
    {
	// advance E & B
	advance(E_Array, E_sol_Array,
		B_Array, B_sol_Array,
		E_Index, B_Index,
		dt, geom, time+dt);

	//advance time
	time = time + dt;

	// ERROR
	// Subtract result from sol (Fab, start comp, num of comp, nghost):
	for (int i=0; i<=2; i++){
	  (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
	  E_error[i] = (*E_sol_Array[i]).norm0();
	  
	  (*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
	  B_error[i] = (*B_sol_Array[i]).norm0();
	}
	
	cout << "step " << n << endl;
	cout << "Ex error: " << E_error[0] << " |Ey error: " << E_error[1] <<
	  " |Ez error: " << E_error[2] << endl;
	cout << "Bx error: " << B_error[0] << " |By error: " << B_error[1] <<
	  " |Bz error: " << B_error[2] << endl;
      
    }
  
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    main_main();
    
    amrex::Finalize();
}




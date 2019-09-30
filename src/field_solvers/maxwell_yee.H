#include <AMReX.H>
#include <AMReX_Print.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <AMReX_CoordSys.H>
#include <AMReX_MFIter.H>

class maxwell_yee
{
  public:

  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int nsteps; // number of simulation steps
  int is_periodic[3];
  amrex::RealBox real_box; // domain like {min_x,min_y,min_z}{max_x,max_y,max_z}
  std::array<std::unique_ptr<amrex::MultiFab>, 3 > E_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, 3 > B_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, 3 > E_sol_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, 3 > B_sol_Array;

  private:
  
  amrex::BoxArray grid;
  amrex::Geometry geom;
  amrex::IntVect dom_lo;
  amrex::IntVect dom_hi;
  std::array<std::unique_ptr<amrex::IndexType>, 3> E_Index;
  std::array<std::unique_ptr<amrex::IndexType>, 3> B_Index;
  amrex::Box domain;
  int Nghost;
  int Ncomp;
  amrex::DistributionMapping distriMap;
  const amrex::Real* dx;
  amrex::Real dt;
  amrex::Real time;

  public:

  // The default constructor.
  maxwell_yee ();

  // Explicit const
  maxwell_yee (int nc, int max_s, int ns, int is_per[], amrex::RealBox rb){
    n_cell = nc;
    max_grid_size = max_s;
    nsteps = ns;
    is_periodic[0] = is_per[0];
    is_periodic[1] = is_per[1];
    is_periodic[2] = is_per[2];
    real_box = rb;

    dom_lo = {0, 0, 0};
    dom_hi = {n_cell-1, n_cell-1, n_cell-1};
    
    E_Index[0].reset(new amrex::IndexType ({0,1,1}));
    E_Index[1].reset(new amrex::IndexType ({1,0,1}));
    E_Index[2].reset(new amrex::IndexType ({1,1,0}));

    B_Index[0].reset(new amrex::IndexType ({1,0,0}));
    B_Index[1].reset(new amrex::IndexType ({0,1,0}));
    B_Index[2].reset(new amrex::IndexType ({0,0,1}));

    amrex::Box domain(dom_lo, dom_hi);
    grid.define(domain);
    grid.maxSize(max_grid_size);
    geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic);

    Nghost = 1;
    Ncomp = 1;

    amrex::DistributionMapping distriMap(grid);
    dx = geom.CellSize();
    dt = 0.001; //0.9*dx[0]*dx[0] / (2.0*3);
    time = 0.0;

    for (int i = 0; i<=2; i++) {
      E_Array[i].reset(new amrex::MultiFab (convert(grid, *E_Index[i]),distriMap,Ncomp,Nghost));
      B_Array[i].reset(new amrex::MultiFab (convert(grid, *B_Index[i]),distriMap,Ncomp,Nghost));

      E_sol_Array[i].reset(new amrex::MultiFab (convert(grid, *E_Index[i]),distriMap,Ncomp,Nghost));
      B_sol_Array[i].reset(new amrex::MultiFab (convert(grid, *B_Index[i]),distriMap,Ncomp,Nghost));
    }
  }

  void init_E_B (double (*fields[6]) (double x,double y,double z, double t)){

    int k,j,l;
    double x,y,z;
    double half_dt = dt/2.0; // for leap frog initialization

    // E_field
    for (int i = 0; i<=2; i++) {
      for ( amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi )
      {
	const amrex::Box& bx = mfi.validbox();
	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

	for(int k=lo[2]; k<=hi[2]; k++){
	  z = geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*dx[2];
	  for(int j=lo[1]; j<=hi[1]; j++){
	    y = geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*dx[1];
	    for(int l=lo[0]; l<=hi[0]; l++){
	      x = geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*dx[0];
	      // the box for these values:
	      amrex::Box cc({l,j,k}, {l,j,k}, *E_Index[i]);
	      (*E_Array[i])[mfi].setVal((*fields[i])(x,y,z,0), cc, 0, 1);
	    }
	  }
	}
	// next mfi
      }
    }


    // B_field
    for (int i = 0; i<=2; i++) {
      for ( amrex::MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi )
      {
	const amrex::Box& bx = mfi.validbox();
	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};
	
	for(int k=lo[2]; k<=hi[2]; k++){
	  z = geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*dx[2];
	  for(int j=lo[1]; j<=hi[1]; j++){
	    y = geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*dx[1];
	    for(int l=lo[0]; l<=hi[0]; l++){
	      x = geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*dx[0];
	      // the box for these values:
	      amrex::Box cc({l,j,k}, {l,j,k}, *B_Index[i]);
	      (*B_Array[i])[mfi].setVal((*fields[i+3])(x,y,z,half_dt), cc, 0, 1);
	    }
	  }
	}
	// next mfi
      }
    }
  };
  
  
  void advance (){

    int l,j,k;
    amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1];

    time = time+dt;
    
    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    for (int i=0; i<=2; i++) {
      (*E_Array[i]).FillBoundary(geom.periodicity());
      (*B_Array[i]).FillBoundary(geom.periodicity());
    }

    // E-field
    for (int i=0; i<=2; i++) {
      for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
	const amrex::Box& bx = mfi.validbox();

	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

	for(int k=lo[2]; k<=hi[2]; k++){
	  for(int j=lo[1]; j<=hi[1]; j++){
	    for(int l=lo[0]; l<=hi[0]; l++){

	      // the box for these values:
	      amrex::Box cc({l,j,k}, {l,j,k}, *E_Index[i]);

	      // old field_out value
	      (*E_Array[i])[mfi].getVal(c1, {l,j,k},0,1);

	      // B-derivatives
	      (*B_Array[(i+2)%3])[mfi].getVal(c2,{  l,  j  ,k},0,1);
	      (*B_Array[(i+2)%3])[mfi].getVal(c3,{l-(i%3==2?1:0),j-(i%3==0?1:0),k-(i%3==1?1:0)},0,1);
	      (*B_Array[(i+1)%3])[mfi].getVal(c4, {  l,  j,  k},0,1);
	      (*B_Array[(i+1)%3])[mfi].getVal(c5,{l-(i%3==1?1:0),j-(i%3==2?1:0),k-(i%3==0?1:0)},0,1);

	      (*E_Array[i])[mfi].setVal(*c1+dt*((*c2-*c3)/dx[(i+1)%3]-
						(*c4-*c5)/dx[(i+2)%3]), cc, 0, 1);
	    }
	  }
	}
	// next mfi
      }
    }

    // B-field
    for (int i=0; i<=2; i++) {
      for (amrex::MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi ) {
	const amrex::Box& bx = mfi.validbox();

	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

	for(int k=lo[2]; k<=hi[2]; k++){
	  for(int j=lo[1]; j<=hi[1]; j++){
	    for(int l=lo[0]; l<=hi[0]; l++){

	      // the box for these values:
	      amrex::Box cc({l,j,k}, {l,j,k}, *B_Index[i]);

	      // old field_out value
	      (*B_Array[i])[mfi].getVal(c1, {l,j,k},0,1);

	      // E-derivatives
	      (*E_Array[(i+2)%3])[mfi].getVal(c2,{l+(i%3==2?1:0),j+(i%3==0?1:0),k+(i%3==1?1:0)},0,1);
	      (*E_Array[(i+2)%3])[mfi].getVal(c3,{  l,  j  ,k},0,1);
	      (*E_Array[(i+1)%3])[mfi].getVal(c4,{l+(i%3==1?1:0),j+(i%3==2?1:0),k+(i%3==0?1:0)},0,1);
	      (*E_Array[(i+1)%3])[mfi].getVal(c5, {  l,  j,  k},0,1);

	      (*B_Array[i])[mfi].setVal(*c1-dt*((*c2-*c3)/dx[(i+1)%3]-
						(*c4-*c5)/dx[(i+2)%3]), cc, 0, 1);
	    }
	  }
	}
	// next mfi
      }
    }  
  };
  
  std::array<amrex::Real,6> computeError(double (*fields[6]) (double x,double y,double z, double t)){
    std::array<amrex::Real,6> E_B_error;
    
    exactSolution(fields);

    for (int i=0; i<=2; i++){
      (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
      E_B_error[i] = (*E_sol_Array[i]).norm0();

      (*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
      E_B_error[i+3] = (*B_sol_Array[i]).norm0();
    }
    return(E_B_error);
  };

  private:

  void exactSolution(double (*fields[6]) (double x,double y,double z, double t)){
    int l,j,k;
    double x,y,z;
    amrex::Real btime = time + dt/2.0;

    // E-field
    for (int i=0; i<=2; i++) {
      for (amrex::MFIter mfi(*E_sol_Array[i]); mfi.isValid(); ++mfi ) {
	const amrex::Box& bx = mfi.validbox();

	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

	for(int k=lo[2]; k<=hi[2]; k++){
	  z = geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*dx[2];
	  for(int j=lo[1]; j<=hi[1]; j++){
	    y = geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*dx[1];
	    for(int l=lo[0]; l<=hi[0]; l++){
	      x = geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*dx[0];

	      // the box for these values:
	      amrex::Box cc({l,j,k}, {l,j,k}, *E_Index[i]);
	      (*E_sol_Array[i])[mfi].setVal((*fields[i])(x,y,z,time), cc, 0, 1);
	    }
	  }
	}
	// next mfi
      }
    }

    // B-field
    for (int i=0; i<=2; i++) {
      for (amrex::MFIter mfi(*B_sol_Array[i]); mfi.isValid(); ++mfi ) {
	const amrex::Box& bx = mfi.validbox();

	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

	for(int k=lo[2]; k<=hi[2]; k++){
	  z = geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*dx[2];
	  for(int j=lo[1]; j<=hi[1]; j++){
	    y = geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*dx[1];
	    for(int l=lo[0]; l<=hi[0]; l++){
	      x = geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*dx[0];

	      // the box for these values:
	      amrex::Box cc({l,j,k}, {l,j,k}, *B_Index[i]);
	      (*B_sol_Array[i])[mfi].setVal((*fields[i+3])(x,y,z,btime), cc, 0, 1);
	    }
	  }
	}
	// next mfi
      }
    }  
  };
  
};

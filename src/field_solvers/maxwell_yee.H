#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_ParmParse.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MLMG.H>

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
#include <infrastructure.H>

#ifndef MAXWELL_YEE_H_
#define MAXWELL_YEE_H_

class maxwell_yee
{
  public:

  int nsteps; // number of simulation steps
  amrex::Real dt;
  std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_SPACEDIM > E_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_SPACEDIM > J_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_SPACEDIM > B_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_SPACEDIM > E_sol_Array;
  std::array<std::unique_ptr<amrex::MultiFab>, GEMPIC_SPACEDIM > B_sol_Array;
  std::array<std::unique_ptr<amrex::IndexType>, GEMPIC_SPACEDIM> E_Index;
  std::array<std::unique_ptr<amrex::IndexType>, GEMPIC_SPACEDIM> B_Index;
  amrex::MultiFab phi_init; // for Poisson
  amrex::MultiFab rho; // for Poisson
  int Nghost;
  amrex::MultiFab phi; // for Poisson
  
  private:
  
  int Ncomp;
  amrex::Real time;
  amrex::MultiFab sigma; //for Poisson

  public:

  // The default constructor.
  maxwell_yee ();

  // Explicit const
  maxwell_yee (initializer init, infrastructure ifr){
    nsteps = init.n_steps;

    // set index indicating nodal-/cell-centered
    // 0 -> cell-centered | 1 -> node-centered

    E_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,1,1)}));
#if (GEMPIC_SPACEDIM > 1)
    E_Index[1].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,0,1)}));
#endif
#if (GEMPIC_SPACEDIM > 2)
    E_Index[2].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,1,0)}));
#endif

    B_Index[0].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(1,0,0)}));
#if (GEMPIC_SPACEDIM > 1)
    B_Index[1].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,1,0)}));
#endif
#if (GEMPIC_SPACEDIM > 2)
    B_Index[2].reset(new amrex::IndexType (amrex::IntVect{AMREX_D_DECL(0,0,1)}));
#endif

    Nghost = 1;
    Ncomp = 1;

    dt = init.dt;
    time = 0.0;

    // make the multifabs nodal-/cell-centered according to indices
    for (int i = 0; i<GEMPIC_SPACEDIM; i++) {
      E_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
      J_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
      B_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *B_Index[i]),ifr.distriMap,Ncomp,Nghost));

      E_sol_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *E_Index[i]),ifr.distriMap,Ncomp,Nghost));
      B_sol_Array[i].reset(new amrex::MultiFab (convert(ifr.grid, *B_Index[i]),ifr.distriMap,Ncomp,Nghost));
    }

    // for Poisson:
    //node centered BA:
    const amrex::BoxArray& nba = amrex::convert(ifr.grid,amrex::IntVect::TheNodeVector());
    phi     .define(nba        , ifr.distriMap, 1, Nghost); //box, distriMap, nVar, nGrow
    rho     .define(nba        , ifr.distriMap, 1, Nghost);
    phi_init.define(nba        , ifr.distriMap, 1, Nghost);
    // sigma is cell-centered (default, so no need to pass IndexVector)
    sigma   .define(ifr.grid, ifr.distriMap, 1, 0);
    sigma.setVal(1.0); // sigma is the identity (lapl = nabla dot ID nabla)
  }

  void init_E_B (double (*fields[2*GEMPIC_SPACEDIM]) (std::array<double,GEMPIC_SPACEDIM> x, double t),
		 infrastructure ifr){

    std::array<double,GEMPIC_SPACEDIM> x;
    double half_dt = dt/2.0; // for leap frog initialization

    // E_field
    for (int i = 0; i<GEMPIC_SPACEDIM; i++) {
      for ( amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi )
      {
	const amrex::Box& bx = mfi.validbox();
	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
	for(int k=lo[2]; k<=hi[2]; k++){
      x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
	  for(int j=lo[1]; j<=hi[1]; j++){
        x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*ifr.dx[1];
#endif
	    for(int l=lo[0]; l<=hi[0]; l++){
          x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*ifr.dx[0];
	      // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *E_Index[i]);
          (*E_Array[i])[mfi].setVal((*fields[i])(x,0), cc, 0, 1);
	    }
#if (GEMPIC_SPACEDIM > 1)
	  }
#endif
#if (GEMPIC_SPACEDIM > 2)
	}
#endif
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

#if (GEMPIC_SPACEDIM > 2)
	for(int k=lo[2]; k<=hi[2]; k++){
      x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
	  for(int j=lo[1]; j<=hi[1]; j++){
        x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*ifr.dx[1];
#endif
	    for(int l=lo[0]; l<=hi[0]; l++){
          x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*ifr.dx[0];
	      // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *B_Index[i]);
          (*B_Array[i])[mfi].setVal((*fields[i+GEMPIC_SPACEDIM])(x,half_dt), cc, 0, 1);
	    }
#if (GEMPIC_SPACEDIM > 1)
	  }
#endif
#if (GEMPIC_SPACEDIM > 2)
	}
#endif
	// next mfi
      }
    }
  }

  void FillBD (infrastructure ifr){
      for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
          (*E_Array[cc]).FillBoundary(ifr.geom.periodicity());
      }
      for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
          (*J_Array[cc]).FillBoundary(ifr.geom.periodicity());
      }
      for(int cc=0;cc<GEMPIC_SPACEDIM;cc++){
          (*B_Array[cc]).FillBoundary(ifr.geom.periodicity());
      }
      rho.FillBoundary(ifr.geom.periodicity());
  }
  
  void advance_time () {
      time = time+dt;
  }

  void advance_E (infrastructure ifr, amrex::Real coefficient){

    amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1], cj[1];

    // E-field
    for (int i=0; i<GEMPIC_SPACEDIM; i++) {
      for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
    const amrex::Box& bx = mfi.validbox();

    amrex::IntVect lo = {bx.smallEnd()};
    amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
    for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
      for(int j=lo[1]; j<=hi[1]; j++){
#endif
        for(int l=lo[0]; l<=hi[0]; l++){

          // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *E_Index[i]);

          // old field_out value
          (*E_Array[i])[mfi].getVal(c1, {AMREX_D_DECL(l,j,k)},0,1);
          // J value
          (*J_Array[i])[mfi].getVal(cj, {AMREX_D_DECL(l,j,k)},0,1);

          // B-derivatives
#if (GEMPIC_SPACEDIM == 1)
          (*B_Array[0])[mfi].getVal(c2,{ l },0,1);
          (*B_Array[0])[mfi].getVal(c3,{l-1},0,1);
          (*E_Array[0])[mfi].setVal(*c1+dt*coefficient*((*c3-*c2)/ifr.dx[0]-*cj), cc, 0, 1);
#elif (GEMPIC_SPACEDIM == 2)
#elif (GEMPIC_SPACEDIM == 3)
          (*B_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c2,{  l,  j  ,k},0,1);
          (*B_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c3,{l-(i%GEMPIC_SPACEDIM==2?1:0),j-(i%GEMPIC_SPACEDIM==0?1:0),k-(i%GEMPIC_SPACEDIM==1?1:0)},0,1);
          (*B_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c4, {  l,  j,  k},0,1);
          (*B_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c5,{l-(i%GEMPIC_SPACEDIM==1?1:0),j-(i%GEMPIC_SPACEDIM==2?1:0),k-(i%GEMPIC_SPACEDIM==0?1:0)},0,1);

          (*E_Array[i])[mfi].setVal(*c1+dt*coefficient*((*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]-
                        (*c4-*c5)/ifr.dx[(i+2)%GEMPIC_SPACEDIM]-*cj), cc, 0, 1);
#endif
        }
#if (GEMPIC_SPACEDIM > 1)
      }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    // next mfi
      }
    }
  }

  void advance_B (infrastructure ifr, amrex::Real coefficient){

    amrex::Real c1[1], c2[1], c3[1], c4[1], c5[1];

    // B-field
    for (int i=0; i<GEMPIC_SPACEDIM; i++) {
      for (amrex::MFIter mfi(*B_Array[i]); mfi.isValid(); ++mfi ) {
    const amrex::Box& bx = mfi.validbox();

    amrex::IntVect lo = {bx.smallEnd()};
    amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
    for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
      for(int j=lo[1]; j<=hi[1]; j++){
#endif
        for(int l=lo[0]; l<=hi[0]; l++){

          // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *B_Index[i]);

          // old field_out value
          (*B_Array[i])[mfi].getVal(c1, {AMREX_D_DECL(l,j,k)},0,1);

          // E-derivatives
#if (GEMPIC_SPACEDIM == 1)
          (*E_Array[0])[mfi].getVal(c2,{ l },0,1);
          (*E_Array[0])[mfi].getVal(c3,{l+1},0,1);
          (*B_Array[i])[mfi].setVal(*c1+dt*coefficient*((*c2-*c3)/ifr.dx[0]), cc, 0, 1);
#elif (GEMPIC_SPACEDIM == 2)
#elif (GEMPIC_SPACEDIM == 3)
          (*E_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c2,{l+(i%GEMPIC_SPACEDIM==2?1:0),j+(i%GEMPIC_SPACEDIM==0?1:0),k+(i%GEMPIC_SPACEDIM==1?1:0)},0,1);
          (*E_Array[(i+2)%GEMPIC_SPACEDIM])[mfi].getVal(c3,{  l,  j  ,k},0,1);
          (*E_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c4,{l+(i%GEMPIC_SPACEDIM==1?1:0),j+(i%GEMPIC_SPACEDIM==2?1:0),k+(i%GEMPIC_SPACEDIM==0?1:0)},0,1);
          (*E_Array[(i+1)%GEMPIC_SPACEDIM])[mfi].getVal(c5, {  l,  j,  k},0,1);

          (*B_Array[i])[mfi].setVal(*c1-dt*coefficient*((*c2-*c3)/ifr.dx[(i+1)%GEMPIC_SPACEDIM]-
                        (*c4-*c5)/ifr.dx[(i+2)%GEMPIC_SPACEDIM]), cc, 0, 1);
#endif
        }
#if (GEMPIC_SPACEDIM > 1)
      }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    // next mfi
      }
    }
  }

  void advance (infrastructure ifr){

    time = time+dt;
    FillBD(ifr);

    advance_E(ifr, 1.);
    advance_B(ifr, 1.);
  }

  void init_rho_phi(double phi_fun(std::array<double,GEMPIC_SPACEDIM> x),
            double rho_fun(std::array<double,GEMPIC_SPACEDIM> x),
		    infrastructure ifr){

      for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi ) {
          const amrex::Box& bx = mfi.validbox();

          amrex::IntVect lo = {bx.smallEnd()};
          amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
    for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
      for(int j=lo[1]; j<=hi[1]; j++){
#endif
        for(int l=lo[0]; l<=hi[0]; l++){

          // the box for these values:
          std::array<double,GEMPIC_SPACEDIM> x = {AMREX_D_DECL(l*ifr.dx[0], j*ifr.dx[1], k*ifr.dx[2])};
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, amrex::IntVect::TheNodeVector());

          phi[mfi].setVal(phi_fun(x), cc, 0, 1);
          rho[mfi].setVal(rho_fun(x), cc, 0, 1);
        }
#if (GEMPIC_SPACEDIM > 1)
      }
#endif
#if (GEMPIC_SPACEDIM > 2)
    }
#endif
    // next mfi
      }

    // box with all interior nodes:
    const amrex::Box& interior = amrex::surroundingNodes(amrex::grow(ifr.geom.Domain(), -1));
    phi.setVal(0.0, interior, 0, 1, 0); // set sol to 0 on interior
  }

  void solve_poisson(infrastructure ifr){
    //--------------------------------------------------------------------------------
    // rho
    
    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 100;
    int max_fmg_iter = 0;
    amrex::Real reltol = 1.e-11;
    
    sigma.setVal(1.0); // sigma is the identity (lapl = nabla dot ID nabla)
    amrex::MLNodeLaplacian linop({ifr.geom}, {ifr.grid}, {ifr.distriMap}); // linear operator class

    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, // for lower ends
	               amrex::LinOpBCType::Periodic,
                       amrex::LinOpBCType::Periodic)},
                      {AMREX_D_DECL(amrex::LinOpBCType::Periodic, // for higher ends
                       amrex::LinOpBCType::Periodic,
                       amrex::LinOpBCType::Periodic)});

    linop.setSigma(0, sigma); // first argument: level-nr

    amrex::MLMG mlmg(linop); // solver class
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    //mlmg.setBottomSolver (amrex::MLMG::BottomSolver::smoother);
  
    mlmg.solve({&phi}, {&rho}, reltol, 0.0);

    //----------------------------------------------------------------------------------
    // E

    amrex::Real c1[1], c2[1];
    
    for (int i=0; i<GEMPIC_SPACEDIM; i++) {
      for (amrex::MFIter mfi(*E_Array[i]); mfi.isValid(); ++mfi ) {
	const amrex::Box& bx = mfi.validbox();

	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
	for(int k=lo[2]; k<=hi[2]; k++){
#endif
#if (GEMPIC_SPACEDIM > 1)
	  for(int j=lo[1]; j<=hi[1]; j++){
#endif
	    for(int l=lo[0]; l<=hi[0]; l++){

	      // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *E_Index[i]);

	      // old field_out value
          (*E_Array[i])[mfi].getVal(c1, {AMREX_D_DECL(l,j,k)},0,1);

	      // phi-derivative
          phi[mfi].getVal(c1,{AMREX_D_DECL(l+(i%GEMPIC_SPACEDIM==0?1:0),j+(i%GEMPIC_SPACEDIM==1?1:0),k+(i%GEMPIC_SPACEDIM==2?1:0))},0,1);
          phi[mfi].getVal(c2,{AMREX_D_DECL(l,j,k)},0,1);

	      (*E_Array[i])[mfi].setVal((*c1-*c2)/ifr.dx[i], cc, 0, 1);
	    }
#if (GEMPIC_SPACEDIM > 2)
	  }
#endif
#if (GEMPIC_SPACEDIM > 1)
	}
#endif
	// next mfi
      }
    }
    
  }
  
  std::array<amrex::Real,2*GEMPIC_SPACEDIM> computeError(double (*fields[]) (std::array<double,GEMPIC_SPACEDIM> x, double t),
					 bool bool_B,
					 infrastructure ifr){
    std::array<amrex::Real,2*GEMPIC_SPACEDIM> E_B_error;
    
    exactSolution(fields, bool_B, ifr);

    for (int i=0; i<=2; i++){
      (*E_sol_Array[i]).minus(*E_Array[i], 0, 1, 0);
      E_B_error[i] = (*E_sol_Array[i]).norm0();

      if (bool_B) {
	(*B_sol_Array[i]).minus(*B_Array[i], 0, 1, 0);
    E_B_error[i+GEMPIC_SPACEDIM] = (*B_sol_Array[i]).norm0();
      }
    }
    return(E_B_error);
  }

  private:

  void exactSolution(double (*fields[]) (std::array<double,GEMPIC_SPACEDIM> x, double t),
		     bool bool_B,
		     infrastructure ifr){
    std::array<double,GEMPIC_SPACEDIM> x;
    amrex::Real btime = time + dt/2.0;

    // E-field
    for (int i=0; i<GEMPIC_SPACEDIM; i++) {
      for (amrex::MFIter mfi(*E_sol_Array[i]); mfi.isValid(); ++mfi ) {
	const amrex::Box& bx = mfi.validbox();

	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
	for(int k=lo[2]; k<=hi[2]; k++){
      x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*E_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
	  for(int j=lo[1]; j<=hi[1]; j++){
        x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*E_Index[i])[1])*ifr.dx[1];
#endif
	    for(int l=lo[0]; l<=hi[0]; l++){
          x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*E_Index[i])[0])*ifr.dx[0];

	      // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *E_Index[i]);
          (*E_sol_Array[i])[mfi].setVal((*fields[i])(x,time), cc, 0, 1);
	    }
#if (GEMPIC_SPACEDIM > 1)
	  }
#endif
#if (GEMPIC_SPACEDIM > 2)
	}
#endif
	// next mfi
      }
    }

    if (bool_B){
      // B-field
      for (int i=0; i<GEMPIC_SPACEDIM; i++) {
        for (amrex::MFIter mfi(*B_sol_Array[i]); mfi.isValid(); ++mfi ) {
	  const amrex::Box& bx = mfi.validbox();

	  amrex::IntVect lo = {bx.smallEnd()};
	  amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
	  for(int k=lo[2]; k<=hi[2]; k++){
        x[2] = ifr.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*B_Index[i])[2])*ifr.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
	    for(int j=lo[1]; j<=hi[1]; j++){
          x[1] = ifr.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*B_Index[i])[1])*ifr.dx[1];
#endif
	      for(int l=lo[0]; l<=hi[0]; l++){
            x[0] = ifr.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*B_Index[i])[0])*ifr.dx[0];

	        // the box for these values:
            amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *B_Index[i]);
            (*B_sol_Array[i])[mfi].setVal((*fields[i+GEMPIC_SPACEDIM])(x,btime), cc, 0, 1);
	      }
#if (GEMPIC_SPACEDIM > 1)
	    }
#endif
#if (GEMPIC_SPACEDIM > 2)
	  }
#endif
	  // next mfi
        }
      }
    }
  }
  
};

#endif

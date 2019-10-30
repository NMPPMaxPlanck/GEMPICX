
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <particle_mesh_coupling.H>
#include <sampler.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;
double weight_fun(double x,double y,double z,double v_x,double v_y,double v_z,int Np){return(1.0/Np);}
void main_main ()
{
  //------------------------------------------------------------------------------
  // Initialize Infrastructure
  
  int n_cell; // number of cells
  int max_grid_size; // maximum number of cells in each direction
  int is_periodic[3]; // periodicity: 1 -> periodic

  n_cell = 8;
  max_grid_size = 4;
  Real dt = 0.01;
  int n_steps = 5;
  // periodic in all directions:
  is_periodic[0] = 1;
  is_periodic[1] = 1;
  is_periodic[2] = 1;

  // physical box (geometry)
  double twopi = 2.0*3.14159265359;
  RealBox real_box({0.0, 0.0, 0.0},
		   {twopi, twopi, twopi});
  // build infrastructure
  infrastructure infra(n_cell, max_grid_size, is_periodic, real_box);

  // Length of domain in x,y,z directions
  array<Real,3> L;
  for(int cc=0;cc<3;cc++){
    L[cc] = (infra.real_box.hi(cc)-infra.real_box.lo(cc));
  }

  //------------------------------------------------------------------------------
  //Initialize Maxwell Yee
  maxwell_yee mw_yee(n_steps, real_box, infra, dt);
  
  //------------------------------------------------------------------------------
  //Initialize Particle Groups
  const int n_species = 1;
  array<Real, n_species> charge = {1.0};
  array<Real, n_species> mass = {1.0};

  particle_groups part_gr(infra, n_species, charge, mass);
  //ppg.n_species;
  MFIter mfi(*mw_yee.B_Array[0]);
  int Np_cell = 1000; //number of particles per cell
  int species = 0; // all particles are same species for now
  init_particles_cellwise(infra, &part_gr, Np_cell, &mfi, species, {0.0}, {1.0}, {0.25}, weight_fun);

//------------------------------------------------------------------------------
  // Set variables for loop

  GpuArray<amrex::Real,3> plo;
  plo[0] = infra.geom.ProbLo()[0];
  plo[1] = infra.geom.ProbLo()[1];
  plo[2] = infra.geom.ProbLo()[2];
  GpuArray<amrex::Real,3> dxi;
  dxi[0] = 1.0/infra.dx[0];
  dxi[1] = 1.0/infra.dx[1];
  dxi[2] = 1.0/infra.dx[2];
  int nc = 3; // number of components for J and E
  
  // initial rho by: deposition of particle charge
  //Set rho to 0
  mw_yee.rho.setVal(0.0, 0); // value and component
    // Deposit charges:
    for (int spec=0;spec<n_species;spec++) {
      (*part_gr.mypc[spec]).Redistribute(); // assign particles to the tile they are in
      for (ParIter<4,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
	const Box& box = pti.validbox();
	auto& particles = pti.GetArrayOfStructs();
	const long np  = pti.numParticles();

	Array4<Real> const& rhoarr = (mw_yee.rho)[pti].array();
	  for (int pp=0;pp<np;pp++) {
	    gempic_deposit_cic(particles[pp], charge[spec], rhoarr, plo, dxi);
	  }
      }
    }

    //assign particles to tiles they are in
    for(int spec=0;spec<n_species;spec++){
      (*part_gr.mypc[spec]).Redistribute();
    }

    //initial E by rho
    mw_yee.phi_init.setVal(0.0, 0);
    //mw_yee.solve_poisson(infra);
    //while poisson solver is not working, set E to 0
    (*mw_yee.E_Array[0]).setVal(0.0, 0);
    (*mw_yee.E_Array[1]).setVal(0.0, 0);
    (*mw_yee.E_Array[2]).setVal(0.0, 0);
    //initial B  is 0
    (*mw_yee.B_Array[0]).setVal(0.0, 0);
    (*mw_yee.B_Array[1]).setVal(0.0, 0);
    (*mw_yee.B_Array[2]).setVal(0.0, 0);
    
//------------------------------------------------------------------------------
  //Time loop
  std::ofstream ofs("PIC.output", std::ofstream::out);
  std::ofstream ofs2("PIC_6dim.output", std::ofstream::out);
  Real time = 0.0;
  for (int t_step=0;t_step<mw_yee.nsteps;t_step++) {
    time += mw_yee.dt;
    
    //--------------------------------------------------------------------------
    //push particles
    mw_yee.FillBD(infra);

    array<Real,3> eres, bres;
    for (int spec=0;spec<n_species;spec++) {
      Real chargemass = charge[spec]/mass[spec];

      for (ParIter<4,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
	const Box& box = pti.validbox();
	auto& particles = pti.GetArrayOfStructs(); // get particles
	const long np  = pti.numParticles();

	// loop over particles
	for (int pp=0;pp<np;pp++){
	  for (int cc=0;cc<nc;cc++){
	    Array4<Real> const& earr = (*mw_yee.E_Array[cc])[pti].array();
	    Array4<Real> const& barr = (*mw_yee.B_Array[cc])[pti].array(); 

	    //E-field at particle position
	    eres[cc] = gempic_interpolate_cic(particles[pp], earr, plo, dxi,
					      *mw_yee.E_Index[cc]);

	    //B-field at particle position
	    bres[cc] = gempic_interpolate_cic(particles[pp], barr, plo, dxi,
					      *mw_yee.B_Index[cc]);
	  }
	  array<Real,6> newPos = push_particle(particles[pp], mw_yee.dt, chargemass, eres, bres, infra);
	  copy(begin(newPos),end(newPos),&particles.data()[pp*8]);
	  Print(ofs2) << newPos[0] << "," << newPos[1] << "," << newPos[2] << "," << newPos[3] << "," << newPos[4] << "," << newPos[5] << endl;
	}
      }
      (*part_gr.mypc[spec]).Redistribute(); // assign particles to the tile they are in
    }
    //--------------------------------------------------------------------------
    //update fields
    
    // deposit charge in J one component and particle at a time

    // set J to 0 before we start depositing values from current particles
    (*mw_yee.J_Array[0]).setVal(0.0, 0); // value and component
    (*mw_yee.J_Array[1]).setVal(0.0, 0);
    (*mw_yee.J_Array[2]).setVal(0.0, 0);
    
    for (int spec=0;spec<n_species;spec++) {
      (*part_gr.mypc[spec]).Redistribute(); // assign particles to the tile they are in
      for (ParIter<4,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
	const Box& box = pti.validbox();
	auto& particles = pti.GetArrayOfStructs();
	const long np  = pti.numParticles();

	for(int cc=0;cc<nc;cc++){
	  Array4<Real> const& jarr = (*mw_yee.J_Array[cc])[pti].array();
	  for (int pp=0;pp<np;pp++) {
	    gempic_deposit_J_cic(particles[pp], charge[spec], cc, jarr, plo,
				 infra.ploE[cc], dxi,
				 *mw_yee.E_Index[cc]);
	  }
	}
      }
    }
    // field updates:
    mw_yee.advance(infra);

    // compute electric energy norms
    array<Real,3> E_n;
    array<Real,3> B_n;
    for(int cc=0;cc<nc;cc++){
      E_n[cc] = infra.dx[cc]*pow((*mw_yee.E_Array[cc]).norm2(),2)/L[cc];
      B_n[cc] = infra.dx[cc]*pow((*mw_yee.B_Array[cc]).norm2(),2)/L[cc];
    }
    cout << E_n[0] << "|" << E_n[1] << "|" << E_n[2] << endl;
    cout << B_n[0] << "|" << B_n[1] << "|" << B_n[2] << endl;

    // compute kinetic energy norm and momentum
    Real kin = 0;
    Real vel;
    array<Real,3> mom = {0.0,0.0,0.0};
    for (int spec=0;spec<n_species;spec++){
      for (ParIter<4,0,0,0> pti(*part_gr.mypc[spec], 0); pti.isValid(); ++pti) {
	auto& particles = pti.GetArrayOfStructs();
	const long np = pti.numParticles();
	for (int pp=0;pp<np;pp++) {
	  vel = abs(particles[pp].rdata(0))+abs(particles[pp].rdata(1))+abs(particles[pp].rdata(2));
	  kin += particles[pp].rdata(3)*pow(vel,2);
	  for (int cmp=0;cmp<3;cmp++) {
	    mom[cmp] += particles[pp].rdata(3)*abs(particles[pp].rdata(cmp));
	  }
	}
      }
    }
    cout << kin << endl;
    cout << "total energy: " << E_n[0]+E_n[1]+E_n[2]+kin << endl;
    Print(ofs) << time << "," << E_n[0] << "," << E_n[1] << "," << E_n[2] << "," << kin << "," << mom[0] << "," << mom[1] << "," << mom[2] << endl;
    
  } // end time loop
  ofs.close();
  ofs2.close();
  
//------------------------------------------------------------------------------
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




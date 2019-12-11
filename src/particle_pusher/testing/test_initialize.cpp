
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <initializer.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (double x,double y,double z,double v_x,double v_y,double v_z,int Np) {
  return(0.0);
}

void main_main ()
{
//------------------------------------------------------------------------------
  // Initialize Infrastructure

  initializer init;
  int is_periodic[3] = {1,1,1};
  int n_cell[3] = {64,64,64};
  init.initialize_from_parameters(n_cell,32,is_periodic,0.01,5,1,{1.0},{1.0},1000,1,
                  {0.0},{1.0},{1.0},WF);
  infrastructure infra(init);

  //need a multifab to be able to iterate later:
  maxwell_yee mw_yee(init, infra);

//------------------------------------------------------------------------------
  //Initialize Particle Groups
  const int n_species = 1;
  array<Real, n_species> charge = {1.0};
  array<Real, n_species> mass = {1.0};
  particle_groups part_gr(init, infra);

  //set particles for first cell (and copies in remaining cells)
  int Np_cell = 100; //number of particles per cell
  int species = 0; // all particles are same species for now
  array<Real,3> position;
  array<Real,3> shifted_position;
  array<Real,3> velocity;
  Real weight;

  // normally distributed random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normD(0,1);

  Real x,y,z;
  
  for (int pp=0;pp<Np_cell;pp++) {
    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = ((Real) rand() / (RAND_MAX))/infra.dx[0];
    position[1] = ((Real) rand() / (RAND_MAX))/infra.dx[1];
    position[2] = ((Real) rand() / (RAND_MAX))/infra.dx[2];

    velocity[0] = normD(gen);
    velocity[1] = normD(gen);
    velocity[2] = normD(gen);
    weight = 1.0;

    //MFI that adds particle from the modell cell to all cells
    for ( MFIter mfi(*mw_yee.B_Array[0]); mfi.isValid(); ++mfi ){
	const amrex::Box& bx = mfi.validbox();
	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};
	for(int k=lo[2]; k<=hi[2]; k++){
	  z = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
	  for(int j=lo[1]; j<=hi[1]; j++){
	    y = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
	    for(int l=lo[0]; l<=hi[0]; l++){
	      x = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
	      shifted_position[0] = position[0] + x;
	      shifted_position[1] = position[1] + y;
	      shifted_position[2] = position[2] + z;
	      part_gr.add_particle(species, shifted_position, velocity, weight);
	    }
	  }
	}
    }
  }
}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);

  main_main();

  amrex::Finalize();
}




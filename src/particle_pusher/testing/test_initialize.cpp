
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <initializer.H>
#include <gempic_Config.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_SPACEDIM> v,int Np,double k) {
  return(0.0);
}

void main_main ()
{
//------------------------------------------------------------------------------
  // Initialize Infrastructure

  initializer init;
  amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
  amrex::IntVect n_cell(AMREX_D_DECL(64,64,64));
  init.initialize_from_parameters(n_cell,32,is_periodic,0.01,5,{1.0},{1.0},1000,1,
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
  std::array<double,GEMPIC_SPACEDIM> position;
  std::array<double,GEMPIC_SPACEDIM> shifted_position;
  std::array<double,GEMPIC_SPACEDIM> velocity;
  Real weight;

  // normally distributed random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normD(0,1);

  std::array<double,GEMPIC_SPACEDIM> x;
  
  for (int pp=0;pp<Np_cell;pp++) {
    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = ((Real) rand() / (RAND_MAX))/infra.dx[0];
#if (GEMPIC_SPACEDIM > 1)
    position[1] = ((Real) rand() / (RAND_MAX))/infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
    position[2] = ((Real) rand() / (RAND_MAX))/infra.dx[2];
#endif

    velocity[0] = normD(gen);
#if (GEMPIC_SPACEDIM > 1)
    velocity[1] = normD(gen);
#endif
#if (GEMPIC_SPACEDIM > 2)
    velocity[2] = normD(gen);
#endif
    weight = 1.0;

    //MFI that adds particle from the modell cell to all cells
    for ( MFIter mfi(*mw_yee.B_Array[0]); mfi.isValid(); ++mfi ){
	const amrex::Box& bx = mfi.validbox();
	amrex::IntVect lo = {bx.smallEnd()};
	amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
	for(int k=lo[2]; k<=hi[2]; k++){
      x[2] = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
      shifted_position[2] = position[2] + x[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
	  for(int j=lo[1]; j<=hi[1]; j++){
        x[1] = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
        shifted_position[1] = position[1] + x[1];
#endif
	    for(int l=lo[0]; l<=hi[0]; l++){
          x[0] = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
          shifted_position[0] = position[0] + x[0];
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




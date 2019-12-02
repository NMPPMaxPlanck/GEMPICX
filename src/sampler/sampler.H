#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <infrastructure.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#ifndef SAMPLER_H_
#define SAMPLER_H_

void init_particles_cellwise(infrastructure infra,
                 particle_groups *ppg,
                             initializer init,
			     int species,
			     amrex::MultiFab *IteratorFab){
  amrex::Real x,y,z;
  std::array<amrex::Real,3> position;
  std::array<amrex::Real,3> shifted_position;
  std::array<amrex::Real,3> velocity;
  amrex::Real weight;
  const int numDistri = init.Vmean.size();

  std::ofstream ofsfun("weights.output", std::ofstream::out);
  // cumulative sum of vel_weights
  std::array<amrex::Real,numDistri> vel_cum;
  std::partial_sum(init.Vweight.begin(), init.Vweight.end(), vel_cum.begin());

  // normally distributed random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::array<std::normal_distribution<>,numDistri> normArray;
  for (int dis=0;dis<numDistri;dis++){
    std::normal_distribution<> normD(init.Vmean[dis],init.Vdev[dis]);
    normArray[dis] = normD;
  }
  
  for (int pp=0;pp<init.Np_cell;pp++) {
    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[0];
    position[1] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[1];
    position[2] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[2];

    amrex::Real gauss_choice = ((amrex::Real) std::rand() / (RAND_MAX));
    int gauss_idx = 0;
    for (int gg=1;gg<numDistri;gg++){
      if (vel_cum[gg-1]-gauss_choice < 0 & vel_cum[gg]-gauss_choice >= 0){
	gauss_idx = gg;
      }
    }
    
    velocity[0] = normArray[gauss_idx](gen);
    velocity[1] = normArray[gauss_idx](gen);
    velocity[2] = normArray[gauss_idx](gen);

    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi(*(IteratorFab)); mfi.isValid(); ++mfi ){
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
        weight = init.weight_fun(shifted_position[0], shifted_position[1], shifted_position[2],
                velocity[0], velocity[1], velocity[2],
                init.Np_cell*pow(infra.n_cell,3)); //DIM
	    (*ppg).add_particle(species, shifted_position, velocity, weight);
        amrex::Print(ofsfun) << weight << "," << shifted_position[0] << std::endl;
	  }
	}
      }
    }
  }

  ofsfun.close();
}


#endif

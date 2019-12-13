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
  //const int numDistri = init.Vmean[0].size();
  const int numDistri = 1;

  std::ofstream ofsfun("weights.output", std::ofstream::out);
  // cumulative sum of vel_weights
  std::array<std::array<amrex::Real,numDistri>,3> vel_cum;
  for(int i=0;i<3;i++) {
      std::partial_sum(init.Vweight[i].begin(), init.Vweight[i].end(), vel_cum[i].begin());
  }

  // normally distributed random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::array<std::array<std::normal_distribution<>,numDistri>,3> normArray;
  for(int i=0;i<3;i++){
      for (int dis=0;dis<numDistri;dis++){
        std::normal_distribution<> normD(init.Vmean[i][dis],init.Vdev[i][dis]);
        normArray[i][dis] = normD;
      }
  }
  
  for (int pp=0;pp<init.Np_cell;pp++) {
    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[0];
    position[1] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[1];
    position[2] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[2];

    amrex::Real gauss_choice[3];
    for(int i=0;i<3;i++){
        gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
    }
    int gauss_idx[3] = {0,0,0};
    for(int i=0;i<3;i++){
        for (int gg=1;gg<numDistri;gg++){
          if (vel_cum[i][gg-1]-gauss_choice[i] < 0 & vel_cum[i][gg]-gauss_choice[i] >= 0){
              gauss_idx[i] = gg;
          }
        }
    }
    
    velocity[0] = normArray[0][gauss_idx[0]](gen);
    velocity[1] = normArray[1][gauss_idx[1]](gen);
    velocity[2] = normArray[2][gauss_idx[2]](gen);

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
                init.Np_cell*infra.n_cell[0]*infra.n_cell[1]*infra.n_cell[2],init.k); //DIM
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

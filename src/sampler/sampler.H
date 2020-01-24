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
                 amrex::MultiFab *IteratorFab,
                             int rSeed = 925){
  amrex::Real x,y,z;
  std::array<amrex::Real,GEMPIC_SPACEDIM> position;
  std::array<amrex::Real,GEMPIC_SPACEDIM> shifted_position;
  std::array<amrex::Real,GEMPIC_VDIM> velocity;
  amrex::Real weight;

  amrex::Real dom_vol = infra.Length[0]
#if (GEMPIC_SPACEDIM > 1)
          *infra.Length[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
          *infra.Length[2]
#endif
          ;

  srand(rSeed);
  // cumulative sum of vel_weights
  std::array<std::array<amrex::Real,GEMPIC_NUMGAUS>,GEMPIC_VDIM> vel_cum;
  for(int i=0;i<GEMPIC_VDIM;i++) {
      std::partial_sum(init.Vweight[i].begin(), init.Vweight[i].end(), vel_cum[i].begin());
  }

  // normally distributed random number generator:
  std::mt19937 gen(rSeed);
  std::array<std::array<std::normal_distribution<>,GEMPIC_NUMGAUS>,GEMPIC_VDIM> normArray;
  for(int i=0;i<GEMPIC_VDIM;i++){
      for (int dis=0;dis<GEMPIC_NUMGAUS;dis++){
        std::normal_distribution<> normD(init.Vmean[i][dis],init.Vdev[i][dis]);
        normArray[i][dis] = normD;
      }
  }

  amrex::Real tot_n_cell = infra.n_cell[0];
#if (GEMPIC_SPACEDIM > 1)
  tot_n_cell *= infra.n_cell[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
  tot_n_cell *= infra.n_cell[2];
#endif
amrex::Real tot_vel = 0.;
  for (int pp=0;pp<init.Np_cell;pp++) {
    //position in model cell [0,dx]x[0,dy]x[0,dz]:
    position[0] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[0];
#if (GEMPIC_SPACEDIM > 1)
    position[1] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
    position[2] = ((amrex::Real) std::rand() / (RAND_MAX))*infra.dx[2];
#endif

    amrex::Real gauss_choice[GEMPIC_VDIM];
    for(int i=0;i<GEMPIC_VDIM;i++){
        gauss_choice[i] = ((amrex::Real) std::rand() / (RAND_MAX));
    }
    std::array<int,GEMPIC_VDIM> gauss_idx = {AMREX_V_DECL(0,0,0)};
    for(int i=0;i<GEMPIC_VDIM;i++){
        for (int gg=1;gg<GEMPIC_NUMGAUS;gg++){
          if (vel_cum[i][gg-1]-gauss_choice[i] < 0 & vel_cum[i][gg]-gauss_choice[i] >= 0){
              gauss_idx[i] = gg;
          }
        }
    }
    
    velocity[0] = normArray[0][gauss_idx[0]](gen);
#if (GEMPIC_VDIM > 1)
    velocity[1] = normArray[1][gauss_idx[1]](gen);
#endif
#if (GEMPIC_VDIM > 2)
    velocity[2] = normArray[2][gauss_idx[2]](gen);
#endif
tot_vel += velocity[1];
    //MFI that adds particle from the model cell to all cells
    for (amrex::MFIter mfi(*(IteratorFab)); mfi.isValid(); ++mfi ){
      const amrex::Box& bx = mfi.validbox();
      amrex::IntVect lo = {bx.smallEnd()};
      amrex::IntVect hi = {bx.bigEnd()};

#if (GEMPIC_SPACEDIM > 2)
      for(int k=lo[2]; k<=hi[2]; k++){
	z = infra.geom.ProbLo()[2] + (double)k*infra.dx[2];
    shifted_position[2] = position[2] + z;
#endif
#if (GEMPIC_SPACEDIM > 1)
	for(int j=lo[1]; j<=hi[1]; j++){
	  y = infra.geom.ProbLo()[1] + (double)j*infra.dx[1];
      shifted_position[1] = position[1] + y;
#endif
	  for(int l=lo[0]; l<=hi[0]; l++){
	    x = infra.geom.ProbLo()[0] + (double)l*infra.dx[0];
	    shifted_position[0] = position[0] + x;

        weight = init.weight_fun(shifted_position, velocity,
                init.Np_cell*tot_n_cell,init.k)*dom_vol; //DIM
	    (*ppg).add_particle(species, shifted_position, velocity, weight);
	  }
#if (GEMPIC_SPACEDIM > 1)
	}
#endif
#if (GEMPIC_SPACEDIM > 2)
      }
#endif
    }
  }
}


#endif

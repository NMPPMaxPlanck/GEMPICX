#include <AMReX.H>
#include <AMReX_Print.H>

#include <particle_groups.H>
#include <infrastructure.H>
#include <maxwell_yee.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

class diagnostics
{
  public:
  int n_time; // number of timesteps
  // 5 static so far!! need to fix this
  std::array<std::array<amrex::Real,3>,5> Efield, Bfield, mom;
  std::array<amrex::Real,5> kin;

  // The default constructor.
  diagnostics ();

  // Explicit const
  diagnostics (int n_t){
    n_time = n_t;
  }

  // save position and velocities
  void save_step_data(int step,
		  infrastructure infra,
		  maxwell_yee *mw_yee,
		  particle_groups *part_gr){
    int nc = 3;
    amrex::Real vel;

    // E and B field
    for(int cc=0;cc<nc;cc++){
      Efield[step][cc] = infra.dx[cc]*pow((*(*mw_yee).E_Array[cc]).norm2(),2)/infra.Length[cc];
      Bfield[step][cc] = infra.dx[cc]*pow((*(*mw_yee).B_Array[cc]).norm2(),2)/infra.Length[cc];
    }

    // compute kinetic energy norm and momentum
    kin[step] = 0;
    mom[step][0] = 0.0;
    mom[step][1] = 0.0;
    mom[step][2] = 0.0;
    
     for (int spec=0;spec<(*part_gr).n_species;spec++){
       for (amrex::ParIter<4,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
	auto& particles = pti.GetArrayOfStructs();
	const long np = pti.numParticles();
	for (int pp=0;pp<np;pp++) {
	  vel = abs(particles[pp].rdata(0))+abs(particles[pp].rdata(1))+abs(particles[pp].rdata(2));
	  kin[step] += particles[pp].rdata(3)*pow(vel,2);
	  for (int cmp=0;cmp<3;cmp++) {
	    mom[step][cmp] += particles[pp].rdata(3)*abs(particles[pp].rdata(cmp));
	  }
	}
      }
    }
  }

  void save_all_to_textfile(amrex::Real dt){
    std::ofstream ofsfun("PIC_save.output", std::ofstream::out);
    for (int step=0;step<n_time;step++){
      amrex::Print(ofsfun) << dt*(step+1) << "," << Efield[step][0] << "," << Efield[step][1] << "," << Efield[step][2] << "," << kin[step] << "," << mom[step][0] << "," << mom[step][1] << "," << mom[step][2] << std::endl;
    }
    ofsfun.close();
  }
};

#endif

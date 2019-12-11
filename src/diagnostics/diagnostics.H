#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>

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
  // 9 static so far!! need to fix this
  std::array<std::array<amrex::Real,3>,9> Efield, Bfield, mom;
  std::array<amrex::Real,9> kin;

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
          particle_groups *part_gr,
                      bool bl_plotfile){
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

     // save Plotfile
     if(bl_plotfile) {
         amrex::Vector<std::string> varnames = {"rho"};
         if (step == 0) {
             WriteSingleLevelPlotfile ("SliceData/rho_plotfile_0", (*mw_yee).rho, varnames,
                                       infra.geom, step*(*mw_yee).dt, 0); //level_step
         }

         varnames[0] = "Ex";
         WriteSingleLevelPlotfile ("SliceData/Ex_plotfile_" + std::to_string(step), (*(*mw_yee).E_Array[0]), varnames, infra.geom, 0.0, 0); // last two args: time and level_step
         varnames[0] = "Ey";
         WriteSingleLevelPlotfile ("SliceData/Ey_plotfile_" + std::to_string(step), (*(*mw_yee).E_Array[1]), varnames, infra.geom, 0.0, 0);
         varnames[0] = "Ez";
         WriteSingleLevelPlotfile ("SliceData/Ez_plotfile_" + std::to_string(step), (*(*mw_yee).E_Array[2]), varnames, infra.geom, 0.0, 0);

         varnames[0] = "Jx";
         WriteSingleLevelPlotfile ("SliceData/Jx_plotfile_" + std::to_string(step), (*(*mw_yee).J_Array[0]), varnames, infra.geom, 0.0, 0); // last two args: time and level_step
         varnames[0] = "Jy";
         WriteSingleLevelPlotfile ("SliceData/Jy_plotfile_" + std::to_string(step), (*(*mw_yee).J_Array[1]), varnames, infra.geom, 0.0, 0);
         varnames[0] = "Jz";
         WriteSingleLevelPlotfile ("SliceData/Jz_plotfile_" + std::to_string(step), (*(*mw_yee).J_Array[2]), varnames, infra.geom, 0.0, 0);

     }
  }

  void save_all_to_textfile(amrex::Real dt, std::string filename = "PIC_save.output", int partial = 0){
      if (partial == 0){
          partial = n_time;
      }
      std::ofstream ofsfun(filename, std::ofstream::out);
    for (int step=0;step<=partial;step++){
      amrex::Print(ofsfun) << dt*(step) << "," << Efield[step][0] << "," << Efield[step][1] << "," << Efield[step][2] << "," << kin[step] << "," << mom[step][0] << "," << mom[step][1] << "," << mom[step][2] << std::endl;
    }
    ofsfun.close();
  }
};

#endif

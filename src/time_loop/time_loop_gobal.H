#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <infrastructure.H>
#include <maxwell_yee.H>
#include <diagnostics.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#include <AMReX_PlotFileUtil.H>

#ifndef TIME_LOOP_GLOBAL_H_
#define TIME_LOOP_GLOBAL_H_

void time_loop_global (infrastructure infra,
                maxwell_yee *mw_yee,
                particle_groups *part_gr) {

  diagnostics diagn((*mw_yee).nsteps);
  
  amrex::GpuArray<amrex::Real,3> plo;
  plo[0] = infra.geom.ProbLo()[0];
  plo[1] = infra.geom.ProbLo()[1];
  plo[2] = infra.geom.ProbLo()[2];
  amrex::GpuArray<amrex::Real,3> dxi;
  dxi[0] = 1.0/infra.dx[0];
  dxi[1] = 1.0/infra.dx[1];
  dxi[2] = 1.0/infra.dx[2];
  int nc = 3; // number of components for J and E
  
  // initial rho by: deposition of particle charge
  //Set rho to 0
  (*mw_yee).rho.setVal(0.0, 0); // value and component
  // Deposit charges:
  for (int spec=0;spec<(*part_gr).n_species;spec++) {
      (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
      for (amrex::ParIter<4,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
          amrex::Box tilebox;
          amrex::FArrayBox local_rho;
          const amrex::Box& box = pti.validbox();

          tilebox = pti.tilebox();
          tilebox.grow((*mw_yee).Nghost);
          const amrex::Box tb = amrex::convert(tilebox, amrex::IntVect::TheUnitVector());

          local_rho.resize(tb,1); // second arg: number of comps
          local_rho.setVal(0.0);

          auto& particles = pti.GetArrayOfStructs();
          const long np  = pti.numParticles();

          amrex::Array4<amrex::Real> const& rhoarr = local_rho.array();

          for (int pp=0;pp<np;pp++) {
              gempic_deposit_cic(particles[pp], (*part_gr).charge[spec], rhoarr, plo, dxi);
          }
          ((*mw_yee).rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
      }
  }

  //assign particles to tiles they are in
  for(int spec=0;spec<(*part_gr).n_species;spec++){
      (*(*part_gr).mypc[spec]).Redistribute();
  }

  //add values from ghost cells and periodic boundary
  (*mw_yee).rho.SumBoundary(infra.geom.periodicity());
  
  amrex::Real rhomean = (*mw_yee).rho.sum(0)/1000.0;
  (*mw_yee).rho.plus(-rhomean,0,1);

  //initial E by rho
  (*mw_yee).phi_init.setVal(0.0, 0);

  std::ofstream ofs("rho.output", std::ofstream::out);
  for (amrex::MFIter mfi((*mw_yee).rho); mfi.isValid(); ++mfi ) {
      amrex::Print(ofs) << (*mw_yee).rho[mfi] << std::endl;
  }
  ofs.close();

  (*mw_yee).solve_poisson(infra);
  //initial B  is 0
  (*(*mw_yee).B_Array[0]).setVal(0.0, 0);
  (*(*mw_yee).B_Array[1]).setVal(0.0, 0);
  (*(*mw_yee).B_Array[2]).setVal(0.0, 0);

  //-----------------------------------------------------------------------------
  //Time loop
  std::ofstream ofs2("PIC_6dim.output", std::ofstream::out);
  amrex::Real time = 0.0;
  diagn.save_step_data(0, infra, mw_yee, part_gr, true);

  for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
      time += (*mw_yee).dt;

      //-------------------------------------------------------------------------
      //push particles & get J
      (*mw_yee).FillBD(infra);

      // set J to 0 before we start depositing values from current particles
      (*(*mw_yee).J_Array[0]).setVal(0.0, 0); // value and component
      (*(*mw_yee).J_Array[1]).setVal(0.0, 0);
      (*(*mw_yee).J_Array[2]).setVal(0.0, 0);

      std::array<amrex::Real,3> eres, bres;
      for (int spec=0;spec<(*part_gr).n_species;spec++) {
          amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

          for (amrex::ParIter<4,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
              const amrex::Box& box = pti.validbox();
              auto& particles = pti.GetArrayOfStructs(); // get particles
              const long np  = pti.numParticles();

              amrex::Array4<amrex::Real> const& jarrx = (*(*mw_yee).J_Array[0])[pti].array();
              amrex::Array4<amrex::Real> const& jarry = (*(*mw_yee).J_Array[1])[pti].array();
              amrex::Array4<amrex::Real> const& jarrz = (*(*mw_yee).J_Array[2])[pti].array();
              // loop over particles
              for (int pp=0;pp<np;pp++){
                  for (int cc=0;cc<nc;cc++){
                      amrex::Array4<amrex::Real> const& earr = (*(*mw_yee).E_Array[cc])[pti].array();
                      amrex::Array4<amrex::Real> const& barr = (*(*mw_yee).B_Array[cc])[pti].array();

                      //E-field at particle position
                      eres[cc] = gempic_interpolate_cic(particles[pp], earr, infra.ploE[cc], dxi);

                      //B-field at particle position
                      bres[cc] = gempic_interpolate_cic(particles[pp], barr, infra.ploB[cc], dxi);
                  }
                  amrex::Real part_weight = particles[pp].rdata(3);
                  std::array<amrex::Real,6> newPos = push_particle(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                  std::copy(begin(newPos),end(newPos),&particles.data()[pp*8]);
                  amrex::Print(ofs2) << newPos[0] << "," << newPos[1] << "," << newPos[2] << "," << newPos[3] << "," << newPos[4] << "," << newPos[5] << std::endl;

                  gempic_deposit_J_cic_global(newPos, part_weight, (*part_gr).charge[spec], 0, mw_yee, infra.ploE[0], dxi);
                  gempic_deposit_J_cic_global(newPos, part_weight, (*part_gr).charge[spec], 1, mw_yee, infra.ploE[1], dxi);
                  gempic_deposit_J_cic_global(newPos, part_weight, (*part_gr).charge[spec], 2, mw_yee, infra.ploE[2], dxi);
              }
          }
          (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
      }
      //--------------------------------------------------------------------------
      //update fields

      //add values from ghost cells and periodic boundary
      (*(*mw_yee).J_Array[0]).SumBoundary(infra.geom.periodicity());
      (*(*mw_yee).J_Array[1]).SumBoundary(infra.geom.periodicity());
      (*(*mw_yee).J_Array[2]).SumBoundary(infra.geom.periodicity());
      std::cout << "finished step " << t_step+1 << std::endl;

      // field updates:
      (*mw_yee).advance(infra);
      // save norms for diagnostics
      diagn.save_step_data(t_step+1, infra, mw_yee, part_gr, true);

  } // end time loop
  ofs2.close();
  diagn.save_all_to_textfile((*mw_yee).dt);
}


#endif

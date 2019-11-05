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
#ifndef TIME_LOOP_H_
#define TIME_LOOP_H_

void time_loop (infrastructure infra,
		maxwell_yee *mw_yee,
		particle_groups *part_gr) {
  
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
	const amrex::Box& box = pti.validbox();
	auto& particles = pti.GetArrayOfStructs();
	const long np  = pti.numParticles();

	amrex::Array4<amrex::Real> const& rhoarr = ((*mw_yee).rho)[pti].array();
	  for (int pp=0;pp<np;pp++) {
	    gempic_deposit_cic(particles[pp], (*part_gr).charge[spec], rhoarr, plo, dxi);
	  }
      }
    }
    //assign particles to tiles they are in
  for(int spec=0;spec<(*part_gr).n_species;spec++){
    (*(*part_gr).mypc[spec]).Redistribute();
    }

    //initial E by rho
    (*mw_yee).phi_init.setVal(0.0, 0);
    //(*mw_yee).solve_poisson(infra);
    //while poisson solver is not working, set E to 0
    (*(*mw_yee).E_Array[0]).setVal(0.0, 0);
    (*(*mw_yee).E_Array[1]).setVal(0.0, 0);
    (*(*mw_yee).E_Array[2]).setVal(0.0, 0);
    //initial B  is 0
    (*(*mw_yee).B_Array[0]).setVal(0.0, 0);
    (*(*mw_yee).B_Array[1]).setVal(0.0, 0);
    (*(*mw_yee).B_Array[2]).setVal(0.0, 0);
    
//------------------------------------------------------------------------------
  //Time loop
  std::ofstream ofs("PIC.output", std::ofstream::out);
  std::ofstream ofs2("PIC_6dim.output", std::ofstream::out);
  amrex::Real time = 0.0;
  for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
    time += (*mw_yee).dt;
    
    //--------------------------------------------------------------------------
    //push particles
    (*mw_yee).FillBD(infra);

    std::array<amrex::Real,3> eres, bres;
    for (int spec=0;spec<(*part_gr).n_species;spec++) {
      amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

      for (amrex::ParIter<4,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
	const amrex::Box& box = pti.validbox();
	auto& particles = pti.GetArrayOfStructs(); // get particles
	const long np  = pti.numParticles();

	// loop over particles
	for (int pp=0;pp<np;pp++){
	  for (int cc=0;cc<nc;cc++){
	    amrex::Array4<amrex::Real> const& earr = (*(*mw_yee).E_Array[cc])[pti].array();
	    amrex::Array4<amrex::Real> const& barr = (*(*mw_yee).B_Array[cc])[pti].array(); 

	    //E-field at particle position
	    eres[cc] = gempic_interpolate_cic(particles[pp], earr, plo, dxi,
					      *(*mw_yee).E_Index[cc]);

	    //B-field at particle position
	    bres[cc] = gempic_interpolate_cic(particles[pp], barr, plo, dxi,
					      *(*mw_yee).B_Index[cc]);
	  }
	  std::array<amrex::Real,6> newPos = push_particle(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
	  std::copy(begin(newPos),end(newPos),&particles.data()[pp*8]);
	  amrex::Print(ofs2) << newPos[0] << "," << newPos[1] << "," << newPos[2] << "," << newPos[3] << "," << newPos[4] << "," << newPos[5] << std::endl;
	}
      }
      (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
    }
    //--------------------------------------------------------------------------
    //update fields
    
    // deposit charge in J one component and particle at a time

    // set J to 0 before we start depositing values from current particles
    (*(*mw_yee).J_Array[0]).setVal(0.0, 0); // value and component
    (*(*mw_yee).J_Array[1]).setVal(0.0, 0);
    (*(*mw_yee).J_Array[2]).setVal(0.0, 0);
    
     for (int spec=0;spec<(*part_gr).n_species;spec++) {
       (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
       for (amrex::ParIter<4,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
	const amrex::Box& box = pti.validbox();
	auto& particles = pti.GetArrayOfStructs();
	const long np  = pti.numParticles();

	for(int cc=0;cc<nc;cc++){
	  amrex::Array4<amrex::Real> const& jarr = (*(*mw_yee).J_Array[cc])[pti].array();
	  for (int pp=0;pp<np;pp++) {
	    gempic_deposit_J_cic(particles[pp], (*part_gr).charge[spec], cc, jarr, plo,
				 infra.ploE[cc], dxi,
				 *(*mw_yee).E_Index[cc]);
	  }
	}
      }
    }
    // field updates:
     (*mw_yee).advance(infra);

    // compute electric energy norms
    std::array<amrex::Real,3> E_n;
    std::array<amrex::Real,3> B_n;
    for(int cc=0;cc<nc;cc++){
      E_n[cc] = infra.dx[cc]*pow((*(*mw_yee).E_Array[cc]).norm2(),2)/infra.Length[cc];
      B_n[cc] = infra.dx[cc]*pow((*(*mw_yee).B_Array[cc]).norm2(),2)/infra.Length[cc];
    }
    std::cout << E_n[0] << "|" << E_n[1] << "|" << E_n[2] << std::endl;
    std::cout << B_n[0] << "|" << B_n[1] << "|" << B_n[2] << std::endl;

    // compute kinetic energy norm and momentum
    amrex::Real kin = 0;
    amrex::Real vel;
    std::array<amrex::Real,3> mom = {0.0,0.0,0.0};
     for (int spec=0;spec<(*part_gr).n_species;spec++){
       for (amrex::ParIter<4,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
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
    std::cout << kin << std::endl;
    std::cout << "total energy: " << E_n[0]+E_n[1]+E_n[2]+kin << std::endl;
    amrex::Print(ofs) << time << "," << E_n[0] << "," << E_n[1] << "," << E_n[2] << "," << kin << "," << mom[0] << "," << mom[1] << "," << mom[2] << std::endl;
    
  } // end time loop
  ofs.close();
  ofs2.close();
}


#endif

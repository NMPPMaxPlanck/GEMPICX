#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <GEMPIC_diagnostics.H>
#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_SumBoundary.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Infra;
using namespace Particles;

#ifndef TIME_LOOP_HS_ZIGZAG_C2_
#define TIME_LOOP_HS_ZIGZAG_C2_

namespace Gempic {
namespace Time_Loop {

/*
    This is the time propagator with a Hamiltonian splitting in a FEM formulation with mass lumping for Lagrange elements of order one and zero

    This propagator is currently not implemented for the 2d2v case. In this case, the operator H_p3 is dismissed and the operators H_p1 and H_p2
    must be reimplemented only producing the b3 updates. In principle the operators are exactly the same but they should just return one bfield
    value - the one that concerns b_z - and it should read out the first component of the bfield vector (which saves the b_z in this case).

   */


template<int vdim, int numspec, int degx, int degy, int degz, bool electromagnetic=true>
void time_loop_hs_zigzag_C2 (infrastructure infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr,
                       diagnostics<vdim, numspec, degx, degy, degz> *diagn,
                       bool ctest,
                       std::string simulation_name,
                       std::ofstream *ofs) {

    splines_at_particles<degx,degy,degz> spline;
    spline1d_at_particles<degx> spline_new_x;
    spline1d_at_particles<degx> spline_old_x;
    spline1d_at_particles<degy> spline_new_y;
    spline1d_at_particles<degy> spline_old_y;
    spline1d_at_particles<degz> spline_new_z;
    spline1d_at_particles<degz> spline_old_z;

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
        (*mw_yee).advance_time();

        // Operator H_B for dt/2
        if ( electromagnetic ) {
          (*mw_yee).advance_E_from_B(infra, (*mw_yee).dt*0.5);

        //--------------------------------------------------------------------------
        // Operator H_E for dt/2 (field part)
          (*mw_yee).advance_B(infra, (*mw_yee).dt*0.5);
          }

        //--------------------------------------------------------------------------
        //deposit charge in J and push particles

        // set J to 0 before we start depositing values from current particles
        for (int i=0; i<vdim; i++) {
            (*(*mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }

        for (int spec=0;spec<numspec;spec++) {
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];
            amrex::Real charge = (*part_gr).charge[spec];

            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs(); // get particles
                const long np  = pti.numParticles();

                std::array<amrex::Array4<amrex::Real>,vdim> jA;
                std::array<amrex::Array4<amrex::Real>,vdim> eA;
                std::array<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bA;

                for (int cc=0;cc < vdim;cc++) {
                    jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                    eA[cc] = (*(*mw_yee).E_Array[cc])[pti].array();
                }
                for (int cc=0;cc < int(vdim/2.5)*2+1;cc++) {
                    bA[cc] = (*(*mw_yee).B_Array[cc])[pti].array();
                }

                // loop over particles: add contribution of old particle position to J and push

                for (int pp=0;pp<np;pp++){
                    // Read out particle position
                    std::array<amrex::Real,GEMPIC_SPACEDIM> pos;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                      pos[d]= particles[pp].pos(d);
                    // evaluate the spline
                    spline.init_particles(pos , infra.plo, infra.dxi);
                    // evaluate the electric field
                    std::array<amrex::Real, vdim> efield = evaluate_efield<vdim,degz,degy,degz>(spline,  eA);

                    // push v with the electric field
                    std::array<amrex::Real,vdim> newPosE = push_v_efield<amrex::Particle<vdim+1>,vdim>(particles[pp], (*mw_yee).dt*0.5, chargemass, efield, infra);
                    std::copy(begin(newPosE),end(newPosE),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)+GEMPIC_SPACEDIM]);

                    amrex::Real weight = charge*particles[pp].rdata(vdim)*infra.dxi[GEMPIC_SPACEDIM];


                    std::array<amrex::Real,vdim> vel;
                    for (unsigned int d=0;d<vdim;++d)
                      vel[d] = particles[pp].rdata(d);
                    std::array<amrex::Real, 2> bfields;
                    // push x1
                    amrex::Real x1_new = pos[0] +  (*mw_yee).dt*vel[0];
                    spline_new_x.init_position( x1_new, infra.plo[0], infra.dxi[0] );
                    spline_old_x.init_position( pos[0], infra.plo[0], infra.dxi[0] );
                    spline.template init_position<0,degx>( x1_new, infra.plo[0], infra.dxi[0] );

                    accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degx, degy, degz, 0>( spline,
                                                                                                               spline_new_x,
                                                                                                               spline_old_x,
                                                                                                               weight,
                                                                                                               infra.geom.CellSize(),
                                                                                                               bA,
                                                                                                               jA,
                                                                                                               bfields);


                    // Push v2 and v3
                    if (vdim > 1)
                      vel[1] = vel[1] - chargemass*bfields[0];
                    if (vdim > 2)
                      vel[2] = vel[2] + chargemass*bfields[1];
                    pos[0] = x1_new;

                    if ( vdim > 1 )
                      {
#if GEMPIC_SPACEDIM > 1
                        // push x2
                      amrex::Real x2_new = pos[1] +  (*mw_yee).dt*vel[1];

                      spline_new_y.init_position( x2_new, infra.plo[1], infra.dxi[1] );
                      spline_old_y.init_position( pos[1], infra.plo[1], infra.dxi[1] );
                      spline.template init_position<1,degy>( x2_new, infra.plo[1], infra.dxi[1] );

                      accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degy, degz, degx, 1>( spline,
                                                        spline_new_y,
                                                        spline_old_y,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        bA,
                                                        jA,
                                                        bfields);

                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[1];
                      if (vdim > 2 )
                        vel[2] = vel[2] - chargemass*bfields[0];
                      pos[1] = x2_new;
#else
                      std::array<amrex::Real, int(vdim/2.5)*2+1> bfield = evaluate_fem_bfield<vdim,degx,degy,degz>(pos,infra.dxi, infra.plo,  bA);
                      gempic_deposit_rho<degx,degy,degz> (spline, weight*vel[1]*(*mw_yee).dt,
                                               jA[1]);
                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[0]*vel[1]*(*mw_yee).dt;
  #endif
}
                    if(vdim > 2 ) {
#if GEMPIC_SPACEDIM == 3
                    // push x3
                    amrex::Real x3_new = pos[2] +  (*mw_yee).dt*vel[2];

                    spline_new_z.init_position( x3_new, infra.plo[2], infra.dxi[2] );
                    spline_old_z.init_position( pos[2], infra.plo[2], infra.dxi[2] );
                    spline.template init_position<2,degz>( x3_new, infra.plo[2], infra.dxi[2] );

                    accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degz, degx, degy, 2>( spline,
                                                      spline_new_z,
                                                      spline_old_z,
                                                      weight,
                                                      infra.geom.CellSize(),
                                                      bA,
                                                      jA,
                                                      bfields);
                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass*bfields[0];
                    vel[1] = vel[1] + chargemass*bfields[1];
                    pos[2] = x3_new;
#else
                    std::array<amrex::Real, int(vdim/2.5)*2+1> bfield = evaluate_bfield<vdim,degx,degy,degz>(spline,  bA);
                    gempic_deposit_rho<degx,degy,degz> (spline, weight*vel[2]*(*mw_yee).dt,
                                             jA[2]);
                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass*bfields[1]*vel[2]*(*mw_yee).dt;
                    vel[1] = vel[1] + chargemass*bfields[0]*vel[2]*(*mw_yee).dt;
#endif


}
                    std::copy(begin(pos),end(pos),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                    std::copy(begin(vel),end(vel),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)+GEMPIC_SPACEDIM]);
//exit(0);
                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }


        //add values from ghost cells and periodic boundary
        Gempic_SumBoundary(mw_yee, &infra); // Sums Boundary of J and Overwrites overlap
        for (int i=0; i<vdim; i++) {
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }
        amrex::Vector<std::string> varnames = { "J_x"};
        amrex::Real timeo = t_step*(*mw_yee).dt;
        WriteSingleLevelPlotfile("current_C2", *(*mw_yee).J_Array[0], varnames,
        infra.geom, timeo, 0);
        //abort();


        // advance E from j (field part of H_p)
        (*mw_yee).advance_E_from_J(infra, 1.0);

        // Second particle loop for particle part from H_E
        for (int spec=0;spec<numspec;spec++) {
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs(); // get particles
                const long np  = pti.numParticles();

                std::array<amrex::Array4<amrex::Real>,vdim> eA;

                for (int cc=0;cc < vdim;cc++) {
                    eA[cc] = (*(*mw_yee).E_Array[cc])[pti].array();
                }

                // loop over particles: add contribution of old particle position to J and push

                for (int pp=0;pp<np;pp++){
                    std::array<amrex::Real,GEMPIC_SPACEDIM> pos;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                      pos[d]= particles[pp].pos(d);

                    // evaluate the spline
                    spline.init_particles(pos , infra.plo, infra.dxi);
                    // evaluate the electric field
                    std::array<amrex::Real, vdim> efield = evaluate_efield<vdim,degx,degy,degz>(spline,  eA);

                    // push v with the electric field
                    std::array<amrex::Real,vdim> newPosE = push_v_efield<amrex::Particle<vdim+1>,vdim>(particles[pp], (*mw_yee).dt*0.5, chargemass, efield, infra);
                    std::copy(begin(newPosE),end(newPosE),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)+GEMPIC_SPACEDIM]);
                  }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }

        //--------------------------------------------------------------------------
        // Operator H_E for dt/2 (field part)
        if ( electromagnetic ) {
          (*mw_yee).advance_B(infra, (*mw_yee).dt*0.5);

        // Operator H_B for dt/2
          (*mw_yee).advance_E_from_B(infra, (*mw_yee).dt*0.5);
        }

        // save norms for diagnostics
        (*diagn).save_step_data(t_step+1, infra, mw_yee, part_gr); // false: slices are not saved

        // save temporary results every n steps
        if ((t_step+1)%1 == 0) {
            (*diagn).save_all_to_textfile((*mw_yee).dt, simulation_name+"_tmp.output", t_step+1);
        }

        amrex::Print() << "finished step " << t_step+1 << std::endl;

        if (ctest) {
            switch (vdim) {
            case 1:
                amrex::AllPrintToFile(simulation_name) << t_step << " " << (*diagn).Efield[t_step][0];
                break;
            case 2:
                amrex::AllPrintToFile(simulation_name) << t_step << " " << (*diagn).Efield[t_step][0] << " " << (*diagn).Efield[t_step][1];
                break;
            case 3:
                amrex::AllPrintToFile(simulation_name) << t_step << " " << (*diagn).Efield[t_step][0] << " " << (*diagn).Efield[t_step][1] << " " << (*diagn).Efield[t_step][2];
                break;
            }
            switch (vdim) {
            case 1: // no break, so that case 1 and 2 are handled the same way
            case 2:
                amrex::AllPrintToFile(simulation_name) << " " << (*diagn).Bfield[t_step][0];
                break;
            case 3:
                amrex::AllPrintToFile(simulation_name) << " " << (*diagn).Bfield[t_step][0] << " " << (*diagn).Bfield[t_step][1] << " " << (*diagn).Bfield[t_step][2];
                break;
            }
            amrex::AllPrintToFile(simulation_name) << " " << (*diagn).kin[t_step] << " ";
            switch (vdim) {
            case 1:
                amrex::AllPrintToFile(simulation_name) << (*diagn).mom[t_step][0];
                break;
            case 2:
                amrex::AllPrintToFile(simulation_name) << (*diagn).mom[t_step][0] << " " << (*diagn).mom[t_step][1];
                break;
            case 3:
                amrex::AllPrintToFile(simulation_name) << (*diagn).mom[t_step][0] << " " << (*diagn).mom[t_step][1] << " " << (*diagn).mom[t_step][2];
                break;
            }
            amrex::AllPrintToFile(simulation_name) << std::endl;
        }
    } // end time loop

    (*diagn).save_all_to_textfile((*mw_yee).dt, simulation_name+"_tmp.output");
}
}
}


#endif

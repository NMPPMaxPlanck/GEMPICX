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
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_SumBoundary.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Infra;
using namespace Particles;

#ifndef TIME_LOOP_HSALL_FEM_
#define TIME_LOOP_HSALL_FEM_

namespace Gempic {
namespace Time_Loop {

/*
    This is the time propagator with a Hamiltonian splitting in a FEM formulation with mass lumping for Lagrange elements of order one and zero
   */


template<int vdim, int numspec, int degx, int degy, int degz, bool electromagnetic=true>
void time_loop_hsall_fem (infrastructure infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr,
                       diagnostics<vdim, numspec, degx, degy, degz> *diagn,
                       bool ctest,
                       std::ofstream *ofs) {

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
                    // evaluate the electric field
                    std::array<amrex::Real, vdim> efield = evaluate_fem_efield<amrex::Particle<vdim+1>,vdim,degz,degy,degz>(particles[pp],infra.dxi, infra.plo,  eA);

                    // push v with the electric field
                    std::array<amrex::Real,vdim> newPosE = push_v_efield<amrex::Particle<vdim+1>,vdim>(particles[pp], (*mw_yee).dt*0.5, chargemass, efield, infra);
                    std::copy(begin(newPosE),end(newPosE),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)+GEMPIC_SPACEDIM]);

                    amrex::Real weight = charge*particles[pp].rdata(vdim)*infra.dxi[GEMPIC_SPACEDIM];

                    std::array<amrex::Real,GEMPIC_SPACEDIM> pos;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                      pos[d]= particles[pp].pos(d);
                    std::array<amrex::Real,vdim> vel;
                    for (unsigned int d=0;d<vdim;++d)
                      vel[d] = particles[pp].rdata(d);
                    std::array<amrex::Real, 2> bfields;
                    // push x1
                    amrex::Real x1_new = pos[0] +  (*mw_yee).dt*vel[0];
                    /*accumulate_j_update_v_x1_fem_integral<vdim, degx, degy, degz>( pos,
                                                        x1_new,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        infra.dxi,
                                                        infra.plo,
                                                        bA,
                                                        jA,
                                                        bfields);*/
                    accumulate_j_update_v_fem_integral<vdim, degx, degy, degz, 0>( pos,
                                                      x1_new,
                                                      weight,
                                                      infra.geom.CellSize(),
                                                      infra.dxi,
                                                      infra.plo,
                                                      bA,
                                                      jA,
                                                      bfields);

                    // Push v2 and v3
                    vel[1] = vel[1] - chargemass*bfields[0];
                    vel[2] = vel[2] + chargemass*bfields[1];
                    pos[0] = x1_new;

                    if ( vdim > 1 )
                      {
                      // push x2
                      amrex::Real x2_new = pos[1] +  (*mw_yee).dt*vel[1];
                      /*accumulate_j_update_v_x2_fem_integral<vdim, degx, degy, degz>( pos,
                                                        x2_new,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        infra.dxi,
                                                        infra.plo,
                                                        bA,
                                                        jA,
                                                        bfields);*/
                      accumulate_j_update_v_fem_integral<vdim, degy, degz, degx, 1>( pos,
                                                        x2_new,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        infra.dxi,
                                                        infra.plo,
                                                        bA,
                                                        jA,
                                                        bfields);

                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[1];
                      vel[2] = vel[2] - chargemass*bfields[0];
                      pos[1] = x2_new;
}
                    if(vdim > 2 ) {
                    // push x3
                    amrex::Real x3_new = pos[2] +  (*mw_yee).dt*vel[2];
                    /*accumulate_j_update_v_x3_fem_integral<vdim, degx, degy, degz>( pos,
                                                        x3_new,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        infra.dxi,
                                                        infra.plo,
                                                        bA,
                                                        jA,
                                                        bfields);*/
                    accumulate_j_update_v_fem_integral<vdim, degz, degx, degy, 2>( pos,
                                                      x3_new,
                                                      weight,
                                                      infra.geom.CellSize(),
                                                      infra.dxi,
                                                      infra.plo,
                                                      bA,
                                                      jA,
                                                      bfields);

                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass*bfields[0];
                    vel[1] = vel[1] + chargemass*bfields[1];
                    pos[2] = x3_new;
}
                    std::copy(begin(pos),end(pos),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                    std::copy(begin(vel),end(vel),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)+GEMPIC_SPACEDIM]);

                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }


        //add values from ghost cells and periodic boundary
        Gempic_SumBoundary(mw_yee, &infra); // Sums Boundary of J and Overwrites overlap
        for (int i=0; i<vdim; i++) {
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }
        /*amrex::Vector<std::string> varnames = { "J_x"};
        amrex::Real timeo = t_step*(*mw_yee).dt;
        WriteSingleLevelPlotfile("current", *(*mw_yee).J_Array[0], varnames,
        infra.geom, timeo, 0);
        abort();*/

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
                    // evaluate the electric field
                    std::array<amrex::Real, vdim> efield = evaluate_fem_efield<amrex::Particle<vdim+1>,vdim,degx,degy,degz>(particles[pp],infra.dxi, infra.plo,  eA);

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
            (*diagn).save_all_to_textfile((*mw_yee).dt, "vlasov_maxwell_save_step1.output", t_step+1);
        }

        amrex::Print() << "finished step " << t_step+1 << std::endl;

        if (ctest) {
            switch (vdim) {
            case 1:
                amrex::AllPrintToFile("test_output_pre_rename.output") << t_step << " " << (*diagn).Efield[t_step][0];
                break;
            case 2:
                amrex::AllPrintToFile("test_output_pre_rename.output") << t_step << " " << (*diagn).Efield[t_step][0] << " " << (*diagn).Efield[t_step][1];
                break;
            case 3:
                amrex::AllPrintToFile("test_output_pre_rename.output") << t_step << " " << (*diagn).Efield[t_step][0] << " " << (*diagn).Efield[t_step][1] << " " << (*diagn).Efield[t_step][2];
                break;
            }
            switch (vdim) {
            case 1: // no break, so that case 1 and 2 are handled the same way
            case 2:
                amrex::AllPrintToFile("test_output_pre_rename.output") << " " << (*diagn).Bfield[t_step][0];
                break;
            case 3:
                amrex::AllPrintToFile("test_output_pre_rename.output") << " " << (*diagn).Bfield[t_step][0] << " " << (*diagn).Bfield[t_step][1] << " " << (*diagn).Bfield[t_step][2];
                break;
            }
            amrex::AllPrintToFile("test_output_pre_rename.output") << " " << (*diagn).kin[t_step] << " ";
            switch (vdim) {
            case 1:
                amrex::AllPrintToFile("test_output_pre_rename.output") << (*diagn).mom[t_step][0];
                break;
            case 2:
                amrex::AllPrintToFile("test_output_pre_rename.output") << (*diagn).mom[t_step][0] << " " << (*diagn).mom[t_step][1];
                break;
            case 3:
                amrex::AllPrintToFile("test_output_pre_rename.output") << (*diagn).mom[t_step][0] << " " << (*diagn).mom[t_step][1] << " " << (*diagn).mom[t_step][2];
                break;
            }
            amrex::AllPrintToFile("test_output_pre_rename.output") << std::endl;
        }
    } // end time loop

    (*diagn).save_all_to_textfile((*mw_yee).dt);
}
}
}


#endif

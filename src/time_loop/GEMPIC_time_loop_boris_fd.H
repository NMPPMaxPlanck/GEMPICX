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

#include <GEMPIC_checkpoint.H>
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

#ifndef TIME_LOOP_BORIS_FD_H_
#define TIME_LOOP_BORIS_FD_H_

namespace Gempic {
namespace Time_Loop {

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool electromagnetic=true>
void time_loop_boris_fd (infrastructure infra,
                         maxwell_yee<vdim> *mw_yee,
                         particle_groups<vdim,numspec> *part_gr,
                         diagnostics<vdim, numspec, degx, degy, degz,degmw> *diagn,
                         bool ctest,
                         std::string simulation_name,
                         std::ofstream *ofs) {

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
        (*mw_yee).advance_time();

        if ( electromagnetic ){
          //--------------------------------------------------------------------------
          // update B
          mw_yee->template hodge_full<degmw>(infra, &(mw_yee->E_Array), &(mw_yee->HE_Array), true);
          mw_yee->advance_B(infra, mw_yee->dt, &(mw_yee->HE_Array), &(mw_yee->B_Array));
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

                    std::array<amrex::Real, vdim+int(vdim/2.5)*2+1> field_interp = gempic_deposit_J_and_interpolate<amrex::Particle<vdim+1>,vdim,degx,degy,degz> (particles[pp], (*part_gr).charge[spec],
                                                                                                                                                          infra.dxi, infra.plo, jA, eA, bA, 0.5, true);

                    std::array<amrex::Real,vdim> eres;
                    for (int comp = 0; comp < vdim; comp++) {
                        eres[comp] = field_interp[comp];
                    }
                    std::array<amrex::Real,int(vdim/2.5)*2+1> bres;
                    for (int comp = 0; comp < int(vdim/2.5)*2+1; comp++) {
                        bres[comp] = field_interp[vdim+comp];
                    }

                    std::array<amrex::Real,GEMPIC_SPACEDIM+vdim> newPos = push_particle_boris<amrex::Particle<vdim+1>,vdim>(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                    std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }

        //--------------------------------------------------------------------------
        //update fields
        // deposit remaining charge in J one particle at a time

        for (int spec=0;spec<numspec;spec++) {
            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                std::array<amrex::Array4<amrex::Real>,vdim> jA;
                std::array<amrex::Array4<amrex::Real>,vdim> eA;
                std::array<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bA;

                for (int cc=0;cc < vdim;cc++) {
                    jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                }

                for (int pp=0;pp<np;pp++){
                    std::array<amrex::Real, vdim+int(vdim/2.5)*2+1> field_interp = gempic_deposit_J_and_interpolate<amrex::Particle<vdim+1>,vdim,degx,degy,degz> (particles[pp], (*part_gr).charge[spec],
                                                                                                                                                          infra.dxi, infra.plo, jA, eA, bA, 0.5, false);
                }
            }
        }


        Gempic_SumBoundary(mw_yee, &infra); // Sums Boundary of J and Overwrites overlap

        for (int i=0; i<vdim; i++) {
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }

        if (electromagnetic){
          // advance E
            mw_yee->template hodge_full<degmw>(infra, &(mw_yee->B_Array), &(mw_yee->HB_Array), false);
            mw_yee->advance_E(infra, mw_yee->dt, true, true, &(mw_yee->HB_Array), &(mw_yee->E_Array));
          }
        else
          {
            // advance E from j
            mw_yee->advance_E(infra, mw_yee->dt, false, true, &(mw_yee->HB_Array), &(mw_yee->E_Array));
          }


        // save norms for diagnostics
        (*diagn).save_step_data(t_step+1, infra, mw_yee, part_gr); // false: slices are not saved

        // save temporary results every n steps
        if ((t_step+1)%5 == 0) {
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

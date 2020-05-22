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

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Infra;
using namespace Particles;

#ifndef TIME_LOOP_AVG_H_
#define TIME_LOOP_AVG_H_

namespace Gempic {
namespace Time_Loop {

void time_loop_avg (infrastructure infra,
                    maxwell_yee *mw_yee,
                    particle_groups *part_gr,
                    diagnostics *diagn,
                    bool ctest,
                    std::ofstream *ofs) {

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
        (*mw_yee).advance_time();

        //--------------------------------------------------------------------------
        // update B
        (*mw_yee).advance_B(infra, (*mw_yee).dt);

        //--------------------------------------------------------------------------
        //deposit charge in J and push particles

        // set J to 0 before we start depositing values from current particles
        for (int i=0; i<GEMPIC_VDIM; i++) {
            (*(*mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }

        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs(); // get particles
                const long np  = pti.numParticles();

                std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> jA;
                std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> eA;
                std::array<amrex::Array4<amrex::Real>,GEMPIC_BDIM> bA;

                for (int cc=0;cc < GEMPIC_VDIM;cc++) {
                    jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                    eA[cc] = (*(*mw_yee).E_Array[cc])[pti].array();
                }
                for (int cc=0;cc < GEMPIC_VDIM;cc++) {
                    bA[cc] = (*(*mw_yee).B_Array[cc])[pti].array();
                }

                // loop over particles: add contribution of old particle position to J and push

                for (int pp=0;pp<np;pp++){

                    std::array<amrex::Real, GEMPIC_VDIM+GEMPIC_BDIM> field_interp = gempic_deposit_J_and_interpolate (particles[pp], (*part_gr).charge[spec],
                                                                                                                      infra.dxi, infra.plo, jA, eA, bA, 0.5, true);

                    std::array<amrex::Real,GEMPIC_VDIM> eres = {AMREX_D_DECL(field_interp[0],field_interp[1],field_interp[2])};
                    std::array<amrex::Real,GEMPIC_BDIM> bres = {AMREX_B_DECL(field_interp[GEMPIC_VDIM],field_interp[GEMPIC_VDIM+1],field_interp[GEMPIC_VDIM+2])};

                    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle_boris(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                    std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);
                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }

        //--------------------------------------------------------------------------
        //update fields
        // deposit remaining charge in J one particle at a time

        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> jA;
                std::array<amrex::Array4<amrex::Real>,GEMPIC_VDIM> eA;
                std::array<amrex::Array4<amrex::Real>,GEMPIC_BDIM> bA;

                for (int cc=0;cc < GEMPIC_VDIM;cc++) {
                    jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                }

                for (int pp=0;pp<np;pp++){
                    std::array<amrex::Real, GEMPIC_VDIM+GEMPIC_BDIM> field_interp = gempic_deposit_J_and_interpolate (particles[pp], (*part_gr).charge[spec],
                                                                                                                      infra.dxi, infra.plo, jA, eA, bA, 0.5, false);
                }
            }
        }
        //add values from ghost cells and periodic boundary
        for (int i=0; i<GEMPIC_VDIM; i++) {
            (*(*mw_yee).J_Array[i]).SumBoundary(infra.geom.periodicity());
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }

        // advance E
        (*mw_yee).advance_E(infra, (*mw_yee).dt);

        // save norms for diagnostics
        (*diagn).save_step_data(t_step+1, infra, mw_yee, part_gr); // false: slices are not saved

        // save temporary results every n steps
        if ((t_step+1)%5 == 0) {
            (*diagn).save_all_to_textfile((*mw_yee).dt, "PIC_save_tmp.output", t_step+1);
        }
        if ((t_step+1)%diagn->freq_x == 0) {
            save_particle_positions(part_gr, diagn->sim_name + "_" + std::to_string(t_step+1));
        }
        if ((t_step+1)%diagn->freq_v == 0) {
            save_particle_velocities(part_gr, diagn->sim_name + "_" + std::to_string(t_step+1));
        }
        amrex::Print() << "finished step " << t_step+1 << std::endl;

        if (ctest) {
            amrex::Print(*ofs) << t_step << " " << (*diagn).Efield[t_step][0]
                        #if (GEMPIC_VDIM > 1)
                                 << " " << (*diagn).Efield[t_step][1]
                        #endif
                        #if (GEMPIC_VDIM > 2)
                                 << " " << (*diagn).Efield[t_step][2]
                        #endif
                                 << " " << (*diagn).Bfield[t_step][0]
                        #if (GEMPIC_BDIM > 1)
                                 << " " << (*diagn).Bfield[t_step][1]
                        #endif
                        #if (GEMPIC_BDIM > 2)
                                 << " " << (*diagn).Bfield[t_step][2]
                        #endif
                                 << " " << (*diagn).kin[t_step] << " " << (*diagn).mom[t_step][0]
                        #if (GEMPIC_VDIM > 1)
                                 << " " << (*diagn).mom[t_step][1]
                        #endif
                        #if (GEMPIC_VDIM > 2)
                                 << " " << (*diagn).mom[t_step][2]
                        #endif
                                 << std::endl;
        }
    } // end time loop

    (*diagn).save_all_to_textfile((*mw_yee).dt);
}
}
}


#endif

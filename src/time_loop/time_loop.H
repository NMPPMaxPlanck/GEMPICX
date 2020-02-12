#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <infrastructure.H>
#include <maxwell_yee.H>
#include <diagnostics.H>
#include <particle_mesh_coupling.H>

#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_CoordSys.H>
#include <AMReX_PlotFileUtil.H>

#ifndef TIME_LOOP_H_
#define TIME_LOOP_H_

void time_loop (infrastructure infra,
                maxwell_yee *mw_yee,
                particle_groups *part_gr,
                diagnostics *diagn,
                bool output_bool = false) {

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {

        //--------------------------------------------------------------------------
        //push particles

        std::array<amrex::Real,GEMPIC_VDIM> eres;
        std::array<amrex::Real,GEMPIC_BDIM> bres;

        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                auto& particles = pti.GetArrayOfStructs(); // get particles
                const long np  = pti.numParticles();

                // loop over particles
                for (int pp=0;pp<np;pp++){
                    for (int cc=0;cc<GEMPIC_VDIM;cc++){
                        amrex::Array4<amrex::Real> const& earr = (*(*mw_yee).E_Array[cc])[pti].array();

                        //E-field at particle position
                        eres[cc] = gempic_interpolate_cic(particles[pp], earr, infra.ploE[cc], infra.dxi);
                    }

                    for (int cc=0;cc<GEMPIC_BDIM;cc++){
                        amrex::Array4<amrex::Real> const& barr = (*(*mw_yee).B_Array[cc])[pti].array();


                        //B-field at particle position
                        bres[cc] = gempic_interpolate_cic(particles[pp], barr, infra.ploB[cc], infra.dxi);
                    }
                    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                    std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);
                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }
        //--------------------------------------------------------------------------
        //update fields
        // deposit charge in J one component and particle at a time

        // set J to 0 before we start depositing values from current particles
        for (int i=0; i<GEMPIC_VDIM; i++) {
            (*(*mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }

        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                for(int cc=0;cc<GEMPIC_VDIM;cc++){
                    amrex::Array4<amrex::Real> const& jarr = (*(*mw_yee).J_Array[cc])[pti].array();
                    for (int pp=0;pp<np;pp++) {
                        gempic_deposit_J_cic(particles[pp], (*part_gr).charge[spec], cc, jarr,
                                             infra.ploE[cc], infra.dxi);
                    }
                }
            }
        }
        //add values from ghost cells and periodic boundary
        for (int i=0; i<GEMPIC_VDIM; i++) {
            (*(*mw_yee).J_Array[i]).SumBoundary(infra.geom.periodicity());
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }

        // field updates:
        (*mw_yee).advance(infra, (*mw_yee).dt);

        // save norms for diagnostics
        (*diagn).save_step_data(t_step+1, infra, mw_yee, part_gr, output_bool);

    } // end time loop
    (*diagn).save_all_to_textfile((*mw_yee).dt);
}


#endif

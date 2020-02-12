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
                       particle_groups *part_gr,
                       diagnostics *diagn) {

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {

        //-------------------------------------------------------------------------
        //push particles & get J
        (*mw_yee).FillBD(infra);

        // set J to 0 before we start depositing values from current particles
        (*(*mw_yee).J_Array[0]).setVal(0.0, 0); // value and component
#if (GEMPIC_VDIM > 1)
        (*(*mw_yee).J_Array[1]).setVal(0.0, 0);
#endif
#if (GEMPIC_VDIM > 2)
        (*(*mw_yee).J_Array[2]).setVal(0.0, 0);
#endif

        std::array<amrex::Real,GEMPIC_VDIM> eres;
        std::array<amrex::Real,GEMPIC_BDIM> bres;
        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                auto& particles = pti.GetArrayOfStructs(); // get particles
                const long np  = pti.numParticles();

                amrex::Array4<amrex::Real> const& jarrx = (*(*mw_yee).J_Array[0])[pti].array();
#if (GEMPIC_VDIM > 1)
                amrex::Array4<amrex::Real> const& jarry = (*(*mw_yee).J_Array[1])[pti].array();
#endif
#if (GEMPIC_VDIM > 2)
                amrex::Array4<amrex::Real> const& jarrz = (*(*mw_yee).J_Array[2])[pti].array();
#endif

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
                    amrex::Real part_weight = particles[pp].rdata(GEMPIC_VDIM);
                    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> avgPos = newPos;
                    for (int cc=0;cc<GEMPIC_SPACEDIM;cc++) {
                        avgPos[cc] = 0.5*(avgPos[cc] + particles[pp].pos(cc));
                        avgPos[cc+GEMPIC_SPACEDIM] = 0.5*(avgPos[cc+GEMPIC_SPACEDIM] + particles[pp].rdata(cc));
                    }
                    std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);

                    gempic_deposit_J_cic_global(avgPos, part_weight, (*part_gr).charge[spec], 0, jarrx, infra.ploE[0], infra.dxi);
#if (GEMPIC_VDIM > 1)
                    gempic_deposit_J_cic_global(avgPos, part_weight, (*part_gr).charge[spec], 1, jarry, infra.ploE[1], infra.dxi);
#endif
#if (GEMPIC_VDIM > 2)
                    gempic_deposit_J_cic_global(avgPos, part_weight, (*part_gr).charge[spec], 2, jarrz, infra.ploE[2], infra.dxi);
#endif
                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }
        //--------------------------------------------------------------------------
        //update fields

        //add values from ghost cells and periodic boundary
        (*(*mw_yee).J_Array[0]).SumBoundary(infra.geom.periodicity());
#if (GEMPIC_VDIM > 1)
        (*(*mw_yee).J_Array[1]).SumBoundary(infra.geom.periodicity());
#endif
#if (GEMPIC_VDIM > 2)
        (*(*mw_yee).J_Array[2]).SumBoundary(infra.geom.periodicity());
#endif
        std::cout << "finished step " << t_step+1 << std::endl;

        // field updates:
        (*mw_yee).advance(infra, (*mw_yee).dt);
        // save norms for diagnostics
        (*diagn).save_step_data(t_step+1, infra, mw_yee, part_gr, true);

    } // end time loop
    (*diagn).save_all_to_textfile((*mw_yee).dt);
}


#endif

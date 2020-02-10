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

#ifndef TIME_LOOP_AVG_H_
#define TIME_LOOP_AVG_H_

void time_loop_avg (infrastructure infra,
                    maxwell_yee *mw_yee,
                    particle_groups *part_gr,
                    double (*initB[GEMPIC_BDIM]) (std::array<double,GEMPIC_SPACEDIM> x,double k),
                    amrex::Real initk) {

    bool output_bool = false;
    diagnostics diagn((*mw_yee).nsteps);

    std::array<amrex::Real,GEMPIC_SPACEDIM> plo;
    std::array<amrex::Real,GEMPIC_SPACEDIM> dxi;
    for (int cc=0;cc<GEMPIC_SPACEDIM;cc++){
        plo[cc] = infra.geom.ProbLo()[cc];
        dxi[cc] = 1.0/infra.dx[cc];
    }

    // initial rho by: deposition of particle charge
    //Set rho to 0
    (*mw_yee).rho.setVal(0.0, 0); // value and component

    // Deposit charges:
    for (int spec=0;spec<(*part_gr).n_species;spec++) {
        (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
            amrex::Box tilebox;
            amrex::FArrayBox local_rho;

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
    (*mw_yee).FillBD(infra);

    // output rho
    if (output_bool){
        std::ofstream ofs("rho.output", std::ofstream::out);
        for (amrex::MFIter mfi((*mw_yee).rho); mfi.isValid(); ++mfi ) {
            amrex::Print(ofs) << (*mw_yee).rho[mfi] << std::endl;
        }
        ofs.close();
    }

    //initial E by rho
    (*mw_yee).phi_init.setVal(0.0, 0);
    (*mw_yee).solve_poisson(infra, output_bool);

    // initial B_field
    (*mw_yee).initB(infra, initB, initk);
    (*mw_yee).FillBD(infra);

    //------------------------------------------------------------------------------
    //approximate e^1/2
    (*mw_yee).advance_E(infra, 0.5);
    //------------------------------------------------------------------------------
    //approximate x^1/2
    for (int spec=0;spec<(*part_gr).n_species;spec++) {
        for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
            auto& particles = pti.GetArrayOfStructs(); // get particles
            const long np  = pti.numParticles();
            for (int pp=0;pp<np;pp++){
                std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle_pos(particles[pp], (*mw_yee).dt, infra, 0.5);
                std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);
            }
        }
        (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
    }

    //------------------------------------------------------------------------------
    //Time loop
    diagn.save_step_data(0, infra, mw_yee, part_gr, output_bool); //false: slices are not saved

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
        (*mw_yee).advance_time();

        //--------------------------------------------------------------------------
        // update B
        (*mw_yee).FillBD(infra);
        (*mw_yee).advance_B(infra, 1.);

        //--------------------------------------------------------------------------
        //deposit charge in J and push particles

        // set J to 0 before we start depositing values from current particles
        for (int i=0; i<GEMPIC_VDIM; i++) {
            (*(*mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
        }

        (*mw_yee).FillBD(infra);

        std::array<amrex::Real,GEMPIC_VDIM> eres; //field values interpolated at particle position
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
                // loop over particles: add contribution of old particle position to J and push

                for (int pp=0;pp<np;pp++){
                    // charge deposition
                    gempic_deposit_J_cic_class(particles[pp], (*part_gr).charge[spec], 0, jarrx,
                                               infra.ploE[0], dxi, *((*mw_yee).E_Index[0]), 0.5);
#if (GEMPIC_VDIM > 1)
                    gempic_deposit_J_cic_class(particles[pp], (*part_gr).charge[spec], 1, jarry,
                                               infra.ploE[1], dxi, *((*mw_yee).E_Index[1]), 0.5);
#endif
#if (GEMPIC_VDIM > 2)
                    gempic_deposit_J_cic_class(particles[pp], (*part_gr).charge[spec], 2, jarrz,
                                               infra.ploE[2], dxi, *((*mw_yee).E_Index[2]), 0.5);
#endif
                    //field interpolation
                    for (int cc=0;cc<GEMPIC_VDIM;cc++){
                        amrex::Array4<amrex::Real> const& earr = (*(*mw_yee).E_Array[cc])[pti].array();

                        //E-field at particle position
                        eres[cc] = gempic_interpolate_cic(particles[pp], earr, infra.ploE[cc], dxi);
                    }
                    for (int cc=0;cc<GEMPIC_BDIM;cc++){
                        amrex::Array4<amrex::Real> const& barr = (*(*mw_yee).B_Array[cc])[pti].array();

                        //B-field at particle position
                        bres[cc] = gempic_interpolate_cic(particles[pp], barr, infra.ploB[cc], dxi);
                    }

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

                amrex::Array4<amrex::Real> const& jarrx = (*(*mw_yee).J_Array[0])[pti].array();
#if (GEMPIC_VDIM > 1)
                amrex::Array4<amrex::Real> const& jarry = (*(*mw_yee).J_Array[1])[pti].array();
#endif
#if (GEMPIC_VDIM > 2)
                amrex::Array4<amrex::Real> const& jarrz = (*(*mw_yee).J_Array[2])[pti].array();
#endif

                for (int pp=0;pp<np;pp++) {
                    gempic_deposit_J_cic_class(particles[pp], (*part_gr).charge[spec], 0, jarrx,
                                               infra.ploE[0], dxi, *((*mw_yee).E_Index[0]), 0.5);
#if (GEMPIC_VDIM > 1)
                    gempic_deposit_J_cic_class(particles[pp], (*part_gr).charge[spec], 1, jarry,
                                               infra.ploE[1], dxi, *((*mw_yee).E_Index[1]), 0.5);
#endif
#if (GEMPIC_VDIM > 2)
                    gempic_deposit_J_cic_class(particles[pp], (*part_gr).charge[spec], 2, jarrz,
                                               infra.ploE[2], dxi, *((*mw_yee).E_Index[2]), 0.5);
#endif

                }
            }
        }
        //add values from ghost cells and periodic boundary
        for (int i=0; i<GEMPIC_VDIM; i++) {
            (*(*mw_yee).J_Array[i]).SumBoundary(infra.geom.periodicity());
        }

        (*mw_yee).FillBD(infra);

        // advance E
        (*mw_yee).advance_E(infra, 1.);

        // save norms for diagnostics
        diagn.save_step_data(t_step+1, infra, mw_yee, part_gr, output_bool); // false: slices are not saved

        // save temporary results every n steps
        if ((t_step+1)%5 == 0) { //& output_bool
            diagn.save_all_to_textfile((*mw_yee).dt, "tmp2/PIC_save.output"+std::to_string(t_step+1), t_step+1);
        }
        amrex::Print() << "finished step " << t_step+1 << std::endl;
    } // end time loop

    diagn.save_all_to_textfile((*mw_yee).dt);
}


#endif

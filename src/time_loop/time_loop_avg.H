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
    amrex::Real rhosum = (*mw_yee).rho.sum(0);

    //add values from ghost cells and periodic boundary
    (*mw_yee).rho.SumBoundary(infra.geom.periodicity());
    //(*mw_yee).FillBD(infra);

    amrex::Real rhomean = rhosum/(infra.n_cell[0]
        #if (GEMPIC_SPACEDIM > 1)
            *infra.n_cell[1]
        #endif
        #if (GEMPIC_SPACEDIM > 2)
            *infra.n_cell[2]
        #endif
            );
    //(*mw_yee).rho.plus(-rhomean,0,1);
    (*mw_yee).FillBD(infra);
    amrex::Vector<std::string> varnames = {"rho"};
    WriteSingleLevelPlotfile ("SliceData/rho_plotfile_0", (*mw_yee).rho, varnames,
                              infra.geom, 0, 0);

    //initial E by rho
    (*mw_yee).phi_init.setVal(0.0, 0);

    std::ofstream ofs("rho.output", std::ofstream::out);
    for (amrex::MFIter mfi((*mw_yee).rho); mfi.isValid(); ++mfi ) {
        amrex::Print(ofs) << (*mw_yee).rho[mfi] << std::endl;
    }
    ofs.close();
    
    (*mw_yee).solve_poisson(infra);

    // initial B_field
    std::array<double,GEMPIC_SPACEDIM> x;
    for (int i = 0; i<GEMPIC_BDIM; i++) {
      for ( amrex::MFIter mfi(*(*mw_yee).B_Array[i]); mfi.isValid(); ++mfi )
      {
    const amrex::Box& bx = mfi.validbox();
    amrex::IntVect lo = {bx.smallEnd()};
    amrex::IntVect hi = {bx.bigEnd()};
#if (GEMPIC_SPACEDIM > 2)
    for(int k=lo[2]; k<=hi[2]; k++){
      x[2] = infra.geom.ProbLo()[2] + ((double)k+0.5-0.5*(*(*mw_yee).B_Index[i])[2])*infra.dx[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
      for(int j=lo[1]; j<=hi[1]; j++){
        x[1] = infra.geom.ProbLo()[1] + ((double)j+0.5-0.5*(*(*mw_yee).B_Index[i])[1])*infra.dx[1];
#endif
        for(int l=lo[0]; l<=hi[0]; l++){
          x[0] = infra.geom.ProbLo()[0] + ((double)l+0.5-0.5*(*(*mw_yee).B_Index[i])[0])*infra.dx[0];
          // the box for these values:
          amrex::Box cc({AMREX_D_DECL(l,j,k)}, {AMREX_D_DECL(l,j,k)}, *(*mw_yee).B_Index[i]);
          (*(*mw_yee).B_Array[i])[mfi].setVal((*initB[i])(x,initk), cc, 0, 1);
        }
#if (GEMPIC_SPACEDIM > 2)
      }
#endif
#if (GEMPIC_SPACEDIM > 1)
    }
#endif
    // next mfi
      }
    }

    //------------------------------------------------------------------------------
    //approximate x^1/2 and e^1/2
    (*mw_yee).advance_E(infra, 0.5);
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
    std::ofstream ofs2("PIC_6dim.output", std::ofstream::out);
    amrex::Real time = 0.0;
    diagn.save_step_data(0, infra, mw_yee, part_gr, true);

    for (int t_step=0;t_step<(*mw_yee).nsteps;t_step++) {
        time += (*mw_yee).dt;
        (*mw_yee).advance_time();

        //--------------------------------------------------------------------------
        // update B
        (*mw_yee).FillBD(infra);
        (*mw_yee).advance_B(infra, 1.);

        //--------------------------------------------------------------------------
        //push particles

        // set J to 0 before we start depositing values from current particles
        (*(*mw_yee).J_Array[0]).setVal(0.0, 0); // value and component
#if (GEMPIC_VDIM > 1)
        (*(*mw_yee).J_Array[1]).setVal(0.0, 0);
#endif
#if (GEMPIC_VDIM > 2)
        (*(*mw_yee).J_Array[2]).setVal(0.0, 0);
#endif

        (*mw_yee).FillBD(infra);

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
                // loop over particles: add contribution of old particle position and push
                for (int pp=0;pp<np;pp++){
                    gempic_deposit_J_cic(particles[pp], (*part_gr).charge[spec], 0, jarrx,
                                              infra.ploE[0], dxi, 0.5);
#if (GEMPIC_VDIM > 1)
                    gempic_deposit_J_cic(particles[pp], (*part_gr).charge[spec], 1, jarry,
                                              infra.ploE[1], dxi, 0.5);
#endif
#if (GEMPIC_VDIM > 2)
                    gempic_deposit_J_cic(particles[pp], (*part_gr).charge[spec], 2, jarrz,
                                              infra.ploE[2], dxi, 0.5);
#endif
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
                    //std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                    std::array<amrex::Real,GEMPIC_SPACEDIM+GEMPIC_VDIM> newPos = push_particle_boris(particles[pp], (*mw_yee).dt, chargemass, eres, bres, infra);
                    std::copy(begin(newPos),end(newPos),&particles.data()[pp*(GEMPIC_SPACEDIM+GEMPIC_VDIM+2)]);
                    amrex::Print(ofs2) << newPos[0] << "," << newPos[1] << "," <<
                                   #if (GEMPIC_SPACEDIM > 1)
                                                       newPos[2] << "," << newPos[3] << "," <<
                                   #endif
                                   #if (GEMPIC_SPACEDIM > 2)
                                                       newPos[4] << "," << newPos[5] <<
                                   #endif
                                                       std::endl;
                }
            }
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        }
        //--------------------------------------------------------------------------
        //update fields
        // deposit remaining charge in J one component and particle at a time

        for (int spec=0;spec<(*part_gr).n_species;spec++) {
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
                auto& particles = pti.GetArrayOfStructs();
                const long np  = pti.numParticles();

                for(int cc=0;cc<GEMPIC_VDIM;cc++){
                    amrex::Array4<amrex::Real> const& jarr = (*(*mw_yee).J_Array[cc])[pti].array();
                    for (int pp=0;pp<np;pp++) {
                        gempic_deposit_J_cic(particles[pp], (*part_gr).charge[spec], cc, jarr,
                                             infra.ploE[cc], dxi, 0.5);
                    }
                }
            }
        }
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
        (*mw_yee).FillBD(infra);

        (*mw_yee).advance_E(infra, 1.);
        // save norms for diagnostics
        diagn.save_step_data(t_step+1, infra, mw_yee, part_gr, true);

        // save temporary results every n steps
        if ((t_step+1)%3 == 0) {
            diagn.save_all_to_textfile((*mw_yee).dt, "tmp/PIC_save.output"+std::to_string(t_step+1), t_step+1);
        }

    } // end time loop
    ofs2.close();
    diagn.save_all_to_textfile((*mw_yee).dt);
}


#endif

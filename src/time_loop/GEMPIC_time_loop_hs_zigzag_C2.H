#ifndef TIME_LOOP_HS_ZIGZAG_C2_
#define TIME_LOOP_HS_ZIGZAG_C2_

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
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_SumBoundary.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Profiling;

namespace Gempic {
namespace Time_Loop {

/*
    This is the time propagator with a Hamiltonian splitting in a FEM formulation with mass lumping for Lagrange elements of order one and zero

    This propagator is currently not implemented for the 2d2v case. In this case, the operator H_p3 is dismissed and the operators H_p1 and H_p2
    must be reimplemented only producing the b3 updates. In principle the operators are exactly the same but they should just return one bfield
    value - the one that concerns b_z - and it should read out the first component of the bfield vector (which saves the b_z in this case).

   */


template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool electromagnetic=true, bool profiling=true>
void time_loop_hs_zigzag_C2_step (computational_domain infra,
                       maxwell_yee<vdim> *mw_yee,
                       amrex::Real coeff,
                       particle_groups<vdim, numspec> *part_gr,
                       timers *profiling_timers) {


    // Initialize the counters for timing the various parts of the algorithm
    bool profiling_count = true;

        (*mw_yee).advance_time(coeff);

        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_fields -= MPI_Wtime();
        // Operator H_B for dt/2
        if ( electromagnetic ) {
            mw_yee->template hodge_full<degmw>(infra, &(mw_yee->B_Array), &(mw_yee->HB_Array), false);
            mw_yee->advance_E(infra, mw_yee->dt*0.5*coeff, true, false, &(mw_yee->HB_Array), &(mw_yee->E_Array));

        //--------------------------------------------------------------------------
        // Operator H_E for dt/2 (field part)
            mw_yee->template hodge_full<degmw>(infra, &(mw_yee->E_Array), &(mw_yee->HE_Array), true);
            mw_yee->advance_B(infra, mw_yee->dt*0.5*coeff, &(mw_yee->HE_Array), &(mw_yee->B_Array));
          }
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_fields += MPI_Wtime();

        //--------------------------------------------------------------------------
        //deposit charge in J and push particles

        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_jboundary -= MPI_Wtime();
        // set J to 0 before we start depositing values from current particles
        for (int i=0; i<vdim; i++) {
            (*(*mw_yee).J_Array[i]).setVal(0.0, 0); // value and component
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_jboundary += MPI_Wtime();

        for (int spec=0;spec<numspec;spec++) {
            if (profiling_timers->profiling && profiling_count)
              profiling_timers->counter_deposition -= MPI_Wtime();
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];
            amrex::Real charge = (*part_gr).charge[spec];

            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                amrex::Particle<vdim+1,0>*  AMREX_RESTRICT particles= &(pti.GetArrayOfStructs()[0]); // get particles
                const long np  = pti.numParticles();

                amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> eA;
                amrex::GpuArray<amrex::Array4<amrex::Real>,int(vdim/2.5)*2+1> bA;

                for (int cc=0;cc < vdim;cc++) {
                    jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                    eA[cc] = (*(*mw_yee).HE_Array[cc])[pti].array();
                }
                for (int cc=0;cc < int(vdim/2.5)*2+1;cc++) {
                    bA[cc] = (*(*mw_yee).B_Array[cc])[pti].array();
                }

                // loop over particles: add contribution of old particle position to J and push

                amrex::Real dt = (*mw_yee).dt;
                   amrex::ParallelFor(np,
                                                                [=] AMREX_GPU_DEVICE (long pp )
                                                                 {
                //for ( int pp=0; pp<np;pp++){


                    splines_at_particles<degx,degy,degz> spline;
                    spline1d_at_particles<degx> spline_new_x;
                    spline1d_at_particles<degx> spline_old_x;
                    spline1d_at_particles<degy> spline_new_y;
                    spline1d_at_particles<degy> spline_old_y;
                    spline1d_at_particles<degz> spline_new_z;
                    spline1d_at_particles<degz> spline_old_z;
                    // Read out particle position
                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> pos;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                      pos[d]= particles[pp].pos(d);
                    // evaluate the spline
                    spline.init_particles(pos , infra.plo, infra.dxi);
                    // evaluate the electric field
                    amrex::GpuArray<amrex::Real, vdim> efield = evaluate_efield<vdim,degx,degy,degz>(spline,  eA);

                    // push v with the electric field
                    amrex::GpuArray<amrex::Real,vdim> newPosE = push_v_efield<amrex::Particle<vdim+1>,vdim>(particles[pp], dt*0.5*coeff, chargemass, efield);
                    for (int vd=0;vd<vdim; vd++)
                        (particles[pp].rdata(vd)) = newPosE[vd];

                    amrex::Real weight = charge*particles[pp].rdata(vdim)*infra.dxi[GEMPIC_SPACEDIM];


                    amrex::GpuArray<amrex::Real,vdim> vel;
                    for (unsigned int d=0;d<vdim;++d)
                      vel[d] = particles[pp].rdata(d);
                    amrex::GpuArray<amrex::Real, 2> bfields;
                    // push x1
                    amrex::Real x1_new = pos[0] +  dt*0.5*coeff*vel[0];
                    spline_new_x.init_position( x1_new, infra.plo[0], infra.dxi[0] );
                    spline_old_x.init_position( pos[0], infra.plo[0], infra.dxi[0] );
                    spline.template init_position<0,degx>( x1_new, infra.plo[0], infra.dxi[0] );

                    amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz))+4> primitive;
                    accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degx, degy, degz, 0, std::max(degx, std::max(degy, degz))+4>( spline,
                                                                                                               spline_new_x,
                                                                                                               spline_old_x,
                                                                                                               weight,
                                                                                                               infra.geom.CellSize(),
                                                                                                               bA,
                                                                                                               jA,
                                                                                                               bfields,
                                                                                                               &primitive);


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
                      amrex::Real x2_new = pos[1] + dt*0.5*coeff*vel[1];

                      spline_new_y.init_position( x2_new, infra.plo[1], infra.dxi[1] );
                      spline_old_y.init_position( pos[1], infra.plo[1], infra.dxi[1] );
                      spline.template init_position<1,degy>( x2_new, infra.plo[1], infra.dxi[1] );

                      accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degy, degz, degx, 1, std::max(degx, std::max(degy, degz))+4>( spline,
                                                        spline_new_y,
                                                        spline_old_y,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        bA,
                                                        jA,
                                                        bfields,
                                                        &primitive);

                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[1];
                      if (vdim > 2 )
                        vel[2] = vel[2] - chargemass*bfields[0];
                      pos[1] = x2_new;
#else
                      amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> bfield = evaluate_fem_bfield<vdim,degx,degy,degz>(pos,infra.dxi, infra.plo,  bA);
                      gempic_deposit_rho<degx,degy,degz> (spline, weight*vel[1]*dt*coeff,
                                               jA[1]);
                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[0]*vel[1]*dt*coeff;
  #endif
}
                    if(vdim > 2 ) {
#if GEMPIC_SPACEDIM == 3
                    // push x3
                    amrex::Real x3_new = pos[2] +  dt*coeff*vel[2];

                    spline_new_z.init_position( x3_new, infra.plo[2], infra.dxi[2] );
                    spline_old_z.init_position( pos[2], infra.plo[2], infra.dxi[2] );
                    spline.template init_position<2,degz>( x3_new, infra.plo[2], infra.dxi[2] );

                    accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degz, degx, degy, 2, std::max(degx, std::max(degy, degz))+4>( spline,
                                                      spline_new_z,
                                                      spline_old_z,
                                                      weight,
                                                      infra.geom.CellSize(),
                                                      bA,
                                                      jA,
                                                      bfields,
                                                      &primitive);
                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass*bfields[0];
                    vel[1] = vel[1] + chargemass*bfields[1];
                    pos[2] = x3_new;
#else
                    amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> bfield = evaluate_bfield<vdim,degx,degy,degz>(spline,  bA);
                    gempic_deposit_rho<degx,degy,degz> (spline, weight*vel[2]*dt*coeff,
                                             jA[2]);
                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass*bfields[1]*vel[2]*dt*coeff;
                    vel[1] = vel[1] + chargemass*bfields[0]*vel[2]*dt*coeff;
#endif


}

                    if ( vdim > 1 )
                      {
#if GEMPIC_SPACEDIM > 1
                        // push x2
                      amrex::Real x2_new = pos[1] +  dt*0.5*coeff*vel[1];

                      spline_new_y.init_position( x2_new, infra.plo[1], infra.dxi[1] );
                      spline_old_y.init_position( pos[1], infra.plo[1], infra.dxi[1] );
                      spline.template init_position<1,degy>( x2_new, infra.plo[1], infra.dxi[1] );

                      accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degy, degz, degx, 1, std::max(degx, std::max(degy, degz))+4>( spline,
                                                        spline_new_y,
                                                        spline_old_y,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        bA,
                                                        jA,
                                                        bfields,
                                                        &primitive);

                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[1];
                      if (vdim > 2 )
                        vel[2] = vel[2] - chargemass*bfields[0];
                      pos[1] = x2_new;
#else
                      amrex::GpuArray<amrex::Real, int(vdim/2.5)*2+1> bfield = evaluate_fem_bfield<vdim,degx,degy,degz>(pos,infra.dxi, infra.plo,  bA);
                      gempic_deposit_rho<degx,degy,degz> (spline, weight*vel[1]*dt*coeff,
                                               jA[1]);
                      // Push v2 and v3
                      vel[0] = vel[0] + chargemass*bfields[0]*vel[1]*dt*coeff;
  #endif
}

                    // second half step of Hp1
                    x1_new = pos[0] +  dt*0.5*coeff*vel[0];
                                        spline_new_x.init_position( x1_new, infra.plo[0], infra.dxi[0] );
                                        spline_old_x.init_position( pos[0], infra.plo[0], infra.dxi[0] );
                                        spline.template init_position<0,degx>( x1_new, infra.plo[0], infra.dxi[0] );

                                        accumulate_j_update_v_C2<splines_at_particles<degx,degy,degz>, vdim, degx, degy, degz, 0, std::max(degx, std::max(degy, degz))+4>( spline,
                                                                                                                                   spline_new_x,
                                                                                                                                   spline_old_x,
                                                                                                                                   weight,
                                                                                                                                   infra.geom.CellSize(),
                                                                                                                                   bA,
                                                                                                                                   jA,
                                                                                                                                   bfields,
                                                                                                                                   &primitive);


                                        // Push v2 and v3
                                        if (vdim > 1)
                                          vel[1] = vel[1] - chargemass*bfields[0];
                                        if (vdim > 2)
                                          vel[2] = vel[2] + chargemass*bfields[1];
                                        pos[0] = x1_new;

                    for (int xd=0;xd<GEMPIC_SPACEDIM; xd++)
                        particles[pp].pos(xd) = pos[xd];
                    //std::copy(pos.begin(),pos.end(),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                    for (int vd=0;vd<vdim; vd++)
                        (particles[pp].rdata(vd)) = vel[vd];
                    //std::copy(vel.begin(),vel.end(),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)+GEMPIC_SPACEDIM]);
//exit(0);
                }
            );
            }
            if (profiling_timers->profiling && profiling_count)
              profiling_timers->counter_deposition += MPI_Wtime();
            if (profiling_timers->profiling && profiling_count)
              profiling_timers->counter_redistribute -= MPI_Wtime();
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in            
            if (profiling_timers->profiling && profiling_count)
              profiling_timers->counter_redistribute += MPI_Wtime();
          }


        //add values from ghost cells and periodic boundary        
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_jboundary -= MPI_Wtime();
        Gempic_SumBoundary(mw_yee, &infra); // Sums Boundary of J and Overwrites overlap
        for (int i=0; i<vdim; i++) {
            (*(*mw_yee).J_Array[i]).FillBoundary(infra.geom.periodicity());
        }
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_jboundary += MPI_Wtime();
        /*amrex::Vector<std::string> varnames = { "J_x"};
        amrex::Real timeo = t_step*(*mw_yee).dt*coeff;
        WriteSingleLevelPlotfile("current_C2", *(*mw_yee).J_Array[0], varnames,
        infra.geom, timeo, 0);
        //abort();*/


        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_fields -= MPI_Wtime();
        // advance E from j (field part of H_p)
        mw_yee->advance_E(infra, 1.0, false, true, &(mw_yee->HB_Array), &(mw_yee->E_Array));      
        mw_yee->template hodge_full<degmw>(infra, &(mw_yee->E_Array), &(mw_yee->HE_Array), true);
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_fields += MPI_Wtime();

        // Second particle loop for particle part from H_E
        for (int spec=0;spec<numspec;spec++) {
            if (profiling_timers->profiling && profiling_count)
             profiling_timers->counter_deposition -= MPI_Wtime();
            amrex::Real chargemass = (*part_gr).charge[spec]/(*part_gr).mass[spec];

            for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {

                amrex::Particle<vdim+1,0>*  AMREX_RESTRICT particles= &(pti.GetArrayOfStructs()[0]);
		//amrex::ArrayOfStructs<4, 0, amrex::DefaultAllocator>*  AMREX_RESTRICT particles = pti.GetArrayOfStructs(); // get particles
                const long np  = pti.numParticles();

                amrex::GpuArray<amrex::Array4<amrex::Real>,vdim> eA;

                for (int cc=0;cc < vdim;cc++) {
                    eA[cc] = (*(*mw_yee).HE_Array[cc])[pti].array();
                }

                amrex::Real dt = (*mw_yee).dt;
                // loop over particles: add contribution of old particle position to J and push
                amrex::ParallelFor(np,
                    [=] AMREX_GPU_DEVICE (long pp)
                {
                    amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> pos;
                    for (unsigned int d=0;d<GEMPIC_SPACEDIM;++d)
                      pos[d]= particles[pp].pos(d);

                    splines_at_particles<degx,degy,degz> splines;
                    // evaluate the spline
                    splines.init_particles(pos , infra.plo, infra.dxi);
                    // evaluate the electric field
                    amrex::GpuArray<amrex::Real, vdim> efield = evaluate_efield<vdim,degx,degy,degz>(splines,  eA);
                    // push v with the electric field
		    amrex::GpuArray<amrex::Real,vdim> newPosE = push_v_efield<amrex::Particle<vdim+1>,vdim>(particles[pp], dt*0.5*coeff, chargemass, efield);
		   for (int vd=0;vd<vdim; vd++)
			   (particles[pp].rdata(vd)) = newPosE[vd];
                });
            }
            if (profiling_timers->profiling && profiling_count)
             profiling_timers->counter_deposition += MPI_Wtime();
            if (profiling_timers->profiling && profiling_count)
              profiling_timers->counter_redistribute -= MPI_Wtime();
            (*(*part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
            if (profiling_timers->profiling && profiling_count)
              profiling_timers->counter_redistribute += MPI_Wtime();
          }

        //--------------------------------------------------------------------------
        // Operator H_E for dt/2 (field part)       
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_fields -= MPI_Wtime();
        if ( electromagnetic ) {
            //mw_yee->template hodge_full<degmw>(infra, &(mw_yee->E_Array), &(mw_yee->HE_Array), true);
            mw_yee->advance_B(infra, mw_yee->dt*0.5*coeff, &(mw_yee->HE_Array), &(mw_yee->B_Array));

          // Operator H_B for dt/2
            mw_yee->template hodge_full<degmw>(infra, &(mw_yee->B_Array), &(mw_yee->HB_Array), false);
            mw_yee->advance_E(infra, mw_yee->dt*0.5*coeff, true, false, &(mw_yee->HB_Array), &(mw_yee->E_Array));
        }
        if (profiling_timers->profiling && profiling_count)
          profiling_timers->counter_fields += MPI_Wtime();

}

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, bool electromagnetic=true, bool profiling=true, bool output=true>
void time_loop_hs_zigzag_C2 (computational_domain infra,
                       maxwell_yee<vdim> *mw_yee,
                       particle_groups<vdim, numspec> *part_gr,
                       diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn,
                       bool ctest,
                       std::string simulation_name,
                       int strang_order) {

    timers profiling_timers(true);

    if (output)
        amrex::AllPrintToFile(simulation_name+".tmp") << std::endl;

    if (profiling_timers.profiling)
        // profiling_timers.counter_all -= MPI_Wtime();
    for (int t_step=0;t_step<mw_yee->nsteps;t_step++) {
        if (profiling_timers.profiling && t_step == 0)
            profiling_timers.counter_all -= MPI_Wtime();
        switch (strang_order) {
        case 2:
            time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic, profiling>(infra, mw_yee, 1.0, part_gr, &profiling_timers);
            break;
        case 4:
            amrex::Real alpha = 1./(2.-pow(2.,1./3.));
            amrex::Real beta = 1. - 2.*alpha;

            time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic, profiling>(infra, mw_yee, alpha, part_gr, &profiling_timers);
            time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic, profiling>(infra, mw_yee, beta, part_gr, &profiling_timers);
            time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic, profiling>(infra, mw_yee, alpha, part_gr, &profiling_timers);
            break;
        }

        if (output)
            diagn->end_of_timestep(&profiling_timers, t_step, infra, mw_yee, part_gr, simulation_name+".tmp", ctest);

    }

    if (profiling_timers.profiling){
      profiling_timers.counter_all += MPI_Wtime();
      amrex::Print() << "Deposition time: " << profiling_timers.counter_deposition << ", redistribute time: " << profiling_timers.counter_redistribute
                     << ", field solver time: " << profiling_timers.counter_fields << ", sum boundary j time: " << profiling_timers.counter_jboundary
                     << ", diagnostics time: " << profiling_timers.counter_diagnostics << ", total time: " << profiling_timers.counter_all << std::endl;
      }
    if (output)
        diagn->save_all_to_textfile(mw_yee->dt, simulation_name+".tmp");


}

}
}


#endif

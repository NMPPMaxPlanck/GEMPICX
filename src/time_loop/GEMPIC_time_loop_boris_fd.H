#ifndef TIME_LOOP_BORIS_FD_H_
#define TIME_LOOP_BORIS_FD_H_

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Array.H>
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
#include <GEMPIC_SumBoundary.H>
#include <GEMPIC_checkpoint.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_profiling.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Profiling;

namespace Gempic
{
namespace Time_Loop
{
template <int vdim, int numspec, int degx, int degy, int degz, int degmw,
          bool electromagnetic = true, bool profiling = true>
void time_loop_boris_fd_step(computational_domain infra, maxwell_yee<vdim> *mw_yee,
                             amrex::Real coeff, particle_groups<vdim, numspec> *part_gr,
                             timers *profiling_timers)
{
    mw_yee->advance_time(coeff);

    if (profiling_timers->profiling) profiling_timers->counter_fields -= MPI_Wtime();
    if (electromagnetic)
    {
        //--------------------------------------------------------------------------
        // update B
        mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
        mw_yee->advance_B(infra, mw_yee->dt * coeff, mw_yee->HE_Array, mw_yee->B_Array);
    }
    if (profiling_timers->profiling) profiling_timers->counter_fields += MPI_Wtime();

    //--------------------------------------------------------------------------
    // deposit charge in J and push particles

    if (profiling_timers->profiling) profiling_timers->counter_jboundary -= MPI_Wtime();
    // set J to 0 before we start depositing values from current particles
    for (int i = 0; i < vdim; i++)
    {
        mw_yee->J_Array[i]->setVal(0.0, 0);  // value and component
        mw_yee->J_Array[i]->FillBoundary(infra.geom.periodicity());
    }
    if (profiling_timers->profiling) profiling_timers->counter_jboundary += MPI_Wtime();
    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real spec_charge = part_gr->charge[spec];
        if (profiling_timers->profiling) profiling_timers->counter_deposition -= MPI_Wtime();
        amrex::Real chargemass = part_gr->charge[spec] / part_gr->mass[spec];

        for (amrex::ParIter<vdim + 1, 0, 0, 0> pti(*part_gr->mypc[spec], 0); pti.isValid(); ++pti)
        {
            amrex::Particle<vdim + 1, 0> *AMREX_RESTRICT particles =
                &(pti.GetArrayOfStructs()[0]);  // get particles
            const long np = pti.numParticles();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

            for (int cc = 0; cc < vdim; cc++)
            {
                jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                eA[cc] = (*(*mw_yee).E_Array[cc])[pti].array();
            }
            for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
            {
                bA[cc] = (*(*mw_yee).B_Array[cc])[pti].array();
            }

            amrex::Real time_step = mw_yee->dt;

            // loop over particles: add contribution of old particle position to J and push
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = particles[pp].pos(d);
                splines_at_particles<degx, degy, degz> spline;
                spline.init_particles(pos, infra.plo, infra.dxi);

                amrex::GpuArray<amrex::Real, vdim> particle_vel;
                for (int comp = 0; comp < vdim; comp++)
                    particle_vel[comp] = particles[pp].rdata(comp);
                amrex::Real charge_weight_dxi_coeff =
                    spec_charge * particles[pp].rdata(vdim) * infra.dxi[GEMPIC_SPACEDIM] * 0.5;

                amrex::GpuArray<amrex::Real, vdim + int(vdim / 2.5) * 2 + 1> field_interp =
                    gempic_deposit_J_and_interpolate<vdim, degx, degy, degz>(
                        charge_weight_dxi_coeff, particle_vel, spline, jA, eA, bA, true);

                amrex::GpuArray<amrex::Real, vdim> eres;
                for (int comp = 0; comp < vdim; comp++)
                {
                    eres[comp] = field_interp[comp];
                }
                amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> bres;
                for (int comp = 0; comp < int(vdim / 2.5) * 2 + 1; comp++)
                {
                    bres[comp] = field_interp[vdim + comp];
                }
                for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                {
                    pos[comp] = particles[pp].pos(comp);
                }
                amrex::GpuArray<amrex::Real, vdim> vel;
                for (int comp = 0; comp < vdim; comp++)
                {
                    vel[comp] = particles[pp].rdata(comp);
                }

                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> newPos =
                    push_particle_boris<vdim>(pos, infra.plo, infra.phi, vel, time_step * coeff,
                                              chargemass, eres, bres);
                for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) particles[pp].pos(xd) = newPos[xd];
                for (int vd = 0; vd < vdim; vd++)
                    (particles[pp].rdata(vd)) = newPos[GEMPIC_SPACEDIM + vd];
            });
        }
        if (profiling_timers->profiling) profiling_timers->counter_deposition += MPI_Wtime();
        if (profiling_timers->profiling) profiling_timers->counter_redistribute -= MPI_Wtime();
        part_gr->mypc[spec]->Redistribute();  // assign particles to the tile they are in
        if (profiling_timers->profiling) profiling_timers->counter_redistribute += MPI_Wtime();
    }

    /*// Here we check the load imbalance
    {
      long parts_per_cpu = 0;
      long max_per_cpu = 0;
      long min_per_cpu = 0;
      for (int spec=0;spec<numspec;spec++) {
        for (amrex::ParIter<vdim+1,0,0,0> pti(*(*part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
          parts_per_cpu += pti.numParticles();
          }
        }
      MPI_Allreduce(&parts_per_cpu, &max_per_cpu, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&parts_per_cpu, &min_per_cpu, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);

      amrex::PrintToFile("particles") << t_step << " " << max_per_cpu << " " << min_per_cpu << " "
    << parts_per_cpu << std::endl;
    }*/

    //--------------------------------------------------------------------------
    // update fields
    // deposit remaining charge in J one particle at a time

    if (profiling_timers->profiling) profiling_timers->counter_deposition -= MPI_Wtime();
    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real spec_charge = part_gr->charge[spec];
        for (amrex::ParIter<vdim + 1, 0, 0, 0> pti(*part_gr->mypc[spec], 0); pti.isValid(); ++pti)
        {
            amrex::Particle<vdim + 1, 0> *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
            const long np = pti.numParticles();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

            for (int cc = 0; cc < vdim; cc++)
            {
                jA[cc] = (*(mw_yee->J_Array)[cc])[pti].array();
            }

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = particles[pp].pos(d);
                splines_at_particles<degx, degy, degz> spline;
                spline.init_particles(pos, infra.plo, infra.dxi);

                amrex::GpuArray<amrex::Real, vdim> particle_vel;
                for (int comp = 0; comp < vdim; comp++)
                    particle_vel[comp] = particles[pp].rdata(comp);
                amrex::Real charge_weight_dxi_coeff =
                    spec_charge * particles[pp].rdata(vdim) * infra.dxi[GEMPIC_SPACEDIM] * 0.5;

                amrex::GpuArray<amrex::Real, vdim + int(vdim / 2.5) * 2 + 1> field_interp =
                    gempic_deposit_J_and_interpolate<vdim, degx, degy, degz>(
                        charge_weight_dxi_coeff, particle_vel, spline, jA, eA, bA, false);
            });
        }
    }
    if (profiling_timers->profiling)
    {
        profiling_timers->counter_deposition += MPI_Wtime();

        profiling_timers->counter_jboundary -= MPI_Wtime();
    }
    Gempic_SumBoundary(mw_yee, &infra);  // Sums Boundary of J and Overwrites overlap

    for (int i = 0; i < vdim; i++)
    {
        mw_yee->J_Array[i]->FillBoundary(infra.geom.periodicity());
    }
    if (profiling_timers->profiling) profiling_timers->counter_jboundary += MPI_Wtime();

    if (profiling_timers->profiling) profiling_timers->counter_fields -= MPI_Wtime();
    if (electromagnetic)
    {
        // advance E
        mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        mw_yee->advance_E(infra, mw_yee->dt * coeff, true, true, mw_yee->HB_Array, mw_yee->E_Array);
    }
    else
    {
        // advance E from j
        mw_yee->advance_E(infra, mw_yee->dt * coeff, false, true, mw_yee->HB_Array,
                          mw_yee->E_Array);
    }
    if (profiling_timers->profiling) profiling_timers->counter_fields += MPI_Wtime();
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw,
          bool electromagnetic = true, bool profiling = true>
void time_loop_boris_fd(computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        particle_groups<vdim, numspec> *part_gr,
                        diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool ctest,
                        std::string simulation_name, int strang_order)
{
    BL_PROFILE("time_loop_boris_fd");

    timers profiling_timers(true);

    if (profiling_timers.profiling) profiling_timers.counter_all -= MPI_Wtime();
    for (int t_step = 0; t_step < mw_yee->nsteps; t_step++)
    {
        BL_PROFILE("time_loop_boris_fd::step");

        switch (strang_order)
        {
            case 2:
                time_loop_boris_fd_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                        profiling>(infra, mw_yee, 1.0, part_gr, &profiling_timers);
                break;
            case 4:
                amrex::Real alpha = 1. / (2. - pow(2., 1. / 3.));
                amrex::Real beta = 1. - 2. * alpha;

                time_loop_boris_fd_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                        profiling>(infra, mw_yee, alpha, part_gr,
                                                   &profiling_timers);
                time_loop_boris_fd_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                        profiling>(infra, mw_yee, beta, part_gr, &profiling_timers);
                time_loop_boris_fd_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                        profiling>(infra, mw_yee, alpha, part_gr,
                                                   &profiling_timers);
                break;
        }

        diagn->end_of_timestep(&profiling_timers, t_step, infra, mw_yee, part_gr, simulation_name,
                               ctest);
    }

    if (profiling_timers.profiling){
      profiling_timers.counter_all += MPI_Wtime();
      amrex::Print() << "Deposition time: " << profiling_timers.counter_deposition << ", redistribute time: " << profiling_timers.counter_redistribute
                     << ", field solver time: " << profiling_timers.counter_fields << ", sum boundary j time: " << profiling_timers.counter_jboundary
                     << ", diagnostics time: " << profiling_timers.counter_diagnostics << ", total time: " << profiling_timers.counter_all << std::endl;
      }
}
}  // namespace Time_Loop
}  // namespace Gempic

#endif

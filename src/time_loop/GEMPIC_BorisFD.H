#ifndef GEMPIC_BORISFD_
#define GEMPIC_BORISFD_

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
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
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_Splitting.H>
#include <GEMPIC_SumBoundary.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Profiling;

namespace Gempic
{
namespace Time_Loop
{

template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
class BorisFD: public Splitting<vdim, numspec, degx, degy, degz, degmw, ndata,
                                electromagnetic, profiling>
{
    private:
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> push_particle(
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> phi, amrex::GpuArray<amrex::Real, vdim> vel,
            amrex::Real dt,
            amrex::Real chargemass,  // charge/mass
            amrex::GpuArray<amrex::Real, vdim> &Ep,
            amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> &Bp
        )
        {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> newPos;
            amrex::GpuArray<amrex::Real, vdim> vm;
            amrex::GpuArray<amrex::Real, vdim> vt;
            amrex::GpuArray<amrex::Real, vdim> vp;
            amrex::Real S;

            // v-
            for (int vd = 0; vd < vdim; vd++)
            {
                vm[vd] = vel[vd] + dt * chargemass / 2 * Ep[vd];
            }

            if (vdim == 1)
            {
                vp[0] = vm[0];
            }
            else if (vdim == 2)
            {
                amrex::Real alpha = chargemass * dt / 2 * Bp[0];
                vp[0] = (1 - pow(alpha, 2)) / (1 + pow(alpha, 2)) * vm[0] +
                        2 * alpha / (1 + pow(alpha, 2)) * vm[1];
                vp[1] = -2 * alpha / (1 + pow(alpha, 2)) * vm[0] +
                        (1 - pow(alpha, 2)) / (1 + pow(alpha, 2)) * vm[1];
            }
            else if (vdim == 3)
            {
                // v'
                amrex::Real coeff_t = dt * chargemass / 2;
                vt[0] = vm[0] + coeff_t * (vm[1] * Bp[2] - vm[2] * Bp[1]);
                vt[1] = vm[1] + coeff_t * (vm[2] * Bp[0] - vm[0] * Bp[2]);
                vt[2] = vm[2] + coeff_t * (vm[0] * Bp[1] - vm[1] * Bp[0]);

                S = pow(dt * chargemass / 2 * Bp[0], 2.) + pow(dt * chargemass / 2 * Bp[1], 2.) +
                    pow(dt * chargemass / 2 * Bp[2], 2.);

                // v+
                amrex::Real coeff = dt * chargemass / (1. + S);
                vp[0] = vm[0] + coeff * (vt[1] * Bp[2] - vt[2] * Bp[1]);
                vp[1] = vm[1] + coeff * (vt[2] * Bp[0] - vt[0] * Bp[2]);
                vp[2] = vm[2] + coeff * (vt[0] * Bp[1] - vt[1] * Bp[0]);
            }

            // v
            newPos[GEMPIC_SPACEDIM] = vp[0] + dt * chargemass / 2 * Ep[0];
            if (vdim > 1)
            {
                newPos[GEMPIC_SPACEDIM + 1] = vp[1] + dt * chargemass / 2 * Ep[1];
            }
            if (vdim > 2)
            {
                newPos[GEMPIC_SPACEDIM + 2] = vp[2] + dt * chargemass / 2 * Ep[2];
            }

            // new positions (modulo the upper bound of the box)
            newPos[0] = mod_dom(pos[0] + newPos[GEMPIC_SPACEDIM] * dt, phi[0], plo[0]);
        #if (GEMPIC_SPACEDIM > 1)
            newPos[1] = mod_dom(pos[1] + newPos[GEMPIC_SPACEDIM + 1] * dt, phi[1], plo[1]);
        #endif
        #if (GEMPIC_SPACEDIM > 2)
            newPos[2] = mod_dom(pos[2] + newPos[GEMPIC_SPACEDIM + 2] * dt, phi[2], plo[2]);
        #endif

            return (newPos);
        }

        inline void particle_loops(
            computational_domain &infra, maxwell_yee<vdim>* mw_yee, amrex::Real const coeff,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        ) override final
        {
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real spec_charge = part_gr[spec]->getCharge();
                if (profiling_timers->profiling) profiling_timers->counter_deposition -= MPI_Wtime();
                amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
                {
                    const long np = pti.numParticles();
                    auto& particles = pti.GetArrayOfStructs();
                    auto partData = particles().data();
                    auto& particleAttributes = pti.GetStructOfArrays();
                    auto weight = particleAttributes.GetRealData(vdim).data();
                    auto velx = particleAttributes.GetRealData(0).data();
                    auto vely = particleAttributes.GetRealData(1).data();
                    auto velz = particleAttributes.GetRealData(2).data();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (*mw_yee->J_Array[cc])[pti].array();
                        eA[cc] = (*mw_yee->E_Array[cc])[pti].array();
                    }
                    for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
                    {
                        bA[cc] = (*mw_yee->B_Array[cc])[pti].array();
                    }

                    amrex::Real time_step = mw_yee->dt;

                    // loop over particles: add contribution of old particle position to J and push
                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = partData[pp].pos(d);
                            splines_at_particles<degx, degy, degz> spline;
                            spline.init_particles(pos, infra.plo, infra.dxi);

                            amrex::GpuArray<amrex::Real, vdim> particle_vel{velx[pp], vely[pp], velz[pp]};
                            amrex::Real charge_weight_dxi_coeff =
                                spec_charge * weight[pp] * infra.dxi[GEMPIC_SPACEDIM] * 0.5;

                            amrex::GpuArray<amrex::Real, vdim + int(vdim / 2.5)* 2 + 1> field_interp =
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
                                pos[comp] = partData[pp].pos(comp);
                            }
                            amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> newPos =
                                push_particle(pos, infra.plo, infra.phi, vel, time_step * coeff,
                                                        chargemass, eres, bres);
                            for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) partData[pp].pos(xd) = newPos[xd];
                            velx[pp] = newPos[GEMPIC_SPACEDIM];
                            vely[pp] = newPos[GEMPIC_SPACEDIM + 1];
                            velz[pp] = newPos[GEMPIC_SPACEDIM + 2];
                        });
                }
                if (profiling_timers->profiling) profiling_timers->counter_deposition += MPI_Wtime();
                if (profiling_timers->profiling) profiling_timers->counter_redistribute -= MPI_Wtime();
                part_gr[spec]->Redistribute();  // assign particles to the tile they are in
                if (profiling_timers->profiling) profiling_timers->counter_redistribute += MPI_Wtime();
            }
            //--------------------------------------------------------------------------
            // update fields
            // deposit remaining charge in J one particle at a time

            if (profiling_timers->profiling) profiling_timers->counter_deposition -= MPI_Wtime();
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real spec_charge = part_gr[spec]->getCharge();
                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
                {
                    const long np = pti.numParticles();
                    const auto& particles = pti.GetArrayOfStructs();
                    const auto partData = particles().data();
                    auto& particleAttributes = pti.GetStructOfArrays();
                    auto weight = particleAttributes.GetRealData(vdim).data();
                    auto velx = particleAttributes.GetRealData(0).data();
                    auto vely = particleAttributes.GetRealData(1).data();
                    auto velz = particleAttributes.GetRealData(2).data();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (*(*mw_yee).J_Array[cc])[pti].array();
                    }

                    amrex::ParallelFor(
                        np,
                        [=] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = partData[pp].pos(d);
                            splines_at_particles<degx, degy, degz> spline;
                            spline.init_particles(pos, infra.plo, infra.dxi);

                            amrex::GpuArray<amrex::Real, vdim> particle_vel{velx[pp], vely[pp], velz[pp]};
                            amrex::Real charge_weight_dxi_coeff =
                                spec_charge * weight[pp] * infra.dxi[GEMPIC_SPACEDIM] * 0.5;

                            amrex::GpuArray<amrex::Real, vdim + int(vdim / 2.5)* 2 + 1> field_interp =
                                gempic_deposit_J_and_interpolate<vdim, degx, degy, degz>(
                                    charge_weight_dxi_coeff, particle_vel, spline, jA, eA, bA, false);
                        });
                }
            }
        }

        virtual void time_step(
            computational_domain infra, maxwell_yee<vdim>* mw_yee, amrex::Real coeff,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        )
        {
            BL_PROFILE("time_loop_boris_fd_step");

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
                mw_yee->J_Array[i]->setVal(0.0);  // value and component
            }
            if (profiling_timers->profiling) profiling_timers->counter_jboundary += MPI_Wtime();

            particle_loops(infra, mw_yee, coeff, part_gr, profiling_timers, profiling_count);

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
};

}
}

#endif

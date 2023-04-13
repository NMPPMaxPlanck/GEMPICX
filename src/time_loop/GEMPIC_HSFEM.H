#ifndef HSFEM_
#define HSFEM_

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
class HSFEM: public HamiltonianSplitting<vdim, numspec, degx, degy, degz, degmw, ndata,
                                   electromagnetic, profiling>
{
    private:
        inline void particle_loops(
            computational_domain &infra, maxwell_yee<vdim>* mw_yee, amrex::Real const coeff,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        ) override final
        {
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();
                amrex::Real charge = part_gr[spec]->getCharge();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
                {
                    auto& particles = pti.GetArrayOfStructs();  // get particles
                    const long np = pti.numParticles();
                    auto& particle_attributes = pti.GetStructOfArrays();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (*(mw_yee->J_Array)[cc])[pti].array();
                        eA[cc] = (*(mw_yee->E_Array)[cc])[pti].array();
                    }
                    for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
                    {
                        bA[cc] = (*(mw_yee->B_Array)[cc])[pti].array();
                    }

                    // loop over particles: add contribution of old particle position to J and push

                    for (int pp = 0; pp < np; pp++)
                    {
                        // evaluate the electric field
                        splines_at_particles<degx, degy, degz> spline;
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                        for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                        {
                            pos[comp] = particles[pp].pos(comp);
                        }
                        spline.init_particles(pos, infra.plo, infra.dxi);
                        amrex::GpuArray<amrex::Real, vdim> efield =
                            evaluate_fem_efield<vdim, degx, degy, degz>(spline, eA);

                        // push v with the electric field
                        amrex::GpuArray<amrex::Real, vdim> vel;
                        for (int comp = 0; comp < vdim; comp++)
                        {
                            vel[comp] = particle_attributes.GetRealData(comp)[pp];
                        }
                        
                        amrex::GpuArray<amrex::Real, vdim> newPosE =
                        HamiltonianSplitting<vdim, numspec, degx, degy, degz, degmw, ndata,
                                             electromagnetic, profiling>::push_v_efield(vel, (*mw_yee).dt * 0.5 * coeff, chargemass, efield);

                        for (int comp = 0; comp < vdim; comp++)
                        {
                            particle_attributes.GetRealData(comp)[pp] = newPosE[comp];
                        }

                        // push x and v with the magnetic field
                        for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                        {
                            pos[comp] = particles[pp].pos(comp);
                        }
                        for (int comp = 0; comp < vdim; comp++)
                        {
                            vel[comp] = particle_attributes.GetRealData(comp)[pp];
                        }
                        spline.init_particles(pos, infra.plo, infra.dxi);
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> newPos =
                            push_x_v_bfield<vdim, degx, degy, degz>(
                                pos, vel, (*mw_yee).dt * coeff, chargemass, bA, infra.tolerance_particle,
                                infra.plo, infra.phi, infra.dxi, spline);
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> xNew, xOld;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        {
                            xOld[d] = particles[pp].pos(d);
                            xNew[d] = newPos[d];
                        }

                        amrex::GpuArray<amrex::Real, vdim> velocity;
                        for (unsigned int d = 0; d < vdim; ++d)
                            velocity[d] = 0.5 * (newPos[d + GEMPIC_SPACEDIM] +
                                                particle_attributes.GetRealData(d)[pp]);

                        accumulate_j_fem_integral<vdim, degx, degy, degz>(
                            spline, xOld, xNew, velocity,
                            charge * particle_attributes.GetRealData(vdim)[pp], infra.dxi, infra.plo, jA);
                        for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) particles[pp].pos(xd) = newPos[xd];
                    }
                }
                part_gr[spec]->Redistribute();  // assign particles to the tile they are in
            }

            // add values from ghost cells and periodic boundary
            Gempic_SumBoundary(mw_yee, &infra);  // Sums Boundary of J and Overwrites overlap
            for (int i = 0; i < vdim; i++)
            {
                mw_yee->J_Array[i]->FillBoundary(infra.geom.periodicity());
            }

            // advance E from j (field part of H_p)
            mw_yee->advance_E(infra, mw_yee->dt * coeff, false, true, mw_yee->HB_Array, mw_yee->E_Array);

            // Second particle loop for particle part from H_E
            for (int spec = 0; spec < numspec; spec++)
            {
                amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
                {
                    const long np = pti.numParticles();
                    auto& particles = pti.GetArrayOfStructs();  // get particles
                    auto partData = particles().data();
                    auto& particle_attributes = pti.GetStructOfArrays();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        eA[cc] = (*(mw_yee->E_Array)[cc])[pti].array();
                    }

                    // loop over particles: add contribution of old particle position to J and push

                    for (int pp = 0; pp < np; pp++)
                    {
                        // evaluate the electric field
                        splines_at_particles<degx, degy, degz> spline;
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                        for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                        {
                            pos[comp] = partData[pp].pos(comp);
                        }
                        spline.init_particles(pos, infra.plo, infra.dxi);
                        amrex::GpuArray<amrex::Real, vdim> efield =
                            evaluate_fem_efield<vdim, degx, degy, degz>(spline, eA);

                        // push v with the electric field
                        amrex::GpuArray<amrex::Real, vdim> vel;
                        for (int comp = 0; comp < vdim; comp++)
                        {
                            vel[comp] = particle_attributes.GetRealData(comp)[pp];
                        }
                        amrex::GpuArray<amrex::Real, vdim> newPosE =
                        HamiltonianSplitting<vdim, numspec, degx, degy, degz, degmw, ndata,
                                             electromagnetic, profiling>::push_v_efield(vel, (*mw_yee).dt * 0.5 * coeff, chargemass, efield);
                        for (int comp = 0; comp < vdim; comp++)
                        {
                            particle_attributes.GetRealData(comp)[pp] = newPosE[comp];
                        }
                    }
                }
                part_gr[spec]->Redistribute();  // assign particles to the tile they are in
            }
        }

        virtual void time_step(
            computational_domain infra, maxwell_yee<vdim>* mw_yee, amrex::Real coeff,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        )
        {
            mw_yee->advance_time(coeff);

            if (electromagnetic)
            {
                // Operator H_B for dt/2
                mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
                mw_yee->advance_E(infra, mw_yee->dt * 0.5 * coeff, true, false, mw_yee->HB_Array,
                                mw_yee->E_Array);

                //--------------------------------------------------------------------------
                // Operator H_E for dt/2 (field part)
                mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
                mw_yee->advance_B(infra, mw_yee->dt * 0.5 * coeff, mw_yee->HE_Array, mw_yee->B_Array);
            }

            //--------------------------------------------------------------------------
            // deposit charge in J and push particles

            // set J to 0 before we start depositing values from current particles
            for (int i = 0; i < vdim; i++)
            {
                mw_yee->J_Array[i]->setVal(0.0);  // value and component
            }

            particle_loops(infra, mw_yee, coeff, part_gr, profiling_timers, profiling_count);

            if (electromagnetic)
            {
                //--------------------------------------------------------------------------
                // Operator H_E for dt/2 (field part)
                mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
                mw_yee->advance_B(infra, mw_yee->dt * 0.5 * coeff, mw_yee->HE_Array, mw_yee->B_Array);

                // Operator H_B for dt/2
                mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
                mw_yee->advance_E(infra, mw_yee->dt * 0.5 * coeff, true, false, mw_yee->HB_Array,
                                mw_yee->E_Array);
            }
        }
};

}  // namespace Time_Loop
}  // namespace Gempic

#endif

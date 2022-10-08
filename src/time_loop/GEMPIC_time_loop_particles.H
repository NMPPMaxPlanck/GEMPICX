#ifndef TIME_LOOP_PARTICLES_H_
#define TIME_LOOP_PARTICLES_H_

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
#include <GEMPIC_SumBoundary.H>
#include <GEMPIC_checkpoint.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_particle_positions.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;

namespace Gempic
{
namespace Time_Loop
{
template <int vdim, int numspec, int degx, int degy, int degz, int degmw, int ndata,
          bool electromagnetic = true>
void time_loop_particles(
    computational_domain infra, maxwell_yee<vdim>* mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw>* diagn, bool ctest,
    std::string simulation_name)
{
    BL_PROFILE("time_loop_particles");

    for (int t_step = 0; t_step < mw_yee->nsteps; t_step++)
    {
        BL_PROFILE("time_loop_particles::step");

        mw_yee->advance_time();

        if (electromagnetic)
        {
            //--------------------------------------------------------------------------
            // update B
            mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
            mw_yee->advance_B(infra, mw_yee->dt, mw_yee->HE_Array, mw_yee->B_Array);
        }

        //--------------------------------------------------------------------------
        // deposit charge in J and push particles

        // set J to 0 before we start depositing values from current particles
        for (int i = 0; i < vdim; i++)
        {
            (mw_yee->J_Array)[i]->setVal(0.0);  // value and component
        }

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real spec_charge = part_gr[spec]->getCharge();

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
                }

                // loop over particles: add contribution of old particle position to J

                for (int pp = 0; pp < np; pp++)
                {
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        pos[d] = particles[pp].pos(d);
                    splines_at_particles<degx, degy, degz> spline;
                    spline.init_particles(pos, infra.plo, infra.dxi);

                    amrex::GpuArray<amrex::Real, vdim> particle_vel;
                    for (int comp = 0; comp < vdim; comp++)
                        particle_vel[comp] = particle_attributes.GetRealData(comp)[pp];
                    amrex::Real charge_weight_dxi_coeff =
                        spec_charge * particle_attributes.GetRealData(vdim)[pp] *
                        infra.dxi[GEMPIC_SPACEDIM] * 0.5;

                    amrex::GpuArray<amrex::Real, vdim + int(vdim / 2.5)* 2 + 1> field_interp =
                        gempic_deposit_J_and_interpolate<vdim, degx, degy, degz>(
                            charge_weight_dxi_coeff, particle_vel, spline, jA, eA, bA, false);
                    amrex::Real d = 1.0;
                    amrex::Real Omega = 1.0;
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + vdim> newPos;
                    newPos[0] = 0.0;
                    newPos[1] = 0.0;
                    newPos[2] = pow(-1.0, spec) * d * sin(Omega * mw_yee->dt);
                    newPos[0] = 0.0;
                    newPos[0] = 0.0;
                    newPos[0] = pow(-1.0, spec) * d * Omega * cos(Omega * mw_yee->dt);
                    // std::copy(newPos.begin(),newPos.end(),&particles.data()[pp*(GEMPIC_SPACEDIM+vdim+2)]);
                    for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) particles[pp].pos(xd) = pos[xd];
                }
            }
            part_gr[spec]->Redistribute();  // assign particles to the tile they are in
        }

        //--------------------------------------------------------------------------
        // deposit remaining charge in J one particle at a time

        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real spec_charge = part_gr[spec]->getCharge();
            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
            {
                auto& particles = pti.GetArrayOfStructs();
                const long np = pti.numParticles();
                auto& particle_attributes = pti.GetStructOfArrays();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

                for (int cc = 0; cc < vdim; cc++)
                {
                    jA[cc] = (*(mw_yee->J_Array)[cc])[pti].array();
                }

                for (int pp = 0; pp < np; pp++)
                {
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        pos[d] = particles[pp].pos(d);
                    splines_at_particles<degx, degy, degz> spline;
                    spline.init_particles(pos, infra.plo, infra.dxi);

                    amrex::GpuArray<amrex::Real, vdim> particle_vel;
                    for (int comp = 0; comp < vdim; comp++)
                        particle_vel[comp] = particle_attributes.GetRealData(comp)[pp];
                    amrex::Real charge_weight_dxi_coeff =
                        spec_charge * particle_attributes.GetRealData(vdim)[pp] *
                        infra.dxi[GEMPIC_SPACEDIM] * 0.5;

                    amrex::GpuArray<amrex::Real, vdim + int(vdim / 2.5)* 2 + 1> field_interp =
                        gempic_deposit_J_and_interpolate<vdim, degx, degy, degz>(
                            charge_weight_dxi_coeff, particle_vel, spline, jA, eA, bA, false);
                }
            }
        }

        Gempic_SumBoundary(mw_yee, &infra);  // Sums Boundary of J and Overwrites overlap

        for (int i = 0; i < vdim; i++)
        {
            (mw_yee->J_Array)[i]->FillBoundary(infra.geom.periodicity());
        }

        if (electromagnetic)
        {
            // advance E
            mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
            mw_yee->advance_E(infra, mw_yee->dt, true, true, mw_yee->HB_Array, mw_yee->E_Array);
        }
        else
        {
            // advance E from j
            mw_yee->advance_E(infra, mw_yee->dt, false, true, mw_yee->HB_Array, mw_yee->E_Array);
        }

        // save norms for diagnostics. At the end of the time step time = (t_step + 1) * dt
        diagn->ComputeReducedDiag(t_step + 1, infra, mw_yee,
                                  part_gr);  // false: slices are not saved

        // save temporary results every n steps
        if ((t_step + 1) % 5 == 0)
        {
            diagn->save_time_history(mw_yee->dt, simulation_name, t_step + 1);
        }
        amrex::Print() << "finished step " << t_step + 1 << std::endl;

        if (ctest) diagn->ctest_output(simulation_name, t_step);
    }  // end time loop

    diagn->save_time_history(mw_yee->dt, simulation_name,
                             0);  // time step as 0 to satisfy arguments needed by function
}
}  // namespace Time_Loop
}  // namespace Gempic

#endif

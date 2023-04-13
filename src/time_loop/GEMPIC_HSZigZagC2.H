/** @file GEMPIC_time_loop_hs_zigzag_C2.H
 * 
 * @brief Implements the time propagator with a Hamiltonian splitting in a FD formulation
 * 
 * Description
 * ----------
 * Implements the time advancement of the particle movements via Hamiltonian Splitting
 * 
 * The Hamiltonian H gets deconstructed into the summands
 * 
 * H = H_b + H_e + H_p
 * 
 * The component H_p gets deconstructed further into
 * 
 * H_p = H_p_1 + H_p_2 + H_p_3
 * 
 * via the Strang splitting
 * 
 * Then we get the solution by applying
 * 
 * H_b * H_e * H_p1 * H_p2 * H_p3 * H_p3 * H_p2 * H_p1 * H_e * H_b
 * 
 * Classes
 * ----------
 * Implements no classes.
 * 
 * Functions
 * ----------
 *  apply_H_b: void
 *      Applies the Hamiltonian for the magnetic field B, H_b
 * 
 *  apply_H_e: void
 *      Applies the Hamiltonian for the electric field E, H_e
 * 
 *  apply_H_p_i: void
 *      Applies one substep of the Strang splitting of the
 *      Hamiltoninan of the particle movement H_p
 * 
 *  particle_loop: void
 *      Iterates through all the particles and applies the Strang 
 *      Splitting to the particle movement H_p by calling the 
 *      substeps H_p_i of the Strang splitting and applies the 
 *      Hamiltonian for the elctric field E, H_e
 * 
 *  time_loop_hs_zigzag_C2_step: void
 *      Executes one time step of the Hamiltonian splitting
 * 
 *  time_loop_hs_zigzag_C2: void
 *      Handles the time stepping for the particle movement by 
 *      calling the time step method
 * 
 * @authors Katharina Kormann, Markus Musch
 */

#ifndef HSZIGZAGC2_
#define HSZIGZAGC2_

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
class HSZigZagC2: public HamiltonianSplitting<vdim, numspec, degx, degy, degz, degmw, ndata,
                                   electromagnetic, profiling>
{
    private:
        /**
        * @brief Apply the Hamiltonian for the magnetic field B.
        * 
        * Description:
        * ----------
        * Assemble the Hodge-operator and advance the electric field E in time.
        * 
        * Parameters:
        * ----------
        * @tparam vdim : int, the number of dimensions of the velocity space v
        * @tparam degmw: int, the degree of the Maxwell-Yee approximation
        * @param infra: computational_domain, object representing the 
        *               computational domain
        * @param mw_yee: maxwell_yee, object representing the Maxwell-Yee solver
        * @param coeff: amrex::Real, coefficient adjusting the time step size
        *               for different splitting methods
        * 
        * Returns:
        * ----------
        * Returns void
        * 
        * Raises:
        * ----------
        * Does not raise any exceptions 
        */
        inline void apply_H_b(
            computational_domain const &infra, maxwell_yee<vdim>* mw_yee, amrex::Real const coeff
        )
        {
            BL_PROFILE_VAR("Apply H_B", hb);
            mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
            mw_yee->advance_E(infra, mw_yee->dt * 0.5 * coeff, true, false, mw_yee->HB_Array,
                                mw_yee->E_Array);
            BL_PROFILE_VAR_STOP(hb);
        }

        /**
        * @brief Apply the Hamiltonian for the electric field E electromagnetic part.
        * 
        * Description:
        * ----------
        * Assemble the Hodge-operator and advance the electric field B in time plus
        * push forward the particles.
        * 
        * Parameters:
        * ----------
        * @tparam vdim: int, the number of dimensions of the velocity space v
        * @tparam degx: int, polynomial degree of the kernel in x-dimension
        * @tparam degy: int, polynomial degree of the kernel in y-dimension
        * @tparam degz: int, polynomial degree of the kernel in z-dimension
        * @tparam degmw: int, the degree of the Maxwell-Yee approximation
        * @tparam electromagnetic: bool, flag signaling if the model is electromagnetic
        * @param infra: computational_domain, object representing the 
        *               computational domain
        * @param mw_yee: maxwell_yee, object representing the Maxwell-Yee solver
        * @param coeff: amrex::Real, coefficient adjusting the time step size
        *               for different splitting methods
        * @param part_gr: amrex::GpuArray, array of particle_group objects
                        containing all particle information.
        * @param profiling_timers: timers, contains all timers for profiling the code
        * @param profiling_count: bool, indicate if it's the first time step or not
        * 
        * Returns:
        * ----------
        * Returns void
        * 
        * Raises:
        * ----------
        * Does not raise any exceptions 
        */
        inline void apply_H_e_field(
            computational_domain const &infra, maxwell_yee<vdim>* mw_yee, amrex::Real const coeff
        )
        {
            BL_PROFILE_VAR("Apply H_E, update B", heB);
            mw_yee->template hodge_full<degmw>(infra, (mw_yee->E_Array), mw_yee->HE_Array, true);
            mw_yee->advance_B(infra, mw_yee->dt * 0.5 * coeff, mw_yee->HE_Array, mw_yee->B_Array);   
            BL_PROFILE_VAR_STOP(heB);
        }

        /**
        * @brief Apply the Hamiltonian for the electric field E particle part.
        * 
        * Description:
        * ----------
        * Assemble the Hodge-operator and advance the electric field B in time plus
        * push forward the particles.
        * 
        * Parameters:
        * ----------
        * @tparam vdim: int, the number of dimensions of the velocity space v
        * @tparam degx: int, polynomial degree of the kernel in x-dimension
        * @tparam degy: int, polynomial degree of the kernel in y-dimension
        * @tparam degz: int, polynomial degree of the kernel in z-dimension
        * @tparam degmw: int, the degree of the Maxwell-Yee approximation
        * @tparam electromagnetic: bool, flag signaling if the model is electromagnetic
        * @param infra: computational_domain, object representing the 
        *               computational domain
        * @param mw_yee: maxwell_yee, object representing the Maxwell-Yee solver
        * @param coeff: amrex::Real, coefficient adjusting the time step size
        *               for different splitting methods
        * @param part_gr: amrex::GpuArray, array of particle_group objects
                        containing all particle information.
        * @param profiling_timers: timers, contains all timers for profiling the code
        * @param profiling_count: bool, indicate if it's the first time step or not
        * 
        * Returns:

        * ----------
        * Returns void
        * 
        * Raises:
        * ----------
        * Does not raise any exceptions 
        */
        inline void apply_H_e_particle(
            computational_domain const &infra, maxwell_yee<vdim>* mw_yee, amrex::Real const coeff, 
            const int spec, amrex::Real const chargemass, amrex::ParIter<0, 0, vdim + ndata, 0> const &pti,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        )
        {   

            amrex::Particle<0, 0>* AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);
            const long np = pti.numParticles();
            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (*(mw_yee->HE_Array)[cc])[pti].array();
            }
            amrex::Real dt = mw_yee->dt;

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = particles[pp].pos(d);
                splines_at_particles<degx, degy, degz> spline;
                spline.init_particles(pos, infra.plo, infra.dxi);
                // evaluate the electric field
                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                // push v with the electric field
                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    HamiltonianSplitting<vdim, numspec, degx, degy, degz, degmw, ndata,
                                   electromagnetic, profiling>::push_v_efield(vel, dt * 0.5 * coeff, chargemass, efield);
                velx[pp] = newPosE[0];
                vely[pp] = newPosE[1];
                velz[pp] = newPosE[2];
            });
        }

        /**
        * @brief Apply a Hamiltonian for the particles p.
        * 
        * Description:
        * ----------
        * Advance the particles p in time.
        * 
        * Parameters:
        * ----------
        * @tparam xidx: int, 
        * @tparam vdim: int, the number of dimensions of the velocity space v
        * @tparam degx: int, polynomial degree of the kernel in x-dimension
        * @tparam degy: int, polynomial degree of the kernel in y-dimension
        * @tparam degz: int, polynomial degree of the kernel in z-dimension
        * @param infra: computational_domain, object representing the 
        *               computational domain
        * @param dt: amrex::Real, the size of the discrete time step
        * @param coeff: amrex::Real, coefficient adjusting the time step size
        *               for different splitting methods
        * @param chargemass: amrex::Real, the quotient of charge over mass
        * @param weight: amrex::Real, charge multiplied with particle weight
        * @param vel: amrex::GpuArray, velocity of the particle in each dimension
        * @param pos: amrex::GpuArray, position of the particle in each dimension
        * @param bfields: amrex::GpuArray, the magnetic field
        * @param dx: amrex::GpuArray, the size of the spatial discretization parameter
        * @param jA: amrex::GpuArray, 
        * @param bA: amrex::GpuArray,
        * @param primitive: amrex::GpuArray,
        * @param spline: splines_at_particles, three dimensional spline function
        * 
        * Returns:
        * ----------
        * Returns void
        * 
        * Raises:
        * ----------
        * Does not raise any exceptions 
        */
        template <int xidx>
        inline void apply_H_p_i(
            computational_domain const &infra, amrex::Real const dt, amrex::Real const coeff, 
            amrex::Real const chargemass, amrex::Real const weight, 
            amrex::GpuArray<amrex::Real, vdim> &vel, 
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos, amrex::GpuArray<amrex::Real, 2> bfields,
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx, 
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jA,
            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const &bA,
            amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4> &primitive,
            splines_at_particles<degx, degy, degz> &spline
        )
        {

            constexpr int degrees[3] = {degx, degy, degz};

            amrex::Real x_new = pos[xidx] + dt * 0.5 * coeff * vel[xidx];

            spline1d_at_particles<degrees[xidx]> spline_new;
            spline1d_at_particles<degrees[xidx]> spline_old;

            spline_new.init_position(x_new, infra.plo[xidx], infra.dxi[xidx]);
            spline_old.init_position(pos[xidx], infra.plo[xidx], infra.dxi[xidx]);
            spline.template init_position<xidx, degrees[xidx]>(x_new, infra.plo[xidx], infra.dxi[xidx]);

            accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>,
                                    vdim, degrees[xidx], degrees[(xidx+1)%3],
                                    degrees[(xidx+2)%3], xidx, std::max(degx, std::max(degy, degz)) + 4>(
                spline, spline_new, spline_old, weight, dx, bA, jA, bfields, primitive);

            if (vdim > 1) vel[(xidx+1)%3] = vel[(xidx+1)%3] - chargemass * bfields[0];
            if (vdim > 2) vel[(xidx+2)%3] = vel[(xidx+2)%3] + chargemass * bfields[1];
            pos[xidx] = x_new;
        }

        inline void particle_loops(
            computational_domain &infra, maxwell_yee<vdim>* mw_yee, amrex::Real const coeff,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        ) override final
        {
            for (int spec = 0; spec < numspec; spec++)
            {
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_deposition -= MPI_Wtime();
                amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();
                amrex::Real charge = part_gr[spec]->getCharge();

                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
                {
                    apply_H_e_particle(infra, mw_yee,
                        coeff, spec, chargemass, pti, part_gr, profiling_timers, profiling_count);

                    amrex::Particle<0, 0>* AMREX_RESTRICT particles =
                        &(pti.GetArrayOfStructs()[0]);
                    const long np = pti.numParticles();
                    auto particle_attributes = &pti.GetStructOfArrays();

                    amrex::ParticleReal* const AMREX_RESTRICT part_weight =
                        particle_attributes->GetRealData(vdim).data();
                    amrex::ParticleReal* const AMREX_RESTRICT velx =
                        particle_attributes->GetRealData(0).data();
                    amrex::ParticleReal* const AMREX_RESTRICT vely =
                        particle_attributes->GetRealData(1).data();
                    amrex::ParticleReal* const AMREX_RESTRICT velz =
                        particle_attributes->GetRealData(2).data();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;

                    for (int cc = 0; cc < vdim; cc++)
                    {
                        jA[cc] = (*(mw_yee->J_Array)[cc])[pti].array();
                    }
                    for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
                    {
                        bA[cc] = (*(mw_yee->B_Array)[cc])[pti].array();
                    }

                    BL_PROFILE_VAR("Apply H_E, update v and apply Hp", heV_hp);

                    amrex::Real const dt = mw_yee->dt;
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx = {
                        infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2]};
                    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
        
                        splines_at_particles<degx, degy, degz> spline;

                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = particles[pp].pos(d);

                        spline.init_particles(pos, infra.plo, infra.dxi);

                        amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                        amrex::Real weight = charge * part_weight[pp] * infra.dxi[GEMPIC_SPACEDIM];

                        vel[0] = velx[pp];
                        vel[1] = vely[pp];
                        vel[2] = velz[pp];

                        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};
                        amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4> primitive;

                        apply_H_p_i<0>(infra, dt, coeff, chargemass, 
                            weight, vel, pos, bfields, dx, jA, bA, primitive, spline);
                        if (vdim > 1)
                        {
                            apply_H_p_i<1>(infra, dt, coeff, chargemass,
                                weight, vel, pos, bfields, dx, jA, bA, primitive, spline);
                        }
                        if (vdim > 2)
                        {
                            apply_H_p_i<2>(infra, 2*dt, coeff, chargemass,
                                weight, vel, pos, bfields, dx, jA, bA, primitive, spline);
                        }
                        if (vdim > 1)
                        {
                            apply_H_p_i<1>(infra, dt, coeff, chargemass,
                                weight, vel, pos, bfields, dx, jA, bA, primitive, spline);
                        }
                        apply_H_p_i<0>(infra, dt, coeff, chargemass, 
                            weight, vel, pos, bfields, dx, jA, bA, primitive, spline);

                        for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) particles[pp].pos(xd) = pos[xd];

                        velx[pp] = vel[0];
                        vely[pp] = vel[1];
                        velz[pp] = vel[2];

                    });
                    BL_PROFILE_VAR_STOP(heV_hp);
                }
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_deposition += MPI_Wtime();
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_redistribute -= MPI_Wtime();

                part_gr[spec]->Redistribute();  // assign particles to the tile they are in

                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_redistribute += MPI_Wtime();
            }

            // add values from ghost cells and periodic boundary
            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_jboundary -= MPI_Wtime();
            BL_PROFILE_VAR("Time Loop Boundary synchronization", bsync);
            Gempic_SumBoundary(mw_yee, &infra);  // Sums Boundary of J and Overwrites overlap

            for (int i = 0; i < vdim; i++)
            {
                (*(mw_yee->J_Array)[i]).FillBoundary(infra.geom.periodicity());
            }

            BL_PROFILE_VAR_STOP(bsync);
            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_jboundary += MPI_Wtime();
            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_fields -= MPI_Wtime();

            BL_PROFILE_VAR("Apply Hp, E part", hpe);
            mw_yee->advance_E(infra, 1.0, false, true, mw_yee->HB_Array, mw_yee->E_Array);
            mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
            BL_PROFILE_VAR_STOP(hpe);

            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_fields += MPI_Wtime();
        
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_fields += MPI_Wtime();

            // Second particle loop for particle part from H_E
            for (int spec = 0; spec < numspec; spec++)
            {
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_deposition -= MPI_Wtime();
                amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();

                // loop over particles: add contribution of old particle position to J and push
                BL_PROFILE_VAR("Apply H_E, update v (second part", heV_2);
                for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
                {
                    apply_H_e_particle(infra, mw_yee, coeff, 
                        spec, chargemass, pti, part_gr, profiling_timers, profiling_count);
                }
                BL_PROFILE_VAR_STOP(heV_2);
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_deposition += MPI_Wtime();
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_redistribute -= MPI_Wtime();
                
                part_gr[spec]->Redistribute();  // assign particles to the tile they are in
                
                if (profiling_timers->profiling && profiling_count) profiling_timers->counter_redistribute += MPI_Wtime();
            }
        }

        virtual void time_step(
            computational_domain infra, maxwell_yee<vdim>* mw_yee, amrex::Real coeff,
            amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& part_gr,
            timers* profiling_timers, bool profiling_count
        )
        {
            BL_PROFILE("Gempic::Time_Loop::time_loop_hs_zigzag_C2_step");

            mw_yee->advance_time(coeff);

            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_jboundary -= MPI_Wtime();

            for (int i = 0; i < vdim; i++)
            {
                mw_yee->J_Array[i]->setVal(0.0, 0);
                mw_yee->J_Array[i]->FillBoundary(infra.geom.periodicity());
            }
            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_jboundary += MPI_Wtime();


            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_fields -= MPI_Wtime();

            if (electromagnetic)
            {
                apply_H_b(infra, mw_yee, coeff);

                apply_H_e_field(infra, mw_yee, coeff);
            }

            particle_loops(infra, mw_yee, coeff, part_gr, profiling_timers, profiling_count);

            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_fields -= MPI_Wtime();
            
            if (electromagnetic)
            {
                apply_H_e_field(infra, mw_yee, coeff);

                apply_H_b(infra, mw_yee, coeff);
            }

            if (profiling_timers->profiling && profiling_count) profiling_timers->counter_fields += MPI_Wtime();
        }
};

}  // namespace Time_Loop
}  // namespace Gempic

#endif

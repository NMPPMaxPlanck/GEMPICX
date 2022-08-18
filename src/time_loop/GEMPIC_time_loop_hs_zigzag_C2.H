#ifndef TIME_LOOP_HS_ZIGZAG_C2_
#define TIME_LOOP_HS_ZIGZAG_C2_

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
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_profiling.H>

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Profiling;

/**
 * @brief Updates the particle part of Hp1 in hamiltonian splitting.
 * The E update can be delayed as J is computed and stored
 *
 * @tparam S
 * @tparam vdim
 * @tparam degx Spline degree in x
 * @tparam degy Spline degree in y
 * @tparam degz Spline degree in z
 * @param infra Computational domain
 * @param dtLoc Local time step
 * @param pos
 * @param vel
 * @param spline
 * @param spline_new_x
 * @param spline_old_x
 * @param primitive
 * @param jA
 * @param bA
 * @param vxbFields The two components of  v x B that are needed for this component
 * @param weight
 * @param chargemass
 */

template <typename S, int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void apply_hp1(
    computational_domain infra, amrex::Real dtLoc,
    amrex::GpuArray<amrex::ParticleReal, GEMPIC_SPACEDIM>& pos,
    amrex::GpuArray<amrex::ParticleReal, vdim>& vel, S& spline,
    spline1d_at_particles<degx>& spline_new_x, spline1d_at_particles<degx>& spline_old_x,
    amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4>& primitive,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const& bA,
    amrex::GpuArray<amrex::Real, 2>& vxbFields, amrex::Real weight, amrex::ParticleReal chargemass)
{
    BL_PROFILE("Apply H_p1 (particles only)");
    // push x1
    amrex::Real x1_new = pos[0] + dtLoc * vel[0];
    spline_new_x.init_position(x1_new, infra.plo[0], infra.dxi[0]);
    spline_old_x.init_position(pos[0], infra.plo[0], infra.dxi[0]);
    spline.template init_position<0, degx>(x1_new, infra.plo[0], infra.dxi[0]);

    accumulate_j_update_v_C2<S, vdim, degx, degy, degz, 0,
                             std::max(degx, std::max(degy, degz)) + 4>(
        spline, spline_new_x, spline_old_x, weight, infra.dx, bA, jA, vxbFields, primitive);

    // Push v2 and v3
    if (vdim > 1) vel[1] = vel[1] - chargemass * vxbFields[0];
    if (vdim > 2) vel[2] = vel[2] + chargemass * vxbFields[1];
    pos[0] = x1_new;
}

/**
 * @brief Updates the particle part of Hp2 in hamiltonian splitting.
 * The E update can be delayed as J is computed and stored
 *
 * @tparam S
 * @tparam vdim
 * @tparam degx
 * @tparam degy
 * @tparam degz
 * @param infra
 * @param dtLoc
 * @param pos
 * @param vel
 * @param spline
 * @param spline_new_y
 * @param spline_old_y
 * @param primitive
 * @param jA
 * @param bA
 * @param vxbFields
 * @param weight
 * @param chargemass
 * @return AMREX_GPU_HOST_DEVICE
 */
template <typename S, int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void apply_hp2(
    computational_domain infra, amrex::Real dtLoc,
    amrex::GpuArray<amrex::ParticleReal, GEMPIC_SPACEDIM>& pos,
    amrex::GpuArray<amrex::ParticleReal, vdim>& vel, S& spline,
    spline1d_at_particles<degy>& spline_new_y, spline1d_at_particles<degy>& spline_old_y,
    amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4>& primitive,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& jA,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const& bA,
    amrex::GpuArray<amrex::Real, 2>& vxbFields, amrex::Real weight, amrex::ParticleReal chargemass)
{
    BL_PROFILE("Apply H_p2");
#if GEMPIC_SPACEDIM > 1
    // push x2
    amrex::Real x2_new = pos[1] + dtLoc * vel[1];

    spline_new_y.init_position(x2_new, infra.plo[1], infra.dxi[1]);
    spline_old_y.init_position(pos[1], infra.plo[1], infra.dxi[1]);
    spline.template init_position<1, degy>(x2_new, infra.plo[1], infra.dxi[1]);

    accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degy, degz, degx, 1,
                             std::max(degx, std::max(degy, degz)) + 4>(
        spline, spline_new_y, spline_old_y, weight, infra.dx, bA, jA, vxbFields, primitive);

    // Push v2 and v3
    vel[0] = vel[0] + chargemass * vxbFields[1];
    if (vdim > 2) vel[2] = vel[2] - chargemass * vxbFields[0];
    pos[1] = x2_new;
#else
    amrex::GpuArray<amrex::Real, int(vdim / 2.5)* 2 + 1> bfield =
        evaluate_fem_bfield<vdim, degx, degy, degz>(pos, infra.dxi, infra.plo, bA);
    gempic_deposit_rho<degx, degy, degz>(spline, weight * vel[1] * dt * coeff, jA[1]);
    // Push v2 and v3
    vel[0] = vel[0] + chargemass * vxbFields[0] * vel[1] * dt * coeff;
#endif
}

namespace Gempic
{
namespace Time_Loop
{
/*
    This is the time propagator with a Hamiltonian splitting in a FEM formulation with mass lumping
   for Lagrange elements of order one and zero

    This propagator is currently not implemented for the 2d2v case. In this case, the operator H_p3
   is dismissed and the operators H_p1 and H_p2 must be reimplemented only producing the b3 updates.
   In principle the operators are exactly the same but they should just return one bfield value -
   the one that concerns b_z - and it should read out the first component of the bfield vector
   (which saves the b_z in this case).

   */

template <int vdim, int numspec, int degx, int degy, int degz, int degmw,
          bool electromagnetic = true, bool profiling = true>
void time_loop_hs_zigzag_C2_step(
    computational_domain infra, maxwell_yee<vdim>* mw_yee, amrex::Real coeff,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec>& part_gr,
    timers* profiling_timers, bool profiling_count)
{
    BL_PROFILE("Gempic::Time_Loop::time_loop_hs_zigzag_C2_step");

    // Initialize the counters for timing the various parts of the algorithm

    mw_yee->advance_time(coeff);

    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_fields -= MPI_Wtime();
    // Operator H_B for dt/2
    if (electromagnetic)
    {
        BL_PROFILE_VAR("Apply H_B", hb);
        mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        mw_yee->advance_E(infra, mw_yee->dt * 0.5 * coeff, true, false, mw_yee->HB_Array,
                          mw_yee->E_Array);
        BL_PROFILE_VAR_STOP(hb);
        //--------------------------------------------------------------------------
        // Operator H_E for dt/2 (field part)
        BL_PROFILE_VAR("Apply H_E, update B", heB);
        mw_yee->template hodge_full<degmw>(infra, (mw_yee->E_Array), mw_yee->HE_Array, true);
        mw_yee->advance_B(infra, mw_yee->dt * 0.5 * coeff, mw_yee->HE_Array, mw_yee->B_Array);
        BL_PROFILE_VAR_STOP(heB);
    }
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_fields += MPI_Wtime();

    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_jboundary -= MPI_Wtime();
    // set J to 0 before we start depositing values from current particles
    for (int i = 0; i < vdim; i++)
    {
        mw_yee->J_Array[i]->setVal(0.0);
    }
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_jboundary += MPI_Wtime();

    for (int spec = 0; spec < numspec; spec++)
    {
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_deposition -= MPI_Wtime();
        amrex::Real charge = part_gr[spec]->getCharge();
        amrex::Real chargemass = charge / part_gr[spec]->getMass();

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
        {
            // transfer data in GPU compatible data structures
            amrex::Real const dt = mw_yee->dt;
            const long np = pti.numParticles();
            auto& particles = pti.GetArrayOfStructs();
            auto partData = particles().data();
            auto& particleAttributes = pti.GetStructOfArrays();
            auto part_weight = particleAttributes.GetRealData(vdim).data();
            auto velx = particleAttributes.GetRealData(0).data();
            auto vely = particleAttributes.GetRealData(1).data();
            auto velz = particleAttributes.GetRealData(2).data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> bA;
            for (int cc = 0; cc < vdim; cc++)
            {
                jA[cc] = (*(mw_yee->J_Array)[cc])[pti].array();
                eA[cc] = (*(mw_yee->HE_Array)[cc])[pti].array();
            }
            for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
            {
                bA[cc] = (*(mw_yee->B_Array)[cc])[pti].array();
            }
            BL_PROFILE_VAR("Apply H_E, update v and apply Hp", heV_hp);
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                amrex::GpuArray<amrex::Real, 2> vxbFields{0., 0.};  // only two comp. needed
                splines_at_particles<degx, degy, degz> spline;
                spline1d_at_particles<degx> spline_new_x;
                spline1d_at_particles<degx> spline_old_x;
                spline1d_at_particles<degy> spline_new_y;
                spline1d_at_particles<degy> spline_old_y;
                spline1d_at_particles<degz> spline_new_z;
                spline1d_at_particles<degz> spline_old_z;
                amrex::GpuArray<amrex::Real, std::max(degx, std::max(degy, degz)) + 4> primitive;

                // Read out particle position, velocities and scaled weight
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = partData[pp].pos(d);
                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                amrex::Real weight = charge * part_weight[pp] * infra.dxi[GEMPIC_SPACEDIM];
                // evaluate the spline at particle position
                spline.init_particles(pos, infra.plo, infra.dxi);

                // evaluate the electric field at particle position
                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                // push v with the electric field
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    push_v_efield<vdim>(vel, dt * 0.5 * coeff, chargemass, efield);
                vel[0] = newPosE[0];
                vel[1] = newPosE[1];
                vel[2] = newPosE[2];

                apply_hp1<splines_at_particles<degx, degy, degz>, vdim, degx, degy, degz>(
                    infra, dt * 0.5 * coeff, pos, vel, spline, spline_new_x, spline_old_x,
                    primitive, jA, bA, vxbFields, weight, chargemass);
                if (vdim > 1)
                {
                    apply_hp2<splines_at_particles<degx, degy, degz>, vdim, degx, degy, degz>(
                        infra, dt * 0.5 * coeff, pos, vel, spline, spline_new_y, spline_old_y,
                        primitive, jA, bA, vxbFields, weight, chargemass);
                }
                if (vdim > 2)
                {
#if GEMPIC_SPACEDIM == 3
                    // push x3
                    amrex::Real x3_new = pos[2] + dt * coeff * vel[2];

                    spline_new_z.init_position(x3_new, infra.plo[2], infra.dxi[2]);
                    spline_old_z.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                    spline.template init_position<2, degz>(x3_new, infra.plo[2], infra.dxi[2]);

                    accumulate_j_update_v_C2<splines_at_particles<degx, degy, degz>, vdim, degz,
                                             degx, degy, 2,
                                             std::max(degx, std::max(degy, degz)) + 4>(
                        spline, spline_new_z, spline_old_z, weight, infra.dx, bA, jA, vxbFields,
                        primitive);

                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass * vxbFields[0];
                    vel[1] = vel[1] + chargemass * vxbFields[1];
                    pos[2] = x3_new;
#else
                        amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> bfield =
                            evaluate_bfield<vdim, degx, degy, degz>(spline, bA);
                        gempic_deposit_rho<degx, degy, degz>(spline, weight * vel[2] * dt * coeff,
                                                             jA[2]);
                        // Push v2 and v3
                        vel[0] = vel[0] - chargemass * vxbFields[1] * vel[2] * dt * coeff;
                        vel[1] = vel[1] + chargemass * vxbFields[0] * vel[2] * dt * coeff;
#endif
                }

                if (vdim > 1)
                {
                    apply_hp2<splines_at_particles<degx, degy, degz>, vdim, degx, degy, degz>(
                        infra, dt * 0.5 * coeff, pos, vel, spline, spline_new_y, spline_old_y,
                        primitive, jA, bA, vxbFields, weight, chargemass);
                }
                apply_hp1<splines_at_particles<degx, degy, degz>, vdim, degx, degy, degz>(
                    infra, dt * 0.5 * coeff, pos, vel, spline, spline_new_x, spline_old_x,
                    primitive, jA, bA, vxbFields, weight, chargemass);

                // update positions and velocities in particle data structure
                for (int xd = 0; xd < GEMPIC_SPACEDIM; xd++) partData[pp].pos(xd) = pos[xd];
                velx[pp] = vel[0];
                vely[pp] = vel[1];
                velz[pp] = vel[2];
            });
            BL_PROFILE_VAR_STOP(heV_hp);
        }
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_deposition += MPI_Wtime();
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_redistribute -= MPI_Wtime();
        part_gr[spec]->Redistribute();  // assign particles to the tile they are in
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_redistribute += MPI_Wtime();
    }

    // add values from ghost cells and periodic boundary
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_jboundary -= MPI_Wtime();
    BL_PROFILE_VAR("Time Loop Boundary synchronization", bsync);
    Gempic_SumBoundary(mw_yee, &infra);  // Sums Boundary of J and Overwrites overlap
    for (int i = 0; i < vdim; i++)
    {
        (*(mw_yee->J_Array)[i]).FillBoundary(infra.geom.periodicity());
    }
    BL_PROFILE_VAR_STOP(bsync);
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_jboundary += MPI_Wtime();

    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_fields -= MPI_Wtime();
    BL_PROFILE_VAR("Apply Hp, E part", hpe);
    // advance E from j (field part of H_p)
    mw_yee->advance_E(infra, 1.0, false, true, mw_yee->HB_Array, mw_yee->E_Array);
    mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
    BL_PROFILE_VAR_STOP(hpe);
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_fields += MPI_Wtime();

    // Second particle loop for particle part from H_E
    for (int spec = 0; spec < numspec; spec++)
    {
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_deposition -= MPI_Wtime();
        amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            auto& particles = pti.GetArrayOfStructs();
            auto partData = particles().data();
            auto& particleAttributes = pti.GetStructOfArrays();
            auto part_weight = particleAttributes.GetRealData(vdim).data();
            auto velx = particleAttributes.GetRealData(0).data();
            auto vely = particleAttributes.GetRealData(1).data();
            auto velz = particleAttributes.GetRealData(2).data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (*(mw_yee->HE_Array)[cc])[pti].array();
            }
            amrex::Real dt = mw_yee->dt;
            // loop over particles: add contribution of old particle position to J and push
            BL_PROFILE_VAR("Apply H_E, update v (second part", heV_2);
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp) {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = partData[pp].pos(d);

                splines_at_particles<degx, degy, degz> splines;

                // evaluate the spline at particle positions
                splines.init_particles(pos, infra.plo, infra.dxi);
                // evaluate the electric field
                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(splines, eA);
                // push v with the electric field
                amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    push_v_efield<vdim>(vel, dt * 0.5 * coeff, chargemass, efield);
                velx[pp] = newPosE[0];
                vely[pp] = newPosE[1];
                velz[pp] = newPosE[2];
            });
            BL_PROFILE_VAR_STOP(heV_2);
        }
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_deposition += MPI_Wtime();
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_redistribute -= MPI_Wtime();
        part_gr[spec]->Redistribute();  // assign particles to the tile they are in
        if (profiling_timers->profiling && profiling_count)
            profiling_timers->counter_redistribute += MPI_Wtime();
    }

    //--------------------------------------------------------------------------
    // Operator H_E for dt/2 (field part)
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_fields -= MPI_Wtime();
    if (electromagnetic)
    {
        BL_PROFILE_VAR("Apply H_E, update B (second step)", heB_2);
        // mw_yee->template hodge_full<degmw>(infra, &(mw_yee->E_Array),
        // &(mw_yee->HE_Array), true);
        // WHY NO HODGE ???
        mw_yee->advance_B(infra, mw_yee->dt * 0.5 * coeff, mw_yee->HE_Array, mw_yee->B_Array);
        BL_PROFILE_VAR_STOP(heB_2);
        // Operator H_B for dt/2
        BL_PROFILE_VAR("Apply H_B, update E (second step)", hb_2);
        mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        mw_yee->advance_E(infra, mw_yee->dt * 0.5 * coeff, true, false, mw_yee->HB_Array,
                          mw_yee->E_Array);
        BL_PROFILE_VAR_STOP(hb_2);
    }
    if (profiling_timers->profiling && profiling_count)
        profiling_timers->counter_fields += MPI_Wtime();
}
template <int vdim, int numspec, int degx, int degy, int degz, int degmw,
          bool electromagnetic = true, bool profiling = true, bool output = true>
void time_loop_hs_zigzag_C2(
    computational_domain infra, maxwell_yee<vdim>* mw_yee,
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec>& part_gr,
    diagnostics<vdim, numspec, degx, degy, degz, degmw>* diagn, bool ctest,
    std::string simulation_name, int strang_order)
{
    BL_PROFILE("time_loop_hs_zigzag_C2");

    bool energy_experiment = true;
    amrex::Real init_energy = 0.5 * (diagn->Efield_energy[0][0] + diagn->Efield_energy[0][1] +
                                     diagn->Efield_energy[0][2] + diagn->Bfield_energy[0][0] +
                                     diagn->Bfield_energy[0][1] + diagn->Bfield_energy[0][2]) +
                              diagn->kin[0] / (infra.Length[0] * infra.Length[1] * infra.Length[2]);
    if (amrex::ParallelDescriptor::MyProc() == 0) remove("energy_error.out.0");
    bool profiling_count = false;
    timers profiling_timers(true);

    // if (profiling_timers.profiling) profiling_timers.counter_all -= MPI_Wtime();
    for (int t_step = 0; t_step < mw_yee->nsteps; t_step++)
    {
        BL_PROFILE("time_loop_hs_zigzag_C2::step");

        if (profiling_timers.profiling && t_step == 0) profiling_timers.counter_all -= MPI_Wtime();
        switch (strang_order)
        {
            case 2:
                time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                            profiling>(infra, mw_yee, 1.0, part_gr,
                                                       &profiling_timers, profiling_count);
                break;
            case 4:
                amrex::Real alpha = 1. / (2. - pow(2., 1. / 3.));
                amrex::Real beta = 1. - 2. * alpha;

                time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                            profiling>(infra, mw_yee, alpha, part_gr,
                                                       &profiling_timers, profiling_count);
                time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                            profiling>(infra, mw_yee, beta, part_gr,
                                                       &profiling_timers, profiling_count);
                time_loop_hs_zigzag_C2_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic,
                                            profiling>(infra, mw_yee, alpha, part_gr,
                                                       &profiling_timers, profiling_count);
                break;
        }

        if (output)
            diagn->end_of_timestep(&profiling_timers, t_step, infra, mw_yee, part_gr,
                                   simulation_name, ctest);
        profiling_count = true;
        if (energy_experiment)
        {
            BL_PROFILE("energy_experiment");
            amrex::Real end_energy =
                0.5 * (diagn->Efield_energy[t_step][0] + diagn->Efield_energy[t_step][1] +
                       diagn->Efield_energy[t_step][2] + diagn->Bfield_energy[t_step][0] +
                       diagn->Bfield_energy[t_step][1] + diagn->Bfield_energy[t_step][2]) +
                diagn->kin[t_step] / (infra.Length[0] * infra.Length[1] * infra.Length[2]);

            if (amrex::ParallelDescriptor::MyProc() == 0)
                amrex::PrintToFile("energy_error.out")
                    << t_step * mw_yee->dt << " " << std::abs(init_energy - end_energy)
                    << std::endl;
        }
    }

    if (profiling_timers.profiling)
    {
        profiling_timers.counter_all += MPI_Wtime();
        amrex::Print() << "Deposition time: " << profiling_timers.counter_deposition
                       << ", redistribute time: " << profiling_timers.counter_redistribute
                       << ", field solver time: " << profiling_timers.counter_fields
                       << ", sum boundary j time: " << profiling_timers.counter_jboundary
                       << ", diagnostics time: " << profiling_timers.counter_diagnostics
                       << ", total time: " << profiling_timers.counter_all << std::endl;
    }
}
}  // namespace Time_Loop
}  // namespace Gempic

#endif

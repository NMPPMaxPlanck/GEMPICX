#ifndef TIME_LOOP_HSALL_FEM_
#define TIME_LOOP_HSALL_FEM_

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
#include <GEMPIC_SumBoundary.H>
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
using namespace Profiling;

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
          bool electromagnetic = true>
void time_loop_hsall_fem_step(computational_domain infra, maxwell_yee<vdim> *mw_yee,
                              amrex::Real coeff, particle_groups<vdim, numspec> *part_gr,
                              timers *profiling_timers)
{
    mw_yee->advance_time(coeff);

    // Operator H_B for dt/2
    if (electromagnetic)
    {
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
        (*(mw_yee->J_Array)[i]).setVal(0.0, 0);  // value and component
        (*(mw_yee->J_Array)[i]).FillBoundary(infra.geom.periodicity());
    }

    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real chargemass = part_gr->charge[spec] / part_gr->mass[spec];
        amrex::Real charge = part_gr->charge[spec];

        for (amrex::ParIter<vdim + 1, 0, 0, 0> pti(*part_gr->mypc[spec], 0); pti.isValid(); ++pti)
        {
            auto &particles = pti.GetArrayOfStructs();  // get particles
            const long np = pti.numParticles();

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
                    vel[comp] = particles[pp].rdata(comp);
                }
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    push_v_efield<vdim>(vel, mw_yee->dt * 0.5 * coeff, chargemass, efield);
                std::copy(newPosE.begin(), newPosE.end(),
                          &particles.data()[pp * (GEMPIC_SPACEDIM + vdim + 2) + GEMPIC_SPACEDIM]);

                amrex::Real weight =
                    charge * particles[pp].rdata(vdim) * infra.dxi[GEMPIC_SPACEDIM];

                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d) pos[d] = particles[pp].pos(d);

                for (unsigned int d = 0; d < vdim; ++d) vel[d] = particles[pp].rdata(d);
                amrex::GpuArray<amrex::Real, 2> bfields;
                amrex::GpuArray<amrex::Real, 2 * std::max(degx, std::max(degy, degz)) + 4>
                    primitive;
                for (int dd = 0; dd < 2 * std::max(degx, std::max(degy, degz)) + 4; dd++)
                    primitive[dd] = 0.0;
                // push x1
                {
                    amrex::Real x1_new = pos[0] + mw_yee->dt * coeff * vel[0];
                    /*accumulate_j_update_v_x1_fem_integral<vdim, degx, degy, degz>( pos,
                                                        x1_new,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        infra.dxi,
                                                        infra.plo,
                                                        bA,
                                                        jA,
                                                        bfields);*/
                    spline1d_at_particles<degx - 1> spline_y_cell;
                    spline_y_cell.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                    spline1d_at_particles<degx> spline_y_node;
                    spline_y_node.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                    spline1d_at_particles<degx - 1> spline_z_cell;
                    spline_z_cell.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                    spline1d_at_particles<degx> spline_z_node;
                    spline_z_node.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                    spline1d_at_particles<degx> spline_new;
                    spline_new.init_position(x1_new, infra.plo[0], infra.dxi[0]);
                    spline1d_at_particles<degx> spline_old;
                    spline_old.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    accumulate_j_update_v_fem_integral<
                        vdim, degx, degy, degz, 0, 2 * std::max(degx, std::max(degy, degz)) + 4>(
                        pos, x1_new, spline_y_cell, spline_y_node, spline_z_cell, spline_z_node,
                        spline_new, spline_old, weight, infra.geom.CellSize(), infra.dxi, infra.plo,
                        bA, jA, bfields, primitive);

                    // Push v2 and v3
                    if (vdim > 1) vel[1] = vel[1] - chargemass * bfields[0];
                    if (vdim > 2) vel[2] = vel[2] + chargemass * bfields[1];
                    pos[0] = x1_new;
                }

                if (vdim > 1)
                {
#if GEMPIC_SPACEDIM > 1
                    // push x2
                    amrex::Real x2_new = pos[1] + (*mw_yee).dt * coeff * vel[1];
                    /*accumulate_j_update_v_x2_fem_integral<vdim, degx, degy, degz>( pos,
                                                      x2_new,
                                                      weight,
                                                      infra.geom.CellSize(),
                                                      infra.dxi,
                                                      infra.plo,
                                                      bA,
                                                      jA,
                                                      bfields);*/
                    spline1d_at_particles<degy - 1> spline_y_cell;
                    spline_y_cell.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                    spline1d_at_particles<degy> spline_y_node;
                    spline_y_node.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                    spline1d_at_particles<degy - 1> spline_z_cell;
                    spline_z_cell.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    spline1d_at_particles<degy> spline_z_node;
                    spline_z_node.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    spline1d_at_particles<degy> spline_new;
                    spline_new.init_position(x2_new, infra.plo[1], infra.dxi[1]);
                    spline1d_at_particles<degy> spline_old;
                    spline_old.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                    accumulate_j_update_v_fem_integral<
                        vdim, degy, degz, degx, 1, 2 * std::max(degx, std::max(degy, degz)) + 4>(
                        pos, x2_new, spline_y_cell, spline_y_node, spline_z_cell, spline_z_node,
                        spline_new, spline_old, weight, infra.geom.CellSize(), infra.dxi, infra.plo,
                        bA, jA, bfields, primitive);

                    // Push v2 and v3
                    vel[0] = vel[0] + chargemass * bfields[1];
                    if (vdim > 2) vel[2] = vel[2] - chargemass * bfields[0];
                    pos[1] = x2_new;
#else
                    amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> bfield =
                        evaluate_fem_bfield<vdim, degx, degy, degz>(pos, infra.dxi, infra.plo, bA);
                    gempic_deposit_rho<amrex::Particle<vdim + 1>, vdim, degx, degy, degz>(
                        particles[pp], weight * vel[1] * (*mw_yee).dt * coeff, jA[1], infra.plo,
                        infra.dxi);
                    // Push v2 and v3
                    vel[0] = vel[0] + chargemass * bfields[0] * vel[1] * (*mw_yee).dt * coeff;
#endif
                }
                if (vdim > 2)
                {
#if GEMPIC_SPACEDIM == 3
                    // push x3
                    amrex::Real x3_new = pos[2] + mw_yee->dt * coeff * vel[2];
                    /*accumulate_j_update_v_x3_fem_integral<vdim, degx, degy, degz>( pos,
                                                        x3_new,
                                                        weight,
                                                        infra.geom.CellSize(),
                                                        infra.dxi,
                                                        infra.plo,
                                                        bA,
                                                        jA,
                                                        bfields);*/
                    spline1d_at_particles<degz - 1> spline_y_cell;
                    spline_y_cell.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    spline1d_at_particles<degz> spline_y_node;
                    spline_y_node.init_position(pos[0], infra.plo[0], infra.dxi[0]);
                    spline1d_at_particles<degz - 1> spline_z_cell;
                    spline_z_cell.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                    spline1d_at_particles<degz> spline_z_node;
                    spline_z_node.init_position(pos[1], infra.plo[1], infra.dxi[1]);
                    spline1d_at_particles<degz> spline_new;
                    spline_new.init_position(x3_new, infra.plo[2], infra.dxi[2]);
                    spline1d_at_particles<degz> spline_old;
                    spline_old.init_position(pos[2], infra.plo[2], infra.dxi[2]);
                    accumulate_j_update_v_fem_integral<
                        vdim, degz, degx, degy, 2, 2 * std::max(degx, std::max(degy, degz)) + 4>(
                        pos, x3_new, spline_y_cell, spline_y_node, spline_z_cell, spline_z_node,
                        spline_new, spline_old, weight, infra.geom.CellSize(), infra.dxi, infra.plo,
                        bA, jA, bfields, primitive);
                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass * bfields[0];
                    vel[1] = vel[1] + chargemass * bfields[1];
                    pos[2] = x3_new;
#else
                    amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1> bfield =
                        evaluate_fem_bfield<vdim, degx, degy, degz>(pos, infra.dxi, infra.plo, bA);
                    gempic_deposit_rho<amrex::Particle<vdim + 1>, vdim, degx, degy, degz>(
                        particles[pp], weight * vel[2] * (*mw_yee).dt * coeff, jA[2], infra.plo,
                        infra.dxi);
                    // Push v2 and v3
                    vel[0] = vel[0] - chargemass * bfields[1] * vel[2] * (*mw_yee).dt * coeff;
                    vel[1] = vel[1] + chargemass * bfields[0] * vel[2] * (*mw_yee).dt * coeff;
#endif
                }
                std::copy(pos.begin(), pos.end(),
                          &particles.data()[pp * (GEMPIC_SPACEDIM + vdim + 2)]);
                std::copy(vel.begin(), vel.end(),
                          &particles.data()[pp * (GEMPIC_SPACEDIM + vdim + 2) + GEMPIC_SPACEDIM]);
            }
        }
        part_gr->mypc[spec]->Redistribute();  // assign particles to the tile they are in
    }

    // add values from ghost cells and periodic boundary
    Gempic_SumBoundary(mw_yee, &infra);  // Sums Boundary of J and Overwrites overlap
    for (int i = 0; i < vdim; i++)
    {
        (*(mw_yee->J_Array)[i]).FillBoundary(infra.geom.periodicity());
    }
    /*amrex::Vector<std::string> varnames = { "J_x"};
    amrex::Real timeo = t_step*(*mw_yee).dt*coeff;
    WriteSingleLevelPlotfile("current", *(*mw_yee).J_Array[0], varnames,
    infra.geom, timeo, 0);
    abort();*/

    // advance E from j (field part of H_p)
    mw_yee->advance_E(infra, 1.0, false, true, mw_yee->HB_Array, mw_yee->E_Array);

    // Second particle loop for particle part from H_E
    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real chargemass = part_gr->charge[spec] / part_gr->mass[spec];

        for (amrex::ParIter<vdim + 1, 0, 0, 0> pti(*part_gr->mypc[spec], 0); pti.isValid(); ++pti)
        {
            auto &particles = pti.GetArrayOfStructs();  // get particles
            const long np = pti.numParticles();

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
                    pos[comp] = particles[pp].pos(comp);
                }
                spline.init_particles(pos, infra.plo, infra.dxi);
                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_fem_efield<vdim, degx, degy, degz>(spline, eA);

                // push v with the electric field
                amrex::GpuArray<amrex::Real, vdim> vel;
                for (int comp = 0; comp < vdim; comp++)
                {
                    vel[comp] = particles[pp].rdata(comp);
                }
                amrex::GpuArray<amrex::Real, vdim> newPosE =
                    push_v_efield<vdim>(vel, mw_yee->dt * 0.5 * coeff, chargemass, efield);
                std::copy(newPosE.begin(), newPosE.end(),
                          &particles.data()[pp * (GEMPIC_SPACEDIM + vdim + 2) + GEMPIC_SPACEDIM]);
            }
        }
        part_gr->mypc[spec]->Redistribute();  // assign particles to the tile they are in
    }

    //--------------------------------------------------------------------------
    // Operator H_E for dt/2 (field part)
    if (electromagnetic)
    {
        mw_yee->template hodge_full<degmw>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
        mw_yee->advance_B(infra, mw_yee->dt * 0.5 * coeff, mw_yee->HE_Array, mw_yee->B_Array);

        // Operator H_B for dt/2
        mw_yee->template hodge_full<degmw>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        mw_yee->advance_E(infra, mw_yee->dt * 0.5 * coeff, true, false, mw_yee->HB_Array,
                          mw_yee->E_Array);
    }
}

template <int vdim, int numspec, int degx, int degy, int degz, int degmw,
          bool electromagnetic = true>
void time_loop_hsall_fem(computational_domain infra, maxwell_yee<vdim> *mw_yee,
                         particle_groups<vdim, numspec> *part_gr,
                         diagnostics<vdim, numspec, degx, degy, degz, degmw> *diagn, bool ctest,
                         std::string simulation_name, int strang_order)
{
    timers profiling_timers(false);

    for (int t_step = 0; t_step < mw_yee->nsteps; t_step++)
    {
        switch (strang_order)
        {
            case 2:
                time_loop_hsall_fem_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic>(
                    infra, mw_yee, 1.0, part_gr, &profiling_timers);
                break;
            case 4:
                amrex::Real alpha = 1. / (2. - pow(2., 1. / 3.));
                amrex::Real beta = 1. - 2. * alpha;

                time_loop_hsall_fem_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic>(
                    infra, mw_yee, alpha, part_gr, &profiling_timers);
                time_loop_hsall_fem_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic>(
                    infra, mw_yee, beta, part_gr, &profiling_timers);
                time_loop_hsall_fem_step<vdim, numspec, degx, degy, degz, degmw, electromagnetic>(
                    infra, mw_yee, alpha, part_gr, &profiling_timers);
                break;
        }

        diagn->end_of_timestep(&profiling_timers, t_step, infra, mw_yee, part_gr, simulation_name,
                               ctest);
    }

}

}  // namespace Time_Loop
}  // namespace Gempic

#endif

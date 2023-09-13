/*------------------------------------------------------------------------------
 functions copied from the AMReX library to make small changes
 AMReX file containing original functions: AMReX_Particle_mod_K.H
 (amrex/Src/Particle/AMReX_Particle_mod_K.H) pulled on 07.10.2019
------------------------------------------------------------------------------*/

#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Math.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_Spline_Class.H>

#include <cmath>

namespace Gempic
{
namespace Particles
{
/**
     Computes for each component i, J_i from \int_t Rho and \int_t v_i x B by integrating over the B-splines
     */

/**
 * @brief Updates J and B along one grid direction
 *
 * @tparam S splines_at_particles type
 * @tparam vdim Number of Components of B and J
 * @tparam degp Degree of spline in direction of integration (x, y or z)
 * @tparam degp1 Degree of spline in next direction (cyclic)
 * @tparam degp2 Degree of spline in second next direction (cyclic)
 * @tparam pdim Direction of integration (x=0, y=1, z=2)
 * @tparam plength
 * @param spline 3D spline at particle position
 * @param spline_new 1D spline at new particle position along given direction
 * @param spline_old 1D spline at old particle position along given direction
 * @param weight Particle charge * weight * dx
 * @param dx
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 * @param[out] fields Contribution of B (the 2 that are not in integration direction)
 * @param primitive Temporary array for holding difference of primitives (used for each direction)
 */
template <typename SplineType, int vdim, int degp, int degp1, int degp2, int pdim>
AMREX_GPU_HOST_DEVICE void accumulate_j_update_v(
    SplineType &spline,
    amrex::Real weight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jArray,
    amrex::GpuArray<amrex::Real, 2> &fields)
{
    int pdim1 = (pdim + 1) % 3;
    int pdim2 = (pdim + 2) % 3;

    fields = {0., 0.};
    // Size of support of difference of primitives
    // check for ghost cells
    int local_size = std::abs(spline.span[pdim] - spline.spanOld[pdim]) + degp;
    int span_start = spline.spanOld[pdim];
    if (spline.span[pdim] < spline.spanOld[pdim]) span_start = spline.span[pdim];

// Add particle contribution to J
GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= local_size; ++ii),
                    for (int jj = 0; jj <= degp1; ++jj),
                    for (int kk = 0; kk <= degp2; ++kk));

                amrex::Real primitiveDifference = spline.template computePrimitiveDifference<pdim, degp>(dx, ii);

                amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index;
                if (pdim == 0)
                    cell_index = {
                        AMREX_D_DECL(span_start + ii, spline.span[1] + jj, spline.span[2] + kk)};
                else if (pdim == 1)
                    cell_index = {
                        AMREX_D_DECL(spline.span[0] + kk, span_start + ii, spline.span[2] + jj)};
                else if (pdim == 2)
                    cell_index = {
                        AMREX_D_DECL(spline.span[0] + jj, spline.span[1] + kk, span_start + ii)};
                double s_coeff =
                    primitiveDifference * spline.splineNode[pdim1][jj] * spline.splineNode[pdim2][kk];
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[pdim].ptr(cell_index[0], 0, 0, 0),
                                               s_coeff * weight);
#else
                amrex::HostDevice::Atomic::Add(
                    jArray[pdim].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0),
                    s_coeff * weight);
#endif
                // Update two components of B
                if (jj < degp1)
                {
                    double s_coeff = primitiveDifference * spline.splineCell[pdim1][jj] *
                                     spline.splineNode[pdim2][kk];

#if (GEMPIC_SPACEDIM == 1)
                    fields[0] += s_coeff * (bArray[pdim2])(cell_index[0], 0, 0, 0);
#else
                    fields[0] +=
                        s_coeff * (bArray[pdim2])(
                                      AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
                if (kk < degp2)
                {
                    double s_coeff = primitiveDifference * spline.splineNode[pdim1][jj] *
                                     spline.splineCell[pdim2][kk];
#if (GEMPIC_SPACEDIM == 1)
                    fields[1] += s_coeff * (bArray[pdim1])(cell_index[0], 0, 0, 0);
#else
                    fields[1] +=
                        s_coeff * (bArray[pdim1])(
                                      AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }

GEMPIC_D_LOOP_END;

}

/*!
 * Push the velocity with a given electric field
 */
template <int vdim>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> push_v_efield(
    amrex::GpuArray<amrex::Real, vdim> vel,
    amrex::Real dt,
    amrex::Real chargemass,  // charge/mass
    amrex::GpuArray<amrex::Real, vdim> &Ep)
{
    amrex::GpuArray<amrex::Real, vdim> newPos;

    AMREX_D_TERM((newPos[0] = vel[0] + dt * chargemass * (Ep[0]));,
                 (newPos[1] = vel[1] + dt * chargemass * (Ep[1]));,
                 (newPos[2] = vel[2] + dt * chargemass * (Ep[2])););
                 
    return (newPos);
}

/**
 * @brief Deposits charge density on cells of dual grid
 *
 * Here nodal splines seem to be used for deposition where integral of cell splines are expected
 *
 * @tparam degx Spline degree in x
 * @tparam degy Spline degree in y
 * @tparam degz Spline degree in z
 * @param spline Splines evaluated at given particle position
 * @param weight Particle weight*charge*dx
 * @param rho charge density to be computed
 */
template <int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE void gempic_deposit_rho(
    Spline::SplineBase<degx, degy, degz> &spline,
    amrex::Real weight,
    amrex::Array4<amrex::Real> const &rho)
{
GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degx; ++ii),
                    for (int jj = 0; jj <= degy; ++jj),
                    for (int kk = 0; kk <= degz; ++kk))

    amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index = {
        AMREX_D_DECL(spline.span[0] + ii, spline.span[1] + jj, spline.span[2] + kk)};
    double s_coeff = spline.template evalBSpline<-1, 0>(ii, jj, kk); 

#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(rho.ptr(cell_index[0], 0, 0, 0), s_coeff * weight);
#else
                amrex::HostDevice::Atomic::Add(
                    rho.ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0),
                    s_coeff * weight);
#endif

GEMPIC_D_LOOP_END;
}

}  // namespace Particles
}  // namespace Gempic

#endif
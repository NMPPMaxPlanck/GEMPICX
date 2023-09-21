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
 * @brief Updates J and computes integrated B on a given particle trajectory along one grid direction
 *  
 * Description:
 * given a particle trajectory x_{p,d} moving along a single dimension only (with velocity v_{p,d} aligned with grid dimension d), this function:
 * 
 * (1) for every dual face (dof) index) i i-th adds the time integrals 
 *     dJ_{i,p,d} = q_p w_p \int_{t_start}^{t_end} t_R2_i( v_{p,d} * S(x-x_{p,d}(t)) ) dt
 *     to the i-th component of the "current density"
 *     where t_R2_i is the dof (reduction operator) corresponding to dual face i
 *     
 *     Note that the resulting current density is a time integral over the time step, not a time average
 *  
 * (2) computes the time integral of magnetic field needed for the time-integrated Lorentz force corresponding to this straight trajectory
 *     dF_{p,d} = (q/m)_p \int_{t_start}^{t_end} v_{p,d} x B(x_{p,d}(t))dt
 *
 *     Specifically this function computes
 *     fields[0] = \int_{t_start}^{t_end} B_{d-1}(x_{p,d}(t))dt
 *     fields[1] = \int_{t_start}^{t_end} B_{d+1}(x_{p,d}(t))dt
 *     So that after this function call the velocity can be updated over the time step 
 *     with 
 *     v_{p,d} += dF_{p,d}
 *     ie
 *     v_{p,d}_{d+1} -= (q/m)_p * v_{p,d}_d * fields[0]
 *     v_{p,d}_{d-1} += (q/m)_p * v_{p,d}_d * fields[1]
 *     (the Lorentz force has no contribution on the d component of v_{p,d})
 *     
 * for details about the algorithm see the documentation
 * https://gempic.pages.mpcdf.de/gempic/latex/particleMesh.html#exact-computation-of-time-integrated-j-x-v-p-x-b-y-x-p-and-v-p-x-b-z-x-p
 * 
 * @tparam SplineType Necessary for mocking to the function can take mock spline types
 * @tparam vDim Number of Components of B and J
 * @tparam degP Degree of spline in direction of integration (x, y or z)
 * @tparam degP1 Degree of spline in next direction (cyclic)
 * @tparam degP2 Degree of spline in second next direction (cyclic)
 * @tparam pDim Direction of integration (x=0, y=1, z=2)
 * 
 * @param spline 3D spline containing spline values and the particle trajectory
 * @param weight Particle weight
 * @param dx Spatial grid size
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 * @param[out] fields Contribution of B (the 2 that are not in integration direction)
 */
template <typename SplineType, int vDim, int pDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_update_v(
    SplineType &spline,
    amrex::Real weight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
    amrex::GpuArray<amrex::Real, 2> &fields)
{
    constexpr int pDim1 = (pDim + 1) % 3;
    constexpr int pDim2 = (pDim + 2) % 3;

    constexpr int degP1 = spline.splineDegrees[pDim1];
    constexpr int degP2 = spline.splineDegrees[pDim2];

    fields = {0., 0.};
    // Size of support of difference of primitives
    // check for ghost cells
    int local_size = std::abs(spline.span[pDim] - spline.spanOld[pDim]) + spline.splineDegrees[pDim];
    int span_start = spline.spanOld[pDim];
    if (spline.span[pDim] < spline.spanOld[pDim]) span_start = spline.span[pDim];

// Add particle contribution to J
GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= local_size; ++ii),
                    for (int jj = 0; jj <= degP1; ++jj),
                    for (int kk = 0; kk <= degP2; ++kk));

                amrex::Real primitiveDifference = spline.template computePrimitiveDifference<pDim>(dx, ii);

                amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index;
                if (pDim == 0)
                    cell_index = {
                        AMREX_D_DECL(span_start + ii, spline.span[1] + jj, spline.span[2] + kk)};
                else if (pDim == 1)
                    cell_index = {
                        AMREX_D_DECL(spline.span[0] + kk, span_start + ii, spline.span[2] + jj)};
                else if (pDim == 2)
                    cell_index = {
                        AMREX_D_DECL(spline.span[0] + jj, spline.span[1] + kk, span_start + ii)};
                double s_coeff =
                    primitiveDifference * spline.splineNode[pDim1][jj] * spline.splineNode[pDim2][kk];
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(jArray[pDim].ptr(cell_index[0], 0, 0, 0),
                                               s_coeff * weight);
#else
                amrex::HostDevice::Atomic::Add(
                    jArray[pDim].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0),
                    s_coeff * weight);
#endif
                // Update two components of B
                if (jj < degP1)
                {
                    double s_coeff = primitiveDifference * spline.splineCell[pDim1][jj] *
                                     spline.splineNode[pDim2][kk];

#if (GEMPIC_SPACEDIM == 1)
                    fields[0] += s_coeff * (bArray[pDim2])(cell_index[0], 0, 0, 0);
#else
                    fields[0] +=
                        s_coeff * (bArray[pDim2])(
                                      AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
                if (kk < degP2)
                {
                    double s_coeff = primitiveDifference * spline.splineNode[pDim1][jj] *
                                     spline.splineCell[pDim2][kk];
#if (GEMPIC_SPACEDIM == 1)
                    fields[1] += s_coeff * (bArray[pDim1])(cell_index[0], 0, 0, 0);
#else
                    fields[1] +=
                        s_coeff * (bArray[pDim1])(
                                      AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }

GEMPIC_D_LOOP_END;

}

/*!
 * @brief Push the velocity with a given electric field
 *
 * @tparam vDim Number of Components of B and J
 * 
 * @param vel Velocity component
 * @param dt Temporal grid size
 * @param chargemass : (q/m)_p ratio
 * @param Ep Electric field
 */
template <int vDim>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> push_v_efield(
    amrex::GpuArray<amrex::Real, vDim> vel,
    amrex::Real dt,
    amrex::Real chargemass,
    amrex::GpuArray<amrex::Real, vDim> &Ep)
{
    amrex::GpuArray<amrex::Real, vDim> newPos;

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
 * @tparam degX Spline degree in x
 * @tparam degY Spline degree in y
 * @tparam degZ Spline degree in z
 * 
 * @param spline Splines evaluated at given particle position
 * @param weight Particle weight*charge*dx
 * @param rho charge density to be computed
 */
template <int degX, int degY, int degZ>
AMREX_GPU_HOST_DEVICE void gempic_deposit_rho(
    Spline::SplineBase<degX, degY, degZ> &spline,
    amrex::Real weight,
    amrex::Array4<amrex::Real> const &rho)
{
GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii),
                    for (int jj = 0; jj <= degY; ++jj),
                    for (int kk = 0; kk <= degZ; ++kk))

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
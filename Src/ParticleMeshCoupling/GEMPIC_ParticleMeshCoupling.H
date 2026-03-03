/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Math.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"

namespace Gempic::ParticleMeshCoupling
{
namespace Impl
{
/**
 * @brief Computes the integral of the D update

 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam pDir1 First direction not along the trajectory
 * @tparam pDir2 Second direction not along the trajectory
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam vDim Number of components of J
 *
 * @param[out] jArray Degrees of freedom of J as dual 2-form
 * @param spline Object containing spline values and the particle trajectory
 * @param cellIndex Primitive cell index from spline
 * @param j sum index for direction pDir1
 * @param k sum index for direction pDir2
 * @param chargeWeight Particle weight * charge
 * @param primitiveDifference The difference of the primitives at the current and previous time step
 */
template <Direction pDir, int pDir1, int pDir2, typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void update_j_component (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray,
    SplineType const& spline,
    amrex::Dim3 const& cellIndex,
    int const j,
    int const k,
    double const chargeWeight,
    double const primitiveDifference)
{
    double sCoeff =
        chargeWeight * GEMPIC_D_MULT(primitiveDifference, spline.m_nodeSplineVals[pDir1][j],
                                     spline.m_nodeSplineVals[pDir2][k]);
    amrex::HostDevice::Atomic::Add(jArray[pDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                   sCoeff);
}

#if AMREX_SPACEDIM == 3
/**
 * @brief Computes the product for the given integration index of the integral of B
 *
 * @tparam pDir1 First direction not along the trajectory
 * @tparam pDir2 Second direction not along the trajectory
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 *
 * @param[out] bArray Degrees of freedom of B as primal 2-form
 * @param spline Object containing spline values and the particle trajectory
 * @param cellIndex Primitive cell index from spline
 * @param j sum index for direction pDir1
 * @param k sum index for direction pDir2
 * @param primitiveDifference The difference of the primitives at the current and previous time step
 */
template <int pDir1, int pDir2, typename SplineType, unsigned int bDim>
AMREX_GPU_HOST_DEVICE inline amrex::Real update_integrated_b_component (
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    SplineType const& spline,
    amrex::Dim3 const& cellIndex,
    int const j,
    int const k,
    double const primitiveDifference)
{
    double sCoeff = GEMPIC_D_MULT(primitiveDifference, spline.m_cellSplineVals[pDir1][j],
                                  spline.m_nodeSplineVals[pDir2][k]);

    return sCoeff * (bArray[pDir2])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
}

/**
 * @brief computes the integral of B components along the trajectory
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam pDir1 First direction not along the trajectory
 * @tparam pDir2 Second direction not along the trajectory
 * @tparam degP1 Spline degree of direction pDir1
 * @tparam degp2 Spline degree of direction pDir2
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param primitiveDifference The difference of the primitves at the current and previous time step
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this
 * should be fixed
 */
template <Direction pDir,
          int pDir1,
          int pDir2,
          int degP1,
          int degP2,
          typename SplineType,
          unsigned int bDim,
          unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_loop (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray,
    int const i)
{
    for (int j = 0; j <= degP1; ++j)
    {
        for (int k = 0; k <= degP2; ++k)
        {
            amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(i, j, k);

            update_j_component<pDir, pDir1, pDir2>(jArray, spline, cellIndex, j, k, chargeWeight,
                                                   primitiveDifference);

            // Update two components of B
            if (j < degP1)
            {
                fields[0] += update_integrated_b_component<pDir1, pDir2>(bArray, spline, cellIndex,
                                                                         j, k, primitiveDifference);
            }
            if (k < degP2)
            {
                fields[1] += update_integrated_b_component<pDir2, pDir1>(bArray, spline, cellIndex,
                                                                         k, j, primitiveDifference);
            }
        }
    }
}

template <Direction pDir,
          int pDir1,
          int pDir2,
          int degP1,
          int degP2,
          typename SplineType,
          unsigned int vDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_loop (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    int const i)
{
    for (int j = 0; j <= degP1; ++j)
    {
        for (int k = 0; k <= degP2; ++k)
        {
            amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(i, j, k);

            update_j_component<pDir, pDir1, pDir2>(jArray, spline, cellIndex, j, k, chargeWeight,
                                                   primitiveDifference);
        }
    }
}

#elif AMREX_SPACEDIM == 2

/**
 * @brief computes the integral of B components along the trajectory
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1)
 * @tparam degP Spline degree of direction (for xDir: degY, for yDir: degX)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param primitiveDifference The difference of the primitves at the current and previous time step
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <Direction pDir, int degP, typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_loop (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray,
    int const i)
{
    if constexpr (pDir == xDir)
    {
        for (int j = 0; j <= degP; ++j)
        {
            amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(i, j, 0);

            update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, j, 0, chargeWeight,
                                                 primitiveDifference);

            if (j < degP)
            {
                fields[0] += primitiveDifference * spline.m_cellSplineVals[yDir][j] *
                             (bArray[zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
            }
            fields[1] += primitiveDifference * spline.m_nodeSplineVals[yDir][j] *
                         (bArray[yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
        }
    }
    else if constexpr (pDir == yDir)
    {
        for (int k = 0; k <= degP; ++k)
        {
            amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<yDir>(i, 0, k);

            update_j_component<yDir, xDir, zDir>(jArray, spline, cellIndex, k, 0, chargeWeight,
                                                 primitiveDifference);

            fields[0] += primitiveDifference * spline.m_nodeSplineVals[xDir][k] *
                         (bArray[xDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);

            if (k < degP)
            {
                fields[1] += primitiveDifference * spline.m_cellSplineVals[xDir][k] *
                             (bArray[zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
            }
        }
    }
}

#elif AMREX_SPACEDIM == 1

/**
 * @brief computes the integral of B components along the trajectory
 *
 * @tparam pDir Direction of integration (trajectory): x=0
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param primitiveDifference The difference of the primitves at the current and previous time step
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <Direction pDir, typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_loop (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray,
    int const i)
{
    amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(i, 0, 0);

    update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                         primitiveDifference);

    fields[0] += primitiveDifference * (bArray[zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
    fields[1] += primitiveDifference * (bArray[yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
}

#endif
} //namespace Impl

#if AMREX_SPACEDIM == 1
/**
 * @brief Updates J and computes an integrated B on a given particle for the H_p_y update
 * in one spatial dimension
 *
 * Since for AMREX_SPACEDIM == 1, there is not primitive Difference available for the y-step.
 * Therefore, v_{p,x} and v_{p,z} are updated using an explicit Euler step.
 *
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param dtv timestep * v_y
 * @param chargeWeight Particle weight * charge
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_euler_y (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const& dtv,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    using namespace Impl; // utility functions for this function
    fields = {0., 0.};
    constexpr int degX = SplineType::s_splineDegrees[xDir];

    for (int ii = 0; ii <= degX; ++ii)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][ii];
        amrex::Real xCellVal = spline.m_cellSplineVals[xDir][ii];

        amrex::Dim3 cellIndex = {spline.m_firstIndex[xDir] + ii};

        amrex::HostDevice::Atomic::Add(jArray[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       chargeWeight * dtv * xNodeVal);

        fields[0] += xNodeVal * (bArray[xDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);

        fields[1] += xCellVal * (bArray[zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
    }
}
#endif

#if AMREX_SPACEDIM == 1 || AMREX_SPACEDIM == 2
/**
 * @brief Updates J and computes an integrated B on a given particle for the H_p_z update
 * in one and two spatial dimension
 *
 * For AMREX_SPACEDIM == 1 and AMREX_SPACEDIM == 2 there is no primitive difference available
 * for the z-step. Therefore, v_{p,x} and v_{p,y} are updated using an explicit Euler step.
 *
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param dtv timestep * v_z
 * @param chargeWeight Particle weight * charge
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_euler_z (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const& dtv,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    using namespace Impl; // utility functions for this function
    fields = {0., 0.};

    constexpr int degX = SplineType::s_splineDegrees[xDir];

#if AMREX_SPACEDIM == 1
    for (int ii = 0; ii <= degX; ++ii)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][ii];

        amrex::Dim3 cellIndex = {spline.m_firstIndex[xDir] + ii};

        double sCoeff = chargeWeight * dtv * xNodeVal;

        amrex::HostDevice::Atomic::Add(jArray[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       sCoeff);

        if (ii < degX)
        {
            amrex::Real xCellVal = spline.m_cellSplineVals[xDir][ii];
            fields[0] += xCellVal * (bArray[yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
        }
        fields[1] += xNodeVal * (bArray[xDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
    }

#elif AMREX_SPACEDIM == 2
    constexpr int degY = SplineType::s_splineDegrees[yDir];
    for (int ii = 0; ii <= degX; ++ii)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][ii];

        for (int jj = 0; jj <= degY; ++jj)
        {
            amrex::Real yNodeVal = spline.m_nodeSplineVals[yDir][jj];
            amrex::Dim3 cellIndex = {
                GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii, spline.m_firstIndex[yDir] + jj, 0)};

            double sCoeff = chargeWeight * dtv * xNodeVal * yNodeVal;

            amrex::HostDevice::Atomic::Add(
                jArray[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0), sCoeff);

            if (ii < degX)
            {
                amrex::Real xCellVal = spline.m_cellSplineVals[xDir][ii];
                fields[0] +=
                    xCellVal * yNodeVal * (bArray[yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
            }
            if (jj < degY)
            {
                amrex::Real yCellVal = spline.m_cellSplineVals[yDir][jj];
                fields[1] +=
                    xNodeVal * yCellVal * (bArray[xDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
            }
        }
    }
#endif
}
#endif

#if AMREX_SPACEDIM == 3
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along one grid
 * direction in 3 spatial dimensions
 *
 * @details
 * given a trajectory \f$ xx_{p,d} \f$ moving along a single direction d (with velocity
 * \f$ vv_{p,d} \f$ parallel to \f$ e_d \f$), this function
 *
 * (1) computes the time integrals
 * \f$ dJ_{i,p,d} = q_p w_p\int_{t_{start}}^{t_{end}} t_{R2_i}(v_{p,d} * S(xx - xx_{p,d}(t))) dt \f$
 *     for every dual face (dof) with index i, where \f$ t_{R2_i} \f$ is corresponding dof
 *     (reduction) operator and adds them to the i-th component of the "current density"
 *
 *     Note that the resulting current density is a time integral over the time step, not a time
 * average
 *
 * (2) computes the time integral of the magnetic field
 *     needed for the time-integrated Lorentz force corresponding to this straight trajectory,
 *     \f$ dF_{p,d} = (q/m)_p \int_{t_{start}}^{t_{end}} vv_{p,d} \times B(xx_{p,d}(t))dt \f$
 *
 *     Specifically, denoting \f$ v_p = vv_{{p,d}_d} \f$ the d-th component of
 *     \f$ vv_{p,d} = v_p * e_d \f$, the function computes
 *
 *     fields[xDir] = \f$ \int_{t_{start}}^{t_{end}} v_p * B_{d-1}(xx_{p,d}(t))dt \f$ =
 *     \f$ \int_{0}^{v_p*dt} B_{d-1}(xx_{p,d}(0)+s*e_d) ds \f$ fields[yDir] =
 *     \f$ \int_{t_{start}}^{t_{end}} v_p * B_{d+1}(xx_{p,d}(t))dt \f$ =
 *     \f$ \int_{0}^{v_p*dt} B_{d+1}(xx_{p,d}(0)+s*e_d) ds \f$
 *
 *     So that after this function call the velocity can be updated over the time step
 *     with
 *
 *     \f$ vv_{p,d} += dF_{p,d} \f$, i.e.
 *
 *     \f$ vv_{{p,d}_{d+1}} -= (q/m)_p \f$ * fields[xDir]
 *
 *     \f$ vv_{{p,d}_{d-1}} += (q/m)_p \f$ * fields[yDir]
 *
 *     (here the Lorentz force has no contribution on the d component of \f$ vv_{p,d}) \f$
 *
 * for details about the algorithm see the documentation
 * https://gempic.pages.mpcdf.de/gempic/latex/particleMesh.html#exact-computation-of-time-integrated-j-x-v-p-x-b-y-x-p-and-v-p-x-b-z-x-p
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J

 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param dx Spatial grid size
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */

template <Direction pDir, typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::accumulate_j_integrate_b()");
    using namespace Impl; // utility functions for this function
    constexpr int pDir1 = (pDir + 1) % 3;
    constexpr int pDir2 = (pDir + 2) % 3;

    constexpr int degP = SplineType::s_splineDegrees[pDir];
    constexpr int degP1 = SplineType::s_splineDegrees[pDir1];
    constexpr int degP2 = SplineType::s_splineDegrees[pDir2];

    fields = {0., 0.};
    // Size of support of difference of primitives
    // check for ghost cells
    int localSize = std::abs(spline.m_firstIndex[pDir] - spline.m_firstIndexOld[pDir]) +
                    spline.s_splineDegrees[pDir];

    // Region of only first spline
    for (int ii = 0; ii < std::min(degP, localSize - degP); ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<pDir, pDir1, pDir2, degP1, degP2>(
            fields, spline, chargeWeight, primitiveDifference, bArray, jArray, ii);
    }

    // Region of no splines (may not exist)
    for (int ii = degP; ii < localSize - degP; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<pDir, pDir1, pDir2, degP1, degP2>(
            fields, spline, chargeWeight, primitiveDifference, bArray, jArray, ii);
    }

    // Region of both splines (may not exist)
    for (int ii = localSize - degP; ii < degP; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<pDir, pDir1, pDir2, degP1, degP2>(
            fields, spline, chargeWeight, primitiveDifference, bArray, jArray, ii);
    }

    // Region of only last spline
    for (int ii = std::max(degP, localSize - degP); ii < localSize; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<pDir, pDir1, pDir2, degP1, degP2>(
            fields, spline, chargeWeight, primitiveDifference, bArray, jArray, ii);
    }
}

//time_integrated_j
template <Direction pDir, typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE void time_integrated_j (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::time_integrated_j()");
    using namespace Impl; // utility functions for this function
    constexpr int pDir1 = (pDir + 1) % 3;
    constexpr int pDir2 = (pDir + 2) % 3;

    constexpr int degP = SplineType::s_splineDegrees[pDir];
    constexpr int degP1 = SplineType::s_splineDegrees[pDir1];
    constexpr int degP2 = SplineType::s_splineDegrees[pDir2];

    // Size of support of difference of primitives
    // check for ghost cells
    int localSize = std::abs(spline.m_firstIndex[pDir] - spline.m_firstIndexOld[pDir]) +
                    spline.s_splineDegrees[pDir];

    // Region of only first spline
    for (int ii = 0; ii < std::min(degP, localSize - degP); ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_loop<pDir, pDir1, pDir2, degP1, degP2>(jArray, spline, chargeWeight,
                                                            primitiveDifference, ii);
    }

    // Region of no splines (may not exist)
    for (int ii = degP; ii < localSize - degP; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_loop<pDir, pDir1, pDir2, degP1, degP2>(jArray, spline, chargeWeight,
                                                            primitiveDifference, ii);
    }

    // Region of both splines (may not exist)
    for (int ii = localSize - degP; ii < degP; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_loop<pDir, pDir1, pDir2, degP1, degP2>(jArray, spline, chargeWeight,
                                                            primitiveDifference, ii);
    }

    // Region of only last spline
    for (int ii = std::max(degP, localSize - degP); ii < localSize; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        accumulate_j_loop<pDir, pDir1, pDir2, degP1, degP2>(jArray, spline, chargeWeight,
                                                            primitiveDifference, ii);
    }
}

#elif AMREX_SPACEDIM == 2
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along one grid
 * direction in 2 spatial dimensions (namely the x- and y-direction)
 *
 * @details
 * given a trajectory \f$ xx_{p,d} \f$ moving along a single direction d (with velocity
 * \f$ vv_{p,d} \f$ parallel to \f$ e_d \f$), this function
 *
 * (1) computes the time integrals
 * \f$ dJ_{i,p,d} = q_p w_p\int_{t_{start}}^{t_{end}} t_{R2_i}(v_{p,d} * S(xx - xx_{p,d}(t))) dt \f$
 *     for every dual face (dof) with index i, where \f$ t_{R2_i} \f$ is corresponding dof
 *     (reduction) operator and adds them to the i-th component of the "current density"
 *
 *     Note that the resulting current density is a time integral over the time step, not a time
 * average
 *
 * (2) computes the time integral of the magnetic field
 *     needed for the time-integrated Lorentz force corresponding to this straight trajectory,
 *     \f$ dF_{p,d} = (q/m)_p \int_{t_{start}}^{t_{end}} vv_{p,d} \times B(xx_{p,d}(t))dt \f$
 *
 *     Specifically, denoting \f$ v_p = vv_{{p,d}_d} \f$ the d-th component of
 *     \f$ vv_{p,d} = v_p * e_d \f$, the function computes
 *     fields[xDir] = \f$ \int_{t_{start}}^{t_{end}} v_p * B_{d-1}(xx_{p,d}(t))dt \f$ =
 *     \f$ \int_{0}^{v_p*dt} B_{d-1}(xx_{p,d}(0)+s*e_d) ds \f$ fields[yDir] =
 *     \f$ \int_{t_{start}}^{t_{end}} v_p * B_{d+1}(xx_{p,d}(t))dt \f$ =
 *     \f$ \int_{0}^{v_p*dt} B_{d+1}(xx_{p,d}(0)+s*e_d) ds \f$
 *
 *     So that after this function call the velocity can be updated over the time step
 *     with
 *
 *     \f$ vv_{p,d} += dF_{p,d} \f$, i.e.
 *
 *     \f$ vv_{{p,d}_{d+1}} -= (q/m)_p \f$ * fields[xDir]
 *
 *     \f$ vv_{{p,d}_{d-1}} += (q/m)_p \f$ * fields[yDir]
 *
 *     (here the Lorentz force has no contribution on the d component of \f$ vv_{p,d}) \f$
 *
 * for details about the algorithm see the documentation
 * https://gempic.pages.mpcdf.de/gempic/latex/particleMesh.html#exact-computation-of-time-integrated-j-x-v-p-x-b-y-x-p-and-v-p-x-b-z-x-p
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param dx Spatial grid size
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <Direction pDir, typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_integrate_b (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::accumulate_j_integrate_b()");
    using namespace Impl; // utility functions for this function
    constexpr int degX = SplineType::s_splineDegrees[xDir];
    constexpr int degY = SplineType::s_splineDegrees[yDir];

    fields = {0., 0.};

    if constexpr (pDir == xDir)
    {
        // Size of support of difference of primitives
        // check for ghost cells
        int localSize = std::abs(spline.m_firstIndex[xDir] - spline.m_firstIndexOld[xDir]) +
                        spline.s_splineDegrees[xDir];

        // Region of only first spline
        for (int ii = 0; ii < std::min(degX, localSize - degX); ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<xDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<xDir, degY>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }

        // Region of no splines (may not exist)
        for (int ii = degX; ii < localSize - degX; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<xDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<xDir, degY>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }

        // Region of both splines (may not exist)
        for (int ii = localSize - degX; ii < degX; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<pDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<xDir, degY>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }

        // Region of only last spline
        for (int ii = std::max(degX, localSize - degX); ii < localSize; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<pDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<xDir, degY>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }
    }

    else if constexpr (pDir == yDir)
    {
        // Size of support of difference of primitives
        // check for ghost cells
        int localSize = std::abs(spline.m_firstIndex[yDir] - spline.m_firstIndexOld[yDir]) +
                        spline.s_splineDegrees[yDir];

        // Region of only first spline
        for (int ii = 0; ii < std::min(degY, localSize - degY); ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<yDir, degX>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }

        // Region of no splines (may not exist)
        for (int ii = degY; ii < localSize - degY; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<yDir, degX>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }

        // Region of both splines (may not exist)
        for (int ii = localSize - degY; ii < degY; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<yDir, degX>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }

        // Region of only last spline
        for (int ii = std::max(degY, localSize - degY); ii < localSize; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            accumulate_j_integrate_b_loop<yDir, degX>(fields, spline, chargeWeight,
                                                      primitiveDifference, bArray, jArray, ii);
        }
    }
}

#elif AMREX_SPACEDIM == 1
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along the grid
 * x-direction
 *
 * @details
 * given a trajectory \f$ xx_{p,d} \f$ moving along a single direction d (with velocity
 * \f$ vv_{p,d} \f$ parallel to \f$ e_d \f$), this function
 *
 * (1) computes the time integrals
 * \f$ dJ_{i,p,d} = q_p w_p\int_{t_{start}}^{t_{end}} t_{R2_i}(v_{p,d} * S(xx - xx_{p,d}(t))) dt \f$
 *     for every dual face (dof) with index i, where \f$ t_{R2_i} \f$ is corresponding dof
 *     (reduction) operator and adds them to the i-th component of the "current density"
 *
 *     Note that the resulting current density is a time integral over the time step, not a time
 * average
 *
 * (2) computes the time integral of the magnetic field
 *     needed for the time-integrated Lorentz force corresponding to this straight trajectory,
 *     \f$ dF_{p,d} = (q/m)_p \int_{t_{start}}^{t_{end}} vv_{p,d} \times B(xx_{p,d}(t))dt \f$
 *
 *     Specifically, denoting \f$ v_p = vv_{{p,d}_d} \f$ the d-th component of
 *     \f$ vv_{p,d} = v_p * e_d \f$, the function computes
 *     fields[xDir] = \f$ \int_{t_{start}}^{t_{end}} v_p * B_{d-1}(xx_{p,d}(t))dt \f$ =
 *     \f$ \int_{0}^{v_p*dt} B_{d-1}(xx_{p,d}(0)+s*e_d) ds \f$
 *
 *     So that after this function call the velocity can be updated over the time step
 *     with
 *
 *     \f$ vv_{p,d} += dF_{p,d} \f$, i.e.
 *
 *     \f$ vv_{{p,d}_{d+1}} -= (q/m)_p \f$ * fields[xDir]
 *
 *     \f$ vv_{{p,d}_{d-1}} += (q/m)_p \f$ * fields[yDir]
 *
 *     (here the Lorentz force has no contribution on the d component of \f$ vv_{p,d}) \f$
 *
 * for details about the algorithm see the documentation
 * https://gempic.pages.mpcdf.de/gempic/latex/particleMesh.html#exact-computation-of-time-integrated-j-x-v-p-x-b-y-x-p-and-v-p-x-b-z-x-p
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 * @tparam vDim Number of components of J
 *
 * @param[out] fields Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param dx Spatial grid size
 * @param bArray Degrees of freedom of B as primal 2-form
 * @param jArray Degrees of freedom of J as dual 2-form
 */

template <Direction pDir, typename SplineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_integrate_b (
    amrex::GpuArray<amrex::Real, 2>& fields,
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::accumulate_j_integrate_b()");
    using namespace Impl; // utility functions for this function

    constexpr int degPx = SplineType::s_splineDegrees[xDir];
    fields = {0., 0.};

    int localSize = std::abs(spline.m_firstIndex[xDir] - spline.m_firstIndexOld[xDir]) +
                    spline.s_splineDegrees[xDir];

    // Region of only first spline
    for (int ii = 0; ii < std::min(degPx, localSize - degPx); ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<xDir>(fields, spline, chargeWeight, primitiveDifference,
                                            bArray, jArray, ii);
    }

    // Region of no splines (may not exist)
    for (int ii = degPx; ii < localSize - degPx; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<xDir>(fields, spline, chargeWeight, primitiveDifference,
                                            bArray, jArray, ii);
    }
    // Region of both splines (may not exist)
    for (int ii = localSize - degPx; ii < degPx; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<xDir>(fields, spline, chargeWeight, primitiveDifference,
                                            bArray, jArray, ii);
    }
    // Region of only last spline
    for (int ii = std::max(degPx, localSize - degPx); ii < localSize; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};

        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        accumulate_j_integrate_b_loop<xDir>(fields, spline, chargeWeight, primitiveDifference,
                                            bArray, jArray, ii);
    }
}
#endif

#if AMREX_SPACEDIM == 3
/**
 * @brief Updates J
 *
 * Description:
 * given a trajectory \f$ xx_{p,d} \f$ moving along a single direction d (with velocity
 * \f$ vv_{p,d} \f$ parallel to \f$ e_d \f$), this function computes the time integrals
 * \f$ dJ_{i,p,d} = q_p w_p\int_{t_{start}}^{t_{end}} t_{R2_i}(v_{p,d} * S(xx - xx_{p,d}(t))) dt \f$
 * and adds them to the i-th component of the "current density" for every dual face (dof) with
 * index i, where \f$ t_{R2_i} \f$ is corresponding dof (reduction) operator
 *
 * Note that the resulting current density is a time integral over the time step, not a time
 * average
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam vDim Number of components of J

 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param dx Spatial grid size
 * @param jArray Degrees of freedom of J as dual 2-form
 */

template <Direction pDir, typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j (
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::accumulate_j()");
    using namespace Impl; // utility functions for this function
    constexpr int pDir1 = (pDir + 1) % 3;
    constexpr int pDir2 = (pDir + 2) % 3;

    constexpr int degP = SplineType::s_splineDegrees[pDir];
    constexpr int degP1 = SplineType::s_splineDegrees[pDir1];
    constexpr int degP2 = SplineType::s_splineDegrees[pDir2];

    // Size of support of difference of primitives
    // check for ghost cells
    int localSize = std::abs(spline.m_firstIndex[pDir] - spline.m_firstIndexOld[pDir]) +
                    spline.s_splineDegrees[pDir];

    // Region of only first spline
    for (int ii = 0; ii < std::min(degP, localSize - degP); ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        for (int j = 0; j <= degP1; ++j)
        {
            for (int k = 0; k <= degP2; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(ii, j, k);

                update_j_component<pDir, pDir1, pDir2>(jArray, spline, cellIndex, j, k,
                                                       chargeWeight, primitiveDifference);
            }
        }
    }

    // Region of no splines (may not exist)
    for (int ii = degP; ii < localSize - degP; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        for (int j = 0; j <= degP1; ++j)
        {
            for (int k = 0; k <= degP2; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(ii, j, k);

                update_j_component<pDir, pDir1, pDir2>(jArray, spline, cellIndex, j, k,
                                                       chargeWeight, primitiveDifference);
            }
        }
    }

    // Region of both splines (may not exist)
    for (int ii = localSize - degP; ii < degP; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        for (int j = 0; j <= degP1; ++j)
        {
            for (int k = 0; k <= degP2; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(ii, j, k);

                update_j_component<pDir, pDir1, pDir2>(jArray, spline, cellIndex, j, k,
                                                       chargeWeight, primitiveDifference);
            }
        }
    }

    // Region of only last spline
    for (int ii = std::max(degP, localSize - degP); ii < localSize; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<pDir, region>(dx, ii);

        for (int j = 0; j <= degP1; ++j)
        {
            for (int k = 0; k <= degP2; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(ii, j, k);

                update_j_component<pDir, pDir1, pDir2>(jArray, spline, cellIndex, j, k,
                                                       chargeWeight, primitiveDifference);
            }
        }
    }
}
#elif AMREX_SPACEDIM == 2

/**
 * @brief Updates J in 2 spatial dimensions (namely the x- and y- direction)
 *
 * Description:
 * given a trajectory \f$ xx_{p,d} \f$ moving along a single direction d (with velocity
 * \f$ vv_{p,d} \f$ parallel to \f$ e_d \f$), this function computes the time integrals
 * \f$ dJ_{i,p,d} = q_p w_p\int_{t_{start}}^{t_{end}} t_{R2_i}(v_{p,d} * S(xx - xx_{p,d}(t))) dt \f$
 * and adds them to the i-th component of the "current density" for every dual face (dof) with
 * index i, where \f$ t_{R2_i} \f$ is corresponding dof (reduction) operator
 *
 * Note that the resulting current density is a time integral over the time step, not a time
 * average
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam vDim Number of components of J

 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param dx Spatial grid size
 * @param jArray Degrees of freedom of J as dual 2-form
 */

template <Direction pDir, typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j (
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::accumulate_j()");
    using namespace Impl; // utility functions for this function
    constexpr int degX = SplineType::s_splineDegrees[xDir];
    constexpr int degY = SplineType::s_splineDegrees[yDir];

    if constexpr (pDir == xDir)
    {
        // Size of support of difference of primitives
        // check for ghost cells
        int localSize = std::abs(spline.m_firstIndex[xDir] - spline.m_firstIndexOld[xDir]) +
                        spline.s_splineDegrees[xDir];

        // Region of only first spline
        for (int ii = 0; ii < std::min(degX, localSize - degX); ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<xDir, region>(dx, ii);

            for (int j = 0; j <= degY; ++j)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, j, 0);

                update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, j, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }

        // Region of no splines (may not exist)
        for (int ii = degX; ii < localSize - degX; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<xDir, region>(dx, ii);

            for (int j = 0; j <= degY; ++j)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, j, 0);

                update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, j, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }

        // Region of both splines (may not exist)
        for (int ii = localSize - degX; ii < degX; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<xDir, region>(dx, ii);

            for (int j = 0; j <= degY; ++j)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, j, 0);

                update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, j, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }

        // Region of only last spline
        for (int ii = std::max(degX, localSize - degX); ii < localSize; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<xDir, region>(dx, ii);

            for (int j = 0; j <= degY; ++j)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, j, 0);

                update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, j, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }
    }
    else if constexpr (pDir == yDir)
    {
        // Size of support of difference of primitives
        // check for ghost cells
        int localSize = std::abs(spline.m_firstIndex[yDir] - spline.m_firstIndexOld[yDir]) +
                        spline.s_splineDegrees[yDir];

        // Region of only first spline
        for (int ii = 0; ii < std::min(degY, localSize - degY); ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            for (int k = 0; k <= degX; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<yDir>(ii, 0, k);

                update_j_component<yDir, xDir, zDir>(jArray, spline, cellIndex, k, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }

        // Region of no splines (may not exist)
        for (int ii = degY; ii < localSize - degY; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            for (int k = 0; k <= degX; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<yDir>(ii, 0, k);

                update_j_component<yDir, xDir, zDir>(jArray, spline, cellIndex, k, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }

        // Region of both splines (may not exist)
        for (int ii = localSize - degY; ii < degY; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            for (int k = 0; k <= degX; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<yDir>(ii, 0, k);

                update_j_component<yDir, xDir, zDir>(jArray, spline, cellIndex, k, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }

        // Region of only last spline
        for (int ii = std::max(degY, localSize - degY); ii < localSize; ++ii)
        {
            constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
            amrex::Real primitiveDifference =
                spline.template compute_primitive_difference<yDir, region>(dx, ii);

            for (int k = 0; k <= degX; ++k)
            {
                amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<yDir>(ii, 0, k);

                update_j_component<yDir, xDir, zDir>(jArray, spline, cellIndex, k, 0, chargeWeight,
                                                     primitiveDifference);
            }
        }
    }
}

#elif AMREX_SPACEDIM == 1
/**
 * @brief Updates J along the grid x-direction
 *
 * Description:
 * given a trajectory \f$ xx_{p,d} \f$ moving along a single direction d (with velocity
 * \f$ vv_{p,d} \f$ parallel to \f$ e_d \f$), this function computes the time integrals
 * \f$ dJ_{i,p,d} = q_p w_p\int_{t_{start}}^{t_{end}} t_{R2_i}(v_{p,d} * S(xx - xx_{p,d}(t))) dt \f$
 * and adds them to the i-th component of the "current density" for every dual face (dof) with
 * index i, where \f$ t_{R2_i} \f$ is corresponding dof (reduction) operator
 *
 * Note that the resulting current density is a time integral over the time step, not a time
 * average
 *
 * @tparam pDir Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam vDim Number of components of J

 * @param spline Object containing spline values and the particle trajectory
 * @param chargeWeight Particle weight * charge
 * @param dx Spatial grid size
 * @param jArray Degrees of freedom of J as dual 2-form
 */

template <Direction pDir, typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j (
    SplineType& spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::accumulate_j()");
    using namespace Impl; // utility functions for this function

    constexpr int degPx = SplineType::s_splineDegrees[xDir];

    int localSize = std::abs(spline.m_firstIndex[xDir] - spline.m_firstIndexOld[xDir]) +
                    spline.s_splineDegrees[xDir];

    // Region of only first spline
    for (int ii = 0; ii < std::min(degPx, localSize - degPx); ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::firstSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, 0, 0);

        update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                             primitiveDifference);
    }

    // Region of no splines (may not exist)
    for (int ii = degPx; ii < localSize - degPx; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::noSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, 0, 0);

        update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                             primitiveDifference);
    }

    // Region of both splines (may not exist)
    for (int ii = localSize - degPx; ii < degPx; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::bothSplines};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, 0, 0);

        update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                             primitiveDifference);
    }

    // Region of only last spline
    for (int ii = std::max(degPx, localSize - degPx); ii < localSize; ++ii)
    {
        constexpr PrimitiveDifferenceRegion region{PrimitiveDifferenceRegion::lastSpline};
        amrex::Real primitiveDifference =
            spline.template compute_primitive_difference<xDir, region>(dx, ii);

        amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<xDir>(ii, 0, 0);

        update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                             primitiveDifference);
    }
}
#endif

#if AMREX_SPACEDIM == 1
/**
 * @brief Updates J in one spatial dimension
 *
 * For AMREX_SPACEDIM == 1 there is no primitive difference available for the y-step.
 * Therefore, v_{p,x} and v_{p,z} are updated using an explicit Euler step.
 *
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam vDim Number of components of J
 *
 * @param spline Object containing spline values and the particle trajectory
 * @param dtv timestep * v_y
 * @param chargeWeight Particle weight * charge
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_euler_y (
    SplineType& spline,
    amrex::Real const& dtv,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    constexpr int degX = SplineType::s_splineDegrees[xDir];

    for (int ii = 0; ii <= degX; ++ii)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][ii];
        amrex::Dim3 cellIndex = {spline.m_firstIndex[xDir] + ii};

        amrex::HostDevice::Atomic::Add(jArray[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       chargeWeight * dtv * xNodeVal);
    }
}
#endif

#if AMREX_SPACEDIM == 1 || AMREX_SPACEDIM == 2
/**
 * @brief Updates J in one and two spatial dimension
 *
 * For AMREX_SPACEDIM == 1 and AMREX_SPACEDIM == 2 there is no primitive difference available
 * for the z-step. Therefore, v_{p,x} and v_{p,y} are updated using an explicit Euler step.

 *
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam vDim Number of components of J
 *
 * @param spline Object containing spline values and the particle trajectory
 * @param dtv timestep * v_z
 * @param chargeWeight Particle weight * charge
 * @param jArray Degrees of freedom of J as dual 2-form
 */
template <typename SplineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_euler_z (
    SplineType& spline,
    amrex::Real const& dtv,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& jArray)
{
    constexpr int degX = SplineType::s_splineDegrees[xDir];

#if AMREX_SPACEDIM == 1
    for (int ii = 0; ii <= degX; ++ii)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][ii];
        amrex::Dim3 cellIndex = {spline.m_firstIndex[xDir] + ii};

        double sCoeff = chargeWeight * dtv * xNodeVal;
        amrex::HostDevice::Atomic::Add(jArray[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       sCoeff);
    }

#elif AMREX_SPACEDIM == 2
    constexpr int degY = SplineType::s_splineDegrees[yDir];
    for (int ii = 0; ii <= degX; ++ii)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][ii];

        for (int jj = 0; jj <= degY; ++jj)
        {
            amrex::Real yNodeVal = spline.m_nodeSplineVals[yDir][jj];
            amrex::Dim3 cellIndex = {
                GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii, spline.m_firstIndex[yDir] + jj, 0)};

            double sCoeff = chargeWeight * dtv * xNodeVal * yNodeVal;
            amrex::HostDevice::Atomic::Add(
                jArray[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0), sCoeff);
        }
    }
#endif
}
#endif

/**
 * @brief Evaluates the B-field for the rotational matrix
 *
 * @tparam SplineType Necessary for mocking so the function can take mock spline types
 * @tparam bDim Number of components of B
 *
 * @param[out] bValues resulting values for the rotational matrix
 * @param[out] normB resulting norm
 * @param bArray array of B field coefficients (primal face dofs)
 * @param spline object containing the values of non-zero grid splines at the particle
 * position
 */
template <typename SplineType, unsigned int bDim>
AMREX_GPU_HOST_DEVICE inline void compute_b (
    amrex::GpuArray<amrex::Real, 3>& bValues,
    amrex::Real& normB,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const& bArray,
    SplineType& spline)
{
    amrex::GpuArray<amrex::Real, bDim> bfield =
        spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bArray);

    normB = std::max(sqrt(bfield[0] * bfield[0] + bfield[1] * bfield[1] + bfield[2] * bfield[2]),
                     1e-15);

    bValues[0] = bfield[0] / normB;
    bValues[1] = bfield[1] / normB;
    bValues[2] = bfield[2] / normB;
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
 * @param[out] rho charge density to be computed
 * @param spline Splines evaluated at given particle position
 * @param chargeWeight Particle weight * charge
 */
template <int degX, int degY, int degZ>
AMREX_GPU_HOST_DEVICE inline void deposit_rho (amrex::Array4<amrex::Real> const& rho,
                                               SplineBase<degX, degY, degZ> const& spline,
                                               amrex::Real const chargeWeight)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::deposit_rho()");
    GEMPIC_D_EXCL(int jj{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};
        double sCoeff = spline.template eval_b_spline<Forms::Field::DualThreeForm>(ii, jj, kk);

        amrex::HostDevice::Atomic::Add(rho.ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       sCoeff * chargeWeight);
    GEMPIC_D_LOOP_END
}

/**
 * @brief
 *
 * @tparam degX Spline degree in x
 * @tparam degY Spline degree in y
 * @tparam degZ Spline degree in z
 *
 * @param[out] twoForm generic twoForm field to be computed
 * @param spline Splines evaluated at given particle position
 * @param dep_vector The deposit vector
 * @param coeff
 */
template <int degX, int degY, int degZ, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void deposit_twoform (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& twoForm,
    SplineBase<degX, degY, degZ> const& spline,
    amrex::GpuArray<amrex::Real, vDim> const& depVector,
    amrex::Real chargeWeight)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::deposit_twoform()")
    AMREX_D_PICK(int jj{0}; int kk{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};

        // Ensure that we only access existing elements of cellSplineVals
        if (ii < degX)
        {
            double sCoeff =
                spline.template eval_b_spline<Forms::Field::DualTwoForm, xDir>(ii, jj, kk);
            amrex::HostDevice::Atomic::Add(
                twoForm[xDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                sCoeff * chargeWeight * depVector[xDir]);
        }

        if constexpr (vDim > 1)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (jj < degY)
            {
                double sCoeff =
                    spline.template eval_b_spline<Forms::Field::DualTwoForm, yDir>(ii, jj, kk);
                amrex::HostDevice::Atomic::Add(
                    twoForm[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * chargeWeight * depVector[yDir]);
            }
        }
        if constexpr (vDim == 3)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (kk < degZ)
            {
                double sCoeff =
                    spline.template eval_b_spline<Forms::Field::DualTwoForm, zDir>(ii, jj, kk);
                amrex::HostDevice::Atomic::Add(
                    twoForm[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * chargeWeight * depVector[zDir]);
            }
        }
    GEMPIC_D_LOOP_END
}

/**
 * @brief
 *
 * @tparam degX Spline degree in x
 * @tparam degY Spline degree in y
 * @tparam degZ Spline degree in z
 *
 * @param[out] currentDensity current density to be computed
 * @param spline Splines evaluated at given particle position
 * @param vel The velocity vector
 * @param coeff
 */

template <Direction dir, int degX, int degY, int degZ, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline amrex::Real deposit_deldot_s_coefficient (
    SplineWithFirstDerivative<degX, degY, degZ> const& spline,
    amrex::GpuArray<amrex::Real, vDim> const& vel,
    int ii,
    int jj,
    int kk)
{
    static_assert(vDim <= 3 && vDim >= 1, "unknown vDim");
    if constexpr (vDim < dir)
    {
        return 0;
    }
    constexpr auto form{Forms::Field::DualTwoForm};
    if constexpr (vDim == 1)
    {
        return vel[xDir] * spline.template eval_b_spline_derivative<form, dir, xDir>(ii, jj, kk);
    }
    else if constexpr (vDim == 2)
    {
        return GEMPIC_D_ADD(
            (vel[xDir] * spline.template eval_b_spline_derivative<form, dir, xDir>(ii, jj, kk)),
            (vel[yDir] * spline.template eval_b_spline_derivative<form, dir, yDir>(ii, jj, kk)), 0);
    }
    else
    {
        return GEMPIC_D_ADD(
            (vel[xDir] * spline.template eval_b_spline_derivative<form, dir, xDir>(ii, jj, kk)),
            (vel[yDir] * spline.template eval_b_spline_derivative<form, dir, yDir>(ii, jj, kk)),
            (vel[zDir] * spline.template eval_b_spline_derivative<form, dir, zDir>(ii, jj, kk)));
    }
}

template <int degX, int degY, int degZ, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void deposit_deldot_s (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& delDotS,
    SplineWithFirstDerivative<degX, degY, degZ> const& spline,
    amrex::GpuArray<amrex::Real, vDim> const& vel,
    amrex::Real massWeight)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::deposit_deldot_s()")
    AMREX_D_PICK(int jj{0}; int kk{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};

        // Ensure that we only access existing elements of cellSplineVals
        if (ii < degX)
        {
            amrex::Real sCoeff{deposit_deldot_s_coefficient<xDir>(spline, vel, ii, jj, kk)};
            amrex::HostDevice::Atomic::Add(
                delDotS[xDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                sCoeff * massWeight * vel[xDir]);
        }
        if constexpr (vDim > 1)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (jj < degY)
            {
                amrex::Real sCoeff{deposit_deldot_s_coefficient<yDir>(spline, vel, ii, jj, kk)};
                amrex::HostDevice::Atomic::Add(
                    delDotS[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * massWeight * vel[yDir]);
            }
        }
        if constexpr (vDim == 3)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (kk < degZ)
            {
                amrex::Real sCoeff{deposit_deldot_s_coefficient<zDir>(spline, vel, ii, jj, kk)};
                amrex::HostDevice::Atomic::Add(
                    delDotS[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * massWeight * vel[zDir]);
            }
        }
    GEMPIC_D_LOOP_END
}

template <Forms::Field form, Direction dir, int degX, int degY, int degZ, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline amrex::Real deposit_curlj_coefficient (
    SplineWithFirstDerivative<degX, degY, degZ> const& spline,
    amrex::GpuArray<amrex::Real, vDim> const& vel,
    int ii,
    int jj,
    int kk)
{
    static_assert(vDim <= 3 && vDim >= 1, "unknown vDim");
    if constexpr (vDim < dir)
    {
        return 0;
    }

    if constexpr (vDim == 1)
    {
        if (dir == xDir)
        {
            return 0.0;
        }
        else if (dir == yDir)
        {
            return GEMPIC_D_ADD(
                0.0, 0.0,
                (vel[xDir] *
                 spline.template eval_b_spline_derivative<form, dir, zDir>(ii, jj, kk)));
        }
        else if (dir == zDir)
        {
            return GEMPIC_D_ADD(
                0.0,
                (-vel[xDir] *
                 spline.template eval_b_spline_derivative<form, dir, yDir>(ii, jj, kk)),
                0.0);
        }
    }
    else if constexpr (vDim == 2)
    {
        if (dir == xDir)
        {
            return GEMPIC_D_ADD(
                0.0, 0.0,
                (-vel[yDir] *
                 spline.template eval_b_spline_derivative<form, dir, zDir>(ii, jj, kk)));
        }
        else if (dir == yDir)
        {
            return GEMPIC_D_ADD(
                0.0, 0.0,
                (vel[xDir] *
                 spline.template eval_b_spline_derivative<form, dir, zDir>(ii, jj, kk)));
        }
        else if (dir == zDir)
        {
            return GEMPIC_D_ADD(
                (vel[yDir] * spline.template eval_b_spline_derivative<form, dir, xDir>(ii, jj, kk)),
                (-vel[xDir] *
                 spline.template eval_b_spline_derivative<form, dir, yDir>(ii, jj, kk)),
                0.0);
        }
    }
    else
    {
        if (dir == xDir)
        {
            return GEMPIC_D_ADD(
                0.0,
                (vel[zDir] * spline.template eval_b_spline_derivative<form, dir, yDir>(ii, jj, kk)),
                (-vel[yDir] *
                 spline.template eval_b_spline_derivative<form, dir, zDir>(ii, jj, kk)));
        }
        else if (dir == yDir)
        {
            return GEMPIC_D_ADD(
                (-vel[zDir] *
                 spline.template eval_b_spline_derivative<form, dir, xDir>(ii, jj, kk)),
                0.0,
                (vel[xDir] *
                 spline.template eval_b_spline_derivative<form, dir, zDir>(ii, jj, kk)));
        }
        else if (dir == zDir)
        {
            return GEMPIC_D_ADD(
                (vel[yDir] * spline.template eval_b_spline_derivative<form, dir, xDir>(ii, jj, kk)),
                (-vel[xDir] *
                 spline.template eval_b_spline_derivative<form, dir, yDir>(ii, jj, kk)),
                0.0);
        }
    }
}

template <Forms::Field form, int degX, int degY, int degZ, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void deposit_curlj (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const& curlJ,
    SplineWithFirstDerivative<degX, degY, degZ> const& spline,
    amrex::GpuArray<amrex::Real, vDim> const& vel,
    amrex::Real chargeWeight)
{
    BL_PROFILE_HOST_DEVICE("Gempic::ParticleMeshCoupling::deposit_curlj()")

    bool xCond{false};
    bool yCond{false};
    bool zCond{false};

    AMREX_D_PICK(int jj{0}; int kk{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};

        if constexpr (form == Forms::Field::DualTwoForm)
        {
            xCond = (ii < degX);
        }
        else if constexpr (form == Forms::Field::DualOneForm)
        {
            xCond = AMREX_D_TERM(true, &&(jj < degY), &&(kk < degZ));
        }

        if (xCond)
        {
            amrex::Real sCoeff{deposit_curlj_coefficient<form, xDir>(spline, vel, ii, jj, kk)};
            amrex::HostDevice::Atomic::Add(
                curlJ[xDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0), sCoeff * chargeWeight);
        }
        if constexpr (vDim > 1)
        {
            if constexpr (form == Forms::Field::DualTwoForm)
            {
                yCond = AMREX_D_TERM(true, &&(jj < degY), );
            }
            else if constexpr (form == Forms::Field::DualOneForm)
            {
                yCond = AMREX_D_TERM((ii < degX), , &&(kk < degZ));
            }

            if (yCond)
            {
                amrex::Real sCoeff{deposit_curlj_coefficient<form, yDir>(spline, vel, ii, jj, kk)};
                amrex::HostDevice::Atomic::Add(
                    curlJ[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * chargeWeight);
            }
        }
        if constexpr (vDim == 3)
        {
            if constexpr (form == Forms::Field::DualTwoForm)
            {
                zCond = AMREX_D_TERM(true, , &&(kk < degZ));
            }
            else if constexpr (form == Forms::Field::DualOneForm)
            {
                zCond = AMREX_D_TERM((ii < degX), &&(jj < degY), );
            }

            if (zCond)
            {
                amrex::Real sCoeff{deposit_curlj_coefficient<form, zDir>(spline, vel, ii, jj, kk)};
                amrex::HostDevice::Atomic::Add(
                    curlJ[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * chargeWeight);
            }
        }
    GEMPIC_D_LOOP_END
}

template <int degX, int degY, int degZ, unsigned int vDim, unsigned int nData>
void deposit_particle_density (
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& particleDensity,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const& particles,
    ComputationalDomain const& compDom,
    bool doSync = true)
{
    for (auto const& particleSpecies : particles)
    {
        amrex::Real charge = particleSpecies->get_charge();

        for (auto& particleGrid : *particleSpecies)
        {
            long const np = particleGrid.numParticles();
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::Array4<amrex::Real> const& pdArr = particleDensity.m_data[particleGrid].array();
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lowerCorner{
                compDom.geometry().ProbLoArray()};

            amrex::ParallelFor(np,
                               [=] AMREX_GPU_DEVICE(long pp)
                               {
                                   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                                   AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                                positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                                positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);
                                   SplineBase<degX, degY, degZ> spline(
                                       positionParticle, lowerCorner,
                                       compDom.inv_cell_size_array());
                                   deposit_rho(pdArr, spline, charge * ptd.rdata(ii.m_iweight)[pp]);
                               });
        }
    }

    if (doSync)
    {
        particleDensity.post_particle_loop_sync();
    }
}

template <int degX, int degY, int degZ, unsigned int vDim, unsigned int nData>
void deposit_individual_particle_density (
    std::vector<Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec,
    std::vector<Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoMassVec,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const& particles,
    ComputationalDomain const& compDom,
    bool doSync = true)
{
    int index = 0;
    for (auto& particleSpecies : particles)
    {
        rhoVec[index].m_data.setVal(0.0);
        rhoMassVec[index].m_data.setVal(0.0);

        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::Array4<amrex::Real> rhoA;
            amrex::Array4<amrex::Real> rhoMassA;

            rhoA = rhoVec[index].m_data[pti].array();
            rhoMassA = rhoMassVec[index].m_data[pti].array();

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{compDom.geometry().ProbLoArray()};

            amrex::ParallelFor(np,
                               [=] AMREX_GPU_DEVICE(long pp)
                               {
                                   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                                   AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                                positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                                positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                                   SplineBase<degX, degY, degZ> spline(
                                       positionParticle, plo, compDom.inv_cell_size_array());

                                   deposit_rho(rhoA, spline, charge * ptd.rdata(ii.m_iweight)[pp]);
                                   deposit_rho(rhoMassA, spline,
                                               mass * ptd.rdata(ii.m_iweight)[pp]);
                               });
        }
        if (doSync)
        {
            rhoVec[index].post_particle_loop_sync();
            rhoMassVec[index].post_particle_loop_sync();
        }

        index++;
    }
}

template <int degX, int degY, int degZ, unsigned int vDim, unsigned int nData>
void deposit_current_density (
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const& particles,
    ComputationalDomain const& compDom,
    bool doSync = true)
{
    for (int comp = 0; comp < vDim; ++comp)
    {
        J.m_data[comp].setVal(0.0);
    }
    for (auto& particleSpecies : particles)
    {
        amrex::Real charge = particleSpecies->get_charge();

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;

            for (int cc = 0; cc < vDim; cc++)
            {
                jA[cc] = (J.m_data[cc])[pti].array();
            }
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{compDom.geometry().ProbLoArray()};

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                    AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                 positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                 positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                    SplineBase<degX, degY, degZ> spline(positionParticle, plo,
                                                        compDom.inv_cell_size_array());

                    amrex::GpuArray<amrex::Real, vDim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                           ptd.rdata(ii.m_ively)[pp],
                                                           ptd.rdata(ii.m_ivelz)[pp]};
                    deposit_twoform(jA, spline, vel, charge * ptd.rdata(ii.m_iweight)[pp]);
                });
        }
    }

    if (doSync)
    {
        J.post_particle_loop_sync();
    }
}

template <int degX, int degY, int degZ, unsigned int vDim, unsigned int nData>
void deposit_rho_weighted_electric_field (
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhoE,
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const& particles,
    ComputationalDomain const& compDom,
    bool doSync = true)
{
    for (int comp = 0; comp < vDim; ++comp)
    {
        rhoE.m_data[comp].setVal(0.0);
    }

    // Fill up E_coeffs
    for (auto& particleSpecies : particles)
    {
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeSqrByMass = charge * charge / mass;

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> rhoeA;

            for (int cc = 0; cc < vDim; cc++)
            {
                eA[cc] = (E.m_data[cc])[pti].array();
                rhoeA[cc] = (rhoE.m_data[cc])[pti].array();
            }
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{compDom.geometry().ProbLoArray()};

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                    AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                 positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                 positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                    SplineBase<degX, degY, degZ> spline(positionParticle, plo,
                                                        compDom.inv_cell_size_array());

                    amrex::GpuArray<amrex::Real, vDim> efield =
                        spline.template eval_spline_field<Forms::Field::PrimalOneForm>(eA);
                    deposit_twoform(rhoeA, spline, efield,
                                    chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
                });
        }
    }

    if (doSync)
    {
        rhoE.post_particle_loop_sync();
    }
}

template <int degX, int degY, int degZ, unsigned int vDim, unsigned int nData>
void deposit_rho_j_jcrossb_deldots (
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rho,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
    Gempic::Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& jcrossB,
    Gempic::Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& delDotS,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const& particles,
    ComputationalDomain const& compDom,
    bool doSync = true)
{
    BL_PROFILE("deposit_rho_j_jcrossb_deldots()");
    rho.m_data.setVal(0.0);
    for (int comp = 0; comp < vDim; ++comp)
    {
        J.m_data[comp].setVal(0.0);
        jcrossB.m_data[comp].setVal(0.0);
        delDotS.m_data[comp].setVal(0.0);
    }
    for (auto& particleSpecies : particles)
    {
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeByMass = charge / mass;

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::Array4<amrex::Real> rhoA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> bA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jcrossbA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> deldotsA;

            rhoA = rho.m_data[pti].array();
            for (int cc = 0; cc < vDim; cc++)
            {
                jA[cc] = (J.m_data[cc])[pti].array();
                bA[cc] = (B.m_data[cc])[pti].array();
                jcrossbA[cc] = (jcrossB.m_data[cc])[pti].array();
                deldotsA[cc] = (delDotS.m_data[cc])[pti].array();
            }
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{compDom.geometry().ProbLoArray()};

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                    AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                 positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                 positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                    SplineBase<degX, degY, degZ> spline(positionParticle, plo,
                                                        compDom.inv_cell_size_array());
                    SplineWithFirstDerivative<degX, degY, degZ> splineWithDeriv(
                        positionParticle, plo, compDom.inv_cell_size_array(),
                        compDom.cell_size_array());

                    amrex::GpuArray<amrex::Real, vDim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                           ptd.rdata(ii.m_ively)[pp],
                                                           ptd.rdata(ii.m_ivelz)[pp]};

                    deposit_rho(rhoA, splineWithDeriv, charge * ptd.rdata(ii.m_iweight)[pp]);
                    deposit_twoform(jA, splineWithDeriv, vel, charge * ptd.rdata(ii.m_iweight)[pp]);

                    amrex::GpuArray<amrex::Real, vDim> bfield =
                        spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);
                    amrex::GpuArray<amrex::Real, vDim> vcrossb{
                        (vel[yDir] * bfield[zDir] - vel[zDir] * bfield[yDir]),
                        (vel[zDir] * bfield[xDir] - vel[xDir] * bfield[zDir]),
                        (vel[xDir] * bfield[yDir] - vel[yDir] * bfield[xDir])};
                    deposit_twoform(jcrossbA, splineWithDeriv, vcrossb,
                                    chargeByMass * charge * ptd.rdata(ii.m_iweight)[pp]);

                    deposit_deldot_s(deldotsA, splineWithDeriv, vel,
                                     charge * ptd.rdata(ii.m_iweight)[pp]);
                });
        }
    }

    if (doSync)
    {
        rho.post_particle_loop_sync();
        J.post_particle_loop_sync();
        jcrossB.post_particle_loop_sync();
        delDotS.post_particle_loop_sync();
    }
}

template <unsigned int vDim>
inline void push_x_using_v (amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
                            amrex::GpuArray<amrex::Real, vDim> const& vel,
                            amrex::Real dt)
{
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        pos[i] += vel[i] * dt;
    }
}

template <unsigned int vDim>
inline void push_v_using_e (amrex::GpuArray<amrex::Real, vDim>& vel,
                            amrex::GpuArray<amrex::Real, vDim> const& efield,
                            amrex::Real dt,
                            amrex::Real chargeByMass)
{
    for (int i = 0; i < vDim; i++)
    {
        vel[i] += chargeByMass * efield[i] * dt;
    }
}

template <unsigned int vDim>
inline void rotate_v_using_b (amrex::GpuArray<amrex::Real, vDim>& vel,
                              amrex::GpuArray<amrex::Real, vDim> const& bfield,
                              amrex::Real dt,
                              amrex::Real chargeByMass)
{
    amrex::Real bmag = std::sqrt(bfield[xDir] * bfield[xDir] + bfield[yDir] * bfield[yDir] +
                                 bfield[zDir] * bfield[zDir]);
    if (bmag != 0.0)
    {
        amrex::GpuArray<amrex::Real, vDim> b{bfield[xDir] / bmag, bfield[yDir] / bmag,
                                             bfield[zDir] / bmag};
        amrex::Real a = chargeByMass * bmag * dt;
        amrex::Real ca = cos(a);
        amrex::Real sa = sin(a);
        amrex::GpuArray<amrex::Real, vDim> vtemp{vel[xDir], vel[yDir], vel[zDir]};

        vel[xDir] =
            (b[xDir] * b[xDir] + (b[yDir] * b[yDir] + b[zDir] * b[zDir]) * ca) * vtemp[xDir] +
            (b[zDir] * sa + b[yDir] * b[xDir] * (1.0 - ca)) * vtemp[yDir] +
            (-b[yDir] * sa + b[zDir] * b[xDir] * (1.0 - ca)) * vtemp[zDir];

        vel[yDir] =
            (-b[zDir] * sa + b[yDir] * b[xDir] * (1.0 - ca)) * vtemp[xDir] +
            (b[yDir] * b[yDir] + (b[xDir] * b[xDir] + b[zDir] * b[zDir]) * ca) * vtemp[yDir] +
            (b[xDir] * sa + b[zDir] * b[yDir] * (1.0 - ca)) * vtemp[zDir];

        vel[zDir] =
            (b[yDir] * sa + b[zDir] * b[xDir] * (1.0 - ca)) * vtemp[xDir] +
            (-b[xDir] * sa + b[zDir] * b[yDir] * (1.0 - ca)) * vtemp[yDir] +
            (b[zDir] * b[zDir] + (b[xDir] * b[xDir] + b[yDir] * b[yDir]) * ca) * vtemp[zDir];
    }
}

} // namespace Gempic::ParticleMeshCoupling

#endif

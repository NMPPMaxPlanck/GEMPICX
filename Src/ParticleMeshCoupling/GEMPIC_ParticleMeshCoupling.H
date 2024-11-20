#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Math.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"

namespace Gempic::ParticleMeshCoupling
{
namespace Impl
{
/**
 * @brief Computes the integral of the D update

 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam pDir1 : First direction not along the trajectory
 * @tparam pDir2 : Second direction not along the trajectory
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam vDim : Number of components of J
 *
 * @param[out] jArray : Degrees of freedom of J as dual 2-form
 * @param spline : Object containing spline values and the particle trajectory
 * @param cellIndex : Primitive cell index from spline
 * @param j : sum index for direction pDir1
 * @param k : sum index for direction pDir2
 * @param chargeWeight : Particle weight * charge
 * @param primitiveDifference: The difference of the primitves at the current and previous time step
 */
template <Direction pDir, int pDir1, int pDir2, typename splineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void update_j_component (
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
    splineType const &spline,
    amrex::Dim3 const &cellIndex,
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

#if GEMPIC_SPACEDIM == 3
/**
 * @brief Computes the product for the given integration index of the integral of B
 *
 * @tparam pDir1 : First direction not along the trajectory
 * @tparam pDir2 : Second direction not along the trajectory
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 *
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param spline : Object containing spline values and the particle trajectory
 * @param cellIndex : Primitive cell index from spline
 * @param j : sum index for direction pDir1
 * @param k : sum index for direction pDir2
 * @param primitiveDifference: The difference of the primitves at the current and previous time step
 */
template <int pDir1, int pDir2, typename splineType, unsigned int bDim>
AMREX_GPU_HOST_DEVICE inline amrex::Real update_integrated_b_component (
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    splineType const &spline,
    amrex::Dim3 const &cellIndex,
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
 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam pDir1 : First direction not along the trajectory
 * @tparam pDir2 : Second direction not along the trajectory
 * @tparam degP1 : Spline degree of direction pDir1
 * @tparam degp2 : Spline degree of direction pDir2
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param primitiveDifference: The difference of the primitves at the current and previous time step
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this
 * should be fixed
 */
template <Direction pDir,
          int pDir1,
          int pDir2,
          int degP1,
          int degP2,
          typename splineType,
          unsigned int bDim,
          unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_loop (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
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
#elif GEMPIC_SPACEDIM == 2

/**
 * @brief computes the integral of B components along the trajectory
 *
 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1)
 * @tparam degP : Spline degree of direction (for xDir: degY, for yDir: degX)
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param primitiveDifference: The difference of the primitves at the current and previous time step
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this
 * should be fixed
 */
template <Direction pDir, int degP, typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_loop (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
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

#elif GEMPIC_SPACEDIM == 1

/**
 * @brief computes the integral of B components along the trajectory
 *
 * @tparam pDir : Direction of integration (trajectory): x=0
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param primitiveDifference: The difference of the primitves at the current and previous time step
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this
 * should be fixed
 */
template <Direction pDir, typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_loop (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const chargeWeight,
    amrex::Real const primitiveDifference,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
    int const i)
{
    amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<pDir>(i, 0, 0);

    update_j_component<xDir, yDir, zDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                         primitiveDifference);

    fields[0] += primitiveDifference * (bArray[zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
    fields[1] += primitiveDifference * (bArray[yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
}

#endif
}  //namespace Impl

#if GEMPIC_SPACEDIM == 1
/**
 * @brief Updates J and computes an integrated B on a given particle for the H_p_y update
 * in one spatial dimension
 *
 * Since for GEMPIC_SPACEDIM == 1, there is not primitive Difference available for the y-step.
 * Therefore, v_{p,x} and v_{p,z} are updated using an explicit Euler step.
 *
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param dtv : timestep * v_y
 * @param chargeWeight : Particle weight * charge
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 */
template <typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_euler_y (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const &dtv,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray)
{
    using namespace Impl;  // utility functions for this function
    fields = {0., 0.};
    constexpr int degX = splineType::s_splineDegrees[xDir];

    for (int k = 0; k <= degX; ++k)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][k];
        amrex::Real xCellVal = spline.m_cellSplineVals[xDir][k];
        amrex::Real primitiveDifferenceEuler = dtv * xNodeVal;

        amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<yDir>(0, 0, k);
        update_j_component<yDir, zDir, xDir>(jArray, spline, cellIndex, 0, 0, chargeWeight,
                                             primitiveDifferenceEuler);

        fields[0] += xNodeVal * (bArray[xDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);

        fields[1] += xCellVal * (bArray[zDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
    }
}
#endif

#if GEMPIC_SPACEDIM == 1 || GEMPIC_SPACEDIM == 2
/**
 * @brief Updates J and computes an integrated B on a given particle for the H_p_z update
 * in one and two spatial dimension
 *
 * Since for GEMPIC_SPACEDIM == 1 and GEMPIC_SPACEDIM == 2, there is not primitive Difference
 * available for the z-step.Therefore, v_{p,x} and v_{p,y} are updated using an explicit Euler step.
 *
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param dtv : timestep * v_z
 * @param chargeWeight : Particle weight * charge
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 */
template <typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b_euler_z (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const &dtv,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray)
{
    using namespace Impl;  // utility functions for this function
    fields = {0., 0.};

    constexpr int degX = splineType::s_splineDegrees[xDir];
#if GEMPIC_SPACEDIM == 2
    constexpr int degY = splineType::s_splineDegrees[yDir];
#elif GEMPIC_SPACEDIM == 1
    constexpr int k = 0;
    constexpr int yNodeVal = 1;
    constexpr int yCellVal = 1;
#endif

    for (int j = 0; j <= degX; ++j)
    {
        amrex::Real xNodeVal = spline.m_nodeSplineVals[xDir][j];
        amrex::Real xCellVal = spline.m_cellSplineVals[xDir][j];

#if GEMPIC_SPACEDIM == 2
        for (int k = 0; k <= degY; ++k)
        {
            amrex::Real yNodeVal = spline.m_nodeSplineVals[yDir][k];
            amrex::Real yCellVal = spline.m_cellSplineVals[yDir][k];
#endif
            amrex::Real primitiveDifferenceEuler = dtv * xNodeVal;

            amrex::Dim3 cellIndex = spline.template select_primitive_cellindex<zDir>(0, j, k);
            update_j_component<zDir, yDir, xDir>(jArray, spline, cellIndex, k, 0, chargeWeight,
                                                 primitiveDifferenceEuler);
#if GEMPIC_SPACEDIM == 2
            if (j < degX)
            {
#endif
                fields[0] +=
                    xCellVal * yNodeVal * (bArray[yDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
#if GEMPIC_SPACEDIM == 2
            }
            if (k < degY)
            {
#endif
                fields[1] +=
                    xNodeVal * yCellVal * (bArray[xDir])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
#if GEMPIC_SPACEDIM == 2
            }
        }
#endif
    }
}
#endif

#if GEMPIC_SPACEDIM == 3
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along one grid
 * direction in 3 spatial dimensions
 *
 * Description:
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
 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J

 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param dx : Spatial grid size
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this
 * should be fixed
 * @todo Rewrite to avoid excessive branching
 */

template <Direction pDir, typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void accumulate_j_integrate_b (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray)
{
    BL_PROFILE("Gempic::ParticleMeshCoupling::accumulate_j_integrate_b()");
    using namespace Impl;  // utility functions for this function
    constexpr int pDir1 = (pDir + 1) % 3;
    constexpr int pDir2 = (pDir + 2) % 3;

    constexpr int degP = splineType::s_splineDegrees[pDir];
    constexpr int degP1 = splineType::s_splineDegrees[pDir1];
    constexpr int degP2 = splineType::s_splineDegrees[pDir2];

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

#elif GEMPIC_SPACEDIM == 2
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along one grid
 * direction in 2 spatial dimensions (namely the x- and y-direction)
 *
 * Description:
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
 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param dx : Spatial grid size
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 */
template <Direction pDir, typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_integrate_b (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray)
{
    BL_PROFILE("Gempic::ParticleMeshCoupling::accumulate_j_integrate_b()");
    using namespace Impl;  // utility functions for this function
    constexpr int degX = splineType::s_splineDegrees[xDir];
    constexpr int degY = splineType::s_splineDegrees[yDir];

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

#elif GEMPIC_SPACEDIM == 1
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along the grid
 * x-direction
 *
 * Description:
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
 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1, z=2)
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam bDim : Number of components of B
 * @tparam vDim : Number of components of J
 *
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param dx : Spatial grid size
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 */

template <Direction pDir, typename splineType, unsigned int bDim, unsigned int vDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_integrate_b (
    amrex::GpuArray<amrex::Real, 2> &fields,
    splineType &spline,
    amrex::Real const chargeWeight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, bDim> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray)
{
    BL_PROFILE("Gempic::ParticleMeshCoupling::accumulate_j_integrate_b()");
    using namespace Impl;  // utility functions for this function

    constexpr int degPx = splineType::s_splineDegrees[xDir];
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

/**
 * @brief Deposits charge density on cells of dual grid
 *
 * Here nodal splines seem to be used for deposition where integral of cell splines are expected
 *
 * @tparam degX Spline degree in x
 * @tparam degY Spline degree in y
 * @tparam degZ Spline degree in z
 *
 * @param rho charge density to be computed
 * @param spline Splines evaluated at given particle position
 * @param chargeWeight Particle weight*charge
 */
template <int degX, int degY, int degZ>
AMREX_GPU_HOST_DEVICE inline void gempic_deposit_rho (amrex::Array4<amrex::Real> const &rho,
                                                      SplineBase<degX, degY, degZ> &spline,
                                                      amrex::Real chargeWeight)
{
    BL_PROFILE("Gempic::ParticleMeshCoupling::gempic_deposit_rho()");
    GEMPIC_D_EXCL(int jj{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};
        double sCoeff = spline.template eval_b_spline<Field::DualThreeForm>(ii, jj, kk);

        amrex::HostDevice::Atomic::Add(rho.ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       sCoeff * chargeWeight);
    GEMPIC_D_LOOP_END
}

template <int degX, int degY, int degZ, unsigned int vDim>
AMREX_GPU_HOST_DEVICE inline void gempic_deposit_j (
    SplineBase<degX, degY, degZ> const &spline,
    amrex::GpuArray<amrex::Real, vDim> const &vel,
    amrex::Real coeff,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &currentDensity)
{
    BL_PROFILE("Gempic::ParticleMeshCoupling::gempic_deposit_j()")
    AMREX_D_PICK(int jj{0}; int kk{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};

        // Ensure that we only access existing elements of cellSplineVals
        if (ii < degX)
        {
            double sCoeff = spline.template eval_b_spline<Field::DualTwoForm, xDir>(ii, jj, kk);
            amrex::HostDevice::Atomic::Add(
                currentDensity[xDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                sCoeff * coeff * vel[xDir]);
        }

        if constexpr (vDim > 1)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (jj < degY)
            {
                double sCoeff = spline.template eval_b_spline<Field::DualTwoForm, yDir>(ii, jj, kk);
                amrex::HostDevice::Atomic::Add(
                    currentDensity[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * coeff * vel[yDir]);
            }
        }
        if constexpr (vDim == 3)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (kk < degZ)
            {
                double sCoeff = spline.template eval_b_spline<Field::DualTwoForm, zDir>(ii, jj, kk);
                amrex::HostDevice::Atomic::Add(
                    currentDensity[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * coeff * vel[zDir]);
            }
        }
    GEMPIC_D_LOOP_END
}

}  // namespace Gempic::ParticleMeshCoupling

#endif
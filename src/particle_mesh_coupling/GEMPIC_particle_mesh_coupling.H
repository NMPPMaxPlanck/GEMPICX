#ifndef PARTICLE_MESH_COUPLING_H_
#define PARTICLE_MESH_COUPLING_H_

#include <cmath>

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Math.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Spline_Class.H"
#include "GEMPIC_particle_mesh_coupling.H"

namespace Gempic
{
namespace Particles
{
/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along one grid
 * direction
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
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam vDim : Number of components of B and J
 * @tparam pDir : Direction of integration (trajectory) (x=0, y=1, z=2)
 *
 * @param spline : Object containing spline values and the particle trajectory
 * @param chargeWeight : Particle weight * charge
 * @param dx : Spatial grid size
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration
 * direction)
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this
 * should be fixed
 * @todo Rewrite to avoid excessive branching
 */

template <Direction pDir, typename splineType, unsigned int vDim>
AMREX_GPU_HOST_DEVICE void accumulate_j_integrate_b (
    splineType &spline,
    amrex::Real chargeWeight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
    amrex::GpuArray<amrex::Real, 2> &fields)
{
    constexpr int pDir1 = (pDir + 1) % 3;
    constexpr int pDir2 = (pDir + 2) % 3;

    const int degP1 = spline.s_splineDegrees[pDir1];
    const int degP2 = spline.s_splineDegrees[pDir2];

    fields = {0., 0.};
    // Size of support of difference of primitives
    // check for ghost cells
    int localSize = std::abs(spline.m_firstIndex[pDir] - spline.m_firstIndexOld[pDir]) +
                    spline.s_splineDegrees[pDir];
    int firstIndexStart = spline.m_firstIndexOld[pDir];
    if (spline.m_firstIndex[pDir] < spline.m_firstIndexOld[pDir])
    {
        firstIndexStart = spline.m_firstIndex[pDir];
    }

    // Add particle contribution to J
    for (int jj = 0; jj <= degP1; ++jj)
    {
        for (int kk = 0; kk <= degP2; ++kk)
        {
            for (int ii = 0; ii <= localSize; ++ii)
            {
                amrex::Real primitiveDifference =
                    spline.template compute_primitive_difference<pDir>(dx, ii);

                amrex::Dim3 cellIndex;
                if (pDir == xDir)
                {
                    cellIndex = {AMREX_D_DECL(firstIndexStart + ii, spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};
                }
                else if (pDir == yDir)
                {
                    cellIndex = {AMREX_D_DECL(spline.m_firstIndex[xDir] + kk, firstIndexStart + ii,
                                              spline.m_firstIndex[zDir] + jj)};
                }
                else if (pDir == zDir)
                {
                    cellIndex = {AMREX_D_DECL(spline.m_firstIndex[xDir] + jj,
                                              spline.m_firstIndex[yDir] + kk,
                                              firstIndexStart + ii)};
                }
                double sCoeff =
                    GEMPIC_D_MULT(primitiveDifference, spline.m_nodeSplineVals[pDir1][jj],
                                  spline.m_nodeSplineVals[pDir2][kk]);
                amrex::HostDevice::Atomic::Add(
                    jArray[pDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * chargeWeight);
                // Update two components of B
                if (jj < degP1)
                {
                    double sCoeff =
                        GEMPIC_D_MULT(primitiveDifference, spline.m_cellSplineVals[pDir1][jj],
                                      spline.m_nodeSplineVals[pDir2][kk]);

                    fields[xDir] +=
                        sCoeff * (bArray[pDir2])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
                }
                if (kk < degP2)
                {
                    double sCoeff =
                        GEMPIC_D_MULT(primitiveDifference, spline.m_nodeSplineVals[pDir1][jj],
                                      spline.m_cellSplineVals[pDir2][kk]);
                    fields[yDir] +=
                        sCoeff * (bArray[pDir1])(cellIndex.x, cellIndex.y, cellIndex.z, 0);
                }
            }
        }
    }
}

/*!
 * @brief Push the velocity with a given electric field
 *
 * @tparam vDim Number of Components of B and J
 *
 * @param vel Velocity component
 * @param dt Temporal grid size
 * @param chargemass : \f$ (q/m)_p \f$ ratio
 * @param E Electric field
 */
template <unsigned int vDim>
AMREX_GPU_HOST_DEVICE void push_v_efield (amrex::GpuArray<amrex::Real, vDim> &vel,
                                          amrex::Real dt,
                                          amrex::Real chargemass,
                                          amrex::GpuArray<amrex::Real, vDim> const &E)
{
    AMREX_D_TERM((vel[xDir] += dt * chargemass * (E[xDir]));
                 , (vel[yDir] += dt * chargemass * (E[yDir]));
                 , (vel[zDir] += dt * chargemass * (E[zDir])););
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
 * @param chargeWeight Particle weight*charge
 * @param rho charge density to be computed
 */
template <int degX, int degY, int degZ>
AMREX_GPU_HOST_DEVICE void gempic_deposit_rho (Spline::SplineBase<degX, degY, degZ> &spline,
                                               amrex::Real chargeWeight,
                                               amrex::Array4<amrex::Real> const &rho)
{
    GEMPIC_D_EXCL(int jj{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {AMREX_D_DECL(spline.m_firstIndex[xDir] + ii,
                                              spline.m_firstIndex[yDir] + jj,
                                              spline.m_firstIndex[zDir] + kk)};
        double sCoeff = spline.template eval_b_spline<Field::DualThreeForm>(ii, jj, kk);

        amrex::HostDevice::Atomic::Add(rho.ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                                       sCoeff * chargeWeight);
    GEMPIC_D_LOOP_END
}

template <int degX, int degY, int degZ, int vDim>
AMREX_GPU_HOST_DEVICE void gempic_deposit_j (
    Spline::SplineBase<degX, degY, degZ> const &spline,
    amrex::GpuArray<amrex::Real, vDim> const &velChargeWeight,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &currentDensity)
{
    AMREX_D_PICK(int jj{0}; int kk{0};, int kk{0};, )
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii), for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cellIndex = {AMREX_D_DECL(spline.firstIndex[xDir] + ii,
                                              spline.firstIndex[yDir] + jj,
                                              spline.firstIndex[zDir] + kk)};

        // Ensure that we only access existing elements of cellSplineVals
        if (ii < degX)
        {
            double sCoeff = spline.template evalBSpline<Field::DualTwoForm, xDir>(ii, jj, kk);
            amrex::HostDevice::Atomic::Add(
                currentDensity[xDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                sCoeff * velChargeWeight[xDir]);
        }

        if constexpr (vDim > 1)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (jj < degY)
            {
                double sCoeff = spline.template evalBSpline<Field::DualTwoForm, yDir>(ii, jj, kk);
                amrex::HostDevice::Atomic::Add(
                    currentDensity[yDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * velChargeWeight[yDir]);
            }
        }
        if constexpr (vDim == 3)
        {
            // Ensure that we only access existing elements of cellSplineVals
            if (kk < degZ)
            {
                double sCoeff = spline.template evalBSpline<Field::DualTwoForm, zDir>(ii, jj, kk);
                amrex::HostDevice::Atomic::Add(
                    currentDensity[zDir].ptr(cellIndex.x, cellIndex.y, cellIndex.z, 0),
                    sCoeff * velChargeWeight[zDir]);
            }
        }
    GEMPIC_D_LOOP_END
}

}  // namespace Particles
}  // namespace Gempic

#endif
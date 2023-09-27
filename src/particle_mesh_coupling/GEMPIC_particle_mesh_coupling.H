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
     Computes for each component i, J_i from \f$ \int_t\rho \f$ and \f$ \int_t v_i \times B \f$ by integrating over the B-splines
     */

/**
 * @brief Updates J and computes an integrated B on a given particle trajectory along one grid direction
 *  
 * Description:
 * given a trajectory xx_{p,d} moving along a single dimension d (with velocity vv_{p,d} parallel to e_d), this function
 * 
 * (1) computes the time integrals 
 *     dJ_{i,p,d} = \f$ q_p w_p \int_{t_start}^{t_end} t_R2_i( v_{p,d} * S(xx - xx_{p,d}(t)) ) dt \f$
 *     for every dual face (dof) with index i, 
 *     where t_R2_i is corresponding dof (reduction) operator
 *     and adds them to the i-th component of the "current density"
 *     
 *     Note that the resulting current density is a time integral over the time step, not a time average
 *  
 * (2) computes the time integral of the magnetic field      
 *     needed for the time-integrated Lorentz force corresponding to this straight trajectory,
 *     \f$ dF_{p,d} = (q/m)_p \int_{t_{start}}^{t_{end}} vv_{p,d} \times B(xx_{p,d}(t))dt \f$
 *      
 *     Specifically, denoting v_p = vv_{p,d}_d the d-th component of vv_{p,d} = v_p * e_d, the function computes
 *     
 *     fields[0] = \f$ \int_{t_start}^{t_end} v_p * B_{d-1}(xx_{p,d}(t))dt = \int_{0}^{v_p*dt} B_{d-1}(xx_{p,d}(0)+s*e_d) ds \f$
 *     fields[1] = \f$ \int _{t_start}^{t_end} v_p * B_{d+1}(xx_{p,d}(t))dt = \int_{0}^{v_p*dt} B_{d+1}(xx_{p,d}(0)+s*e_d) ds \f$
 *     
 *     So that after this function call the velocity can be updated over the time step 
 *     with 
 *     vv_{p,d} += dF_{p,d}
 *     ie
 *     vv_{p,d}_{d+1} -= (q/m)_p * fields[0]
 *     vv_{p,d}_{d-1} += (q/m)_p * fields[1]
 *     (here the Lorentz force has no contribution on the d component of vv_{p,d})
 *     
 * for details about the algorithm see the documentation
 * https://gempic.pages.mpcdf.de/gempic/latex/particleMesh.html#exact-computation-of-time-integrated-j-x-v-p-x-b-y-x-p-and-v-p-x-b-z-x-p
 * 
 * @tparam SplineType : Necessary for mocking so the function can take mock spline types
 * @tparam vDim : Number of components of B and J
 * @tparam pDim : Direction of integration (trajectory) (x=0, y=1, z=2)
 * 
 * @param spline : Object containing spline values and the particle trajectory
 * @param weight : Particle weight
 * @param dx : Spatial grid size
 * @param bArray : Degrees of freedom of B as primal 2-form
 * @param jArray : Degrees of freedom of J as dual 2-form
 * @param[out] fields : Integral of B components along trajectory (the 2 that are not in integration direction)
 *
 * @todo vDim is not always the nb of components of B: (if vDim == 2, then B has 1 component), this should be fixed
 * @todo Rewrite to avoid excessive branching
 */

template <typename SplineType, int vDim, int pDim>
AMREX_GPU_HOST_DEVICE void accumulate_J_integrate_B (
    SplineType &spline,
    amrex::Real weight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> const &jArray,
    amrex::GpuArray<amrex::Real, 2> &fields)
{
    constexpr int pDim1 = (pDim + 1) % 3;
    constexpr int pDim2 = (pDim + 2) % 3;

    const int degP1 = spline.splineDegrees[pDim1];
    const int degP2 = spline.splineDegrees[pDim2];

    fields = {0., 0.};
    // Size of support of difference of primitives
    // check for ghost cells
    int local_size = std::abs(spline.firstIndex[pDim] - spline.firstIndexOld[pDim]) + spline.splineDegrees[pDim];
    int firstIndex_start = spline.firstIndexOld[pDim];
    if (spline.firstIndex[pDim] < spline.firstIndexOld[pDim]) firstIndex_start = spline.firstIndex[pDim];

// Add particle contribution to J
    for (int jj = 0; jj <= degP1; ++jj) {
        for (int kk = 0; kk <= degP2; ++kk) {
            for (int ii = 0; ii <= local_size; ++ii) {

                amrex::Real primitiveDifference = spline.template computePrimitiveDifference<pDim>(dx, ii);

                amrex::Dim3 cell_index;
                if (pDim == 0)
                    cell_index = {
                        AMREX_D_DECL(firstIndex_start + ii, spline.firstIndex[1] + jj, spline.firstIndex[2] + kk)};
                else if (pDim == 1)
                    cell_index = {
                        AMREX_D_DECL(spline.firstIndex[0] + kk, firstIndex_start + ii, spline.firstIndex[2] + jj)};
                else if (pDim == 2)
                    cell_index = {
                        AMREX_D_DECL(spline.firstIndex[0] + jj, spline.firstIndex[1] + kk, firstIndex_start + ii)};
                double s_coeff = GEMPIC_D_MULT(
                    primitiveDifference,
                    spline.nodeSplineVals[pDim1][jj],
                    spline.nodeSplineVals[pDim2][kk]);
                amrex::HostDevice::Atomic::Add(
                    jArray[pDim].ptr(cell_index.x, cell_index.y, cell_index.z, 0),
                    s_coeff * weight);
                // Update two components of B
                if (jj < degP1)
                {
                    double s_coeff = GEMPIC_D_MULT(
                        primitiveDifference,
                        spline.cellSplineVals[pDim1][jj],
                        spline.nodeSplineVals[pDim2][kk]);

                    fields[0] +=
                        s_coeff * (bArray[pDim2])(cell_index.x, cell_index.y, cell_index.z, 0);
                }
                if (kk < degP2)
                {
                    double s_coeff = GEMPIC_D_MULT(
                        primitiveDifference,
                        spline.nodeSplineVals[pDim1][jj],
                        spline.cellSplineVals[pDim2][kk]);
                    fields[1] +=
                        s_coeff * (bArray[pDim1])(cell_index.x, cell_index.y, cell_index.z, 0);
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
 * @param chargemass : (q/m)_p ratio
 * @param Ep Electric field
 */
template <int vDim>
AMREX_GPU_HOST_DEVICE void push_v_efield (
    amrex::GpuArray<amrex::Real, vDim> &vel,
    amrex::Real dt,
    amrex::Real chargemass,
    amrex::GpuArray<amrex::Real, vDim> &Ep)
{
    amrex::GpuArray<amrex::Real, vDim> newPos;

    AMREX_D_TERM((vel[0] += dt * chargemass * (Ep[0]));,
                 (vel[1] += dt * chargemass * (Ep[1]));,
                 (vel[2] += dt * chargemass * (Ep[2])););
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
AMREX_GPU_HOST_DEVICE void gempic_deposit_rho (
    Spline::SplineBase<degX, degY, degZ> &spline,
    amrex::Real weight,
    amrex::Array4<amrex::Real> const &rho)
{
    AMREX_D_PICK(
    int jj{0}; int kk{0};,
    int kk{0};,)
    GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degX; ++ii),
                        for (int jj = 0; jj <= degY; ++jj),
                        for (int kk = 0; kk <= degZ; ++kk))
        amrex::Dim3 cell_index = {AMREX_D_DECL(
            spline.firstIndex[0] + ii, spline.firstIndex[1] + jj, spline.firstIndex[2] + kk)};
        double s_coeff = spline.template evalBSpline<0>(ii, jj, kk);

        amrex::HostDevice::Atomic::Add(
            rho.ptr(cell_index.x, cell_index.y, cell_index.z, 0),
            s_coeff * weight);
    GEMPIC_D_LOOP_END
}

}  // namespace Particles
}  // namespace Gempic

#endif
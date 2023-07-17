#ifndef PARTICLE_MESH_COUPLING_C2_H_
#define PARTICLE_MESH_COUPLING_C2_H_

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_Spline_Class.H>

#include <cmath>

namespace Gempic
{
namespace Particles
{
/**
 * @brief Computes values of splines at particles in 3D
 *
 * In Finite Difference GEMPIC formalism components of forms in the de Rham diagram are products of 
 * node centered splines of the form \f$S^d(x_i-x_p)\f$ of degree d and cell centered splines
 * of the form \f$S^{d-1}(x_{i+\frac{1}{2}}-x_p)\f$ of degree d-1
 * @tparam degx Spline degree in x
 * @tparam degy Spline degree in y
 * @tparam degz Spline degree in z
 */
template <int degx, int degy, int degz>
class splines_at_particles
{
public:
/// Cell centered spline = \f$S^{d-1}(x-x_{i+\frac 12})\f$
    amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degx, degy), degz)>,
                    GEMPIC_SPACEDIM>
        spline_cell;  
/// node centered spline = \f$S^d(x-x_{i})\f$
    amrex::GpuArray<amrex::GpuArray<amrex::Real, std::max(std::max(degx, degy), degz) + 1>,
                    GEMPIC_SPACEDIM>
        spline_node;

    amrex::GpuArray<int, GEMPIC_SPACEDIM> span;

    // Default constructor
    // splines_at_particles();

    /**
     * @brief Compute splines \f$S_n\f$ and \f$S_c\f$ at particle position in each direction
     *
     * @param position position of particle
     * @param plo left extremity of 1D grid
     * @param dxi inverse cell size
     */
    template <int dim, int deg>
    AMREX_GPU_HOST_DEVICE void init_position(amrex::Real position, amrex::Real plo, amrex::Real dxi)
    {
        // Compute normalized position xint and span
        amrex::Real xint = (position - plo) * dxi;
        span[dim] = std::floor(xint);
        xint = xint - span[dim];

        if (deg % 2 == 0)
        {
            if (xint < 0.5)
            {
                span[dim] -= deg / 2;
                xint = 0.5 - xint;
            }
            else
            {
                span[dim] += -deg / 2 + 1;
                xint = 1.5 - xint;
            }
        }
        else
        {
            span[dim] -= (deg - 1) / 2;
            xint = 1. - xint;
        }

        spline_cell[dim] = spline_eval<deg - 1, std::max(std::max(degx, degy), degz)>(xint);
        spline_node[dim] = spline_eval<deg, std::max(std::max(degx, degy), degz) + 1>(xint);
    }

    AMREX_GPU_HOST_DEVICE void init_particles(
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &position,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxi)
    {
        init_position<0, degx>(position[0], plo[0], dxi[0]);
#if (GEMPIC_SPACEDIM > 1)
        init_position<1, degy>(position[1], plo[1], dxi[1]);
#endif
#if (GEMPIC_SPACEDIM > 2)
        init_position<2, degz>(position[2], plo[2], dxi[2]);
#endif
    }

    template <int degree, int length>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, length> spline_eval(amrex::Real x)
    {
        amrex::GpuArray<amrex::Real, length> s;
        amrex::Real factor;
        switch (degree)
        {
            case 0:
                s[0] = 1.;
                break;
            case 1:
                s[0] = x;
                s[1] = 1. - x;
                break;
            case 2:
                s[0] = pow(x, 2) * 0.5;
                s[1] = (1. - x) * x + 0.5;
                s[2] = pow(x - 1, 2) * 0.5;
                break;
            case 3:
                factor = 1./6.;
                s[0] = pow(x, 3) * factor;
                s[1] = (((-x + 1) * x + 1.) * 3. * x + 1.) * factor;
                s[2] = ((3. * x - 6.) * x * x + 4.) * factor;
                s[3] = pow((1. - x), 3) * factor;
                break;
            case 4:
                factor = 1./24.;
                s[0] = pow(x, 4) * factor;
                s[1] = ((((-x + 1.) * 4. * x + 6.) * x + 4.) * x + 1.) * factor;
                s[2] = ((((6. * x - 12.) * x - 6.) * x + 12.) * x + 11.) * factor;
                s[3] = ((((-4. * x + 12.) * x - 6.) * x - 12.) * x + 11.) * factor;
                s[4] = pow(1. - x, 4) * factor;
                break;
            case 5:
                factor = 1./120.;
                s[0] = pow(x, 5) * factor;
                s[1] = (((((-x + 1.) * x + 2.) * x + 2.) * x + 1.) * 5. * x + 1.) * factor;
                s[2] = (((((x - 2.) * x - 2.) * x + 2.) * x + 5.) * 10. * x + 26.) * factor;
                s[3] = (((-x + 3.) * x * x - 6.) * x * 10. * x + 66.) * factor;
                s[4] = (((((x - 4.) * x + 4.) * x + 4.) * x - 10.) * 5. * x + 26.) * factor;
                s[5] = (((((-x + 5.) * x - 10.) * x + 10.) * x - 5.) * x + 1.) * factor;
                break;
            case 6:
            // TODO: rewrite with Horner's rule
                s[0] = pow(x, 6) / 720.;
                s[1] = -pow(x, 6) / 120. + pow(x, 5) / 120. + pow(x, 4) / 48. + pow(x, 3) / 36. +
                       pow(x, 2) / 48. + x / 120. + 1. / 720.;
                s[2] = pow(x, 6) / 48. - pow(x, 5) / 24. - pow(x, 4) / 16. + pow(x, 3) / 36. +
                       3. * pow(x, 2) / 16. + 5. * x / 24. + 19. / 240.;
                s[3] = -pow(x, 6) / 36. + pow(x, 5) / 12. + pow(x, 4) / 24. - 2. * pow(x, 3) / 9. -
                       5. * pow(x, 2) / 24. + 1. * x / 3. + 151. / 360.;
                s[4] = pow(x, 6) / 48. - pow(x, 5) / 12. + pow(x, 4) / 24. + 2. * pow(x, 3) / 9. -
                       5. * pow(x, 2) / 24. - x / 3. + 151. / 360.;
                s[5] = -pow(x, 6) / 120. + pow(x, 5) / 24. - pow(x, 4) / 16. - pow(x, 3) / 36. +
                       3. * pow(x, 2) / 16. - 5. * x / 24. + 19. / 240.;
                s[6] = pow(x, 6) / 720. - pow(x, 5) / 120. + pow(x, 4) / 48. - pow(x, 3) / 36. +
                       pow(x, 2) / 48. - x / 120. + 1. / 720.;
                break;
        }
        return (s);
    }
};

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
AMREX_GPU_HOST_DEVICE void gempic_deposit_rho_C3(splines_at_particles<degx, degy, degz> &spline,
                                                 amrex::Real weight,
                                                 amrex::Array4<amrex::Real> const &rho)
{
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk)
#endif
    {
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj)
#endif
        {
            for (int ii = 0; ii <= degx; ++ii)
            {
                amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index = {
                    AMREX_D_DECL(spline.span[0] + ii, spline.span[1] + jj, spline.span[2] + kk)};
                double s_coeff = spline.spline_node[0][ii]
#if (GEMPIC_SPACEDIM > 1)
                                 * spline.spline_node[1][jj]
#endif
#if (GEMPIC_SPACEDIM > 2)
                                 * spline.spline_node[2][kk]
#endif
                    ;
#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(rho.ptr(cell_index[0], 0, 0, 0), s_coeff * weight);
#else
                amrex::HostDevice::Atomic::Add(
                    rho.ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0),
                    s_coeff * weight);
#endif
            }
        }
    }
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
AMREX_GPU_HOST_DEVICE void gempic_deposit_rho_C3_new_splines(Spline::SplineBase<degx, degy, degz> &spline,
                                                 amrex::Real weight,
                                                 amrex::Array4<amrex::Real> const &rho)
{
GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii <= degx; ++ii),
                    for (int jj = 0; jj <= degy; ++jj),
                    for (int kk = 0; kk <= degz; ++kk))

    amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index = {
        AMREX_D_DECL(spline.span[0] + ii, spline.span[1] + jj, spline.span[2] + kk)};
    double s_coeff = GEMPIC_D_MULT(spline.splineNode[0][ii],
                                   spline.splineNode[1][jj],
                                   spline.splineNode[2][kk]);

#if (GEMPIC_SPACEDIM == 1)
                amrex::HostDevice::Atomic::Add(rho.ptr(cell_index[0], 0, 0, 0), s_coeff * weight);
#else
                amrex::HostDevice::Atomic::Add(
                    rho.ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0),
                    s_coeff * weight);
#endif

GEMPIC_D_LOOP_END;
}

/**
     Computes for each component i, J_i from \int_t Rho and \int_t v_i x B by integrating over the
   B-splines
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
AMREX_GPU_HOST_DEVICE void accumulate_j_update_v_C2(
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

template <int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE void gempic_accumulate_j(
    amrex::Real charge, amrex::GpuArray<amrex::Real, vdim> &vel,
    splines_at_particles<degx, degy, degz> const &spline,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jArray, amrex::Real dxi)

{
#if (GEMPIC_SPACEDIM > 2)
    for (int kk = 0; kk <= degz; ++kk)
    {
#endif
#if (GEMPIC_SPACEDIM > 1)
        for (int jj = 0; jj <= degy; ++jj)
        {
#endif
            for (int ii = 0; ii <= degx; ++ii)
            {
                // deposit in J
                for (int cc = 0; cc < vdim; cc++)
                {
                    if (!(
#if (GEMPIC_SPACEDIM > 2)
                            ((kk == degz) & (cc % 3 == 2)) |
#endif
#if (GEMPIC_SPACEDIM > 1)
                            ((jj == degy) & (cc % 3 == 1)) |
#endif
                            ((ii == degx) & (cc % 3 == 0))))
                    {
                        double s_coeff = 0;
                        if (cc % 3 == 0)
                        {
                            s_coeff = spline.spline_cell[0][ii];
                        }
                        else
                        {
                            s_coeff = spline.spline_node[0][ii];
                        }
#if (GEMPIC_SPACEDIM > 1)
                        if (cc % 3 == 1)
                        {
                            s_coeff *= spline.spline_cell[1][jj];
                        }
                        else
                        {
                            s_coeff *= spline.spline_node[1][jj];
                        }
#endif
#if (GEMPIC_SPACEDIM > 2)
                        if (cc % 3 == 2)
                        {
                            s_coeff *= spline.spline_cell[2][kk];
                        }
                        else
                        {
                            s_coeff *= spline.spline_node[2][kk];
                        }
#endif
                        amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index = {AMREX_D_DECL(
                            spline.span[0] + ii, spline.span[1] + jj, spline.span[2] + kk)};

#if (GEMPIC_SPACEDIM == 1)
                        amrex::HostDevice::Atomic::Add(jArray[cc].ptr(cell_index[0], 0, 0, 0),
                                                       s_coeff * charge * vel[cc]);
#else
                amrex::HostDevice::Atomic::Add(
                    jArray[cc].ptr(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0),
                    s_coeff * dxi * charge * vel[cc]);
#endif
                    }
                }
#if (GEMPIC_SPACEDIM > 1)
            }
#endif
#if (GEMPIC_SPACEDIM > 2)
        }
#endif
    }
}

}  // namespace Particles
}  // namespace Gempic

#endif

#ifndef PARTICLE_MESH_COUPLING_C2_H_
#define PARTICLE_MESH_COUPLING_C2_H_

#include <AMReX_Array.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>

#include <cmath>

namespace Gempic
{
namespace Particles
{
/**
 * @brief Evaluate all 1D splines and spline primitives at a given particle position
 *
 * @tparam deg spline degree
 */
template <int deg>
class spline1d_at_particles
{
public:
    amrex::GpuArray<amrex::Real, deg + 1> spline; // deg+1 non zero contributions
    amrex::GpuArray<amrex::Real, deg> primitive; // non vanishing primitives for splines of deg-1
    int box;
    /**
     * @brief Find the 1D cell in which the particle is
     *
     * @param position position of particle
     * @param plo left extremity of 1D grid
     * @param dxi cell size
     */
    AMREX_GPU_HOST_DEVICE void init_position(amrex::Real position, amrex::Real plo, amrex::Real dxi)
    {
        // Compute normalized position xint and box
        amrex::Real xint = (position - plo) * dxi;
        box = std::floor(xint);
        xint = xint - box;

        if (deg % 2 == 0)
        {
            if (xint < 0.5)
            {
                box -= deg / 2;
                xint = 0.5 - xint;
            }
            else
            {
                box += -deg / 2 + 1;
                xint = 1.5 - xint;
            }
        }
        else
        {
            box -= (deg - 1) / 2;
            xint = 1. - xint;
        }
        primitive = spline_eval_primitive<deg - 1>(xint);
        spline = spline_eval<deg>(xint);
    }

private:
    /**
     * @brief Computes the values of the (degree + 1) splines that are not zero at any given point x
     * in [0,1]
     *
     * @tparam degree Spline degree
     * @param x Position at which splines are evaluated
     * @return s Array of spline values at x
     */
    template <int degree>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, degree + 1> spline_eval(amrex::Real x)
    {
        amrex::GpuArray<amrex::Real, degree + 1> s;  // there are degree + 1 splines which are
        switch (degree)
        {
            case 0:
                s[0] = 1;
                break;
            case 1:
                s[0] = x;
                s[1] = 1 - x;
                break;
            case 2:
                s[0] = pow(x, 2) / 2.;
                s[1] = 1. / 2. + x - pow(x, 2);
                s[2] = 1. / 2. - x + pow(x, 2) / 2.;
                break;
            case 3:
                s[0] = pow(x, 3) / 6.;
                s[1] = 1. / 6. * (1. + 3. * x + 3. * pow(x, 2) - 3. * pow(x, 3));
                s[2] = 1. / 6. * (4. - 6. * pow(x, 2) + 3. * pow(x, 3));
                s[3] = 1. / 6. * pow((1. - x), 3);
                break;
            case 4:
                s[0] = pow(x, 4) / 24.;
                s[1] = ((((-x + 1.) * 4. * x + 6.) * x + 4.) * x + 1.) / 24.;
                s[2] = ((((6. * x - 12.) * x - 6.) * x + 12.) * x + 11.) / 24.;
                s[3] = ((((-4. * x + 12.) * x - 6.) * x - 12.) * x + 11.) / 24.;
                s[4] = pow(1. - x, 4) / 24.;
                break;
            case 5:
                s[0] = pow(x, 5) / 120.;
                s[1] = (((((-x + 1.) * x + 2.) * x + 2.) * x + 1.) * 5. * x + 1.) / 120.;
                s[2] = (((((x - 2.) * x - 2.) * x + 2.) * x + 5.) * 10. * x + 26.) / 120.;
                s[3] = (((-x + 3.) * x * x - 6.) * x * 10. * x + 66.) / 120.;
                s[4] = (((((x - 4.) * x + 4.) * x + 4.) * x - 10.) * 5. * x + 26.) / 120.;
                s[5] = (((((-x + 5.) * x - 10.) * x + 10.) * x - 5.) * x + 1.) / 120.;
                break;
            case 6:
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
    /**
     * @brief Evaluate primitives of splines of a given degree at given point x
     *
     * @tparam degree
     * @param x
     */
    template <int degree>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, degree + 1> spline_eval_primitive(
        amrex::Real x)
    {
        amrex::GpuArray<amrex::Real, degree + 1> s;
        switch (degree)
        {
            case -1:
                s[0] = 1;
                break;
            case 0:
                s[0] = x;
                break;
            case 1:
                s[0] = pow(x, 2) / 2.;
                s[1] = 1. - pow(x - 1., 2) / 2.;
                break;
            case 2:
                s[0] = pow(x, 3) / 6.;
                s[1] = (((-2. * x + 3) * x + 3.) * x + 1.) / 6.;
                s[2] = pow(x - 1., 3) / 6. + 1.;
                break;
            case 3:
                s[0] = pow(x, 4) / 24.;
                s[1] = -pow(x, 4) / 8. + pow(x, 3) / 6. + pow(x, 2) / 4. + x / 6. + 1. / 24.;
                s[2] = pow(x, 4) / 8. - pow(x, 3) / 3. + x * 2. / 3. + 0.5;
                s[3] = -pow((1. - x), 4) / 24. + 1.;
                break;
            case 4:
                s[0] = pow(x, 5) / 120.;
                s[1] = (((((-4. * x + 5.) * x + 10.) * x + 10.) * x + 5.) * x + 1.) / 120.;
                s[2] = (((((6. * x - 15.) * x - 10.) * x + 30.) * x + 55.) * x + 27.) / 120.;
                s[3] = (((((-4. * x + 15.) * x - 10.) * x - 30.) * x + 55.) * x + 93.) / 120.;
                s[4] = -pow(1. - x, 5) / 120. + 1.;
                break;
            case 5:
                s[0] = pow(x, 6) / 720.;
                s[1] =
                    ((((((-5. * x + 6.) * x + 15.) * x + 20.) * x + 15.) * x + 6.) * x + 1.) / 720.;
                s[2] =
                    ((((((10. * x - 24.) * x - 30.) * x + 40.) * x + 150.) * x + 156.) * x + 58.) /
                    720.;
                s[3] = ((((-10. * x + 36.) * x * x - 120.) * x * x + 396.) * x + 360.) / 720.;
                s[4] =
                    ((((((5. * x - 24.) * x + 30.) * x + 40.) * x - 150.) * x + 156.) * x + 662.) /
                    720.;
                s[5] = ((((((-x + 6.) * x - 15.) * x + 20.) * x - 15.) * x + 6.) * x + 719.) / 720.;
                break;
            case 6:
                s[0] = pow(x, 7) / 5040.;
                s[1] = -pow(x, 7) / 840. + pow(x, 6) / 720. + pow(x, 5) / 240. + pow(x, 4) / 144. +
                       pow(x, 3) / 144. + pow(x, 2) / 240. + x / 720. + 1. / 5040.;
                s[2] = pow(x, 7) / 336. - pow(x, 6) / 144. - pow(x, 5) / 80. + pow(x, 4) / 144. +
                       pow(x, 3) / 16. + 5. * pow(x, 2) / 48. + 19. * x / 240. + 121. / 5040.;
                s[3] = -pow(x, 7) / 252. + pow(x, 6) / 72. + pow(x, 5) / 120. - pow(x, 4) / 18. -
                       5. * pow(x, 3) / 72. + pow(x, 2) / 6. + 151. * x / 360. + 82. / 315.;
                s[4] = pow(x, 7) / 336. - pow(x, 6) / 72. + pow(x, 5) / 120. + pow(x, 4) / 18. -
                       5. * pow(x, 3) / 72. - pow(x, 2) / 6. + 151. * x / 360. + 233. / 315.;
                s[5] = -pow(x, 7) / 840. + pow(x, 6) / 144. - pow(x, 5) / 80. - pow(x, 4) / 144. +
                       pow(x, 3) / 16. - 5. * pow(x, 2) / 48. + 19. * x / 240. + 691. / 5040.;
                s[6] = pow(x, 7) / 5040. - pow(x, 6) / 720. + pow(x, 5) / 240. - pow(x, 4) / 144. +
                       pow(x, 3) / 144. - pow(x, 2) / 240. + x / 720. + 811. / 5040.;
                break;
        }
        return (s);
    }
};
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

    amrex::GpuArray<int, GEMPIC_SPACEDIM> box;

    // Default constructor
    // splines_at_particles();

    /**
     * @brief Find the 1D cell in which the particle is
     *
     * @param position position of particle
     * @param plo left extremity of 1D grid
     * @param dxi cell size
     */
    template <int dim, int deg>
    AMREX_GPU_HOST_DEVICE void init_position(amrex::Real position, amrex::Real plo, amrex::Real dxi)
    {
        // Compute normalized position xint and box
        amrex::Real xint = (position - plo) * dxi;
        box[dim] = std::floor(xint);
        xint = xint - box[dim];

        if (deg % 2 == 0)
        {
            if (xint < 0.5)
            {
                box[dim] -= deg / 2;
                xint = 0.5 - xint;
            }
            else
            {
                box[dim] += -deg / 2 + 1;
                xint = 1.5 - xint;
            }
        }
        else
        {
            box[dim] -= (deg - 1) / 2;
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

private:
    template <int degree, int length>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, length> spline_eval(amrex::Real x)
    {
        amrex::GpuArray<amrex::Real, length> s;
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
                s[0] = pow(x, 2) / 2.;
                s[1] = (1. - x) * x + 0.5;
                s[2] = pow(x - 1, 2) / 2.;
                break;
            case 3:
                s[0] = pow(x, 3) / 6.;
                s[1] = (((-x + 1) * x + 1.) * 3. * x + 1.) / 6.;
                s[2] = ((3. * x - 6.) * x * x + 4.) / 6.;
                s[3] = pow((1. - x), 3) / 6.;
                break;
            case 4:
                s[0] = pow(x, 4) / 24.;
                s[1] = ((((-x + 1.) * 4. * x + 6.) * x + 4.) * x + 1.) / 24.;
                s[2] = ((((6. * x - 12.) * x - 6.) * x + 12.) * x + 11.) / 24.;
                s[3] = ((((-4. * x + 12.) * x - 6.) * x - 12.) * x + 11.) / 24.;
                s[4] = pow(1. - x, 4) / 24.;
                break;
            case 5:
                s[0] = pow(x, 5) / 120.;
                s[1] = (((((-x + 1.) * x + 2.) * x + 2.) * x + 1.) * 5. * x + 1.) / 120.;
                s[2] = (((((x - 2.) * x - 2.) * x + 2.) * x + 5.) * 10. * x + 26.) / 120.;
                s[3] = (((-x + 3.) * x * x - 6.) * x * 10. * x + 66.) / 120.;
                s[4] = (((((x - 4.) * x + 4.) * x + 4.) * x - 10.) * 5. * x + 26.) / 120.;
                s[5] = (((((-x + 5.) * x - 10.) * x + 10.) * x - 5.) * x + 1.) / 120.;
                break;
            case 6:
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
 * @brief Computes electric field at particle position
 *
 * @tparam vdim Number of components of electric field
 * @tparam degx Spline degree in x
 * @tparam degy Spline degree in y
 * @tparam degz Spline degree in z
 * @param spline Spline evaluated at given particle position
 * @param eArray Degrees of freedom of electric field
 * @return amrex::GpuArray<amrex::Real, vdim> values of electric field at particle position
 *
 * Electric field is a one form so its components are
 * \f$ E_x(x,y,z) = \sum e_x S_c(x) S_n(y) S_n(z)/\Delta x \f$
 * \f$ E_y(x,y,z) = \sum e_y S_n(x) S_c(y) S_n(z)/\Delta y \f$
 * \f$ E_y(x,y,z) = \sum e_z S_n(x) S_n(y) S_c(z)/\Delta z \f$
 * where \f$S_c\f$ are the cell centered splines of degree d-1 and \f$S_c\f$ the node centered splines of degree d.
 * 
 * !!! SCALING NEEDS TO BE ADAPTED (divisions by $\Delta x,y,z)
 */
template <int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> evaluate_efield(
    splines_at_particles<degx, degy, degz> &spline,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &eArray)
{
    amrex::GpuArray<amrex::Real, vdim> fields;
    for (int comp = 0; comp < vdim; comp++)
    {
        fields[comp] = 0.;
    }

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
                    AMREX_D_DECL(spline.box[0] + ii, spline.box[1] + jj, spline.box[2] + kk)};
                if (ii < degx)
                {
                    double s_coeff = spline.spline_cell[0][ii]
#if (GEMPIC_SPACEDIM > 1)
                                     * spline.spline_node[1][jj]
#endif
#if (GEMPIC_SPACEDIM > 2)
                                     * spline.spline_node[2][kk]
#endif
                        ;
#if (GEMPIC_SPACEDIM == 1)
                    fields[0] += s_coeff * (eArray[0])(cell_index[0], 0, 0, 0);
#else
                    fields[0] +=
                        s_coeff *
                        (eArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
#if (GEMPIC_SPACEDIM > 1)
                if (vdim > 1)
                {
                    if (jj < degy)
                    {
                        double s_coeff = spline.spline_node[0][ii] * spline.spline_cell[1][jj]
#if (GEMPIC_SPACEDIM > 2)
                                         * spline.spline_node[2][kk]
#endif
                            ;
                        fields[1] +=
                            s_coeff *
                            (eArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]),
                                        0);
                    }
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
                if (kk < degz)
                {
                    double s_coeff = spline.spline_node[0][ii] * spline.spline_node[1][jj] *
                                     spline.spline_cell[2][kk];
                    fields[2] +=
                        s_coeff *
                        (eArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
#endif
            }
        }
    }
    return (fields);
}

/**
 * @brief Evaluated magnetic field (2-form) at particle positions
 *
 * @tparam vdim number of components
 * @tparam degx Spline degree in x
 * @tparam degy Spline degree in y
 * @tparam degz Spline degree in z
 * @param spline Contains splines evaluated at some particle position
 * @param bArray Contains degrees of freedom of 2-form
 * @return amrex::GpuArray<amrex::Real, vdim> Values of components of 2-form at particle positions
 */
template <int vdim, int degx, int degy, int degz>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> evaluate_bfield(
    splines_at_particles<degx, degy, degz> &spline,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &bArray)
{
    amrex::GpuArray<amrex::Real, vdim> fields;
    for (int comp = 0; comp < vdim; comp++)
    {
        fields[comp] = 0.;
    }

    for (int kk = 0; kk <= degz; ++kk)
    {
        for (int jj = 0; jj <= degy; ++jj)
        {
            for (int ii = 0; ii <= degx; ++ii)
            {
                amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index = {
                    AMREX_D_DECL(spline.box[0] + ii, spline.box[1] + jj, spline.box[2] + kk)};
                if (ii < degx)
                {
                    double s_coeff = spline.spline_cell[0][ii] * spline.spline_node[1][jj] *
                                     spline.spline_node[2][kk];
                    fields[0] +=
                        s_coeff *
                        (bArray[0])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
                if (jj < degy)
                {
                    double s_coeff = spline.spline_node[0][ii] * spline.spline_cell[1][jj] *
                                     spline.spline_node[2][kk];
                    fields[1] +=
                        s_coeff *
                        (bArray[1])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
                if (kk < degz)
                {
                    double s_coeff = spline.spline_node[0][ii] * spline.spline_node[1][jj] *
                                     spline.spline_cell[2][kk];
                    fields[2] +=
                        s_coeff *
                        (bArray[2])(AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
                }
            }
        }
    }
    return (fields);
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
                    AMREX_D_DECL(spline.box[0] + ii, spline.box[1] + jj, spline.box[2] + kk)};
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
template <typename S, int vdim, int degp, int degp1, int degp2, int pdim, int plength>
AMREX_GPU_HOST_DEVICE void accumulate_j_update_v_C2(
    S const &spline, spline1d_at_particles<degp> const &spline_new,
    spline1d_at_particles<degp> const &spline_old, amrex::Real weight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jArray,
    amrex::GpuArray<amrex::Real, 2> &fields, amrex::GpuArray<amrex::Real, plength> &primitive)
{
    int pdim1 = (pdim + 1) % 3;
    int pdim2 = (pdim + 2) % 3;

    fields = {0., 0.};
    // Size of support of difference of primitives
    int local_size = std::abs(spline_new.box - spline_old.box) + degp;
    int box_start = spline_old.box;
    if (spline_new.box < spline_old.box) box_start = spline_new.box;

    // Compute primitive at particle position
    for (unsigned int j = 0; j < plength; j++) primitive[j] = 0.0;

    if (spline_old.box < spline_new.box)
    {
        for (unsigned int j = 0; j < degp; j++) primitive[j] = spline_old.primitive[j] * dx[pdim];
        for (unsigned int j = degp; j < local_size; j++) primitive[j] = dx[pdim];

        for (unsigned int j = 0; j < degp; j++)
        {
            primitive[local_size - degp + j] -= spline_new.primitive[j] * dx[pdim];
        }
    }
    else
    {
        for (unsigned int j = degp; j < local_size; j++)
        {
            primitive[j] = -dx[pdim];
        }
        for (unsigned int j = 0; j < degp; j++)
        {
            primitive[j] = -spline_new.primitive[j] * dx[pdim];
        }
        for (unsigned int j = 0; j < degp; j++)
        {
            primitive[local_size - degp + j] += spline_old.primitive[j] * dx[pdim];
        }
    }

    // Add particle contribution to J
    for (int kk = 0; kk <= degp2; ++kk)
    {
        for (int jj = 0; jj <= degp1; ++jj)
        {
            for (int ii = 0; ii <= local_size; ++ii)
            {
                amrex::GpuArray<int, GEMPIC_SPACEDIM> cell_index;
                if (pdim == 0)
                    cell_index = {
                        AMREX_D_DECL(box_start + ii, spline.box[1] + jj, spline.box[2] + kk)};
                else if (pdim == 1)
                    cell_index = {
                        AMREX_D_DECL(spline.box[0] + kk, box_start + ii, spline.box[2] + jj)};
                else if (pdim == 2)
                    cell_index = {
                        AMREX_D_DECL(spline.box[0] + jj, spline.box[1] + kk, box_start + ii)};
                double s_coeff =
                    primitive[ii] * spline.spline_node[pdim1][jj] * spline.spline_node[pdim2][kk];
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
                    double s_coeff = primitive[ii] * spline.spline_cell[pdim1][jj] *
                                     spline.spline_node[pdim2][kk];

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
                    double s_coeff = primitive[ii] * spline.spline_node[pdim1][jj] *
                                     spline.spline_cell[pdim2][kk];
#if (GEMPIC_SPACEDIM == 1)
                    fields[1] += s_coeff * (bArray[pdim1])(cell_index[0], 0, 0, 0);
#else
                    fields[1] +=
                        s_coeff * (bArray[pdim1])(
                                      AMREX_D_DECL(cell_index[0], cell_index[1], cell_index[2]), 0);
#endif
                }
            }
        }
    }
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
                        if (cc % 3 == 1)
                        {
                            s_coeff *= spline.spline_cell[1][jj];
                        }
                        else
                        {
                            s_coeff *= spline.spline_node[1][jj];
                        }
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
                            spline.box[0] + ii, spline.box[1] + jj, spline.box[2] + kk)};

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

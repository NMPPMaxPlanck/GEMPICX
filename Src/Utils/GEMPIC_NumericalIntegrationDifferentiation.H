/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_NUMERICALINTEGRATIONDIFFERENTIATION_H_
#define GEMPIC_NUMERICALINTEGRATIONDIFFERENTIATION_H_

#include <array>
#include <type_traits>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumTools.H"

namespace Gempic
{
// clang-format off
/**
 * @brief Gauss-Legendre quadrature for integration of finite intervals
 *
 * @details Implements integration stencils with points in {@ref s_minPoints,...,@ref s_maxPoints}
 * 
 * @todo Add concepts with C++20 for template parameter `Function`
 */
// clang-format on
class GaussLegendreQuadrature
{
public:
    GaussLegendreQuadrature(int points);
    // clang-format off
    /**
     * @brief Approximates
     * \f[
     *      \int_{m - hw}^{m + hw} f(x) \mathrm{d}x
     * \f]
     * with Gauss-Legendre quadrature
     * 
     * @details Example on how to integrate a 1D subset of a 4D function \f$ \int_{y_0-\Delta y/2}^{y_0+\Delta y/2} f(x_0,y,z_0,t_0) \mathrm{d}y \f$
     * @snippet Utils/GEMPIC_TestNumericalIntegrationDifferentiation.cpp GaussLegendreQuadratureExample.LineIntegral
     */
    // clang-format on
    template <typename Function1D>
    inline AMREX_GPU_HOST_DEVICE amrex::Real line (amrex::Real const& midpoint,
                                                   amrex::Real const& halfWidth,
                                                   Function1D const& f) const
    {
        static_assert(
            std::is_invocable_r_v<amrex::Real, Function1D, amrex::Real>,
            "'Function1D const &f' has no callable interface matching amrex::Real(amrex::Real)");
        amrex::Real integral{};
        for (int ix{0}; ix < m_points; ix++)
        {
            amrex::Real x{midpoint + halfWidth * m_location[ix]};
            integral += m_weights[ix] * f(x);
        }
        return integral * halfWidth;
    }
    // clang-format off
    /**
     * @brief Approximates
     * \f[
     *      \int_{m[0] - hw[0]}^{m[0] + hw[0]} \int_{m[1] - hw[1]}^{m[1] + hw[1]} f(x,y) \mathrm{d}y \mathrm{d}x
     * \f]
     * with Gauss-Legendre quadrature
     * 
     * @details Example on how to integrate a 2D subset of a 4D function \f$ \int_{y_0-\Delta y/2}^{y_0+\Delta y/2} \int_{x_0-\Delta x/2}^{x_0+\Delta x/2} f(x,y,z_0,t_0) \mathrm{d}x\mathrm{d}y \f$
     * @snippet Utils/GEMPIC_TestNumericalIntegrationDifferentiation.cpp GaussLegendreQuadratureExample.SurfaceIntegral
     */
    // clang-format on
    template <typename Function2D>
    inline AMREX_GPU_HOST_DEVICE amrex::Real surface (std::array<amrex::Real, 2> const& midpoint,
                                                      std::array<amrex::Real, 2> const& halfWidth,
                                                      Function2D const& f) const
    {
        static_assert(std::is_invocable_r_v<amrex::Real, Function2D, amrex::Real, amrex::Real>,
                      "'Function2D const &f' has no callable interface matching "
                      "amrex::Real(amrex::Real, amrex::Real)");
        amrex::Real integral{};
        for (int ix{0}; ix < m_points; ix++)
        {
            amrex::Real x{midpoint[0] + halfWidth[0] * m_location[ix]};
            for (int iy{0}; iy < m_points; iy++)
            {
                amrex::Real y{midpoint[1] + halfWidth[1] * m_location[iy]};
                integral += m_weights[ix] * m_weights[iy] * f(x, y);
            }
        }
        return integral * halfWidth[0] * halfWidth[1];
    }
    // clang-format off
    /**
     * @brief Approximates
     * \f[
     *      \int_{m[0] - hw[0]}^{m[0] + hw[0]} \int_{m[1] - hw[1]}^{m[1] + hw[1]} \int_{m[2] - hw[2]}^{m[2] + hw[2]} f(x,y,z) \mathrm{d}z \mathrm{d}y \mathrm{d}x
     * \f]
     * with Gauss-Legendre quadrature
     * 
     * @details Example on how to integrate a 3D subset of a 4D function \f$ \int_{z_0-\Delta z/2}^{z_0+\Delta z/2} \int_{y_0-\Delta y/2}^{y_0+\Delta y/2} \int_{x_0-\Delta x/2}^{x_0+\Delta x/2} f(x,y,z, t_0) \mathrm{d}x\mathrm{d}y\mathrm{d}z \f$
     * @snippet Utils/GEMPIC_TestNumericalIntegrationDifferentiation.cpp GaussLegendreQuadratureExample.VolumeIntegral
     */
    // clang-format on
    template <typename Function3D>
    inline AMREX_GPU_HOST_DEVICE amrex::Real volume (std::array<amrex::Real, 3> const& midpoint,
                                                     std::array<amrex::Real, 3> const& halfWidth,
                                                     Function3D const& f) const
    {
        static_assert(
            std::is_invocable_r_v<amrex::Real, Function3D, amrex::Real, amrex::Real, amrex::Real>,
            "'Function3D const &f' has no callable interface matching amrex::Real(amrex::Real, "
            "amrex::Real, amrex::Real)");
        amrex::Real integral{};
        for (int ix{0}; ix < m_points; ix++)
        {
            amrex::Real x{midpoint[0] + halfWidth[0] * m_location[ix]};
            for (int iy{0}; iy < m_points; iy++)
            {
                amrex::Real y{midpoint[1] + halfWidth[1] * m_location[iy]};
                for (int iz{0}; iz < m_points; iz++)
                {
                    amrex::Real z{midpoint[2] + halfWidth[2] * m_location[iz]};
                    integral += m_weights[ix] * m_weights[iy] * m_weights[iz] * f(x, y, z);
                }
            }
        }
        return integral * halfWidth[0] * halfWidth[1] * halfWidth[2];
    }

    // NOLINTBEGIN(readability-magic-numbers)
    //! Smallest Gauss-Legendre integration stencil
    static constexpr int s_minPoints{1};
    //! Largest Gauss-Legendre integration stencil
    static constexpr int s_maxPoints{10};
    // NOLINTEND(readability-magic-numbers)
    std::array<amrex::Real, s_maxPoints> get_quadrature_nodes () { return m_location; };
    std::array<amrex::Real, s_maxPoints> get_quadrature_weights () { return m_weights; };

private:
    int m_points{};
    static constexpr int s_nGaussFormulas{s_maxPoints - s_minPoints + 1};
    std::array<amrex::Real, s_maxPoints> m_location;
    std::array<amrex::Real, s_maxPoints> m_weights;
    static std::array<amrex::Real, s_maxPoints> location(int points);
    static std::array<amrex::Real, s_maxPoints> weights(int points);
};

// clang-format off
/**
 * @brief Gauss-Legendre quadrature weights and nodes on the unit interval [0,1]
 *
 */
// clang-format on
template <int points>
class GaussQuadratureUnit
{
public:
    static constexpr int s_points{points};
    static constexpr auto s_gl = gauss_legendre<points, 0.0, 1.0>();
    static constexpr std::array<amrex::Real, points> const& s_nodes = s_gl.first;
    static constexpr std::array<amrex::Real, points> const& s_weights = s_gl.second;
    static constexpr int get_n_nodes () { return s_points; };
};

} // namespace Gempic

#endif
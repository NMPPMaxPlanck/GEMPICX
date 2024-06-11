#pragma once

#include <array>

#include <AMReX_Array.H>

namespace Gempic
{
// NOLINTBEGIN(readability-magic-numbers)
// clang-format off
/**
 * @brief Evaluating all non-zero polynomial intervals of a cardinal B-Spline which have been mapped into
 *        \f$x\in[0,1]\f$
 * 
 * @details B-Splines based on the definition of \f$S^{degree}(x)\f$ in
 *          @verbatim embed:rst:inline :ref:`Particle-Mesh Coupling` @endverbatim. 
 *          All B-Spline degrees which have been implemented can be recreated executing the 
 *          [Mathematica script](https://gitlab.mpcdf.mpg.de/gempic/gempic/-/blob/master/Src/ParticleMeshCoupling/GEMPIC_BSplines.wls?ref_type=heads).
 *
 *          Example usage of \f$S^2(x)\f$:
 *          @code{cpp}
 *          auto res = cardinal_b_spline<2>(0.25);
 *          std::cout << "First polynomial interval S^2(-1.25)=" << res[0] << std::endl;
 *          std::cout << "Second polynomial interval S^2(-0.25)=" << res[1] << std::endl;
 *          std::cout << "Third polynomial interval S^2(0.75)=" << res[1] << std::endl;
 *          @endcode
 *          outputs
 *          @code
 *          First polynomial interval S^2(-1.25)=0.03125
 *          Second polynomial interval S^2(-0.25)=0.6875
 *          Third polynomial interval S^2(0.75)=0.28125
 *          @endcode
 * 
 * @param x : \f$x\in[0,1]\f$
 * @tparam degree : \f$degree\in\{0,1,2,3,4,5\}\f$
 * @tparam length : For compatibility with @ref SplineBase. 
 *                  Will be deprecated as soon as possible.
 *                  Every user thinking about using this, DO NOT!
 *
 * @return res : \f$res[i] = S^{degree}(x+i-(degree+1)/2)\f$
 */
// clang-format on
template <size_t degree, size_t length = degree + 1>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, length> cardinal_b_splines (amrex::Real x)
{
    amrex::GpuArray<amrex::Real, length> res{};
    static_assert(degree < 6, "Only B Splines of degree 0 to 5 have been implemented!");
    switch (degree)
    {
        case (0):
        {
            res[0] = 1;
            return res;
        }
        case (1):
        {
            res[0] = x;
            res[1] = 1 - x;
            return res;
        }
        case (2):
        {
            constexpr double inv2{1.0 / 2.0};
            res[0] = x * x * inv2;
            res[1] = (1.0 - x) * x + inv2;
            res[2] = (x - 1.0) * (x - 1.0) * inv2;
            return res;
        }
        case (3):
        {
            constexpr double inv6{1.0 / 6.0};
            res[0] = x * x * x * inv6;
            res[1] = (((-x + 1.0) * x + 1.0) * 3.0 * x + 1.0) * inv6;
            res[2] = ((x - 2.0) * 3.0 * x * x + 4.0) * inv6;
            res[3] = -(x - 1.0) * (x - 1.0) * (x - 1.0) * inv6;
            return res;
        }
        case (4):
        {
            constexpr double inv24{1.0 / 24.0};
            res[0] = x * x * x * x * inv24;
            res[1] = ((((-x + 1.0) * 4.0 * x + 6.0) * x + 4.0) * x + 1.0) * inv24;
            res[2] = ((((x - 2.0) * 6.0 * x - 6.0) * x + 12.0) * x + 11.0) * inv24;
            res[3] = ((((-x + 3.0) * 4.0 * x - 6.0) * x - 12.0) * x + 11.0) * inv24;
            res[4] = (x - 1.0);
            res[4] = res[4] * res[4] * res[4] * res[4] * inv24;
            return res;
        }
        case (5):
        {
            constexpr double inv60{1.0 / 60.0};
            constexpr double inv120{1.0 / 120.0};
            res[0] = x * x * x * x * x * inv120;
            res[1] = ((((((-x + 1.0) * x + 2.0) * x + 2.0) * x + 1.0) * 5.0 * x) + 1.0) * inv120;
            res[2] = (((((x - 2.0) * x - 2.0) * x + 2.0) * x + 5.0) * 5.0 * x + 13.0) * inv60;
            res[3] = (((-x + 3.0) * x * x - 6.0) * 5.0 * x * x + 33.0) * inv60;
            res[4] = (((((x - 4.0) * x + 4.0) * x + 4.0) * x - 10.0) * 5.0 * x + 26.0) * inv120;
            res[5] = (x - 1.0);
            res[5] = -res[5] * res[5] * res[5] * res[5] * res[5] * inv120;
            return res;
        }
    }
}

// clang-format off
/**
 * @brief Evaluating all non-zero polynomial intervals of the primitive of a cardinal B-Spline which
 *        have been mapped into \f$x\in[0,1]\f$
 *
 * @details Implemented \f$\int_{-(degree+1)/2}^x S^{degree}(y) \mathrm{d}y \f$
 *          Otherwise this function executes the same behavior as @ref cardinal_b_spline
 *
 * @param x : \f$x\in[0,1]\f$
 * @tparam degree : \f$degree\in\{0,1,2,3,4,5\}\f$
 * @tparam length : For compatibility with @ref SplineBase.
 *                  Will be deprecated as soon as possible.
 *                  Every user thinking about using this, DO NOT!
 *
 * @return res : \f$res[i] = \int_{(degree+1)/2}^{x+i-(degree+1)/2} S^{degree}(y) \mathrm{d}y\f$
 */
// clang-format on
template <size_t degree, size_t length = degree + 1>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, length> cardinal_b_splines_primitive (
    amrex::Real x)
{
    amrex::GpuArray<amrex::Real, length> res{};
    static_assert(degree < 6,
                  "Only the primitives for Bsplines of degree 0 to 5 have been implemented.");
    switch (degree)
    {
        case (0):
        {
            res[0] = x;
            return res;
        }
        case (1):
        {
            constexpr double inv2{1.0 / 2.0};
            res[0] = x * x * inv2;
            res[1] = ((-x + 2.0) * x + 1.0) * inv2;
            return res;
        }
        case (2):
        {
            constexpr double inv6{1.0 / 6.0};
            res[0] = x * x * x * inv6;
            res[1] = (((-2.0 * x + 3.0) * x + 3.0) * x + 1.0) * inv6;
            res[2] = (((x - 3.0) * x + 3.0) * x + 5.0) * inv6;
            return res;
        }
        case (3):
        {
            constexpr double inv24{1.0 / 24.0};
            res[0] = x * x * x * x * inv24;
            res[1] = ((((-3.0 * x + 4.0) * x + 6.0) * x + 4.0) * x + 1.0) * inv24;
            res[2] = (((3.0 * x - 8.0) * x * x + 16.0) * x + 12) * inv24;
            res[3] = ((((-x + 4.0) * x - 6.0) * x + 4.0) * x + 23.0) * inv24;
            return res;
        }
        case (4):
        {
            constexpr double inv120{1.0 / 120.0};
            res[0] = x * x * x * x * x * inv120;
            res[1] = (((((-4.0 * x + 5.0) * x + 10.0) * x + 10.0) * x + 5.0) * x + 1.0) * inv120;
            res[2] = (((((6.0 * x - 15) * x - 10.0) * x + 30.0) * x + 55.0) * x + 27.0) * inv120;
            res[3] = (((((-4.0 * x + 15.0) * x - 10.0) * x - 30.0) * x + 55.0) * x + 93.0) * inv120;
            res[4] = ((((x - 5.0) * x + 10.0) * x - 10.0 * x) + 5.0 * x + 119.0) * inv120;
            return res;
        }
        case (5):
        {
            constexpr double inv360{1.0 / 360.0};
            constexpr double inv720{1.0 / 720.0};
            res[0] = x * x * x * x * x * x * inv720;
            res[1] =
                ((((((-5.0 * x + 6.0) * x + 15.0) * x + 20.0) * x + 15.0) * x + 6.0) * x + 1.0) *
                inv720;
            res[2] =
                ((((((5.0 * x - 12.0) * x - 15.0) * x + 20.0) * x + 75.0) * x + 78.0) * x + 29.0) *
                inv360;
            res[3] = ((((-5.0 * x + 18.0) * x * x - 60.0) * x * x + 198.0) * x + 180) * inv360;
            res[4] = ((((((5.0 * x - 24.0) * x + 30.0) * x + 40.0) * x - 150.0) * x + 156.0) * x +
                      662.0) *
                     inv720;
            res[5] = ((((((-x + 6.0) * x - 15.0) * x + 20.0) * x - 15.0) * x + 6.0) * x + 719.0) *
                     inv720;
            return res;
        }
    }
}

// clang-format off
/**
 * @brief Evaluating all non-zero polynomial intervals the first derivative of a cardinal B-Spline which have been mapped into
 *        \f$x\in[0,1]\f$
 * 
 * @details Implemented \f$\partial_x S^{degree}(x)\f$
 *          Otherwise this function executes the same behavior as @ref cardinal_b_spline
 * @param x : \f$x\in[0,1]\f$
 * @tparam degree : \f$degree\in\{0,1,2,3,4,5\}\f$
 *
 * @return res : \f$res[i] = \partial_x S^{degree}(x+i-(degree+1)/2)\f$
 * 
 */
// clang-format on
template <size_t degree>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, degree + 1> cardinal_b_splines_first_derivative (
    amrex::Real x)
{
    amrex::GpuArray<amrex::Real, degree + 1> res{};
    static_assert(0 < degree and degree < 6,
                  "Only first derivatives B Splines of degree 1 to 5 have been implemented!");
    switch (degree)
    {
        case (1):
        {
            res[0] = 1;
            res[1] = -1;
            return res;
        }
        case (2):
        {
            res[0] = x;
            res[1] = (1.0 - 2.0 * x);
            res[2] = -1.0 + x;
            return res;
        }
        case (3):
        {
            constexpr double inv2{1.0 / 2.0};
            res[0] = x * x * inv2;
            res[1] = ((-3.0 * x + 2.0) * x + 1.0) * inv2;
            res[2] = (3.0 * x - 4.0) * x * inv2;
            res[3] = (x - 1.0);
            res[3] = -res[3] * res[3] * inv2;
            return res;
        }
        case (4):
        {
            constexpr double inv6{1.0 / 6.0};
            constexpr double inv2{1.0 / 2.0};
            res[0] = x * x * x * inv6;
            res[1] = (((-4.0 * x + 3.0) * x + 3.0) * x + 1.0) * inv6;
            res[2] = (((2.0 * x - 3.0) * x - 1.0) * x + 1.0) * inv2;
            res[3] = (((-4.0 * x + 9.0) * x - 3.0) * x - 3.0) * inv6;
            res[4] = (x - 1.0);
            res[4] = res[4] * res[4] * res[4] * inv6;
            return res;
        }
        case (5):
        {
            constexpr double inv12{1.0 / 12.0};
            constexpr double inv24{1.0 / 24.0};
            res[0] = x * x * x * x * inv24;
            res[1] = ((((-5.0 * x + 4.0) * x + 6.0) * x + 4.0) * x + 1.0) * inv24;
            res[2] = ((((5.0 * x - 8.0) * x - 6.0) * x + 4.0) * x + 5.0) * inv12;
            res[3] = (((-5.0 * x * inv12 + 1.0) * x * x - 1.0) * x);
            res[4] = ((((5.0 * x - 16.0) * x + 12.0) * x + 8.0) * x - 10.0) * inv24;
            res[5] = (x - 1.0);
            res[5] = -res[5] * res[5] * res[5] * res[5] * inv24;
            return res;
        }
    }
}

// clang-format off
/**
 * @brief Evaluating all non-zero polynomial intervals the first derivative of a cardinal B-Spline which have been mapped into
 *        \f$x\in[0,1]\f$
 * 
 * @details Implemented \f$\partial_x^2 S^{degree}(x)\f$
 *          Otherwise this function executes the same behavior as @ref cardinal_b_spline
 * 
 * @param x : \f$x\in[0,1]\f$
 * @tparam degree : \f$degree\in\{0,1,2,3,4,5\}\f$
 *
 * @return res : \f$res[i] = \partial_x^2 S^{degree}(x+i-(degree+1)/2)\f$
 * 
 */
// clang-format on
template <size_t degree>
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, degree + 1>
cardinal_b_splines_second_derivative (amrex::Real x)
{
    amrex::GpuArray<amrex::Real, degree + 1> res{};
    static_assert(1 < degree or degree < 6,
                  "Only second derivatives B Splines of degree 2 to 5 have been implemented!");
    switch (degree)
    {
        case (2):
        {
            res[0] = 1.0;
            res[1] = -2.0;
            res[2] = 1.0;
            return res;
        }
        case (3):
        {
            res[0] = x;
            res[1] = 1.0 - 3.0 * x;
            res[2] = -2.0 + 3.0 * x;
            res[3] = (1.0 - x);
            return res;
        }
        case (4):
        {
            constexpr double inv2{1.0 / 2.0};
            res[0] = x * x * inv2;
            res[1] = (-2.0 * x + 1.0) * x + inv2;
            res[2] = (x - 1.0) * 3.0 * x - inv2;
            res[3] = (-2.0 * x + 3.0) * x - inv2;
            res[4] = (x - 1.0);
            res[4] = res[4] * res[4] * inv2;
            return res;
        }
        case (5):
        {
            constexpr double inv3{1.0 / 3.0};
            constexpr double inv6{1.0 / 6.0};
            res[0] = x * x * x * inv6;
            res[1] = (((-5.0 * x + 3.0) * x + 3.0) * x + 1.0) * inv6;
            res[2] = (((5.0 * x - 6.0) * x - 3.0) * x + 1.0) * inv3;
            res[3] = ((-5.0 * x * inv3 + 3.0) * x * x - 1.0);
            res[4] = ((5.0 * x * inv6 - 2.0) * x + 1.0) * x + inv3;
            res[5] = (x - 1.0);
            res[5] = -res[5] * res[5] * res[5] * inv6;
            return res;
        }
    }
}
// NOLINTEND(readability-magic-numbers)
}  //namespace Gempic
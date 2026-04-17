/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_ARRAY4D_H_
#define GEMPIC_ARRAY4D_H_

namespace Gempic::Utils
{

/**
 * A GPU-compatible four-dimensional array.
 * I didn't find the Array4D in amrex, so I extended the Array3D.
 *
 * \tparam xLo Index for lower bound in \a x dimension. Can be other than 0.
 * \tparam xHi Index for upper bound in \a x dimension.
 * \tparam yLo Index for lower bound in \a y dimension. Can be other than 0.
 * \tparam yHi Index for upper bound in \a y dimension.
 * \tparam zLo Index for lower bound in \a z dimension. Can be other than 0.
 * \tparam zHi Index for upper bound in \a z dimension.
 * \tparam tLo Index for lower bound in \a t dimension. Can be other than 0.
 * \tparam tHi Index for upper bound in \a t dimension.
 * \tparam ORDER Either amrex::Order::C (C/C++ row-major order) or
 *               amrex::Order::F (Fortran column-major order, which is the
 *               default if not given)
 */
template <class T,
          int xLo,
          int xHi,
          int yLo,
          int yHi,
          int zLo,
          int zHi,
          int tLo,
          int tHi,
          amrex::Order order = amrex::Order::F>
struct Array4D
{
    /**
     * Returns the total number of elements in the Array4D object as an
     * unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    size () noexcept
    {
        return (xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1) * (tHi - tLo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array4D object in the
     * \a x direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int xlo () noexcept
    {
        return xLo;
    }

    /**
     * Returns the index of the upper bound of the Array4D object in the
     * \a x direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int xhi () noexcept
    {
        return xHi;
    }

    /**
     * Returns the number of elements of the Array4D object in the
     * \a x direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    xlen () noexcept
    {
        return (xHi - xLo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array4D object in the
     * \a y direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int ylo () noexcept
    {
        return yLo;
    }

    /**
     * Returns the index of the upper bound of the Array4D object in the
     * \a y direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int yhi () noexcept
    {
        return yHi;
    }

    /**
     * Returns the number of elements of the Array4D object in the
     * \a y direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    ylen () noexcept
    {
        return (yHi - yLo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array4D object in the
     * \a z direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int zlo () noexcept
    {
        return zLo;
    }

    /**
     * Returns the index of the upper bound of the Array4D object in the
     * \a z direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int zhi () noexcept
    {
        return zHi;
    }

    /**
     * Returns the number of elements of the Array4D object in the
     * \a z direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    zlen () noexcept
    {
        return (zHi - zLo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array4D object in the
     * \a t direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int tlo () noexcept
    {
        return tLo;
    }

    /**
     * Returns the index of the upper bound of the Array4D object in the
     * \a t direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int thi () noexcept
    {
        return tHi;
    }

    /**
     * Returns the number of elements of the Array4D object in the
     * \a t direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    tlen () noexcept
    {
        return (tHi - tLo + 1);
    }

    /**
     * Returns a \c const pointer address to the first element of the
     * Array4D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T* begin () const noexcept
    {
        return m_arr;
    }

    /**
     * Returns a \c const pointer address right after the last element of the
     * Array4D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T* end () const noexcept
    {
        return m_arr + (xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1) * (tHi - tLo + 1);
    }

    /**
     * Returns a pointer address to the first element of the
     * Array4D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T* begin () noexcept { return m_arr; }

    /**
     * Returns a pointer address right after the last element of the
     * Array4D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T* end () noexcept
    {
        return m_arr + (xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1) * (tHi - tLo + 1);
    }

    /**
     * The elements of an Array4D object are accessed using parentheses,
     * e.g. \c array(i,j,k,l), instead of using square brackets.
     * If the order is not specified, Fortran column-major order is assumed
     * (the index \c i moves the fastest)
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::F, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T& operator()(int i,
                                                                               int j,
                                                                               int k,
                                                                               int l) const noexcept
    {
        return m_arr[i + j * (xHi - xLo + 1) + k * ((xHi - xLo + 1) * (yHi - yLo + 1)) +
                     l * ((xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1)) -
                     (tLo * ((xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1)) +
                      zLo * ((xHi - xLo + 1) * (yHi - yLo + 1)) + yLo * (xHi - xLo + 1) + xLo)];
    }

    /**
     * The elements of an Array4D object are accessed using parentheses,
     * e.g. \c array(i,j,k,l), instead of using square brackets.
     * If the order is not specified, Fortran column-major order is assumed
     * (the index \c i moves the fastest)
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::F, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator()(int i,
                                                                         int j,
                                                                         int k,
                                                                         int l) noexcept
    {
        return m_arr[i + j * (xHi - xLo + 1) + k * ((xHi - xLo + 1) * (yHi - yLo + 1)) +
                     l * ((xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1)) -
                     (tLo * ((xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1)) +
                      zLo * ((xHi - xLo + 1) * (yHi - yLo + 1)) + yLo * (xHi - xLo + 1) + xLo)];
    }

    /**
     * The elements of an Array4D object are accessed using parentheses,
     * e.g. \c array(i,j,k,l), instead of using square brackets.
     * When the order is manually specified as amrex::Order::C, row-major order
     * is used (the index \c k moves the fastest).
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::C, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T& operator()(int i,
                                                                               int j,
                                                                               int k,
                                                                               int l) const noexcept
    {
        return m_arr[l + k * (tHi - tLo + 1) + j * ((tHi - tLo + 1) * (zHi - zLo + 1)) +
                     i * ((tHi - tLo + 1) * (zHi - zLo + 1) * (yHi - yLo + 1)) -
                     (xLo * ((tHi - tLo + 1) * (zHi - zLo + 1) * (yHi - yLo + 1)) +
                      yLo * ((tHi - tLo + 1) * (zHi - zLo + 1)) + zLo * (tHi - tLo + 1) + tLo)];
    }

    /**
     * The elements of an Array4D object are accessed using parentheses,
     * e.g. \c array(i,j,k,l), instead of using square brackets.
     * When the order is manually specified as amrex::Order::C, row-major order
     * is used (the index \c k moves the fastest).
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::C, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator()(int i,
                                                                         int j,
                                                                         int k,
                                                                         int l) noexcept
    {
        return m_arr[l + k * (tHi - tLo + 1) + j * ((tHi - tLo + 1) * (zHi - zLo + 1)) +
                     i * ((tHi - tLo + 1) * (zHi - zLo + 1) * (yHi - yLo + 1)) -
                     (xLo * ((tHi - tLo + 1) * (zHi - zLo + 1) * (yHi - yLo + 1)) +
                      yLo * ((tHi - tLo + 1) * (zHi - zLo + 1)) + zLo * (tHi - tLo + 1) + tLo)];
    }

    /**
     * When called without any arguments, returns the sum of all
     * elements in the Array4D object.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr T sum () const noexcept
    {
        T s = 0;
        for (int i = 0; i < (xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1) * (tHi - tLo + 1);
             ++i)
        {
            s += m_arr[i];
        }
        return s;
    }

    /**
     * When called with four arguments, performs a sum reduction over
     * the specified \c axis, for a particular set of location indices \c loc0,
     * \c loc1 and \c loc2.
     *
     * \param axis The dimension to reduce (0 for \a x dimension,
     *             1 for \a y dimension, 2 for \a z dimension, 3 for \a t dimension)
     * \param loc0 The appropriate location index (either \c i or \c j)
     * \param loc1 The appropriate location index (either \c j or \c k)
     * \param loc2 The appropriate location index (either \c k or \c l)
     *
     * This can be used, for instance, to calculate the sum over the \a x
     * dimension of an Array4D object that was instantiated as
     * \verbatim Array4D<amrex::Real, 1, M, 1, N, 1, K, 1, L> array; \endverbatim
     *
     * One could instantiate an Array3D object to hold the results,
     * \verbatim Array3D<amrex::Real, 1, N, 1, K, 1, L> mat; \endverbatim
     * and then perform the summation for each element of the resulting
     * matrix.
     * \verbatim
     for         (int j = 1; j <= N; ++j) {
         for     (int k = 1; k <= K; ++k) {
             for (int l = 1; l <= L; ++l) {
                 mat(j,k,l) = array.sum(0,j,k,l)
             }
         }
     }
     \endverbatim
     * In this example, the axis is 0 and the location indices are \c loc0 = \c j,
     * \c loc1 = \c k and \c loc2 = \c l. For axis = 1, the location indices are treated as
     * \c loc0 = \c i, \c loc1 = \c k and \c loc2 = \c l; for axis = 2, \c loc0 = \c i,
     * \c loc1 = \c j and \c loc2 = \c l; for axis = 3, \c loc0 = \c i, \c loc1 = \c j and
     * \c loc2 = \c k.
     *
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr T sum (int axis,
                                                                            int loc0,
                                                                            int loc1,
                                                                            int loc2) const noexcept
    {
        T s = 0;
        if (axis == 0)
        {
            int j = loc0;
            int k = loc1;
            int l = loc2;
            for (int i = xLo; i <= xHi; ++i)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        else if (axis == 1)
        {
            int i = loc0;
            int k = loc1;
            int l = loc2;
            for (int j = yLo; j <= yHi; ++j)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        else if (axis == 2)
        {
            int i = loc0;
            int j = loc1;
            int l = loc2;
            for (int k = zLo; k <= zHi; ++k)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        else if (axis == 3)
        {
            int i = loc0;
            int j = loc1;
            int k = loc2;
            for (int l = tLo; l <= tHi; ++l)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        return s;
    }

    /**
     * When called without any arguments, returns the product of all
     * elements in the Array4D object.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr T product () const noexcept
    {
        T p = 1;
        for (int i = 0; i < (xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1) * (tHi - tLo + 1);
             ++i)
        {
            p *= m_arr[i];
        }
        return p;
    }

    /**
     * When called with three arguments, performs a product reduction over
     * the specified \c axis, for a particular set of location indices \c loc0,
     * \c loc1, and \c loc2.
     *
     * \param axis The dimension to reduce (0 for \a x dimension,
     *             1 for \a y dimension, 2 for \a z dimension, 3 for \a t dimension)
     * \param loc0 The appropriate location index (either \c i or \c j)
     * \param loc1 The appropriate location index (either \c j or \c k)
     * \param loc2 The appropriate location index (either \c k or \c l)
     *
     * This can be used, for instance, to calculate the sum over the \a z
     * dimension of an Array4D object that was instantiated as
     * \verbatim Array4D<amrex::Real, 1, M, 1, N, 1, K, 1, L> array; \endverbatim
     *
     * One could instantiate an Array3D object to hold the results,
     * \verbatim Array3D<amrex::Real, 1, M, 1, N, 1, L> mat; \endverbatim
     * and then perform the summation for each element of the resulting
     * matrix.
     * \verbatim
     for         (int l = 1; l <= L; ++l) {
         for     (int j = 1; j <= N; ++j) {
             for (int i = 1; i <= M; ++i) {
                 mat(i,j,l) = array.sum(2,i,j,l)
             }
         }
     }
     \endverbatim
     * In this example, the axis is 2 and the location indices are \c loc0 = \c i, \c loc1 = \c j
     * \c loc2 = \c l. For axis = 0, the location indices are treated as \c loc0 = \c j,
     * \c loc1 = \c k and \c loc2 = \c l; for axis = 1, \c loc0 = \c i, \c loc1 = \c k and
     * \c loc2 = \c l; for axis = 3, \c loc0 = \c i, \c loc1 = \c j and \c loc2 = \c k.
     *
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr T product (
        int const axis, int const loc0, int const loc1, int const loc2) const noexcept
    {
        T p = 1;
        if (axis == 0)
        {
            int j = loc0;
            int k = loc1;
            int l = loc2;
            for (int i = xLo; i <= xHi; ++i)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        else if (axis == 1)
        {
            int i = loc0;
            int k = loc1;
            int l = loc2;
            for (int j = yLo; j <= yHi; ++j)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        else if (axis == 2)
        {
            int i = loc0;
            int j = loc1;
            int l = loc2;
            for (int k = zLo; k <= zHi; ++k)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        else if (axis == 3)
        {
            int i = loc0;
            int j = loc1;
            int k = loc2;
            for (int l = tLo; l <= tHi; ++l)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        return p;
    }

    T m_arr[(xHi - xLo + 1) * (yHi - yLo + 1) * (zHi - zLo + 1) * (tHi - tLo + 1)];
};
} //namespace Gempic::Utils

#endif
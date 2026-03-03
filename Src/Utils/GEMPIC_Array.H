/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_ARRAY_H_
#define GEMPIC_ARRAY_H_

namespace Gempic
{

/**
 * A GPU-compatible four-dimensional array.
 * I didn't find the Array4D in amrex, so I extended the Array3D.
 * many comments are still from the amrex::Array3D
 *
 * \tparam m_XLO Index for lower bound in \a x dimension. Can be other than 0.
 * \tparam m_XHI Index for upper bound in \a x dimension.
 * \tparam m_YLO Index for lower bound in \a y dimension. Can be other than 0.
 * \tparam m_YHI Index for upper bound in \a y dimension.
 * \tparam m_ZLO Index for lower bound in \a z dimension. Can be other than 0.
 * \tparam m_ZHI Index for upper bound in \a z dimension.
 * \tparam m_TLO Index for lower bound in \a t dimension. Can be other than 0.
 * \tparam m_THI Index for upper bound in \a t dimension.
 * \tparam ORDER Either amrex::Order::C (C/C++ row-major order) or
 *               amrex::Order::F (Fortran column-major order, which is the
 *               default if not given)
 */
template <class T,
          int mXlo,
          int mXhi,
          int mYlo,
          int mYhi,
          int mZlo,
          int mZhi,
          int mTlo,
          int mThi,
          amrex::Order order = amrex::Order::F>
struct Array4D
{
    /**
     * Returns the total number of elements in the Array3D object as an
     * unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    size () noexcept
    {
        return (mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1) * (mThi - mTlo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array3D object in the
     * \a x direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int xlo () noexcept
    {
        return mXlo;
    }

    /**
     * Returns the index of the upper bound of the Array3D object in the
     * \a x direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int xhi () noexcept
    {
        return mXhi;
    }

    /**
     * Returns the number of elements of the Array3D object in the
     * \a x direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    xlen () noexcept
    {
        return (mXhi - mXlo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array3D object in the
     * \a y direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int ylo () noexcept
    {
        return mYlo;
    }

    /**
     * Returns the index of the upper bound of the Array3D object in the
     * \a y direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int yhi () noexcept
    {
        return mYhi;
    }

    /**
     * Returns the number of elements of the Array3D object in the
     * \a y direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    ylen () noexcept
    {
        return (mYhi - mYlo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array3D object in the
     * \a z direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int zlo () noexcept
    {
        return mZlo;
    }

    /**
     * Returns the index of the upper bound of the Array3D object in the
     * \a z direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int zhi () noexcept
    {
        return mZhi;
    }

    /**
     * Returns the number of elements of the Array3D object in the
     * \a z direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    zlen () noexcept
    {
        return (mZhi - mZlo + 1);
    }

    /**
     * Returns the index of the lower bound of the Array3D object in the
     * \a z direction.
     * Can be other than 0.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int tlo () noexcept
    {
        return mTlo;
    }

    /**
     * Returns the index of the upper bound of the Array3D object in the
     * \a z direction.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr int thi () noexcept
    {
        return mThi;
    }

    /**
     * Returns the number of elements of the Array3D object in the
     * \a z direction as an unsigned integer.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static constexpr unsigned int
    tlen () noexcept
    {
        return (mThi - mTlo + 1);
    }

    /**
     * Returns a \c const pointer address to the first element of the
     * Array3D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T* begin () const noexcept
    {
        return m_arr;
    }

    /**
     * Returns a \c const pointer address right after the last element of the
     * Array3D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T* end () const noexcept
    {
        return m_arr +
               (mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1) * (mThi - mTlo + 1);
    }

    /**
     * Returns a pointer address to the first element of the
     * Array3D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T* begin () noexcept { return m_arr; }

    /**
     * Returns a pointer address right after the last element of the
     * Array3D object, as if the object is treated as one-dimensional.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T* end () noexcept
    {
        return m_arr +
               (mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1) * (mThi - mTlo + 1);
    }

    /**
     * The elements of an Array3D object are accessed using parentheses,
     * e.g. \c array(i,j,k), instead of using square brackets.
     * If the order is not specified, Fortran column-major order is assumed
     * (the index \c i moves the fastest)
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::F, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T& operator()(int i,
                                                                               int j,
                                                                               int k,
                                                                               int l) const noexcept
    {
        return m_arr[i + j * (mXhi - mXlo + 1) + k * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1)) +
                     l * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1)) -
                     (mTlo * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1)) +
                      mZlo * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1)) + mYlo * (mXhi - mXlo + 1) +
                      mXlo)];
    }

    /**
     * The elements of an Array3D object are accessed using parentheses,
     * e.g. \c array(i,j,k), instead of using square brackets.
     * If the order is not specified, Fortran column-major order is assumed
     * (the index \c i moves the fastest)
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::F, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator()(int i,
                                                                         int j,
                                                                         int k,
                                                                         int l) noexcept
    {
        return m_arr[i + j * (mXhi - mXlo + 1) + k * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1)) +
                     l * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1)) -
                     (mTlo * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1)) +
                      mZlo * ((mXhi - mXlo + 1) * (mYhi - mYlo + 1)) + mYlo * (mXhi - mXlo + 1) +
                      mXlo)];
    }

    /**
     * The elements of an Array3D object are accessed using parentheses,
     * e.g. \c array(i,j,k), instead of using square brackets.
     * When the order is manually specified as amrex::Order::C, row-major order
     * is used (the index \c k moves the fastest).
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::C, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T& operator()(int i,
                                                                               int j,
                                                                               int k,
                                                                               int l) const noexcept
    {
        return m_arr[l + k * (mThi - mTlo + 1) + j * ((mThi - mTlo + 1) * (mZhi - mZlo + 1)) +
                     i * ((mThi - mTlo + 1) * (mZhi - mZlo + 1) * (mYhi - mYlo + 1)) -
                     (mXlo * ((mThi - mTlo + 1) * (mZhi - mZlo + 1) * (mYhi - mYlo + 1)) +
                      mYlo * ((mThi - mTlo + 1) * (mZhi - mZlo + 1)) + mZlo * (mThi - mTlo + 1) +
                      mTlo)];
    }

    /**
     * The elements of an Array3D object are accessed using parentheses,
     * e.g. \c array(i,j,k), instead of using square brackets.
     * When the order is manually specified as amrex::Order::C, row-major order
     * is used (the index \c k moves the fastest).
     */
    template <amrex::Order ord = order, std::enable_if_t<ord == amrex::Order::C, int> = 0>
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator()(int i,
                                                                         int j,
                                                                         int k,
                                                                         int l) noexcept
    {
        return m_arr[l + k * (mThi - mTlo + 1) + j * ((mThi - mTlo + 1) * (mZhi - mZlo + 1)) +
                     i * ((mThi - mTlo + 1) * (mZhi - mZlo + 1) * (mYhi - mYlo + 1)) -
                     (mXlo * ((mThi - mTlo + 1) * (mZhi - mZlo + 1) * (mYhi - mYlo + 1)) +
                      mYlo * ((mThi - mTlo + 1) * (mZhi - mZlo + 1)) + mZlo * (mThi - mTlo + 1) +
                      mTlo)];
    }

    /**
     * When called without any arguments, returns the sum of all
     * elements in the Array3D object.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr T sum () const noexcept
    {
        T s = 0;
        for (int i = 0;
             i < (mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1) * (mThi - mTlo + 1); ++i)
        {
            s += m_arr[i];
        }
        return s;
    }

    /**
     * When called with three arguments, performs a sum reduction over
     * the specified \c axis, for a particular set of location indices \c loc0
     * and \c loc1.
     *
     * \param axis The dimension to reduce (0 for \a x dimension,
     *             1 for \a y dimension, 2 for \a z dimension)
     * \param loc0 The appropriate location index (either \c i or \c j)
     * \param loc1 The appropriate location index (either \c j or \c k)
     *
     * This can be used, for instance, to calculate the sum over the \a x
     * dimension of an Array3D object that was instantiated as
     * \verbatim Array3D<amrex::Real, 1, M, 1, N, 1, K> array; \endverbatim
     *
     * One could instantiate an Array2D object to hold the results,
     * \verbatim Array2D<amrex::Real, 1, N, 1, K> mat; \endverbatim
     * and then perform the summation for each element of the resulting
     * matrix.
     * \verbatim
     for     (int j = 1; j <= N; ++j) {
         for (int k = 1; k <= K; ++k) {
             mat(j,k) = array.sum(0,j,k)
         }
     }
     \endverbatim
     * In this example, the axis is 0 and the location indices are \c loc0 = \c j
     * and \c loc1 = \c k. For axis = 1, the location indices are treated as
     * \c loc0 = \c i and \c loc1 = \c k; for axis = 2, \c loc0 = \c j and \c loc1 = \c k.
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
            for (int i = mXlo; i <= mXhi; ++i)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        else if (axis == 1)
        {
            int i = loc0;
            int k = loc1;
            int l = loc2;
            for (int j = mYlo; j <= mYhi; ++j)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        else if (axis == 2)
        {
            int i = loc0;
            int j = loc1;
            int l = loc2;
            for (int k = mZlo; k <= mZhi; ++k)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        else if (axis == 3)
        {
            int i = loc0;
            int j = loc1;
            int k = loc2;
            for (int l = mTlo; l <= mThi; ++l)
            {
                s += this->operator()(i, j, k, l);
            }
        }
        return s;
    }

    /**
     * When called without any arguments, returns the product of all
     * elements in the Array3D object.
     */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr T product () const noexcept
    {
        T p = 1;
        for (int i = 0;
             i < (mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1) * (mThi - mTlo + 1); ++i)
        {
            p *= m_arr[i];
        }
        return p;
    }

    /**
     * When called with three arguments, performs a product reduction over
     * the specified \c axis, for a particular set of location indices \c loc0
     * and \c loc1.
     *
     * \param axis The dimension to reduce (0 for \a x dimension,
     *             1 for \a y dimension, 2 for \a z dimension)
     * \param loc0 The appropriate location index (either \c i or \c j)
     * \param loc1 The appropriate location index (either \c j or \c k)
     *
     * This can be used, for instance, to calculate the sum over the \a z
     * dimension of an Array3D object that was instantiated as
     * \verbatim Array3D<amrex::Real, 1, M, 1, N, 1, K> array; \endverbatim
     *
     * One could instantiate an Array2D object to hold the results,
     * \verbatim Array2D<amrex::Real, 1, M, 1, N> mat; \endverbatim
     * and then perform the summation for each element of the resulting
     * matrix.
     * \verbatim
     for     (int j = 1; j <= N; ++j) {
         for (int i = 1; i <= M; ++i) {
             mat(i,j) = array.sum(2,i,j)
         }
     }
     \endverbatim
     * In this example, the axis is 2 and the location indices are \c loc0 = \c i
     * and \c loc1 = \c j. For axis = 0, the location indices are treated as
     * \c loc0 = \c j and \c loc1 = \c k; for axis = 1, \c loc0 = \c i and \c loc1 = \c k.
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
            for (int i = mXlo; i <= mXhi; ++i)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        else if (axis == 1)
        {
            int i = loc0;
            int k = loc1;
            int l = loc2;
            for (int j = mYlo; j <= mYhi; ++j)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        else if (axis == 2)
        {
            int i = loc0;
            int j = loc1;
            int l = loc2;
            for (int k = mZlo; k <= mZhi; ++k)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        else if (axis == 3)
        {
            int i = loc0;
            int j = loc1;
            int k = loc2;
            for (int l = mTlo; l <= mThi; ++l)
            {
                p *= this->operator()(i, j, k, l);
            }
        }
        return p;
    }

    T m_arr[(mXhi - mXlo + 1) * (mYhi - mYlo + 1) * (mZhi - mZlo + 1) * (mThi - mTlo + 1)];
};
} //namespace Gempic

#endif
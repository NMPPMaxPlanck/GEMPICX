/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_OFFSETARRAY_H_
#define GEMPIC_OFFSETARRAY_H_

#include <array>
#include <cstddef>

#include <AMReX_GpuQualifiers.H>

namespace Gempic::Utils
{

/**
 * @brief A fixed-size array of @p N values plus a first-index offset.
 *
 * @details `operator[]` takes a *grid* index — i.e. `arr[first_index() + k]`
 *          returns `values()[k]` for `k ∈ [0, N)`. The type is intentionally
 *          lightweight: no bounds checks, no periodicity awareness, no axis
 *          handle. Use it as a "small window into a larger 1-D index space."
 *
 * @tparam T value type
 * @tparam N number of values held
 */
template <typename T, std::size_t n>
class OffsetArray
{
public:
    AMREX_GPU_HOST_DEVICE OffsetArray() = default;

    AMREX_GPU_HOST_DEVICE
    OffsetArray(int firstIndex, std::array<T, n> const& values) :
        m_firstIndex{firstIndex}, m_values{values}
    {
    }

    AMREX_GPU_HOST_DEVICE T const& operator[](int gridIndex) const
    {
        return m_values[gridIndex - m_firstIndex];
    }
    AMREX_GPU_HOST_DEVICE T& operator[](int gridIndex)
    {
        return m_values[gridIndex - m_firstIndex];
    }

    AMREX_GPU_HOST_DEVICE int first_index () const { return m_firstIndex; }
    AMREX_GPU_HOST_DEVICE int last_index () const { return m_firstIndex + static_cast<int>(n) - 1; }
    AMREX_GPU_HOST_DEVICE static constexpr std::size_t size () { return n; }

    AMREX_GPU_HOST_DEVICE std::array<T, n> const& values () const { return m_values; }
    AMREX_GPU_HOST_DEVICE std::array<T, n>& values () { return m_values; }

    bool operator==(OffsetArray const&) const = default;

    struct Entry
    {
        int m_index;
        T& m_value;
    };

    struct ConstEntry
    {
        int m_index;
        T const& m_value;
    };

    struct Iterator
    {
        T* m_ptr;
        int m_index;

        AMREX_GPU_HOST_DEVICE Entry operator*() { return {m_index, *m_ptr}; }
        AMREX_GPU_HOST_DEVICE Iterator& operator++()
        {
            ++m_ptr;
            ++m_index;
            return *this;
        }
        AMREX_GPU_HOST_DEVICE bool operator==(Iterator const& o) const { return m_ptr == o.m_ptr; }
        AMREX_GPU_HOST_DEVICE bool operator!=(Iterator const& o) const { return m_ptr != o.m_ptr; }
    };

    struct ConstIterator
    {
        T const* m_ptr;
        int m_index;

        AMREX_GPU_HOST_DEVICE ConstEntry operator*() const { return {m_index, *m_ptr}; }
        AMREX_GPU_HOST_DEVICE ConstIterator& operator++()
        {
            ++m_ptr;
            ++m_index;
            return *this;
        }
        AMREX_GPU_HOST_DEVICE bool operator==(ConstIterator const& o) const
        {
            return m_ptr == o.m_ptr;
        }
        AMREX_GPU_HOST_DEVICE bool operator!=(ConstIterator const& o) const
        {
            return m_ptr != o.m_ptr;
        }
    };

    AMREX_GPU_HOST_DEVICE Iterator begin () { return {m_values.data(), m_firstIndex}; }
    AMREX_GPU_HOST_DEVICE Iterator end ()
    {
        return {m_values.data() + n, m_firstIndex + static_cast<int>(n)};
    }
    AMREX_GPU_HOST_DEVICE ConstIterator begin () const { return {m_values.data(), m_firstIndex}; }
    AMREX_GPU_HOST_DEVICE ConstIterator end () const
    {
        return {m_values.data() + n, m_firstIndex + static_cast<int>(n)};
    }

private:
    int m_firstIndex{0};
    std::array<T, n> m_values{};
};

} // namespace Gempic::Utils

#endif
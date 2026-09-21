/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FIELDTOOLS_H_
#define GEMPIC_FIELDTOOLS_H_

#include <AMReX_REAL.H>

#include "GEMPIC_DeRhamComplex.H"

AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE bool in_interval (amrex::Real x,
                                                            amrex::Real minVal,
                                                            amrex::Real maxVal)
{
    return x >= minVal && x <= maxVal;
}

template <typename ArrayType>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE bool in_interval (amrex::Real x,
                                                            ArrayType const& minVal,
                                                            ArrayType const& maxVal,
                                                            int i)
{
    return x >= minVal[i] && x <= maxVal[i];
}

template <typename Arr, typename Min, typename Max>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void gempic_assert_array_in_interval (
    Arr const& arr, int size, Min const& minVal, Max const& maxVal, char const* msg)
{
#if defined(AMREX_DEBUG)
    for (int i = 0; i < size; ++i)
    {
        if constexpr (std::is_arithmetic_v<Min>)
        {
            AMREX_ASSERT_WITH_MESSAGE(in_interval(arr[i], minVal, maxVal), msg);
        }
        else
        {
            AMREX_ASSERT_WITH_MESSAGE(in_interval(arr[i], minVal[i], maxVal[i], i), msg);
        }
    }
#endif
}

template <Gempic::Forms::FormC SrcFormType,
          Direction idir = Direction::noDir,
          Direction formDir = Direction::noDir>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr bool is_1d_v1 ()
{
    if constexpr (Gempic::Forms::ZeroFormC<SrcFormType>)
    {
        return false; // node
    }
    else if constexpr (Gempic::Forms::ThreeFormC<SrcFormType>)
    {
        return true; // cell
    }
    else if constexpr (Gempic::Forms::OneFormC<SrcFormType>)
    {
        static_assert(idir != Direction::noDir);
        static_assert(formDir != Direction::noDir);
        return idir == formDir; // aligned edge dir
    }
    else if constexpr (Gempic::Forms::TwoFormC<SrcFormType>)
    {
        static_assert(idir != Direction::noDir);
        static_assert(formDir != Direction::noDir);
        return idir != formDir; // spanning face dirs
    }
}

template <typename SrcFormType,
          Direction idir = Direction::noDir,
          Direction formDir = Direction::noDir>
inline constexpr bool use1dV1 = is_1d_v1<SrcFormType, idir, formDir>();

template <typename SrcFormType,
          Direction idir = Direction::noDir,
          Direction formDir = Direction::noDir,
          typename V0Type,
          typename V1Type>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr decltype(auto) choose_if_1d_v0_or_1d_v1 (
    V0Type const& v0obj, V1Type const& v1obj)
{
    if constexpr (use1dV1<SrcFormType, idir, formDir>)
    {
        return (v1obj);
    }
    else
    {
        return (v0obj);
    }
}

template <typename SrcFormType, auto v0, auto v1, auto v2, auto v3>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr decltype(auto) form_to_struct ()
{
    static_assert(Gempic::Forms::FormC<SrcFormType>, "Unsupported form type in form_to_struct");

    if constexpr (Gempic::Forms::ZeroFormC<SrcFormType>)
    {
        return v0;
    }
    else if constexpr (Gempic::Forms::OneFormC<SrcFormType>)
    {
        return v1;
    }
    else if constexpr (Gempic::Forms::TwoFormC<SrcFormType>)
    {
        return v2;
    }
    else if constexpr (Gempic::Forms::ThreeFormC<SrcFormType>)
    {
        return v3;
    }
}

/**
 * @brief helper for directional shift array
 */
template <typename SrcFormType,
          Direction idir = Direction::noDir,
          Direction formDir = Direction::noDir>
inline constexpr int staggeringDirShiftValue =
    choose_if_1d_v0_or_1d_v1<SrcFormType, idir, formDir>(1, 0);

/**
 * @brief make a directional shift array
 */
template <typename SrcFormType, Direction formDir = Direction::noDir>
inline constexpr std::array<int, 3> staggeringDirShift{
    staggeringDirShiftValue<SrcFormType, Direction::xDir, formDir>,
    staggeringDirShiftValue<SrcFormType, Direction::yDir, formDir>,
    staggeringDirShiftValue<SrcFormType, Direction::zDir, formDir>};

// dual interpolation/histopolation:
// primal grid indexing
//       i-1           i
//  |.....o.....|------o------|
//  i-1         i             i+1
// dual interpolation:
// if  x< x_i => dual interpolation is centered at the i-1-th dual node, in [i-1,i] dual interval
//  if x> x_i => dual interpolation is centered at the i-th dual node, in [i,i+1] dual interval
// dual histopolation: dual barycenters coincide with primal nodes => always [i,i+1]
// let's store the shift for dual reconstructions depending on hist or int. spaces and left or
// right:
//  0 => left  side (x < x_i);
//  1 => right side (x > x_i).
// dual interpolation stencil shift relative to primal index:
AMREX_GPU_DEVICE inline constexpr amrex::GpuArray<int, 2> dual2primalInterpShift{{-1, 0}};
// dual histopolation stencil shift relative to primal index:
AMREX_GPU_DEVICE inline constexpr amrex::GpuArray<int, 2> dual2primalHistoShift{{0, 1}};

// side = 0 (left), 1 (right)
template <typename SrcFormType,
          Direction idir = Direction::noDir,
          Direction formDir = Direction::noDir>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr int get_idx_shift_dual2primal_space (int side)
{
    AMREX_ASSERT(side == 0 || side == 1);
    constexpr auto arr = choose_if_1d_v0_or_1d_v1<SrcFormType, idir, formDir>(
        dual2primalInterpShift, dual2primalHistoShift);
    return arr[side];
}

/**
 * @brief helper object to map the data structure to the primal or dual grid
 */
template <Gempic::Forms::FormC SrcFormType, Direction formDir = Direction::noDir>
struct StaggeredDofMapping;

template <Gempic::Forms::OneTwoFormC SrcFormType, Direction formDir>
struct StaggeredDofMapping<SrcFormType, formDir>
{
    static_assert(formDir < Direction::noDir,
                  "StaggeredDofMapping: OneTwoForm requires formDir in {0,1,2}");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_r;
    // for node-based staggered grid (grid shift 0.5*(dx,dy,dz)), a single integer should be enough
    amrex::GpuArray<int, 3> m_shiftDualGridIdx;
    static consteval int dir_shift (int i) { return staggeringDirShift<SrcFormType, formDir>[i]; }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE
    StaggeredDofMapping (amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& rIn)
    {
        gempic_assert_array_in_interval(
            rIn, AMREX_SPACEDIM, 0.0, 1.0,
            "impossible error in StaggeredDofMapping! rIn must be in [0,1]");
        // --------------------------------------------------------
        // Compute LR
        // LR[d] = round(node coordinate)
        // --------------------------------------------------------
        amrex::GpuArray<int, AMREX_SPACEDIM> lr;
        lr.fill(0);
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> shiftDualGrid;
        shiftDualGrid.fill(0.0);
        m_shiftDualGridIdx.fill(0);
        if constexpr (SrcFormType::s_grid == Gempic::Forms::Grid::dual)
        {
            lr = {AMREX_D_DECL(static_cast<int>(amrex::Math::round(rIn[xDir])),
                               static_cast<int>(amrex::Math::round(rIn[yDir])),
                               static_cast<int>(amrex::Math::round(rIn[zDir])))};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                shiftDualGrid[d] = 0.5 - lr[d];
            }
            m_shiftDualGridIdx = amrex::GpuArray<int, 3>{GEMPIC_D_PAD(
                (get_idx_shift_dual2primal_space<SrcFormType, formDir, xDir>(lr[xDir])),
                (get_idx_shift_dual2primal_space<SrcFormType, formDir, yDir>(lr[yDir])),
                (get_idx_shift_dual2primal_space<SrcFormType, formDir, zDir>(lr[zDir])))};
        }

        m_r = {AMREX_D_DECL(rIn[xDir] + shiftDualGrid[xDir], rIn[yDir] + shiftDualGrid[yDir],
                            rIn[zDir] + shiftDualGrid[zDir])};
    }
};

template <Gempic::Forms::ZeroThreeFormC SrcFormType, Direction formDir>
struct StaggeredDofMapping<SrcFormType, formDir>
{
    static_assert(formDir == Direction::noDir,
                  "StaggeredDofMapping: ZeroThreeForm must use formDir = noDir");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_r;
    amrex::GpuArray<int, 3> m_shiftDualGridIdx;
    static consteval int dir_shift (int i) { return staggeringDirShift<SrcFormType>[i]; }

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE
    StaggeredDofMapping (amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& rIn)
    {
        gempic_assert_array_in_interval(
            rIn, AMREX_SPACEDIM, 0.0, 1.0,
            "impossible error in StaggeredDofMapping! rIn must be in [0,1]");
        // --------------------------------------------------------
        // Compute LR
        // LR[d] = round(node coordinate)
        // --------------------------------------------------------
        amrex::GpuArray<int, AMREX_SPACEDIM> lr;
        lr.fill(0);
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> shiftDualGrid;
        shiftDualGrid.fill(0.0);
        m_shiftDualGridIdx.fill(0);
        if constexpr (SrcFormType::s_grid == Gempic::Forms::Grid::dual)
        {
            lr = {AMREX_D_DECL(static_cast<int>(amrex::Math::round(rIn[xDir])),
                               static_cast<int>(amrex::Math::round(rIn[yDir])),
                               static_cast<int>(amrex::Math::round(rIn[zDir])))};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                shiftDualGrid[d] = 0.5 - lr[d];
            }
            m_shiftDualGridIdx = amrex::GpuArray<int, 3>{
                GEMPIC_D_PAD((get_idx_shift_dual2primal_space<SrcFormType>(lr[xDir])),
                             (get_idx_shift_dual2primal_space<SrcFormType>(lr[yDir])),
                             (get_idx_shift_dual2primal_space<SrcFormType>(lr[zDir])))};
        }

        m_r = {AMREX_D_DECL(rIn[xDir] + shiftDualGrid[xDir], rIn[yDir] + shiftDualGrid[yDir],
                            rIn[zDir] + shiftDualGrid[zDir])};
    }
};
#endif

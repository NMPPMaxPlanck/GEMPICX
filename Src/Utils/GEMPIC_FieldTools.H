/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_FIELDTOOLS_H_
#define GEMPIC_FIELDTOOLS_H_

#include <AMReX_REAL.H>

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

/**
 * @brief make a directional shift array
 */
template <Gempic::Forms::OneTwoFormC SrcFormType, Direction dir>
AMREX_GPU_HOST_DEVICE consteval std::array<int, 3> make_dir_shift ()
{
    if constexpr (SrcFormType::s_space == Gempic::Forms::Space::edge)
    {
        auto tmp = std::array<int, 3>{1, 1, 1};
        tmp[dir] = 0;
        return tmp;
    }
    else // face
    {
        auto tmp = std::array<int, 3>{0, 0, 0};
        tmp[dir] = 1;
        return tmp;
    }
}

/**
 * @brief make a shift array
 */
template <Gempic::Forms::ZeroThreeFormC SrcFormType>
AMREX_GPU_HOST_DEVICE consteval std::array<int, 3> make_dir_shift ()
{
    if constexpr (SrcFormType::s_space == Gempic::Forms::Space::node)
    {
        return std::array<int, 3>{1, 1, 1};
    }
    else // cell
    {
        return std::array<int, 3>{0, 0, 0};
    }
}

// dual interpolation/histopolation: the reference stencil is centered at , is the left neighbor
// primal/dual node. depending if we are at the right of i-th primal barycenter and ith dual node
// primal idxing
//       i-1           i
//  |.....o.....|------o------|
//  i-1         i             i+1
// dual interpolation:
// if  x< x_i => dual interpolation is centered at the i-1-th dual node, in [i-1,i] dual interval
//  if x> x_i => dual interpolation is centered at the i-th dual node, in [i,i+1] dual interval
// dual histopolation: dual barycenters coincide with primal nodes => always [i,i+1]
// let's store the shift for dual reconstructions depending on hist or int. spaces and left or
// right:
//  0 => left  (x < x_i);
//  1 => right (x > x_i).
// dual interpolation shift:
AMREX_GPU_DEVICE inline constexpr amrex::GpuArray<int, 2> dual2primalShiftV0{{-1, 0}};
// dual histopolation shift:
AMREX_GPU_DEVICE inline constexpr amrex::GpuArray<int, 2> dual2primalShiftV1{{0, 1}};

AMREX_GPU_DEVICE inline constexpr amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM>
    idxShiftDual2primalNode{
        {AMREX_D_DECL(dual2primalShiftV0, dual2primalShiftV0, dual2primalShiftV0)}};

AMREX_GPU_DEVICE inline constexpr amrex::
    GpuArray<amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM>, 3>
        idxShiftDual2primalEdge{
            {{{AMREX_D_DECL(dual2primalShiftV1, dual2primalShiftV0, dual2primalShiftV0)}},
             {{AMREX_D_DECL(dual2primalShiftV0, dual2primalShiftV1, dual2primalShiftV0)}},
             {{AMREX_D_DECL(dual2primalShiftV0, dual2primalShiftV0, dual2primalShiftV1)}}}};
AMREX_GPU_DEVICE inline constexpr amrex::
    GpuArray<amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM>, 3>
        idxShiftDual2primalFace{
            {{{AMREX_D_DECL(dual2primalShiftV0, dual2primalShiftV1, dual2primalShiftV1)}},
             {{AMREX_D_DECL(dual2primalShiftV1, dual2primalShiftV0, dual2primalShiftV1)}},
             {{AMREX_D_DECL(dual2primalShiftV1, dual2primalShiftV1, dual2primalShiftV0)}}}};

AMREX_GPU_DEVICE inline constexpr amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM>
    idxShiftDual2primalCell{
        {AMREX_D_DECL(dual2primalShiftV1, dual2primalShiftV1, dual2primalShiftV1)}};

template <Direction idir, Gempic::Forms::ZeroFormC SrcFormType>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr int get_idx_shift_dual2primal_space (int i)
{
    AMREX_ASSERT(i == 0 || i == 1);
    return idxShiftDual2primalNode[idir][i];
};
template <Direction idir, typename SrcFormType>
    requires Gempic::Forms::ThreeFormC<SrcFormType>
AMREX_GPU_DEVICE GEMPIC_FORCE_INLINE constexpr int get_idx_shift_dual2primal_space (int i)
{
    AMREX_ASSERT(i == 0 || i == 1);
    return idxShiftDual2primalCell[idir][i];
};
template <Direction iDim, Direction idir, Gempic::Forms::OneFormC SrcFormType>
AMREX_GPU_DEVICE GEMPIC_FORCE_INLINE constexpr int get_idx_shift_dual2primal_space (int i)
{
    AMREX_ASSERT(i == 0 || i == 1);
    return idxShiftDual2primalEdge[iDim][idir][i];
};
template <Direction iDim, Direction idir, Gempic::Forms::TwoFormC SrcFormType>
AMREX_GPU_DEVICE GEMPIC_FORCE_INLINE constexpr int get_idx_shift_dual2primal_space (int i)
{
    AMREX_ASSERT(i == 0 || i == 1);
    return idxShiftDual2primalFace[iDim][idir][i];
};
/**
 * @brief helper object to map the data structure to the primal or dual grid
 */
template <typename SrcFormType, Direction dir = Direction::noDir>
struct StaggeredDofMapping;

template <Gempic::Forms::OneTwoFormC SrcFormType, Direction dir>
struct StaggeredDofMapping<SrcFormType, dir>
{
    static_assert(dir < Direction::noDir,
                  "StaggeredDofMapping: OneTwoForm requires dir in {0,1,2}");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_r;
    amrex::GpuArray<int, 3>
        m_shiftDualGridIdx; // for node-based staggered grid (grid shift 0.5*(dx,dy,dz)), a single
                            // integer is probably enough.
    inline static constexpr std::array<int, 3> s_mDirShift{make_dir_shift<SrcFormType, dir>()};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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
                GEMPIC_D_PAD((get_idx_shift_dual2primal_space<dir, xDir, SrcFormType>(lr[xDir])),
                             (get_idx_shift_dual2primal_space<dir, yDir, SrcFormType>(lr[yDir])),
                             (get_idx_shift_dual2primal_space<dir, zDir, SrcFormType>(lr[zDir])))};
        }

        m_r = {AMREX_D_DECL(rIn[xDir] + shiftDualGrid[xDir], rIn[yDir] + shiftDualGrid[yDir],
                            rIn[zDir] + shiftDualGrid[zDir])};
    }
};

template <Gempic::Forms::ZeroThreeFormC SrcFormType, Direction dir>
struct StaggeredDofMapping<SrcFormType, dir>
{
    static_assert(dir == Direction::noDir,
                  "StaggeredDofMapping: ZeroThreeForm must use dir = noDir");
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_r;
    amrex::GpuArray<int, 3> m_shiftDualGridIdx;
    inline static constexpr std::array<int, 3> s_mDirShift{make_dir_shift<SrcFormType>()};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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
                (get_idx_shift_dual2primal_space<Direction::xDir, SrcFormType>(lr[xDir])),
                (get_idx_shift_dual2primal_space<Direction::yDir, SrcFormType>(lr[yDir])),
                (get_idx_shift_dual2primal_space<Direction::zDir, SrcFormType>(lr[zDir])))};
        }

        m_r = {AMREX_D_DECL(rIn[xDir] + shiftDualGrid[xDir], rIn[yDir] + shiftDualGrid[yDir],
                            rIn[zDir] + shiftDualGrid[zDir])};
    }
};
#endif

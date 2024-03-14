/* Copyright 2019 Maxence Thevenet, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GEMPIC_FILTER_H_
#define GEMPIC_FILTER_H_

#include <algorithm>

#include <AMReX_Array4.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Config.H"

namespace Gempic::Filter
{

class Filter
{
public:
    Filter() = default;

#ifdef AMREX_USE_GPU
    void apply_stencil (
        amrex::MultiFab& dstmf, const amrex::MultiFab& srcmf, int scomp, int dcomp, int ncomp)
    {
        if (!m_useFilter)
        {
            return;
        }
        BL_PROFILE("Filter::ApplyStencil(MultiFab)");
        ncomp = std::min(ncomp, srcmf.nComp());

        for (amrex::MFIter mfi(dstmf); mfi.isValid(); ++mfi)
        {
            const auto& src = srcmf.array(mfi);
            const auto& dst = dstmf.array(mfi);
            const amrex::Box& tbx = mfi.growntilebox();

            // Apply filter
            do_filter(tbx, src, dst, scomp, dcomp, ncomp);
        }
    }

    /* \brief Apply stencil on FArrayBox (GPU version, 2D/3D).
     * \param dstfab Destination FArrayBox
     * \param srcmf source FArrayBox
     * \param tbx Grown box on which srcfab is defined.
     * \param scomp first component of srcfab on which the filter is applied
     * \param dcomp first component of dstfab on which the filter is applied
     * \param ncomp Number of components on which the filter is applied.
     */
    void apply_stencil (amrex::FArrayBox& dstfab,
                        const amrex::FArrayBox& srcfab,
                        const amrex::Box& tbx,
                        int scomp,
                        int dcomp,
                        int ncomp)
    {
        if (!m_useFilter)
        {
            return;
        }
        BL_PROFILE("Filter::ApplyStencil(FArrayBox)");
        ncomp = std::min(ncomp, srcfab.nComp());
        const auto& src = srcfab.array();
        const auto& dst = dstfab.array();

        // Apply filter
        do_filter(tbx, src, dst, scomp, dcomp, ncomp);
    }

    /* \brief Apply stencil (2D/3D, CPU/GPU)
     */
    void do_filter (const amrex::Box& tbx,
                    amrex::Array4<amrex::Real const> const& src,
                    amrex::Array4<amrex::Real> const& dst,
                    int scomp,
                    int dcomp,
                    int ncomp)
    {
        AMREX_D_TERM(amrex::Real const* AMREX_RESTRICT sz = m_stencilZ.data();
                     , amrex::Real const* AMREX_RESTRICT sx = m_stencilX.data();
                     , amrex::Real const* AMREX_RESTRICT sy = m_stencilY.data();)

        amrex::Dim3 slenLocal = m_slen;

        GEMPIC_D_EXCL(int iy{0};, int iz{0};, )
        AMREX_PARALLEL_FOR_4D(tbx, ncomp, i, j, k, n, {
            amrex::Real d = 0.0;

            // Pad source array with zeros beyond ghost cells
            // for out-of-bound accesses due to large-stencil operations
            const auto src_zeropad =
                [src] (const int jj, const int kk, const int ll, const int nn) noexcept
            { return src.contains(jj, kk, ll) ? src(jj, kk, ll, nn) : 0.0; };

            GEMPIC_D_LOOP_BEGIN(for (int ix = 0; ix < slenLocal.x; ++ix),
                                for (int iy = 0; iy < slenLocal.y; ++iy),
                                for (int iz = 0; iz < slenLocal.z; ++iz))
                amrex::Real sss = GEMPIC_D_MULT(sz[AMREX_D_PICK(ix, iy, iz)], sx[ix], sy[iy]);
                d += sss * (GEMPIC_D_ADD(src_zeropad(i - ix, j - iy, k - iz, scomp + n) +
                                             src_zeropad(i + ix, j - iy, k - iz, scomp + n),
                                         src_zeropad(i - ix, j + iy, k - iz, scomp + n) +
                                             src_zeropad(i + ix, j + iy, k - iz, scomp + n),
                                         src_zeropad(i - ix, j - iy, k + iz, scomp + n) +
                                             src_zeropad(i + ix, j - iy, k + iz, scomp + n) +
                                             src_zeropad(i - ix, j + iy, k + iz, scomp + n) +
                                             src_zeropad(i + ix, j + iy, k + iz, scomp + n)));
            GEMPIC_D_LOOP_END

            dst(i, j, k, dcomp + n) = d;
        });
    }
#else
    /* \brief Apply stencil on MultiFab (CPU version, 2D/3D).
     * \param dstmf Destination MultiFab
     * \param srcmf source MultiFab
     * \param[in] lev mesh refinement level
     * \param scomp first component of srcmf on which the filter is applied
     * \param dcomp first component of dstmf on which the filter is applied
     * \param ncomp Number of components on which the filter is applied.
     */
    void apply_stencil (
        amrex::MultiFab& dstmf, const amrex::MultiFab& srcmf, int scomp, int dcomp, int ncomp)
    {
        if (!m_useFilter)
        {
            return;
        }
        BL_PROFILE("Filter::ApplyStencil(MultiFab)");
        ncomp = std::min(ncomp, srcmf.nComp());
        // amrex::FArrayBox tmpfab;
        for (amrex::MFIter mfi(dstmf); mfi.isValid(); ++mfi)
        {
            const auto& srcfab = srcmf[mfi];
            auto& dstfab = dstmf[mfi];
            /*
            const amrex::Box& tbx = mfi.growntilebox();
            const amrex::Box& gbx = amrex::grow(tbx, stencil_length_each_dir - 1);
            // tmpfab has enough ghost cells for the stencil
            tmpfab.resize(gbx, ncomp);
            tmpfab.setVal(0.0, gbx, 0, ncomp);
            // Copy values in srcfab into tmpfab
            const amrex::Box& ibx = gbx & srcfab.box();
            tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
            // Apply filter
            DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);
            */
            do_filter(mfi.validbox(), srcfab.array(), dstfab.array(), 0, dcomp, ncomp);
        }
    }

    /* \brief Apply stencil on FArrayBox (CPU version, 2D/3D).
     * \param dstfab Destination FArrayBox
     * \param srcmf source FArrayBox
     * \param tbx Grown box on which srcfab is defined.
     * \param scomp first component of srcfab on which the filter is applied
     * \param dcomp first component of dstfab on which the filter is applied
     * \param ncomp Number of components on which the filter is applied.
     */
    void apply_stencil (amrex::FArrayBox& dstfab,
                        const amrex::FArrayBox& srcfab,
                        const amrex::Box& tbx,
                        int scomp,
                        int dcomp,
                        int ncomp)
    {
        if (!m_useFilter)
        {
            return;
        }
        BL_PROFILE("Filter::ApplyStencil(FArrayBox)");
        ncomp = std::min(ncomp, srcfab.nComp());
        /*
        amrex::FArrayBox tmpfab;
        const amrex::Box& gbx = amrex::grow(tbx, stencil_length_each_dir - 1);
        // tmpfab has enough ghost cells for the stencil
        tmpfab.resize(gbx, ncomp);
        tmpfab.setVal(0.0, gbx, 0, ncomp);
        // Copy values in srcfab into tmpfab
        const amrex::Box& ibx = gbx & srcfab.box();
        tmpfab.copy(srcfab, ibx, scomp, ibx, 0, ncomp);
        // Apply filter
        DoFilter(tbx, tmpfab.array(), dstfab.array(), 0, dcomp, ncomp);
        */
        do_filter(tbx, srcfab.array(), dstfab.array(), 0, dcomp, ncomp);
    }

    void do_filter (const amrex::Box& tbx,
                    amrex::Array4<amrex::Real const> const& tmp,
                    amrex::Array4<amrex::Real> const& dst,
                    int scomp,
                    int dcomp,
                    int ncomp)
    {
        const auto lo = amrex::lbound(tbx);
        const auto hi = amrex::ubound(tbx);
        // tmp and dst are of type Array4 (Fortran ordering)
        AMREX_D_TERM(amrex::Real const* AMREX_RESTRICT sz = m_stencilZ.data();
                     , amrex::Real const* AMREX_RESTRICT sx = m_stencilX.data();
                     , amrex::Real const* AMREX_RESTRICT sy = m_stencilY.data();)

        GEMPIC_D_EXCL(int j{0};, int k{0};, )
        GEMPIC_D_EXCL(int iy{0};, int iz{0};, )
        for (int n = 0; n < ncomp; ++n)
        {
            // Set dst value to 0.
            GEMPIC_D_LOOP_BEGIN(for (int i = lo.x; i <= hi.x; ++i),
                                for (int j = lo.y; j <= hi.y; ++j),
                                for (int k = lo.z; k <= hi.z; ++k))
                dst(i, j, k, dcomp + n) = 0.0;
            GEMPIC_D_LOOP_END
            // 3 nested loop on 3D stencil
            GEMPIC_D_LOOP_BEGIN(for (int ix = 0; ix < m_slen.x; ++ix),
                                for (int iy = 0; iy < m_slen.y; ++iy),
                                for (int iz = 0; iz < m_slen.z; ++iz))
                amrex::Real sss = GEMPIC_D_MULT(sz[AMREX_D_PICK(ix, iy, iz)], sx[ix], sy[iy]);

                // 3 nested loop on 3D array
                GEMPIC_D_LOOP_BEGIN(for (int i = lo.x; i <= hi.x; ++i),
                                    for (int j = lo.y; j <= hi.y; ++j),
                                    for (int k = lo.z; k <= hi.z; ++k))
                    dst(i, j, k, dcomp + n) +=
                        sss * (GEMPIC_D_ADD(tmp(i - ix, j - iy, k - iz, scomp + n) +
                                                tmp(i + ix, j - iy, k - iz, scomp + n),
                                            tmp(i - ix, j + iy, k - iz, scomp + n) +
                                                tmp(i + ix, j + iy, k - iz, scomp + n),
                                            tmp(i - ix, j - iy, k + iz, scomp + n) +
                                                tmp(i + ix, j - iy, k + iz, scomp + n) +
                                                tmp(i - ix, j + iy, k + iz, scomp + n) +
                                                tmp(i + ix, j + iy, k + iz, scomp + n)));
                GEMPIC_D_LOOP_END
            GEMPIC_D_LOOP_END
        }
    }
#endif  // #ifdef AMREX_USE_GPU

    // In 2D, stencil_length_each_dir = {length(stencil_x), length(stencil_z)}
    amrex::IntVect m_stencilLengthEachDir;

protected:
    bool m_useFilter;
    // Stencil along each direction.
    // in 2D, stencil_y is not initialized.
    amrex::Gpu::DeviceVector<amrex::Real> m_stencilX, m_stencilY, m_stencilZ;
    // Length of each stencil.
    // In 2D, slen = {length(stencil_x), length(stencil_z), 1}
    amrex::Dim3 m_slen;

private:
};

}  // namespace Gempic::Filter

#endif  // #ifndef GEMPIC_FILTER_H_
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

    void do_filter (const amrex::Box& tbx,
                    amrex::Array4<amrex::Real const> const& src,
                    amrex::Array4<amrex::Real> const& dst,
                    int scomp,
                    int dcomp,
                    int ncomp)
    {
        // src and dst are of type Array4 (Fortran ordering)
        AMREX_D_TERM(amrex::Real const* AMREX_RESTRICT sx = m_stencilX.data();
                     , amrex::Real const* AMREX_RESTRICT sy = m_stencilY.data();
                     , amrex::Real const* AMREX_RESTRICT sz = m_stencilZ.data();)

        amrex::Dim3 slen{AMREX_D_DECL(static_cast<int>(m_stencilX.size()),
                                      static_cast<int>(m_stencilY.size()),
                                      static_cast<int>(m_stencilZ.size()))};

        GEMPIC_D_EXCL(int iy{0};, int iz{0};, )
        amrex::ParallelFor(tbx, ncomp,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                           {
                               amrex::Real d{0.0};

                               // 3 nested loop on 3D stencil
                               GEMPIC_D_LOOP_BEGIN(for (int ix{0}; ix < slen.x; ++ix),
                                                   for (int iy{0}; iy < slen.y; ++iy),
                                                   for (int iz{0}; iz < slen.z; ++iz))
                                   amrex::Real sss = GEMPIC_D_MULT(sx[ix], sy[iy], sz[iz]);

                                   d += sss *
                                        (GEMPIC_D_ADD(src(i - ix, j - iy, k - iz, scomp + n) +
                                                          src(i + ix, j - iy, k - iz, scomp + n),
                                                      src(i - ix, j + iy, k - iz, scomp + n) +
                                                          src(i + ix, j + iy, k - iz, scomp + n),
                                                      src(i - ix, j - iy, k + iz, scomp + n) +
                                                          src(i + ix, j - iy, k + iz, scomp + n) +
                                                          src(i - ix, j + iy, k + iz, scomp + n) +
                                                          src(i + ix, j + iy, k + iz, scomp + n)));
                               GEMPIC_D_LOOP_END

                               dst(i, j, k, dcomp + n) = d;
                           });
    }

    /* \brief Apply stencil on MultiFab (CPU/GPU version, 1D/2D/3D).
     * \param dstmf Destination MultiFab
     * \param srcmf source MultiFab
     * \param scomp first component of srcmf on which the filter is applied
     * \param dcomp first component of dstmf on which the filter is applied
     * \param ncomp Number of components on which the filter is applied.
     */
    void apply_stencil (
        const amrex::MultiFab& srcmf, amrex::MultiFab& dstmf, int scomp, int dcomp, int ncomp)
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
            const auto& srcfab = srcmf.array(mfi);
            const auto& dstfab = dstmf.array(mfi);

            do_filter(mfi.validbox(), srcfab, dstfab, 0, dcomp, ncomp);
        }
    }

protected:
    // If true, a filter is used to smooth charge and currents
    bool m_useFilter{false};
    // Stencil along each direction.
    // in 2D, stencilZ is not initialized.
    amrex::Gpu::DeviceVector<amrex::Real> AMREX_D_DECL(m_stencilX, m_stencilY, m_stencilZ);

private:
};

}  // namespace Gempic::Filter

#endif  // #ifndef GEMPIC_FILTER_H_
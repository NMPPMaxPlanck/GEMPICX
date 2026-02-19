/* Copyright 2019 Maxence Thevenet, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GEMPIC_FILTER_H_
#define GEMPIC_FILTER_H_

#include <limits>

#include <AMReX_Array4.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"

namespace Gempic::Filter
{
class Filter
{
public:
    Filter() = default;

    void do_filter(amrex::Box const& tbx,
                   amrex::Array4<amrex::Real const> const& src,
                   amrex::Array4<amrex::Real> const& dst,
                   int scomp,
                   int dcomp,
                   int ncomp);

    /// @cond EXCLUDE_API_DOC
    template <Forms::Grid grid, Forms::Space space>
        requires Forms::ZeroOrThreeFormSpaceC<space>
    void apply_stencil (Gempic::Forms::DeRhamField<grid, space>& dstField,
                        Gempic::Forms::DeRhamField<grid, space> const& srcField,
                        int scomp = 0,
                        int dcomp = 0,
                        int ncomp = std::numeric_limits<int>::max())
    {
        BL_PROFILE("Gempic::Filter::Filter::apply_stencil()");

        ncomp = std::min(ncomp, srcField.m_data.nComp());
        if (!m_useFilter)
        {
            // just copy
            copy(dstField, srcField);
            return;
        }

        for (amrex::MFIter mfi(dstField.m_data); mfi.isValid(); ++mfi)
        {
            auto const& srcfab = srcField.m_data.array(mfi);
            auto const& dstfab = dstField.m_data.array(mfi);

            do_filter(mfi.validbox(), srcfab, dstfab, 0, dcomp, ncomp);
        }
    }
    /// @endcond

    template <Forms::Grid grid, Forms::Space space>
        requires Forms::OneOrTwoFormSpaceC<space>
    void apply_stencil (Gempic::Forms::DeRhamField<grid, space>& dstField,
                        Gempic::Forms::DeRhamField<grid, space> const& srcField,
                        int scomp = 0,
                        int dcomp = 0,
                        int ncomp = std::numeric_limits<int>::max())
    {
        BL_PROFILE("Gempic::Filter::Filter::apply_stencil()");

        if (!m_useFilter)
        {
            // just copy
            copy(dstField, srcField);
            return;
        }
        // loop over the 3 directions
        for (int d{0}; d < 3; ++d)
        {
            ncomp = std::min(ncomp, srcField.m_data[d].nComp());

            for (amrex::MFIter mfi(dstField.m_data[d]); mfi.isValid(); ++mfi)
            {
                auto const& srcfab = srcField.m_data[d].array(mfi);
                auto const& dstfab = dstField.m_data[d].array(mfi);

                do_filter(mfi.validbox(), srcfab, dstfab, 0, dcomp, ncomp);
            }
        }
    }

protected:
    // If true, a filter is used to smoothe charge and currents
    bool m_useFilter{false};
    // Stencil along each direction.
    // in 2D, stencilZ is not initialized.
    amrex::Gpu::DeviceVector<amrex::Real> AMREX_D_DECL(m_stencilX, m_stencilY, m_stencilZ);
};

} // namespace Gempic::Filter

#endif // #ifndef GEMPIC_FILTER_H_

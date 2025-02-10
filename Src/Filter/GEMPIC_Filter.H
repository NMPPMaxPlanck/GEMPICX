/* Copyright 2019 Maxence Thevenet, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GEMPIC_FILTER_H_
#define GEMPIC_FILTER_H_

#include <algorithm>
#include <limits>

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

    void do_filter(const amrex::Box& tbx,
                   amrex::Array4<amrex::Real const> const& src,
                   amrex::Array4<amrex::Real> const& dst,
                   int scomp,
                   int dcomp,
                   int ncomp);

    /* \brief Apply stencil on MultiFab (CPU/GPU version, 1D/2D/3D).
     * \param dstmf Destination MultiFab
     * \param srcmf source MultiFab
     * \param scomp first component of srcmf on which the filter is applied
     * \param dcomp first component of dstmf on which the filter is applied
     * \param ncomp Number of components on which the filter is applied.
     */
    void apply_stencil(const amrex::MultiFab& srcmf,
                       amrex::MultiFab& dstmf,
                       int scomp = 0,
                       int dcomp = 0,
                       int ncomp = std::numeric_limits<int>::max());

protected:
    // If true, a filter is used to smooth charge and currents
    bool m_useFilter{false};
    // Stencil along each direction.
    // in 2D, stencilZ is not initialized.
    amrex::Gpu::DeviceVector<amrex::Real> AMREX_D_DECL(m_stencilX, m_stencilY, m_stencilZ);

private:
};

} // namespace Gempic::Filter

#endif // #ifndef GEMPIC_FILTER_H_
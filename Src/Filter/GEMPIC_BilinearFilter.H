/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GEMPIC_BILIN_FILTER_H_
#define GEMPIC_BILIN_FILTER_H_

#include <AMReX_Array.H>
#include <AMReX_Config.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Filter.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Filter
{
namespace Impl
{
// Only needs newS for optimization reasons that are a bit excessive
void convolve_filter(amrex::Vector<amrex::Real>& oldS,
                     amrex::Vector<amrex::Real>& newS,
                     amrex::Real const alpha,
                     int const ipass);

} // namespace Impl

class BilinearFilter : public Filter
{
public:
    BilinearFilter();

    void compute_stencils();

    // Filter parameter (0.5 for bilinear filters)
    amrex::Real const m_alpha{0.5};

    // Do a compensation step?
    bool m_compensate{false};

    amrex::Array<int, AMREX_SPACEDIM> m_nPass{AMREX_D_DECL(1, 1, 1)};
};
} // namespace Gempic::Filter
#endif // #ifndef GEMPIC_BILIN_FILTER_H_

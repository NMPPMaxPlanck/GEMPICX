/* Copyright 2019 Andrew Myers, Maxence Thevenet, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef GEMPIC_BILIN_FILTER_H_
#define GEMPIC_BILIN_FILTER_H_

#include <array>
#include <vector>

#include <AMReX_Array.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
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
void convolve_filter (amrex::Vector<amrex::Real> &oldS,
                      amrex::Vector<amrex::Real> &newS,
                      const amrex::Real alpha,
                      const int ipass)
{
    BL_PROFILE("Gempic::Filter::Impl::convolve_filter()");
    const amrex::Real beta{0.5 - 0.5 * alpha};

    // element 0 has to be treated in its own way
    newS[0] = alpha * oldS[0] + (1 - alpha) * oldS[1];
    // For each element j, apply the filter to oldS to get newS[j]
    for (int j{1}; j < ipass; j++)
    {
        newS[j] = alpha * oldS[j] + beta * (oldS[j - 1] + oldS[j + 1]);
    }
    // final element has to be treated in its own way
    newS[ipass] = alpha * oldS[ipass] + beta * oldS[ipass - 1];

    oldS = newS;
}

void compute_stencil (amrex::Gpu::DeviceVector<amrex::Real> &stencil,
                      const unsigned int npass,
                      const amrex::Real alpha,
                      const bool doCompensation)
{
    BL_PROFILE("Gempic::Filter::Impl::compute_stencil()");
    amrex::Vector<amrex::Real> oldS(1u + npass + static_cast<size_t>(doCompensation), 0.);
    amrex::Vector<amrex::Real> newS(oldS.size(), 0.);

    oldS[0] = 1.;
    // Convolve the filter with itself npass times
    int const lastpass = static_cast<int>(npass + 1u);
    for (int ipass{1}; ipass < lastpass; ipass++)
    {
        convolve_filter(oldS, newS, alpha, ipass);
    }

    if (doCompensation)
    {
        amrex::Real compensationAlpha{0.5 * npass};

        // Convolve using different alpha
        convolve_filter(oldS, newS, compensationAlpha, lastpass);
    }

    // we use oldS here to make sure the stencil is correct even when npass = 0
    oldS[0] *= 0.5;  // because we will use it twice
    stencil.resize(oldS.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, oldS.begin(), oldS.end(), stencil.begin());
    /// @todo: Device wide Gpu::synchronize() usually too excessive and might interfere with other
    /// libraries (e.g. MPI)
    /// https://amrex-codes.github.io/amrex/docs_html/GPU.html#stream-and-synchronization
    amrex::Gpu::synchronize();
}
}  // namespace Impl

class BilinearFilter : public Filter
{
public:
    BilinearFilter()
    {
        BL_PROFILE("Gempic::Filter::Impl::BilinearFilter::BilinearFilter()");
        Io::Parameters params("Filter", "class BilinearFilter");
        // Filter parameters
        // If true, a bilinear filter is used to smooth charge and currents
        params.get_or_set("enable", m_useFilter);
        if (m_useFilter)
        {
            params.get("nPass", m_nPass);

            // Do a compensation step?
            params.get_or_set("compensate", m_compensate);

            std::vector<int> nGhostVector;
            params.get("nGhost", nGhostVector);
            if (AMREX_D_TERM((nGhostVector[xDir] < (m_nPass[xDir] + m_compensate)),
                             || (nGhostVector[yDir] < (m_nPass[yDir] + m_compensate)),
                             || (nGhostVector[zDir] < (m_nPass[zDir] + m_compensate))))
            {
                std::cerr
                    << "Grid is not large enough to contain the filter stencil. Try increasing the "
                       "amount of extra ghost cells or decreasing the number of filter passes.\n";
                std::exit(Error::InvalidInput);
            }
            compute_stencils();
        }
    }

    void compute_stencils ()
    {
        BL_PROFILE("Gempic::Filter::Impl::BilinearFilter::compute_stencils()");
        AMREX_D_DECL(Impl::compute_stencil(m_stencilX, m_nPass[xDir], m_alpha, m_compensate),
                     Impl::compute_stencil(m_stencilY, m_nPass[yDir], m_alpha, m_compensate),
                     Impl::compute_stencil(m_stencilZ, m_nPass[zDir], m_alpha, m_compensate));
    }

    // Filter parameter (0.5 for bilinear filters)
    const amrex::Real m_alpha{0.5};

    // Do a compensation step?
    bool m_compensate{false};

    amrex::Array<int, GEMPIC_SPACEDIM> m_nPass{AMREX_D_DECL(1, 1, 1)};
};
}  // namespace Gempic::Filter
#endif  // #ifndef GEMPIC_BILIN_FILTER_H_

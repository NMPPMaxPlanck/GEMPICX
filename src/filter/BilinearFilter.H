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

#include "Filter.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_parameters.H"

namespace
{
void compute_stencil (amrex::Gpu::DeviceVector<amrex::Real> &stencil, unsigned int npass)
{
    amrex::Vector<amrex::Real> oldS(1u + npass, 0.);
    amrex::Vector<amrex::Real> newS(1u + npass, 0.);

    oldS.at(0) = 1.;
    // Convolve the filter with itself npass times
    int const lastpass = static_cast<int>(npass + 1u);
    for (int ipass = 1; ipass < lastpass; ipass++)
    {
        // element 0 has to be treated in its own way
        newS.at(0) = 0.5 * oldS.at(0) + 0.5 * oldS.at(1);
        amrex::Real loc = 0.;
        // For each element j, apply the filter to
        // old_s to get new_s[j]. loc stores the tmp
        // filtered value.
        for (int j = 1; j < ipass; j++)
        {
            loc = 0.5 * oldS[j];
            loc += 0.25 * oldS[j - 1];
            loc += 0.25 * oldS.at(j + 1);
            newS.at(j) = loc;
        }
        // final element has to be treated in its own way
        loc = 0.5 * oldS[ipass] + 0.25 * oldS[ipass - 1];
        newS.at(ipass) = loc;
        // copy new_s into old_s
        oldS = newS;
    }

    // we use old_s here to make sure the stencil is correct even when npass = 0
    oldS.at(0) *= 0.5;  // because we will use it twice
    stencil.resize(oldS.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, oldS.begin(), oldS.end(), stencil.begin());
    /// @todo: Device wide Gpu::synchronize() usually too excessive and might interfere with other
    /// libraries (e.g. MPI)
    /// https://amrex-codes.github.io/amrex/docs_html/GPU.html#stream-and-synchronization
    amrex::Gpu::synchronize();
}
}  // namespace

namespace Gempic
{
class BilinearFilter : public Filter
{
public:
    BilinearFilter()
    {
        Parameters params("filter", "class BilinearFilter");
        // Filter parameters
        // If true, a bilinear filter is used to smooth charge and currents
        m_useFilter = false;
        params.get_or_set("use_filter", m_useFilter);
        if (m_useFilter)
        {
            params.get("filter_npass_each_dir", m_npassEachDir);
            std::vector<int> nGhostVector;
            params.get("n_ghost", nGhostVector);
            if (AMREX_D_TERM((nGhostVector[xDir] < m_npassEachDir[xDir]),
                             || (nGhostVector[yDir] < m_npassEachDir[yDir]),
                             || (nGhostVector[zDir] < m_npassEachDir[zDir])))
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
        BL_PROFILE("BilinearFilter::ComputeStencils()");
        int i = 0;
        for (auto el : m_npassEachDir) m_stencilLengthEachDir[i++] = el;
        m_stencilLengthEachDir += 1.;

        // npass_each_dir = npass_z npass_x npass_y
        AMREX_D_DECL(m_stencilZ.resize(1u + m_npassEachDir[GEMPIC_SPACEDIM - 1]),
                     m_stencilX.resize(1u + m_npassEachDir[xDir]),
                     m_stencilY.resize(1u + m_npassEachDir[yDir]));
        AMREX_D_DECL(compute_stencil(m_stencilZ, m_npassEachDir[GEMPIC_SPACEDIM - 1]),
                     compute_stencil(m_stencilX, m_npassEachDir[xDir]),
                     compute_stencil(m_stencilY, m_npassEachDir[yDir]));

        m_slen = m_stencilLengthEachDir.dim3();
        GEMPIC_D_EXCL(m_slen.y = 1;, m_slen.z = 1;, )
    }

    amrex::Array<int, GEMPIC_SPACEDIM> m_npassEachDir{AMREX_D_DECL(1, 1, 1)};
};
}  // namespace Gempic
#endif  // #ifndef GEMPIC_BILIN_FILTER_H_

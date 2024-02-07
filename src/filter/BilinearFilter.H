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
#include "Filter.H"
#include "GEMPIC_parameters.H"

namespace
{
void compute_stencil(amrex::Gpu::DeviceVector<amrex::Real> &stencil, unsigned int npass)
{
    amrex::Vector<amrex::Real> old_s(1u + npass, 0.);
    amrex::Vector<amrex::Real> new_s(1u + npass, 0.);

    old_s.at(0) = 1.;
    // Convolve the filter with itself npass times
    int const lastpass = static_cast<int>(npass + 1u);
    for (int ipass = 1; ipass < lastpass; ipass++)
    {
        // element 0 has to be treated in its own way
        new_s.at(0) = 0.5 * old_s.at(0) + 0.5 * old_s.at(1);
        amrex::Real loc = 0.;
        // For each element j, apply the filter to
        // old_s to get new_s[j]. loc stores the tmp
        // filtered value.
        for (int j = 1; j < ipass; j++)
        {
            loc = 0.5 * old_s[j];
            loc += 0.25 * old_s[j - 1];
            loc += 0.25 * old_s.at(j + 1);
            new_s.at(j) = loc;
        }
        // final element has to be treated in its own way
        loc = 0.5 * old_s[ipass] + 0.25 * old_s[ipass - 1];
        new_s.at(ipass) = loc;
        // copy new_s into old_s
        old_s = new_s;
    }

    // we use old_s here to make sure the stencil is correct even when npass = 0
    old_s.at(0) *= 0.5;  // because we will use it twice
    stencil.resize(old_s.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, old_s.begin(), old_s.end(), stencil.begin());
    /// @todo: Device wide Gpu::synchronize() usually too excessive and might interfere with other libraries (e.g. MPI) https://amrex-codes.github.io/amrex/docs_html/GPU.html#stream-and-synchronization
    amrex::Gpu::synchronize();
}
}  // namespace

namespace Gempic {
class BilinearFilter : public Filter
{
public:

    BilinearFilter()
    {
        Parameters params("filter", "class BilinearFilter");
        // Filter parameters
        // If true, a bilinear filter is used to smooth charge and currents
        use_filter = false;
        params.getOrSet("use_filter", use_filter);
        if (use_filter)
        {
            params.get("filter_npass_each_dir", npass_each_dir);
            std::vector<int> nGhostVector;
            params.get("n_ghost", nGhostVector);
            if (AMREX_D_TERM((nGhostVector[xDir] < npass_each_dir[xDir]),
                          || (nGhostVector[yDir] < npass_each_dir[yDir]),
                          || (nGhostVector[zDir] < npass_each_dir[zDir])))
            {
                std::cerr << "Grid is not large enough to contain the filter stencil. Try increasing the amount of extra ghost cells or decreasing the number of filter passes.\n";
                std::exit(Error::InvalidInput);
            }
            ComputeStencils();
        }
    }

    void ComputeStencils()
    {
        BL_PROFILE("BilinearFilter::ComputeStencils()");
        int i = 0;
        for (auto el : npass_each_dir) stencil_length_each_dir[i++] = el;
        stencil_length_each_dir += 1.;

        // npass_each_dir = npass_z npass_x npass_y
        AMREX_D_DECL(
            stencil_z.resize(1u + npass_each_dir[GEMPIC_SPACEDIM - 1]),
            stencil_x.resize(1u + npass_each_dir[xDir]),
            stencil_y.resize(1u + npass_each_dir[yDir]));
        AMREX_D_DECL(
            compute_stencil(stencil_z, npass_each_dir[GEMPIC_SPACEDIM - 1]),
            compute_stencil(stencil_x, npass_each_dir[xDir]),
            compute_stencil(stencil_y, npass_each_dir[yDir]));

        slen = stencil_length_each_dir.dim3();
        GEMPIC_D_EXCL(
            slen.y = 1;,
            slen.z = 1;,)
    }

    amrex::Array<int, GEMPIC_SPACEDIM> npass_each_dir{AMREX_D_DECL(1, 1, 1)};
};
} // namespace Gempic
#endif  // #ifndef GEMPIC_BILIN_FILTER_H_

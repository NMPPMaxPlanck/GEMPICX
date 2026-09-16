/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#if defined(GEMPIC_USE_MKL)
#include <mkl.h>
#include <mkl_lapacke.h>
#elif defined(GEMPIC_USE_LAPACK)
#include <lapacke.h>
#endif
#include "GEMPIC_NumTools.H"

namespace Gempic
{
/**
 *  a modified OverlapMask
 *
 * **/
std::unique_ptr<amrex::MultiFab> get_shared_bnd_mask (amrex::MultiFab& thisMF,
                                                      amrex::Periodicity const& period)
{
    amrex::BoxArray const& ba = thisMF.boxArray();
    amrex::DistributionMapping const& dm = thisMF.DistributionMap();

    auto p = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0, amrex::MFInfo(), thisMF.Factory());

    std::vector<amrex::IntVect> const& pshifts = period.shiftIntVect();

    amrex::Vector<amrex::Array4BoxTag<amrex::Real>> tags;

    bool runOnGpu = amrex::Gpu::inLaunchRegion();
    amrex::ignore_unused(runOnGpu, tags);
    {
        std::vector<std::pair<int, amrex::Box>> isects;

        for (amrex::MFIter mfi(*p); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = (*p)[mfi].box();
            amrex::Array4<amrex::Real> const& arr = p->array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k, { arr(i, j, k) = amrex::Real(0.0); });

            for (auto const& iv : pshifts)
            {
                ba.intersections(bx + iv, isects);
                for (auto const& is : isects)
                {
                    amrex::Box const& b = is.second - iv;
#ifdef AMREX_USE_GPU
                    if (runOnGpu)
                    {
                        tags.push_back({arr, b});
                    }
                    else
#endif
                    {
                        amrex::LoopConcurrentOnCpu(b, [=] (int i, int j, int k) noexcept
                                                   { arr(i, j, k) = amrex::Real(1.0); });
                    }
                }
            }
        }
    }

#ifdef AMREX_USE_GPU
    amrex::ParallelFor(tags, 1,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k, int n,
                                            amrex::Array4BoxTag<amrex::Real> const& tag) noexcept
                       { tag.dfab(i, j, k, n) = amrex::Real(1.0); });
#endif

    return p;
}

/**
 * @brief overload from
template \<typename dataStruct\>
void sum_boundary_sync (amrex::FabArray<amrex::BaseFab<dataStruct>>& thisMF,
                            amrex::Periodicity const& period)
 */
void sum_boundary_sync (amrex::MultiFab& thisMF, amrex::Periodicity const& period)
{
    if (thisMF.ixType().cellCentered())
    {
        return;
    }

    int const ncomp = thisMF.nComp();

    amrex::MultiFab tmpmf(thisMF.boxArray(), thisMF.DistributionMap(), ncomp, 0, amrex::MFInfo(),
                          thisMF.Factory());
    tmpmf.setVal(amrex::Real(0.0));
    tmpmf.ParallelCopy(thisMF, period, amrex::FabArrayBase::ADD);
    thisMF.ParallelCopy(tmpmf, period);
    return;
}

/***
 * modified from WeightedSync
 ***/
// from amrex::Add
//template <class FAB, class bar = std::enable_if_t<IsBaseFab<FAB>::value>>
void mult_and_add (amrex::Real const dstVal,
                   amrex::MultiFab& dst,
                   amrex::Real const srcVal,
                   amrex::MultiFab const& src,
                   int srccomp,
                   int dstcomp,
                   int numcomp,
                   amrex::IntVect const& nghost)
{
    //BL_PROFILE("amrex::Add()");

#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion() && dst.isFusingCandidate())
    {
        auto const& dstfa = dst.arrays();
        auto const& srcfa = src.const_arrays();
        amrex::ParallelFor(dst, nghost, numcomp,
                           [=] AMREX_GPU_DEVICE(int boxNo, int i, int j, int k, int n) noexcept
                           {
                               dstfa[boxNo](i, j, k, n + dstcomp) =
                                   dstVal * dstfa[boxNo](i, j, k, n + dstcomp) +
                                   srcVal * srcfa[boxNo](i, j, k, n + srccomp);
                           });
        if (!amrex::Gpu::inNoSyncRegion())
        {
            amrex::Gpu::streamSynchronize();
        }
    }
    else
#endif
    {
        for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.growntilebox(nghost);
            if (bx.ok())
            {
                auto const srcFab = src.array(mfi);
                auto dstFab = dst.array(mfi);
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, numcomp, i, j, k, n, {
                    dstFab(i, j, k, n + dstcomp) = std::fma(srcVal, srcFab(i, j, k, n + srccomp),
                                                            dstVal * dstFab(i, j, k, n + dstcomp));
                });
            }
        }
    }
}

// from amrex::Add
//template <class FAB, class bar = std::enable_if_t<IsBaseFab<FAB>::value>>
void mult_and_add (amrex::MultiFab& dst,
                   amrex::Real const srcVal,
                   amrex::MultiFab const& src,
                   int srccomp,
                   int dstcomp,
                   int numcomp,
                   amrex::IntVect const& nghost)
{
    //BL_PROFILE("amrex::Add()");
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion() && dst.isFusingCandidate())
    {
        auto const& dstfa = dst.arrays();
        auto const& srcfa = src.const_arrays();
        amrex::ParallelFor(dst, nghost, numcomp,
                           [=] AMREX_GPU_DEVICE(int boxNo, int i, int j, int k, int n) noexcept
                           {
                               dstfa[boxNo](i, j, k, n + dstcomp) =
                                   dstfa[boxNo](i, j, k, n + dstcomp) +
                                   srcVal * srcfa[boxNo](i, j, k, n + srccomp);
                           });
        if (!amrex::Gpu::inNoSyncRegion())
        {
            amrex::Gpu::streamSynchronize();
        }
    }
    else
#endif
    {
        for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.growntilebox(nghost);
            if (bx.ok())
            {
                auto const srcFab = src.array(mfi);
                auto dstFab = dst.array(mfi);
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, numcomp, i, j, k, n, {
                    dstFab(i, j, k, n + dstcomp) += srcVal * srcFab(i, j, k, n + srccomp);
                });
            }
        }
    }
}

// from Real MultiFab::Dot( const MultiFab& x, int xcomp, const MultiFab& y, int ycomp, int numcomp,
// int nghost, bool local)
amrex::Real multi_fab_wgt_dot (amrex::MultiFab const& wgt,
                               amrex::MultiFab const& x,
                               int xcomp,
                               amrex::MultiFab const& y,
                               int ycomp,
                               int numcomp,
                               int nghost,
                               bool local)
{
    return Gempic::wgt_dot(wgt, x, xcomp, y, ycomp, numcomp, amrex::IntVect(nghost), local);
}

// from Real MultiFab::Dot(const MultiFab& x, int xcomp, int numcomp, int nghost, bool local)
amrex::Real multi_fab_wgt_dot (amrex::MultiFab const& wgt,
                               amrex::MultiFab const& x,
                               int xcomp,
                               int numcomp,
                               int nghost,
                               bool local)
{
    AMREX_ASSERT(x.nGrowVect().allGE(nghost));

    BL_PROFILE("multi_fab_wgt_dot()");

    auto sm = amrex::Real(0.0);
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion())
    {
        auto const& xma = x.const_arrays();
        auto const& wgtma = wgt.const_arrays();
        sm = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real>{},
                              x, amrex::IntVect (nghost),
                              [=] AMREX_GPU_DEVICE(int boxNo, int i, int j,
                                                   int k) noexcept -> amrex::GpuTuple<amrex::Real>
                              {
                                  auto t = amrex::Real(0.0);
                                  auto const& xfab = xma[boxNo];
                                  auto const& wgtfab = wgtma[boxNo];
                                  for (int n = 0; n < numcomp; ++n)
                                  {
                                      t += wgtfab(i, j, k, n) * xfab(i, j, k, xcomp + n) *
                                           xfab(i, j, k, xcomp + n);
                                  }
                                  return t;
                              });
    }
    else
#endif
    {
        for (amrex::MFIter mfi(x, true); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.tilebox(); //mfi.growntilebox(nghost);
            amrex::Array4<amrex::Real const> const& xfab = x.const_array(mfi);
            amrex::Array4<amrex::Real const> const& wgtfab = wgt.const_array(mfi);
            AMREX_LOOP_4D(bx, numcomp, i, j, k, n, {
                sm += wgtfab(i, j, k, n) * xfab(i, j, k, xcomp + n) * xfab(i, j, k, xcomp + n);
            });
        }
    }

    if (!local)
    {
        amrex::ParallelAllReduce::Sum(sm, amrex::ParallelContext::CommunicatorSub());
    }

    return sm;
}

} //namespace Gempic
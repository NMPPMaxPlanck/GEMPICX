#include "GEMPIC_NumTools.H"

/**
 *  a modified OverlapMask
 *
 * **/
std::unique_ptr<amrex::MultiFab> get_shared_bnd_mask (amrex::MultiFab& thisMF,
                                                      const amrex::Periodicity& period)
{
    const amrex::BoxArray& ba = thisMF.boxArray();
    const amrex::DistributionMapping& dm = thisMF.DistributionMap();

    auto p = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0, amrex::MFInfo(), thisMF.Factory());

    const std::vector<amrex::IntVect>& pshifts = period.shiftIntVect();

    amrex::Vector<amrex::Array4BoxTag<amrex::Real>> tags;

    bool runOnGpu = amrex::Gpu::inLaunchRegion();
    amrex::ignore_unused(runOnGpu, tags);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!runOnGpu)
#endif
    {
        std::vector<std::pair<int, amrex::Box>> isects;

        for (amrex::MFIter mfi(*p); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = (*p)[mfi].box();
            amrex::Array4<amrex::Real> const& arr = p->array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k, { arr(i, j, k) = amrex::Real(0.0); });

            for (const auto& iv : pshifts)
            {
                ba.intersections(bx + iv, isects);
                for (const auto& is : isects)
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
                       {
                           amrex::Real* p = tag.dfab.ptr(i, j, k, n);
                           amrex::Gpu::Atomic::AddNoRet(p, amrex::Real(1.0));
                       });
#endif

    return p;
}

/***
 * modified from WeightedSync
 ***/
void sum_boundary_sync (amrex::MultiFab& thisMF, const amrex::Periodicity& period)
{
    if (thisMF.ixType().cellCentered())
    {
        return;
    }

    auto wgt = get_shared_bnd_mask(thisMF, period);

    const int ncomp = thisMF.nComp();

    amrex::MultiFab tmpmf(thisMF.boxArray(), thisMF.DistributionMap(), ncomp, 0, amrex::MFInfo(),
                          thisMF.Factory());
    tmpmf.setVal(amrex::Real(0.0));
    tmpmf.ParallelCopy(thisMF, period, amrex::FabArrayBase::ADD);

    amrex::MultiFab::Copy(thisMF, tmpmf, 0, 0, ncomp, 0);
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
                   const amrex::IntVect& nghost)
{
    //BL_PROFILE("amrex::Add()");

#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion() && dst.isFusingCandidate())
    {
        auto const& dstfa = dst.arrays();
        auto const& srcfa = src.const_arrays();
        amrex::ParallelFor(dst, nghost, numcomp,
                           [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept
                           {
                               dstfa[box_no](i, j, k, n + dstcomp) =
                                   dstVal * dstfa[box_no](i, j, k, n + dstcomp) +
                                   srcVal * srcfa[box_no](i, j, k, n + srccomp);
                           });
        if (!amrex::Gpu::inNoSyncRegion())
        {
            amrex::Gpu::streamSynchronize();
        }
    }
    else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox(nghost);
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
                   const amrex::IntVect& nghost)
{
    //BL_PROFILE("amrex::Add()");
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion() && dst.isFusingCandidate())
    {
        auto const& dstfa = dst.arrays();
        auto const& srcfa = src.const_arrays();
        amrex::ParallelFor(dst, nghost, numcomp,
                           [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept
                           {
                               dstfa[box_no](i, j, k, n + dstcomp) =
                                   dstfa[box_no](i, j, k, n + dstcomp) +
                                   srcVal * srcfa[box_no](i, j, k, n + srccomp);
                           });
        if (!amrex::Gpu::inNoSyncRegion())
        {
            amrex::Gpu::streamSynchronize();
        }
    }
    else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.growntilebox(nghost);
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

// from
// template <typename FAB, std::enable_if_t<IsBaseFab<FAB>::value, int> FOO = 0>
// typename FAB::value_type Dot (FabArray<FAB> const& x,
//                               int xcomp,
//                               FabArray<FAB> const& y,
//                               int ycomp,
//                               int ncomp,
//                               IntVect const& nghost,
//                               bool local = false)
/**
 * \brief Compute dot products of two FabArrays
 *
 * \param wgt    weight (single component) FabArray
 * \param x      first FabArray
 * \param xcomp  starting component of x
 * \param y      second FabArray
 * \param ycomp  starting component of y
 * \param ncomp  number of components
 * \param nghost number of ghost cells
 * \param local  If true, MPI communication is skipped.
 */
template <typename FAB, std::enable_if_t<amrex::IsBaseFab<FAB>::value, int> foo>
typename FAB::value_type wgt_dot (amrex::FabArray<FAB> const& wgt,
                                  amrex::FabArray<FAB> const& x,
                                  int xcomp,
                                  amrex::FabArray<FAB> const& y,
                                  int ycomp,
                                  int ncomp,
                                  amrex::IntVect const& nghost,
                                  bool local)
{
    BL_ASSERT(x.boxArray() == y.boxArray());
    BL_ASSERT(x.DistributionMap() == y.DistributionMap());
    BL_ASSERT(x.nGrowVect().allGE(nghost) && y.nGrowVect().allGE(nghost));

    BL_PROFILE("amrex::Dot()");

    using T = typename FAB::value_type;
    auto sm = T(0.0);
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion())
    {
        auto const& xma = x.const_arrays();
        auto const& yma = y.const_arrays();
        auto const& wgtma = wgt.const_arrays();
        sm = ParReduce(
            amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<T>{}, x, nghost,
            [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::GpuTuple<T>
            {
                auto t = T(0.0);
                auto const& xfab = xma[box_no];
                auto const& yfab = yma[box_no];
                auto const& wgtfab = wgtma[box_no];
                for (int n = 0; n < ncomp; ++n)
                {
                    t += wgtfab(i, j, k, xcomp + n) * xfab(i, j, k, xcomp + n) *
                         yfab(i, j, k, ycomp + n);
                }
                return t;
            });
    }
    else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+ : sm)
#endif
        for (amrex::MFIter mfi(x, true); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.tilebox(); //growntilebox(nghost);
            auto const& xfab = x.const_array(mfi);
            auto const& yfab = y.const_array(mfi);
            auto const& wgtfab = wgt.const_array(mfi);
            auto smLoc = T(0.0);

            AMREX_LOOP_4D(bx, ncomp, i, j, k, n, {
                smLoc += wgtfab(i, j, k, n) * xfab(i, j, k, xcomp + n) * yfab(i, j, k, ycomp + n);
            });
            sm += smLoc;
        }
    }

    if (!local)
    {
        amrex::ParallelAllReduce::Sum(sm, amrex::ParallelContext::CommunicatorSub());
    }

    return sm;
}

// from Real MultiFab::Dot( const MultiFab& x, int xcomp, const MultiFab& y, int ycomp, int numcomp,
// int nghost, bool local)
amrex::Real multi_fab_wgt_dot (const amrex::MultiFab& wgt,
                               const amrex::MultiFab& x,
                               int xcomp,
                               const amrex::MultiFab& y,
                               int ycomp,
                               int numcomp,
                               int nghost,
                               bool local)
{
    return wgt_dot(wgt, x, xcomp, y, ycomp, numcomp, amrex::IntVect(nghost), local);
}

// from Real MultiFab::Dot(const MultiFab& x, int xcomp, int numcomp, int nghost, bool local)
amrex::Real multi_fab_wgt_dot (const amrex::MultiFab& wgt,
                               const amrex::MultiFab& x,
                               int xcomp,
                               int numcomp,
                               int nghost,
                               bool local)
{
    BL_ASSERT(x.nGrowVect().allGE(nghost));

    BL_PROFILE("MultiFab::Dot()");

    auto sm = amrex::Real(0.0);
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion())
    {
        auto const& xma = x.const_arrays();
        auto const& wgtma = wgt.const_arrays();
        sm = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real>{},
                              x, amrex::IntVect (nghost),
                              [=] AMREX_GPU_DEVICE(int box_no, int i, int j,
                                                   int k) noexcept -> amrex::GpuTuple<amrex::Real>
                              {
                                  amrex::Real t = amrex::Real(0.0);
                                  auto const& xfab = xma[box_no];
                                  auto const& wgtfab = wgtma[box_no];
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+ : sm)
#endif
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

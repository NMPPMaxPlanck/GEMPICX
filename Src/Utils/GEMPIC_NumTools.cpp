/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include "GEMPIC_NumTools.H"

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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!runOnGpu)
#endif
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
 * @ref sum_boundary_sync (amrex::FabArray<amrex::BaseFab<dataStruct>>& thisMF,
 *                         amrex::Periodicity const& period)
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
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
    AMREX_ASSERT(x.boxArray() == y.boxArray());
    AMREX_ASSERT(x.DistributionMap() == y.DistributionMap());
    AMREX_ASSERT(x.nGrowVect().allGE(nghost) && y.nGrowVect().allGE(nghost));

    BL_PROFILE("wgt_dot()");

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
            [=] AMREX_GPU_DEVICE(int boxNo, int i, int j, int k) noexcept -> amrex::GpuTuple<T>
            {
                auto t = T(0.0);
                auto const& xfab = xma[boxNo];
                auto const& yfab = yma[boxNo];
                auto const& wgtfab = wgtma[boxNo];
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
amrex::Real multi_fab_wgt_dot (amrex::MultiFab const& wgt,
                               amrex::MultiFab const& x,
                               int xcomp,
                               amrex::MultiFab const& y,
                               int ycomp,
                               int numcomp,
                               int nghost,
                               bool local)
{
    return wgt_dot(wgt, x, xcomp, y, ycomp, numcomp, amrex::IntVect(nghost), local);
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

#ifdef USE_MKL
#include <vector>

double ConditionNumber (std::vector<std::vector<double>> const& A, int N)
{
    std::vector<double> flatA(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) flatA[i * N + j] = A[i][j];

    int info;
    std::vector<int> ipiv(N);
    dgetrf(&N, &N, flatA.data(), &N, ipiv.data(), &info);

    double normA = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', N, N, flatA.data(), N);
    double normInvA;

    // Allocate workspace for inversion
    int lwork = N * N;
    std::vector<double> work(lwork);

    // Compute the inverse
    dgetri(&N, flatA.data(), &N, ipiv.data(), work.data(), &lwork, &info);
    normInvA = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', N, N, flatA.data(), N);

    return normA * normInvA;
}

void matrix_inverseMKL (std::vector<std::vector<double>>& iA,
                        std::vector<std::vector<double>> const& A,
                        int const N)
{
    // Flatten the matrix A into a one-dimensional array for MKL
    std::vector<double> flatA(N * N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            flatA[i * N + j] = A[i][j];
        }
    }

    std::vector<int> ipiv(N);
    int info;

    // Perform LU decomposition
    dgetrf(&N, &N, flatA.data(), &N, ipiv.data(), &info);
    if (info != 0)
    {
        std::cerr << "Error in LU decomposition: " << info << std::endl;
        return;
    }

    // Allocate workspace for inversion
    int lwork = N * N;
    std::vector<double> work(lwork);

    // Compute the inverse
    dgetri(&N, flatA.data(), &N, ipiv.data(), work.data(), &lwork, &info);
    if (info != 0)
    {
        std::cerr << "Error in matrix inversion: " << info << std::endl;
        return;
    }

    // Reshape the flattened array back to 2D iA
    iA.resize(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            iA[i][j] = flatA[i * N + j];
        }
    }
}
#endif // USE_MKL

/**
 * Compute matrix inverse with Gauss-Jordan Elimination.
 *
 * \param iA
 * \param A
 * \param N
 */
void matrix_inverse (std::vector<std::vector<double>>& iA,
                     std::vector<std::vector<double>> const& A,
                     int const N)
{
    // Initialize C matrix
    std::vector<std::vector<double>> C(N, std::vector<double>(2 * N, 0.0));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i][j] = A[i][j];
        }
        C[i][N + i] = 1.0;
    }

    // Forward elimination and row swapping (if necessary)
    for (int i = 0; i < N; ++i)
    {
        //If pivot element is zero, then swap rows
        int ml = 0;
        for (int k = i + 1; k < N; ++k)
        {
            if (std::abs(C[k][i]) > std::abs(C[i + ml][i]))
            {
                ml = k - i;
            }
        }
        int iswap = i + ml;
        if (iswap != i) std::swap(C[i], C[iswap]);
        if (C[i][i] == 0.0)
        {
            std::cerr << "ERROR. Matrix is singular!" << std::endl;
            for (auto const& row : A)
            {
                for (auto const& val : row)
                {
                    std::cerr << val << " ";
                }
                std::cerr << std::endl;
            }
            std::exit(1);
        }
        double piv = 1.0 / C[i][i];
        for (int l = 0; l < 2 * N; ++l)
        {
            C[i][l] *= piv;
        }
        for (int j = i + 1; j < N; ++j)
        {
            double factor = C[j][i];
            for (int l = 0; l < 2 * N; ++l)
            {
                C[j][l] -= factor * C[i][l];
            }
        }
    }

    // Back substitution
    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = i - 1; j >= 0; --j)
        {
            double factor = C[j][i];
            for (int l = 0; l < 2 * N; ++l)
            {
                C[j][l] -= factor * C[i][l];
            }
        }
    }

    // Compute inverse matrix
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            iA[i][j] = C[i][N + j];
        }
    }
}

void matrix_inverse_ld (std::vector<std::vector<my_precision>>& iA,
                        std::vector<std::vector<my_precision>> const& A,
                        int const N)
{
#if defined(DEBUG) || defined(_DEBUG)
    // Assertions to validate inputs
    BL_ASSERT(N > 0);                              // N must be positive.
    BL_ASSERT(A.size() == static_cast<size_t>(N)); // A must have N rows.
    for (auto const& row : A)
    {
        BL_ASSERT(row.size() == static_cast<size_t>(N)); // Each row in A must have N columns.
    }

    BL_ASSERT(iA.size() == static_cast<size_t>(N)); // iA must have N rows.
    for (auto const& row : iA)
    {
        BL_ASSERT(row.size() == static_cast<size_t>(N)); // Each row in iA must have N columns.
    }
#endif
    // Initialize C matrix
    std::vector<std::vector<my_precision>> C(N, std::vector<my_precision>(2 * N, 0.0));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i][j] = A[i][j];
        }
        C[i][N + i] = 1.0;
    }

    // Forward elimination and row swapping (if necessary)
    for (int i = 0; i < N; ++i)
    {
        //If pivot element is zero, then swap rows
        int ml = 0;
        for (int k = i + 1; k < N; ++k)
        {
            if (std::abs(C[k][i]) > std::abs(C[i + ml][i]))
            {
                ml = k - i;
            }
        }
        int iswap = i + ml;
        if (iswap != i) std::swap(C[i], C[iswap]);
        if (C[i][i] == 0.0)
        {
            std::cerr << "ERROR. Matrix is singular!" << std::endl;
            for (auto const& row : A)
            {
                for (auto const& val : row)
                {
                    std::cerr << val << " ";
                }
                std::cerr << std::endl;
            }
            std::exit(1);
        }
        my_precision piv = 1.0 / C[i][i];
        for (int l = 0; l < 2 * N; ++l)
        {
            C[i][l] *= piv;
        }
        for (int j = i + 1; j < N; ++j)
        {
            my_precision factor = C[j][i];
            for (int l = 0; l < 2 * N; ++l)
            {
                C[j][l] -= factor * C[i][l];
            }
        }
    }

    // Back substitution
    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = i - 1; j >= 0; --j)
        {
            my_precision factor = C[j][i];
            for (int l = 0; l < 2 * N; ++l)
            {
                C[j][l] -= factor * C[i][l];
            }
        }
    }

    // Compute inverse matrix
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            iA[i][j] = C[i][N + j];
        }
    }
}

#ifdef USE_MKL

void solve_system_mkl (std::vector<std::vector<double>> const& M,
                       std::vector<std::vector<double>> const& K,
                       std::vector<std::vector<double>>& A,
                       int N)
{
    // Flatten the matrices M and K for MKL
    std::vector<double> flatM(N * N);
    std::vector<double> flatK(N * N);
    std::vector<double> flatA(N * N);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            flatM[i * N + j] = M[i][j];
            flatK[i * N + j] = K[i][j];
        }
    }

    int info;
    std::vector<int> ipiv(N);

    // Perform LU decomposition on M
    dgetrf(&N, &N, flatM.data(), &N, ipiv.data(), &info);
    if (info != 0)
    {
        std::cerr << "Error in LU decomposition: " << info << std::endl;
        return;
    }

    // Solve the system M * A = K for A
    dgetrs("N", &N, &N, flatM.data(), &N, ipiv.data(), flatK.data(), &N, &info);
    if (info != 0)
    {
        std::cerr << "Error in solving the system: " << info << std::endl;
        return;
    }

    // Copy the solution from flatK to flatA
    std::copy(flatK.begin(), flatK.end(), flatA.begin());

    // Reshape the flattened solution array back to 2D A
    A.resize(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i][j] = flatA[i * N + j];
        }
    }
}

#endif

AMREX_GPU_HOST_DEVICE amrex::Real minmod (amrex::Real const a, amrex::Real const b)
{
    if (a * b <= 0)
    {
        return 0.0;
    }
    else
    {
        amrex::Real absb = std::abs(b);
        amrex::Real absa = std::abs(a);
        if (absa < absb)
        {
            return a;
        }
        else
        {
            return b;
        }
    }
}

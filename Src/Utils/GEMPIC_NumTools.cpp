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
template <typename dataStruct>
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

/**
 * computes n Gauss-Legendre quadrature nodes x in [x1,x2] and the corresponding quadrature weights
 * w this comes directly from numerical recipies, pdf free available on the website. this comes
 * directly from numerical recipies, pdf free available on the website.
 *
 * \param x1
 * \param x2
 * \param x
 * \param w
 */
void gauleg (std::vector<amrex::Real>& x,
             std::vector<amrex::Real>& w,
             amrex::Real const x1,
             amrex::Real const x2,
             int n)
{
    // Ensure valid inputs
    BL_ASSERT(n > 0);                              // n must be greater than zero.
    BL_ASSERT(x.size() == static_cast<size_t>(n)); // x must have size n.
    BL_ASSERT(w.size() == static_cast<size_t>(n)); // w must have size n.
    // Given the lower and upper limits of integration x1 and x2, this routine returns arrays
    // x[0..n-1] and w[0..n-1] of length n, containing the abscissas and weights of the
    // Gauss-Legendre n-point quadrature formula.
    constexpr amrex::Real eps = 1.0e-15; // EPS is the relative precision.
    amrex::Real z1, z, xm, xl, pp, p3, p2, p1;
    int m = (n + 1) / 2;  // The roots are symmetric in the interval, so
    xm = 0.5 * (x2 + x1); // we only have to find half of them.
    xl = 0.5 * (x2 - x1);
    amrex::Real const factor = M_PI / (n + 0.5);
    for (int i = 0; i < m; i++)
    {                                 // Loop over the desired roots.
        z = cos(factor * (i + 0.75)); // Starting with this approximation to the ith
                                      // root, we enter the main
                                      // loop of refinement
        do
        {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 0; j < n; j++)
            {            // Loop up the recurrence relation to get the
                p3 = p2; // Legendre polynomial evaluated at z.
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
            }
            // p1 is now the desired Legendre polynomial.We next compute pp, its derivative,
            //     by a standard relation involving also p2,
            //     the polynomial of one lower order.
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp; // Newton's method.
        } while (std::abs(z - z1) > eps);
        x[i] = xm - xl * z;                          // Scale the root to the desired interval,
        x[n - 1 - i] = xm + xl * z;                  // and put in its symmetric counterpart.
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp); // Compute the weight
        w[n - 1 - i] = w[i];                         // and its symmetric counterpart.
    }
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

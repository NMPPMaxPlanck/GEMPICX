/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_NUMTOOLS_H_
#define GEMPIC_NUMTOOLS_H_

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#if defined(GEMPIC_USE_MKL)
#include <mkl.h>
#include <mkl_lapacke.h>
#elif defined(GEMPIC_USE_LAPACK)
#include <lapacke.h>
#endif
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_FabArray.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

#include "GEMPIC_Config.H"

namespace Gempic
{

/**
 * @brief amrex::GpuArray<T,N> does not provide a std::tuple_size like std::aarray
 * @tparam T
 */
template <typename T>
struct get_array_size;

template <typename T, unsigned int n>
struct get_array_size<amrex::GpuArray<T, n>>
{
    static constexpr unsigned int s_myValue = n;
};

template <typename T, std::size_t n>
struct get_array_size<std::array<T, n>>
{
    static constexpr std::size_t s_myValue = n;
};

std::unique_ptr<amrex::MultiFab> get_shared_bnd_mask(amrex::MultiFab& thisMF,
                                                     amrex::Periodicity const& period);

/**
 * @brief like SumBoundary or WeightedSync, but for amrex::FabArray<amrex::BaseFab<dataStruct>>
 * objects.
 * @tparam DataStruct a data structure having a fill(amrex::Real) function, and an associated
 * amrex::FabArrayBase::ADD function
 * @param thisMF a FabArray of BaseFab of dataStruc. (like a generalized MultiFab)
 * @param period amrex::Periodicity
 */
template <typename DataStruct>
void sum_boundary_sync (amrex::FabArray<amrex::BaseFab<DataStruct>>& thisMF,
                        amrex::Periodicity const& period)
{
    if (thisMF.ixType().cellCentered())
    {
        return;
    }
    int const ncomp = thisMF.nComp();

    amrex::FabArray<amrex::BaseFab<DataStruct>> tmpmf(thisMF.boxArray(), thisMF.DistributionMap(),
                                                      ncomp, 0, amrex::MFInfo(), thisMF.Factory());

    DataStruct locdof0;
    locdof0.fill(0.0); // we assume dataStruct has a fill(amrex::Real) function.
    tmpmf.setVal(locdof0);
    tmpmf.ParallelCopy(thisMF, period, amrex::FabArrayBase::ADD);
    thisMF.ParallelCopy(tmpmf, period);

    return;
}

void sum_boundary_sync(amrex::MultiFab& thisMF, amrex::Periodicity const& period);

/**
 * multidof (AoS) MultiFab to multicomponents (SoA) MultiFab transformation
 * **/
template <typename DataStruct>
void aos2soa_fab (amrex::MultiFab& soaMFA, amrex::FabArray<amrex::BaseFab<DataStruct>>& aosMFA)
{
    constexpr unsigned int size = get_array_size<DataStruct>::s_myValue;
    BL_ASSERT(soaMFA.nComp() == aosMFA.nComp() * size);
    for (amrex::MFIter mfi(aosMFA); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx =
            mfi.growntilebox(); // cache blocking (for cpus) loops over valid boxes

        auto const& soa = (soaMFA).array(mfi);
        auto const& aos = (aosMFA).array(mfi);
        int nComp{aosMFA.nComp()};
        ParallelFor(bx, nComp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        for (int idof = 0; idof < size; ++idof)
                        {
                            int idofcomp = n * size + idof;
                            soa(i, j, k, idofcomp) = aos(i, j, k, n)[idof];
                        }
                    });
    }
}
/**
 * multidof (AoS) MultiFab to multicomponents (SoA) MultiFab transformation
 * **/
template <typename DataStruct, int nDim>
void aos2soa_fab (amrex::Array<amrex::MultiFab, nDim>& soaMFA,
                  amrex::Array<amrex::FabArray<amrex::BaseFab<DataStruct>>, nDim>& aosMFA)
{
    // using signed int because `nComp` always returns an `int`
    constexpr int size = int(get_array_size<DataStruct>::s_myValue);
    BL_ASSERT(soaMFA[xDir].nComp() == aosMFA[xDir].nComp() * size);
#if nDim > 1
    BL_ASSERT(soaMFA[yDir].nComp() == aosMFA[yDir].nComp() * size);
    BL_ASSERT(aosMFA[yDir].nComp() == aosMFA[xDir].nComp());
#endif
#if nDim > 2
    BL_ASSERT(soaMFA[zDir].nComp() == aosMFA[zDir].nComp() * size);
    BL_ASSERT(aosMFA[zDir].nComp() == aosMFA[yDir].nComp());
#endif

    for (int iDir = 0; iDir < nDim; ++iDir)
    {
        for (amrex::MFIter mfi(aosMFA[iDir]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx =
                mfi.growntilebox(); // cache blocking (for cpus) loops over valid boxes

            auto const& soa = (soaMFA[iDir]).array(mfi);
            auto const& aos = (aosMFA[iDir]).array(mfi);
            int nComp{aosMFA[iDir].nComp()};
            ParallelFor(bx, nComp,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            for (int idof = 0; idof < size; ++idof)
                            {
                                int idofcomp = n * size + idof;
                                soa(i, j, k, idofcomp) = aos(i, j, k, n)[idof];
                            };
                        });
        }
    }
}

/**
 * @brief multicomponents (SoA) MultiFab to multidof (AoS) MultiFab transformation
 * @tparam dataStruct the data struct, e.g. a GpuArray...
 * @param aosMFA input FabArray of BaseFab of dataStruct
 * @param soaMFA output multicomponent MultiFab
 */
template <typename DataStruct>
void soa2aos_fab (amrex::FabArray<amrex::BaseFab<DataStruct>>& aosMFA, amrex::MultiFab& soaMFA)
{
    constexpr unsigned int size = get_array_size<DataStruct>::s_myValue;
    BL_ASSERT(soaMFA.nComp() == aosMFA.nComp() * size);
    for (amrex::MFIter mfi(aosMFA); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx =
            mfi.growntilebox(); // cache blocking (for cpus) loops over valid boxes

        auto const& soa = (soaMFA).array(mfi);
        auto const& aos = (aosMFA).array(mfi);
        int nComp{aosMFA.nComp()};
        ParallelFor(bx, nComp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        for (int idof = 0; idof < size; ++idof)
                        {
                            int idofcomp = n * size + idof;
                            aos(i, j, k, n)[idof] = soa(i, j, k, idofcomp);
                        }
                    });
    }
}

/**
 * @brief  multidof (AoS) MultiFab to multicomponents (SoA) MultiFab transformation
 * @tparam dataStruct the data struct, e.g. a GpuArray...
 * @param soaMFA input multicomponent MultiFab
 * @param aosMFA output FabArray of BaseFab of dataStruct
 */
template <typename DataStruct, int nDim>
void soa2aos_fab (amrex::Array<amrex::FabArray<amrex::BaseFab<DataStruct>>, nDim>& aosMFA,
                  amrex::Array<amrex::MultiFab, nDim>& soaMFA)
{
    // using signed int because `nComp` always returns an `int`
    constexpr int size = get_array_size<DataStruct>::s_myValue;
    BL_ASSERT(soaMFA[xDir].nComp() == aosMFA[xDir].nComp() * size);
#if nDim > 1
    BL_ASSERT(soaMFA[yDir].nComp() == aosMFA[yDir].nComp() * size);
    BL_ASSERT(soaMFA[yDir].nComp() == BL_ASSERT(soaMFA[xDir].nComp());
    BL_ASSERT(aosMFA[yDir].nComp() == BL_ASSERT(aosMFA[xDir].nComp());
#endif
#if nDim > 2
    BL_ASSERT(soaMFA[zDir].nComp() == aosMFA[zDir].nComp() * size);
    BL_ASSERT(soaMFA[zDir].nComp() == BL_ASSERT(soaMFA[yDir].nComp());
    BL_ASSERT(aosMFA[zDir].nComp() == BL_ASSERT(aosMFA[yDir].nComp());
#endif
    for (int iDir = 0; iDir < nDim; ++iDir)
    {
        for (amrex::MFIter mfi(aosMFA[iDir]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx =
                mfi.growntilebox(); // cache blocking (for cpus) loops over valid boxes

            auto const& soa = (soaMFA[iDir]).array(mfi);
            auto const& aos = (aosMFA[iDir]).array(mfi);
            int nComp{aosMFA[iDir].nComp()};
            ParallelFor(bx, nComp,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                        {
                            for (int idof = 0; idof < size; ++idof)
                            {
                                int idofcomp = n * size + idof;
                                aos(i, j, k, n)[idof] = soa(i, j, k, idofcomp);
                            }
                        });
        }
    }
}

void mult_and_add(amrex::Real const dstVal,
                  amrex::MultiFab& dst,
                  amrex::Real const srcVal,
                  amrex::MultiFab const& src,
                  int srccomp,
                  int dstcomp,
                  int numcomp,
                  amrex::IntVect const& nghost);

void mult_and_add(amrex::MultiFab& dst,
                  amrex::Real const srcVal,
                  amrex::MultiFab const& src,
                  int srccomp,
                  int dstcomp,
                  int numcomp,
                  amrex::IntVect const& nghost);

/**
 * @brief Compute dot products of two FabArrays: this comes from amrex function
 * @verbatim
template <typename FAB, std::enable_if_t<IsBaseFab<FAB>::value, int> FOO = 0>
typename FAB::value_type Dot (FabArray<FAB> const& x,
                              int xcomp,
                              FabArray<FAB> const& y,
                              int ycomp,
                              int ncomp,
                              IntVect const& nghost,
                              bool local = false)
 * @endverbatim
 *
 * @param wgt    weight, single component
 * @param x      first FabArray
 * @param xcomp  starting component of x
 * @param y      second FabArray
 * @param ycomp  starting component of y
 * @param ncomp  number of components
 * @param nghost number of ghost cells
 * @param local  If true, MPI communication is skipped.
 */
template <typename FAB, std::enable_if_t<amrex::IsBaseFab<FAB>::value, int> foo = 0>
typename FAB::value_type wgt_dot (amrex::FabArray<FAB> const& wgt,
                                  amrex::FabArray<FAB> const& x,
                                  int xcomp,
                                  amrex::FabArray<FAB> const& y,
                                  int ycomp,
                                  int ncomp,
                                  amrex::IntVect const& nghost,
                                  bool local = false)
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
amrex::Real multi_fab_wgt_dot(amrex::MultiFab const& wgt,
                              amrex::MultiFab const& x,
                              int xcomp,
                              amrex::MultiFab const& y,
                              int ycomp,
                              int numcomp,
                              int nghost,
                              bool local);

// from Real MultiFab::Dot(const MultiFab& x, int xcomp, int numcomp, int nghost, bool local)
amrex::Real multi_fab_wgt_dot(amrex::MultiFab const& wgt,
                              amrex::MultiFab const& x,
                              int xcomp,
                              int numcomp,
                              int nghost,
                              bool local);

namespace Impl
{

// Runtime Functor for Legendre polynomial of degree N
struct LegendrePolyRuntime
{
    int m_degree;

    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE std::pair<amrex::Real, amrex::Real> operator()(
        amrex::Real z) const
    {
        amrex::Real p0 = 1.0;
        amrex::Real p1 = z;
        for (int j = 2; j <= m_degree; ++j)
        {
            amrex::Real pj = ((2 * j - 1) * z * p1 - (j - 1) * p0) / j;
            p0 = p1;
            p1 = pj;
        }
        amrex::Real dP = m_degree * (z * p1 - p0) / (z * z - 1.0);
        return {p1, dP};
    }
};

// Compile-time Functor for Legendre polynomial of order N
template <int n>
struct LegendrePoly
{
    // Evaluate P_N(z) and derivative
    AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr std::pair<amrex::Real, amrex::Real>
    operator()(amrex::Real z) const
    {
        amrex::Real p0 = 1.0;
        amrex::Real p1 = z;
        for (int j = 2; j <= n; ++j)
        {
            amrex::Real pj = ((2 * j - 1) * z * p1 - (j - 1) * p0) / j;
            p0 = p1;
            p1 = pj;
        }
        amrex::Real dP = n * (z * p1 - p0) / (z * z - 1.0);
        return {p1, dP};
    }
};

AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr amrex::Real abs_constexpr (amrex::Real x)
{
    return x < 0 ? -x : x;
}

// Generic Newton solver for any functor
template <typename Func>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE constexpr amrex::Real newton_solver (
    amrex::Real z0, Func f, int maxIter = 20, amrex::Real tol = 1e-15)
{
    amrex::Real z = z0;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        auto [fz, dfz] = f(z);
        amrex::Real dz = fz / dfz;
        z -= dz;
        if (abs_constexpr(dz) < tol) break;
    }
    return z;
}

constexpr amrex::Real cos_constexpr (amrex::Real x, int terms = 10)
{
    amrex::Real result = 1.0;
    amrex::Real xPow = 1.0;
    amrex::Real fact = 1.0;
    int sign = -1;
    for (int n = 1; n <= terms; ++n)
    {
        xPow *= x * x;
        fact *= (2 * n - 1) * (2 * n);
        result += sign * xPow / fact;
        sign *= -1;
    }
    return result;
}

// Runtime Gauss-Legendre quadrature returning vectors
GEMPIC_FORCE_INLINE
std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>> gauss_legendre (amrex::Real a,
                                                                              amrex::Real b,
                                                                              int N)
{
    std::vector<amrex::Real> nodes(N);
    std::vector<amrex::Real> weights(N);

    int half = (N + 1) / 2;
    amrex::Real center = 0.5 * (a + b);
    amrex::Real halfLength = 0.5 * (b - a);

    LegendrePolyRuntime P{N};

    for (int i = 0; i < half; ++i)
    {
        amrex::Real zi0 = std::cos(M_PI * (i + 0.75) / (N + 0.5));
        amrex::Real zi = newton_solver(zi0, P);

        nodes[i] = center - halfLength * zi;
        nodes[N - 1 - i] = center + halfLength * zi;

        auto [_, dP] = P(zi);
        amrex::Real wi = 2.0 * halfLength / ((1.0 - zi * zi) * dP * dP);
        weights[i] = wi;
        weights[N - 1 - i] = wi;
    }

    return {nodes, weights};
}

// Compile-time Gauss-Legendre quadrature
template <int n, amrex::Real a, amrex::Real b>
consteval auto gauss_legendre ()
{
    std::array<amrex::Real, n> nodes{};
    std::array<amrex::Real, n> weights{};
    constexpr int half = (n + 1) / 2;
    constexpr amrex::Real factor{M_PI / (n + 0.5)};
    constexpr amrex::Real center = 0.5 * (a + b);
    constexpr amrex::Real halfLength = 0.5 * (b - a);

    LegendrePoly<n> P{};
    // Iterate over half the points at compile-time
    ct_for_index_sequence(std::make_index_sequence<half>{},
                          [&] (auto I)
                          {
                              constexpr std::size_t i = I;
                              constexpr auto zi0 = cos_constexpr(factor * (i + 0.75));
                              auto zi = newton_solver(zi0, P);

                              nodes[i] = center - halfLength * zi;
                              nodes[n - 1 - i] = center + halfLength * zi;

                              auto [_, dP] = P(zi);

                              auto wi = 2.0 * halfLength / ((1.0 - zi * zi) * dP * dP);
                              weights[i] = wi;
                              weights[n - 1 - i] = wi;
                          });

    return std::pair{nodes, weights};
}

} //namespace Impl

template <int n, amrex::Real a, amrex::Real b>
consteval std::pair<std::array<amrex::Real, n>, std::array<amrex::Real, n>> gauss_legendre ()
{
    return Impl::template gauss_legendre<n, a, b>();
}

GEMPIC_FORCE_INLINE std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>> gauss_legendre (
    amrex::Real a, amrex::Real b, int N)
{
    return Impl::gauss_legendre(a, b, N);
}

/**
 * Numerical Scheme handler
 */
template <class FieldsHandler>
class NumericalScheme
{
private:
public:
    /**
     * @brief get the pointer to the field used as initial condition for the explicit time-step, by
     * reference
     * @return shared pointer of FieldsHandlerStruct
     */
    virtual std::shared_ptr<FieldsHandler> const& get_pointer_initial_field_explicit() = 0;
    virtual std::shared_ptr<FieldsHandler> const& get_pointer_initial_field_implicit() = 0;

    virtual std::shared_ptr<FieldsHandler> const& get_pointer_time_derivative_explicit() = 0;
    virtual std::shared_ptr<FieldsHandler> const& get_pointer_time_derivative_implicit() = 0;

    virtual void set_initial_condition() = 0;

    virtual void one_step_solve_dqdt_explicit(amrex::Real rkTime) = 0;

    virtual void one_step_eval_dqdt_implicit(amrex::Real dtLoc) = 0;

    virtual void one_step_solve_q_implicit(amrex::Real rkTime, amrex::Real dtLoc) = 0;

    virtual void one_step_solve_dqdt_semiimplicit(amrex::Real timeI,
                                                  amrex::Real dtI,
                                                  amrex::Real timeE,
                                                  amrex::Real dtE) = 0;

    virtual bool is_finaltime_reached() = 0;

    virtual void init_new_timestep() = 0;

    virtual void finalize_new_timestep() = 0;

    virtual int get_ncomp() = 0;

    virtual amrex::Periodicity get_periodicity() = 0;
};

// Flatten A in row-major
template <std::size_t n, typename T>
void flatten_rm (std::array<T, n * n>& flat, std::array<std::array<T, n>, n> const& nonFlat)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j) flat[i * n + j] = nonFlat[i][j];
    }
}

template <typename T>
void flatten_rm (std::vector<T>& flat, std::vector<std::vector<T>> const& nonFlat)
{
    auto N{nonFlat.size()};
    AMREX_ASSERT_WITH_MESSAGE(N * N == flat.size(), "flat Matrix size must be a NxN square");
    for (std::size_t i = 0; i < N; ++i)
    {
        AMREX_ASSERT_WITH_MESSAGE(nonFlat.size() == nonFlat[i].size(),
                                  "non square matrix in flatten_RM!");
        for (std::size_t j = 0; j < N; ++j) flat[i * N + j] = nonFlat[i][j];
    }
}

// Unflatten to std::array (row-major)
template <std::size_t n, typename T>
void unflatten_rm (std::array<std::array<T, n>, n>& nonFlat, T const* flat)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j) nonFlat[i][j] = flat[i * n + j];
    }
}
template <std::size_t n, typename T>
void unflatten_rm (std::array<std::array<T, n>, n>& nonFlat, std::array<T, n * n> const& flat)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j) nonFlat[i][j] = flat[i * n + j];
    }
}

// Unflatten to std::vector (row-major)
template <typename T>
void unflatten_rm (std::vector<std::vector<T>>& nonFlat, T const* flat, int N = -1)
{
    if (N == -1) N = static_cast<int>(nonFlat.size());
    AMREX_ASSERT_WITH_MESSAGE(N == nonFlat.size(), "nonFlat Matrix size must be a NxN square");
    for (std::size_t i = 0; i < N; ++i)
    {
        AMREX_ASSERT_WITH_MESSAGE(nonFlat[i].size() == N, "non square matrix in unflatten_RM!");
        for (std::size_t j = 0; j < N; ++j) nonFlat[i][j] = flat[i * N + j];
    }
}
template <typename T>
void unflatten_rm (std::vector<std::vector<T>>& nonFlat, std::vector<T> const& flat, int N = -1)
{
    if (N == -1) N = static_cast<int>(nonFlat.size());
    AMREX_ASSERT_WITH_MESSAGE(N * N == flat.size(), "flat Matrix size must be a NxN square");
    AMREX_ASSERT_WITH_MESSAGE(N == nonFlat.size(), "nonFlat Matrix size must be a NxN square");
    for (std::size_t i = 0; i < N; ++i)
    {
        AMREX_ASSERT_WITH_MESSAGE(nonFlat[i].size() == N, "non square matrix in unflatten_RM!");
        for (std::size_t j = 0; j < N; ++j) nonFlat[i][j] = flat[i * N + j];
    }
}

/**
 * @brief LU decomposition with partial pivoting and forward/backward substitution.
 *
 * This class implements an in-place LU factorization of a square matrix of size N x N,
 * with partial pivoting, and provides routines to solve linear systems for single
 * or multiple right-hand sides, as well as to compute the inverse.
 *
 * @tparam T  Numeric type (e.g., double, float) for matrix entries.
 * @tparam nSize  Dimension of the square matrix.
 *
 * @note The input matrix is assumed to be in **row-major order**:
 *       'lu[i][j]' corresponds to the element in row i and column j.
 *       The LU factorization is performed in-place; the input matrix is copied
 *       to the internal 'lu' array, which is then overwritten during factorization.
 *
 * @note Multiple RHS matrices are assumed to be column-wise: 'B[i][j]' is row i, column j.
 */
template <std::size_t nSize, typename T>
struct LUdcmp
{
    enum class Status
    {
        ok,
        singular,
        nearSingular
    };

    Status m_status{Status::ok};

    std::array<T, nSize * nSize> m_lu{}; // LU decomposition
    std::array<int, nSize> m_indx{};     // pivot indices
    T m_rowSwapParity = 1;

    /**
     * @brief Constructor: performs LU factorization with partial pivoting.
     * @param a  Input N x N matrix (row-major). Copied internally.
     */
    explicit AMREX_GPU_HOST_DEVICE LUdcmp (std::array<T, nSize * nSize> const& a) { factorize(a); }
    /**
     * @brief performs LU factorization with partial pivoting.
     * @param a  Input N x N matrix (row-major). Copied internally.
     */
    void AMREX_GPU_HOST_DEVICE factorize (std::array<T, nSize * nSize> const& a)
    {
        // Iniitialize
        m_rowSwapParity = T(1);
        m_status = Status::ok;

        m_lu = a;
        std::array<T, nSize> vv{}; // implicit scaling of each row

        // Compute implicit scaling
        for (std::size_t i = 0; i < nSize; ++i)
        {
            T big = 0.0;
            for (std::size_t j = 0; j < nSize; ++j)
            {
                T absLu = std::abs(m_lu[i * nSize + j]);
                big = (big > absLu) ? big : absLu;
            }
            if (big == T(0))
            {
                //Singular matrix in LUdcmp
                m_status = Status::singular;
                return;
            }
            vv[i] = 1.0 / (big + std::numeric_limits<T>::min());
        }

        // Outer loop over columns (partial pivoting)
        for (std::size_t k = 0; k < nSize; ++k)
        {
            T big = 0.0;
            std::size_t imax = k;
            for (std::size_t i = k; i < nSize; ++i)
            {
                T absLu = std::abs(m_lu[i * nSize + k]);
                T temp = vv[i] * absLu;
                if (temp > big)
                {
                    big = temp;
                    imax = i;
                }
            }

            if (k != imax)
            {
                for (std::size_t j = 0; j < nSize; ++j)
                {
                    T tmp = m_lu[k * nSize + j];
                    m_lu[k * nSize + j] = m_lu[imax * nSize + j];
                    m_lu[imax * nSize + j] = tmp;
                }

                m_rowSwapParity = -m_rowSwapParity;
                T tmp = vv[k];
                vv[k] = vv[imax];
                vv[imax] = tmp;
            }
            m_indx[k] = static_cast<int>(imax);

            // pivot check: relative to the row's own max magnitude (vv[k] before this pivot)
            T const pivotScale = T(1) / vv[k];
            if (std::abs(m_lu[k * nSize + k]) < std::numeric_limits<T>::epsilon() * pivotScale)
            {
                m_status = Status::nearSingular;
                return;
            }

            for (std::size_t i = k + 1; i < nSize; ++i)
            {
                m_lu[i * nSize + k] /= m_lu[k * nSize + k];
                for (std::size_t j = k + 1; j < nSize; ++j)
                {
                    m_lu[i * nSize + j] -= m_lu[i * nSize + k] * m_lu[k * nSize + j];
                }
            }
        }
    }

    // Forward/Back substitution for single RHS
    template <std::size_t... idx>
    static AMREX_GPU_HOST_DEVICE inline void forward_substitute (
        std::array<T, nSize>& x,
        std::array<T, nSize * nSize> const& lu,
        std::array<int, nSize> const& indx,
        std::index_sequence<idx...>)
    {
        int ii = -1;
        ((
             [&]
             {
                 int i = idx;
                 int ip = indx[i];
                 T sum = x[ip];
                 x[ip] = x[i];
                 if (ii >= 0)
                 {
                     for (std::size_t j = ii; j < i; ++j) sum -= lu[i * nSize + j] * x[j];
                 }
                 else if (sum != 0)
                 {
                     ii = i;
                 }
                 x[i] = sum;
             }()),
         ...);
    }

    template <std::size_t... idx>
    static AMREX_GPU_HOST_DEVICE inline void back_substitute (
        std::array<T, nSize>& x,
        std::array<T, nSize * nSize> const& lu,
        std::index_sequence<idx...>)
    {
        ((
             [&]
             {
                 int i = nSize - 1 - idx;
                 T sum = x[i];
                 for (std::size_t j = i + 1; j < nSize; ++j) sum -= lu[i * nSize + j] * x[j];
                 x[i] = sum / lu[i * nSize + i];
             }()),
         ...);
    }

    /**
     * @brief Solve a linear system Ax = b for a single RHS.
     * @param b  Input vector (length N)
     * @param x  Output solution vector (length N)
     *
     * @note Uses the internally stored LU decomposition.
     */
    inline AMREX_GPU_HOST_DEVICE void solve (std::array<T, nSize>& x,
                                             std::array<T, nSize> const& b) const
    {
        AMREX_ASSERT(m_status == Status::ok);
        x = b;
        forward_substitute(x, m_lu, m_indx, std::make_index_sequence<nSize>{});
        back_substitute(x, m_lu, std::make_index_sequence<nSize>{});
    }

    /**
     * @brief Solve a linear system AX = B for multiple RHS (columns of B).
     * @tparam M  Number of RHS vectors (columns)
     * @param B  Input matrix of RHS vectors (row-major, B[i][j] = row i, col j)
     * @param X  Output matrix of solutions (row-major)
     */
    template <std::size_t m>
    inline AMREX_GPU_HOST_DEVICE void solve (std::array<T, m * nSize>& X,
                                             std::array<T, m * nSize> const& B) const
    {
        AMREX_ASSERT(m_status == Status::ok);

        for (std::size_t j = 0; j < m; ++j)
        {
            std::array<T, nSize> col{};
            for (std::size_t i = 0; i < nSize; ++i) col[i] = B[i * nSize + j];
            forward_substitute(col, m_lu, m_indx, std::make_index_sequence<nSize>{});
            back_substitute(col, m_lu, std::make_index_sequence<nSize>{});
            for (std::size_t i = 0; i < nSize; ++i) X[i * nSize + j] = col[i];
        }
    }

    /**
     * @brief Compute the inverse of the original matrix.
     * @param ainv Output inverse matrix (row-major)
     */
    inline AMREX_GPU_HOST_DEVICE void inverse (std::array<T, nSize * nSize>& ainv) const
    {
        AMREX_ASSERT(m_status == Status::ok);
        std::array<T, nSize * nSize> id{};
        for (std::size_t i = 0; i < nSize; ++i) id[i * nSize + i] = 1.0;
        solve<nSize>(ainv, id);
    }

    AMREX_GPU_HOST_DEVICE
    Status status () const { return m_status; }
};

/**
 * @brief runtime version of LUdcmp
 * @tparam T
 */
template <typename T>
struct LUdcmpRuntime
{
    std::vector<T> m_lu; // flattened m_n*m_n row-major
    std::vector<int> m_indx;
    T m_rowSwapParity = 1; // parity of row swaps
    int m_n = 0;

    LUdcmpRuntime(std::vector<T> const& aIn, int nin = -1 // optional, auto-detect from size
    )
    {
        AMREX_ASSERT(!aIn.empty());
        if (nin == -1)
        {
            // Determine size m_n from aIn
            m_n = static_cast<int>(std::sqrt(aIn.size()));
        }
        else
        {
            m_n = nin;
        }

        AMREX_ASSERT_WITH_MESSAGE(m_n * m_n == static_cast<int>(aIn.size()),
                                  "aIn size is not a NxN square");
        m_lu = aIn; // copy input
        m_indx.resize(m_n);
        std::vector<T> vv(m_n);

        // Compute implicit scaling
        for (int i = 0; i < m_n; ++i)
        {
            T big = 0;
            for (int j = 0; j < m_n; ++j) big = std::max(big, std::abs(m_lu[i * m_n + j]));
            if (big == 0) throw std::runtime_error("Singular matrix in LUdcmpRuntime");
            // scaling step: floor to avoid 0/0, magnitude-independent
            vv[i] = 1.0 / (big + std::numeric_limits<T>::min());
        }

        // LU factorization with partial pivoting
        for (int k = 0; k < m_n; ++k)
        {
            T big = 0;
            int imax = k;
            for (int i = k; i < m_n; ++i)
            {
                T temp = vv[i] * std::abs(m_lu[i * m_n + k]);
                if (temp > big)
                {
                    big = temp;
                    imax = i;
                }
            }

            if (k != imax)
            {
                // Swap rows k and imax
                std::swap_ranges(m_lu.begin() + k * m_n, m_lu.begin() + (k + 1) * m_n,
                                 m_lu.begin() + imax * m_n);
                m_rowSwapParity = -m_rowSwapParity;
                std::swap(vv[k], vv[imax]);
            }

            m_indx[k] = imax;

            // pivot check: relative to the row's own max magnitude (vv[k] before this pivot)
            T const pivotScale = T(1) / vv[k]; // ~ the row's original max abs value
            if (std::abs(m_lu[k * m_n + k]) < std::numeric_limits<T>::epsilon() * pivotScale)
            {
                throw std::runtime_error("Near-singular pivot in LUdcmpRuntime");
            }

            for (int i = k + 1; i < m_n; ++i)
            {
                m_lu[i * m_n + k] /= m_lu[k * m_n + k];
                for (int j = k + 1; j < m_n; ++j)
                {
                    m_lu[i * m_n + j] -= m_lu[i * m_n + k] * m_lu[k * m_n + j];
                }
            }
        }
    }

    // Solve single RHS
    void solve (std::vector<T>& x, std::vector<T> const& b) const
    {
        if (m_n == 0)
        {
            throw std::runtime_error("LUdcmpRuntime: solve() called on unfactorized matrix");
        }
        if (b.size() != static_cast<size_t>(m_n)) throw std::runtime_error("Dimension mismatch");
        x = b; // this also allows solve(b,b);
        int ii = -1;

        // Forward substitution with pivoting
        for (int i = 0; i < m_n; ++i)
        {
            int ip = m_indx[i];
            T sum = x[ip];
            x[ip] = x[i];
            if (ii >= 0)
            {
                for (int j = ii; j < i; ++j) sum -= m_lu[i * m_n + j] * x[j];
            }
            else if (sum != 0)
            {
                ii = i;
            }
            x[i] = sum;
        }

        // Backward substitution
        for (int i = m_n - 1; i >= 0; --i)
        {
            T sum = x[i];
            for (int j = i + 1; j < m_n; ++j) sum -= m_lu[i * m_n + j] * x[j];
            x[i] = sum / m_lu[i * m_n + i];
        }
    }

    // Solve multiple RHS (flattened B: m_n*nrhs row-major)
    void solve (std::vector<T>& X, std::vector<T> const& bIn, int nrhs) const
    {
        if (nrhs == -1)
        {
            nrhs = static_cast<int>(bIn.size()) / m_n;
            AMREX_ASSERT_WITH_MESSAGE(m_n * nrhs == static_cast<int>(bIn.size()),
                                      "bIn size inconsistent with m_n");
        }
        AMREX_ASSERT_WITH_MESSAGE(bIn.size() == static_cast<size_t>(m_n * nrhs),
                                  "Dimension mismatch");
        X = bIn;
        std::vector<T> col(m_n);

        for (int j = 0; j < nrhs; ++j)
        {
            for (int i = 0; i < m_n; ++i) col[i] = bIn[i * nrhs + j];
            solve(col, col);
            for (int i = 0; i < m_n; ++i) X[i * nrhs + j] = col[i];
        }
    }

    // Compute inverse
    void inverse (std::vector<T>& ainv) const
    {
        ainv.resize(m_n * m_n, 0);
        std::vector<T> icol(m_n);
        for (int j = 0; j < m_n; ++j)
        {
            // Unit vector column
            std::fill(icol.begin(), icol.end(), 0);
            icol[j] = 1;
            std::vector<T> xcol;
            solve(xcol, icol);
            for (int i = 0; i < m_n; ++i) ainv[i * m_n + j] = xcol[i];
        }
    }
};

//host only helper to solve a linear system using LU decomposition
template <std::size_t nSize, typename T, std::size_t m = 1>
typename LUdcmp<nSize, T>::Status lu_solve (std::array<T, nSize * nSize> const& a,
                                            std::array<T, m * nSize>& x,
                                            std::array<T, m * nSize> const& b)
{
    LUdcmp<nSize, T> const lu(a);
    if (lu.status() == decltype(lu)::Status::ok)
    {
        lu.template solve<m>(x, b);
    }
    return lu.status();
}

//host only helper to compute the inverse of a matrix using LU decomposition
template <std::size_t nSize, typename T>
std::array<T, nSize * nSize> lu_inverse (std::array<T, nSize * nSize> const& a)
{
    LUdcmp<nSize, T> const lu(a);
    std::array<T, nSize * nSize> aInv{};
    if (lu.status() == decltype(lu)::Status::ok)
    {
        lu.inverse(aInv);
    }
    switch (lu.status())
    {
        case LUdcmp<nSize, T>::Status::singular:
            amrex::Warning("LU matrix is singular");
            break;
        case LUdcmp<nSize, T>::Status::nearSingular:
            amrex::Warning("LU matrix is close to singular");
            break;
        default:
            break;
    }
    return aInv;
}

// ------------------------------
template <size_t nSize>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE amrex::Real condition_number (
    std::array<amrex::Real, nSize * nSize> const& aFlat)
{
#ifdef GEMPIC_USE_LAPACK_OR_MKL
    std::array<lapack_int, nSize> ipiv{};
    std::array<amrex::Real, nSize * nSize> A_copy = aFlat;

    lapack_int info =
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, nSize, nSize, A_copy.data(), nSize, ipiv.data());
    if (info != 0) throw std::runtime_error("dgetrf failed");

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, nSize, A_copy.data(), nSize, ipiv.data());
    if (info != 0) throw std::runtime_error("dgetri failed");

    amrex::Real normA = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', nSize, nSize, aFlat.data(), nSize);
    amrex::Real normInvA =
        LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', nSize, nSize, A_copy.data(), nSize);
#else
    LUdcmp<nSize, amrex::Real> lu(aFlat); // templated LU for fixed-size array
    std::array<amrex::Real, nSize * nSize> aInv{};
    lu.inverse(aInv);

    amrex::Real normA = 0.0, normInvA = 0.0;
    for (size_t j = 0; j < nSize; ++j)
    {
        amrex::Real colSumA = 0.0, colSumInv = 0.0;
        for (size_t i = 0; i < nSize; ++i)
        {
            colSumA += std::abs(aFlat[i * nSize + j]);
            colSumInv += std::abs(aInv[i * nSize + j]);
        }
        normA = std::max(normA, colSumA);
        normInvA = std::max(normInvA, colSumInv);
    }
#endif
    return normA * normInvA;
}

// Runtime version with std::vector (fallback using LUdcmp if LAPACK/MKL unavailable)
GEMPIC_FORCE_INLINE amrex::Real condition_number (std::vector<amrex::Real> const& flatA,
                                                  int nSize,
                                                  std::vector<amrex::Real>* flatAinvOut = nullptr)
{
#ifdef GEMPIC_USE_LAPACK_OR_MKL
    std::vector<lapack_int> ipiv(nSize);
    std::vector<amrex::Real> aInv = flatA;
    lapack_int info =
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, nSize, nSize, aInv.data(), nSize, ipiv.data());
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(!info, "Error in LU factorization: " + std::to_string(info));
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, nSize, aInv.data(), nSize, ipiv.data());
    GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(!info, "Error in matrix inversion: " + std::to_string(info));

    amrex::Real normA = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', nSize, nSize, flatA.data(), nSize);
    amrex::Real normInvA = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', nSize, nSize, aInv.data(), nSize);

    if (flatAinvOut != nullptr)
    {
        *flatAinvOut = aInv;
    }

    return normA * normInvA;
#else
    LUdcmpRuntime<amrex::Real> lu(flatA);
    std::vector<amrex::Real> aInv;
    lu.inverse(aInv);

    if (flatAinvOut != nullptr)
    {
        *flatAinvOut = aInv;
    }

    amrex::Real normA = 0.0;
    amrex::Real normInvA = 0.0;
    for (int j = 0; j < nSize; ++j)
    {
        amrex::Real colSumA = 0.0;
        amrex::Real colSumInv = 0.0;
        for (int i = 0; i < nSize; ++i)
        {
            colSumA += std::abs(flatA[i * nSize + j]);
            colSumInv += std::abs(aInv[i * nSize + j]);
        }
        normA = std::max(normA, colSumA);
        normInvA = std::max(normInvA, colSumInv);
    }
    return normA * normInvA;
#endif
}

// Runtime version with std::vector (fallback using LUdcmp if LAPACK/MKL unavailable)
GEMPIC_FORCE_INLINE amrex::Real condition_number (std::vector<std::vector<amrex::Real>> const& aIn,
                                                  int nSize)
{
    std::vector<amrex::Real> flatA(nSize * nSize);
    flatten_rm(flatA, aIn);
    return condition_number(flatA, nSize);
}

// -----------------------------------------------------------------------------
// Matrix inverse using LAPACK / MKL or  (LUdcmp)
// -----------------------------------------------------------------------------
template <std::size_t nSize>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void matrix_inverse (
    std::array<std::array<amrex::Real, nSize>, nSize> const& aIn,
    std::array<std::array<amrex::Real, nSize>, nSize>& aInv)
{
#ifdef GEMPIC_USE_LAPACK_OR_MKL
    lapack_int const n = static_cast<lapack_int>(nSize);
    std::array<amrex::Real, n * n> flatA{};
    std::array<lapack_int, n> ipiv{};

    // Flatten row-major
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) flatA[i * n + j] = aIn[i][j];

    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, flatA.data(), n, ipiv.data());
    if (info != 0)
    {
        std::cerr << "Error in LU decomposition: " << info << std::endl;
        return;
    }

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, flatA.data(), n, ipiv.data());
    if (info != 0)
    {
        std::cerr << "Error in matrix inversion: " << info << std::endl;
        return;
    }

    // Un-flatten
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) aInv[i][j] = flatA[i * n + j];

#else
    // Runtime fallback
    LUdcmp<nSize, amrex::Real> lu(aIn);
    lu.inverse(aInv);
#endif
}

GEMPIC_FORCE_INLINE void matrix_inverse (std::vector<amrex::Real>& aInvFlat,
                                         std::vector<amrex::Real> const& aFlat,
                                         int nSize = -1)
{
    // Determine nSize if not provided
    if (nSize == -1)
    {
        nSize = static_cast<int>(std::sqrt(aFlat.size()));
    }
    AMREX_ASSERT(nSize * nSize == static_cast<int>(aFlat.size()));
#ifdef GEMPIC_USE_LAPACK_OR_MKL
    std::vector<amrex::Real> A_copy = aFlat; // LAPACK overwrites input
    std::vector<lapack_int> ipiv(nSize);

    lapack_int info =
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, nSize, nSize, A_copy.data(), nSize, ipiv.data());
    if (info != 0)
    {
        std::cerr << "Error in LU decomposition: " << info << std::endl;
        return;
    }

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, nSize, A_copy.data(), nSize, ipiv.data());
    if (info != 0)
    {
        std::cerr << "Error in matrix inversion: " << info << std::endl;
        return;
    }

    aInvFlat = std::move(A_copy);

#else
    LUdcmpRuntime<amrex::Real> lu(aFlat);
    lu.inverse(aInvFlat);
#endif
}

GEMPIC_FORCE_INLINE void matrix_inverse (std::vector<std::vector<amrex::Real>>& aInv,
                                         std::vector<std::vector<amrex::Real>> const& A,
                                         int nSize = -1)
{
    // Determine nSize if not provided
    if (nSize == -1)
    {
        nSize = static_cast<int>(A.size());
    }
    AMREX_ASSERT(nSize == static_cast<int>(A[0].size()));
    AMREX_ASSERT(static_cast<int>(A.size()) == static_cast<int>(A[0].size()));
    AMREX_ASSERT(nSize == static_cast<int>(aInv[0].size()));
    AMREX_ASSERT(static_cast<int>(aInv.size()) == static_cast<int>(aInv[0].size()));

    // Flatten input matrix
    std::vector<amrex::Real> aFlat(nSize * nSize);
    flatten_rm(aFlat, A);

    // Prepare flattened output
    std::vector<amrex::Real> aInvFlat(nSize * nSize);

    // Call flattened version
    matrix_inverse(aInvFlat, aFlat, nSize);

    // Unflatten into 2D vector
    aInv.resize(nSize, std::vector<amrex::Real>(nSize));
    unflatten_rm(aInv, aInvFlat, nSize);
}

// ------------------------
// Fixed-size template
// ------------------------
template <std::size_t nSize>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void solve_system (
    std::array<amrex::Real, nSize * nSize>& xOut,
    std::array<amrex::Real, nSize * nSize> const& aIn,
    std::array<amrex::Real, nSize * nSize> const& bIn)
{
#ifdef GEMPIC_USE_LAPACK_OR_MKL
    lapack_int const n = static_cast<lapack_int>(nSize);
    lapack_int const nrhs = 1;

    for (std::size_t j = 0; j < nSize; ++j)
    {
        std::array<amrex::Real, nSize * nSize> a = aIn; // copy because LAPACK modifies A
        std::array<amrex::Real, nSize> b{};
        std::array<lapack_int, nSize> ipiv{};

        for (std::size_t i = 0; i < nSize; ++i) b[i] = bIn[i * nSize + j];

        lapack_int info =
            LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a.data(), n, ipiv.data(), b.data(), nrhs);
        if (info != 0)
        {
            std::cerr << "Error in LAPACK dgesv: " << info << std::endl;
            return;
        }

        for (std::size_t i = 0; i < nSize; ++i) xOut[i * nSize + j] = b[i];
    }

#else
    // Fallback: LUdcmp
    LUdcmp<nSize, amrex::Real> lu(aIn);
    lu.template solve<nSize>(xOut, bIn);
#endif
}

// ------------------------
// Compiletime-size arrays: Gauss Jordan
// ------------------------
template <std::size_t n>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void solve_system_gj (
    std::array<std::array<amrex::Real, n>, n>& xOut,
    std::array<std::array<amrex::Real, n>, n> const& aIn,
    std::array<std::array<amrex::Real, n>, n> const& bIn)
{
    //  Gauss-Jordan elimination
    xOut = bIn;
    std::array<std::array<amrex::Real, n>, n> A = aIn;

    for (std::size_t i = 0; i < n; ++i)
    {
        assert(A[i][i] != 0 && "Singular matrix in Gauss-Jordan solver");

        amrex::Real diag = A[i][i];
        for (std::size_t j = 0; j < n; ++j)
        {
            A[i][j] /= diag;
            xOut[i][j] /= diag;
        }

        for (std::size_t k = 0; k < n; ++k)
        {
            if (k == i) continue;
            amrex::Real factor = A[k][i];
            for (std::size_t j = 0; j < n; ++j)
            {
                A[k][j] -= factor * A[i][j];
                xOut[k][j] -= factor * xOut[i][j];
            }
        }
    }
}
// ------------------------
// Runtime-size vectors: Gauss Jordan (with partial pivoting)
// ------------------------
GEMPIC_FORCE_INLINE void solve_system_gj (std::vector<amrex::Real>& xOut,
                                          std::vector<amrex::Real> const& aIn,
                                          std::vector<amrex::Real> const& bIn)
{
    auto N = static_cast<std::ptrdiff_t>(std::sqrt(static_cast<double>(aIn.size())));
    std::ptrdiff_t nrhs = static_cast<std::ptrdiff_t>(bIn.size()) / N;

    AMREX_ASSERT_WITH_MESSAGE(N * N == static_cast<std::ptrdiff_t>(aIn.size()),
                              "aIn must be square");
    AMREX_ASSERT_WITH_MESSAGE(N * nrhs == static_cast<std::ptrdiff_t>(bIn.size()),
                              "bIn size inconsistent with N");

    std::vector<amrex::Real> A = aIn;
    xOut = bIn;

    // Row-scale estimate for a relative (not absolute) singularity test,
    // consistent with LUdcmp/LUdcmpRuntime's pivot check.
    std::vector<amrex::Real> rowScale(N, amrex::Real(0));
    for (std::ptrdiff_t i = 0; i < N; ++i)
    {
        for (std::ptrdiff_t j = 0; j < N; ++j)
        {
            rowScale[i] = std::max(rowScale[i], std::abs(A[i * N + j]));
        }
    }

    for (std::ptrdiff_t i = 0; i < N; ++i)
    {
        // Partial pivoting: find the best pivot row among i..N-1, scaled
        // by each row's own magnitude (same spirit as LUdcmp's vv[] scaling).
        std::ptrdiff_t pivotRow = i;
        auto best = amrex::Real(0);
        for (std::ptrdiff_t k = i; k < N; ++k)
        {
            amrex::Real const scale = rowScale[k] + std::numeric_limits<amrex::Real>::min();
            amrex::Real const candidate = std::abs(A[k * N + i]) / scale;
            if (candidate > best)
            {
                best = candidate;
                pivotRow = k;
            }
        }

        if (pivotRow != i)
        {
            for (std::ptrdiff_t j = 0; j < N; ++j) std::swap(A[i * N + j], A[pivotRow * N + j]);
            for (std::ptrdiff_t j = 0; j < nrhs; ++j)
            {
                std::swap(xOut[i * nrhs + j], xOut[pivotRow * nrhs + j]);
            }
            std::swap(rowScale[i], rowScale[pivotRow]);
        }

        amrex::Real const pivotScale = rowScale[i] + std::numeric_limits<amrex::Real>::min();
        AMREX_ASSERT_WITH_MESSAGE(
            std::abs(A[i * N + i]) >= std::numeric_limits<amrex::Real>::epsilon() * pivotScale,
            "Singular (or near-singular) matrix in Gauss-Jordan solver");

        amrex::Real diag = A[i * N + i];
        for (std::ptrdiff_t j = 0; j < N; ++j) A[i * N + j] /= diag;
        for (std::ptrdiff_t j = 0; j < nrhs; ++j) xOut[i * nrhs + j] /= diag;

        for (std::ptrdiff_t k = 0; k < N; ++k)
        {
            if (k == i) continue;
            amrex::Real factor = A[k * N + i];
            if (factor == amrex::Real(0)) continue;
            for (std::ptrdiff_t j = 0; j < N; ++j) A[k * N + j] -= factor * A[i * N + j];
            for (std::ptrdiff_t j = 0; j < nrhs; ++j)
            {
                xOut[k * nrhs + j] -= factor * xOut[i * nrhs + j];
            }
        }
    }
}

// Wrapper: unflattened version using flatten/unflatten utilities
GEMPIC_FORCE_INLINE void solve_system_gj (std::vector<std::vector<amrex::Real>>& xOut,
                                          std::vector<std::vector<amrex::Real>> const& aIn,
                                          std::vector<std::vector<amrex::Real>> const& bIn)
{
    AMREX_ASSERT_WITH_MESSAGE(!aIn.empty() && !bIn.empty(), "aIn/bIn must not be empty");
    auto N = static_cast<std::ptrdiff_t>(std::sqrt(static_cast<double>(aIn.size())));
    AMREX_ASSERT_WITH_MESSAGE(N == static_cast<std::ptrdiff_t>(aIn[0].size()),
                              "Matrix size must be a NxN square");
    AMREX_ASSERT_WITH_MESSAGE(N == static_cast<std::ptrdiff_t>(bIn.size()),
                              "Matrix size must be a Nxnrhs square");
    auto nrhs = static_cast<std::ptrdiff_t>(bIn[0].size());

    std::vector<amrex::Real> aFlat(N * N);
    std::vector<amrex::Real> bFlat(N * nrhs);
    std::vector<amrex::Real> xFlat(N * nrhs);

    flatten_rm(aFlat, aIn);
    flatten_rm(bFlat, bIn);

    // Call flattened solver
    solve_system_gj(xFlat, aFlat, bFlat);

    // Unflatten output
    xOut.resize(N, std::vector<amrex::Real>(nrhs));
    unflatten_rm(xOut, xFlat);
}

// -----------------------------------------------------------------------------
// Gauss-Jordan inversion
// -----------------------------------------------------------------------------
GEMPIC_FORCE_INLINE void matrix_inverse_gj (std::vector<std::vector<amrex::Real>>& iA,
                                            std::vector<std::vector<amrex::Real>> const& A,
                                            int N)
{
    std::vector<std::vector<amrex::Real>> C(N, std::vector<amrex::Real>(2 * N, 0.0));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) C[i][j] = A[i][j];
        C[i][N + i] = 1.0;
    }

    for (int i = 0; i < N; ++i)
    {
        int ml = 0;
        for (int k = i + 1; k < N; ++k)
        {
            if (std::abs(C[k][i]) > std::abs(C[i + ml][i])) ml = k - i;
        }
        int iswap = i + ml;
        if (iswap != i) std::swap(C[i], C[iswap]);
        if (C[i][i] == 0.0)
        {
            std::cerr << "ERROR: Matrix is singular!" << std::endl;
            std::exit(1);
        }
        amrex::Real piv = 1.0 / C[i][i];
        for (int l = 0; l < 2 * N; ++l) C[i][l] *= piv;
        for (int j = i + 1; j < N; ++j)
        {
            amrex::Real factor = C[j][i];
            for (int l = 0; l < 2 * N; ++l) C[j][l] -= factor * C[i][l];
        }
    }

    for (int i = N - 1; i >= 0; --i)
    {
        for (int j = i - 1; j >= 0; --j)
        {
            amrex::Real factor = C[j][i];
            for (int l = 0; l < 2 * N; ++l) C[j][l] -= factor * C[i][l];
        }
    }

    iA.resize(N, std::vector<amrex::Real>(N));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) iA[i][j] = C[i][N + j];
    }
}

GEMPIC_FORCE_INLINE void solve_system (std::vector<amrex::Real>& xOut,      // flat row-major N*nrhs
                                       std::vector<amrex::Real> const& aIn, // flat row-major N*N
                                       std::vector<amrex::Real> const& bIn, // flat row-major N*nrhs
                                       int N = -1,   // optional, auto-detect from size
                                       int nrhs = -1 // optional, auto-detect from size
)
{
#ifdef GEMPIC_USE_LAPACK_OR_MKL
    // Auto-detect N if not provided
    assert(!aIn.empty());
    if (N == -1)
    {
        N = static_cast<int>(std::sqrt(aIn.size()));
        assert(N * N == static_cast<int>(aIn.size()) && "aIn size is not a perfect square");
    }

    if (nrhs == -1)
    {
        nrhs = static_cast<int>(bIn.size()) / N;
        assert(N * nrhs == static_cast<int>(bIn.size()) && "bIn size inconsistent with N");
    }

    // Copy inputs into mutable arrays for LAPACK
    std::vector<amrex::Real> A(aIn); // LAPACKE_dgesv modifies A
    std::vector<amrex::Real> B(bIn); // LAPACKE_dgesv overwrites B with solution
    std::vector<lapack_int> ipiv(N);

    // Solve linear system
    lapack_int info =
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, nrhs, A.data(), N, ipiv.data(), B.data(), nrhs);
    if (info != 0)
    {
        std::cerr << "Error in LAPACK dgesv: " << info << std::endl;
        return;
    }

    // Output solution
    xOut = std::move(B);

#else
    // Fallback: use LUdcmpRuntime
    nrhs = static_cast<int>(bIn.size()) / N;
    std::vector<amrex::Real> xFlat(bIn); // copy bIn as starting X

    LUdcmpRuntime<amrex::Real> lu(aIn, N);
    lu.solve(xFlat, bIn, nrhs); // solve multiple RHS in-place

    xOut = std::move(xFlat); // result in flat row-major vector
#endif
}

// ------------------------
// Runtime-size vectors
// ------------------------
GEMPIC_FORCE_INLINE void solve_system (std::vector<std::vector<amrex::Real>>& xOut,
                                       std::vector<std::vector<amrex::Real>> const& aIn,
                                       std::vector<std::vector<amrex::Real>> const& bIn,
                                       int N = -1,   // optional, auto-detect from size
                                       int nrhs = -1 // optional, auto-detect from size
)
{
    // Auto-detect N if not provided
    AMREX_ASSERT(!aIn.empty());
    if (N == -1)
    {
        N = static_cast<int>(std::sqrt(aIn.size()));
    }

    if (nrhs == -1)
    {
        nrhs = static_cast<int>(bIn[0].size());
    }
    AMREX_ASSERT_WITH_MESSAGE(N == static_cast<int>(aIn.size()), "aIn size is not NxN");
    AMREX_ASSERT_WITH_MESSAGE(N == static_cast<int>(bIn.size()), "bIn size is not Nxnrhs");

    // Flatten A (NxN) and B (Nxnrhs)
    std::vector<amrex::Real> aFlat(N * N);
    std::vector<amrex::Real> bFlat(N * nrhs);
    flatten_rm(aFlat, aIn); // flatten square matrix A
    flatten_rm(bFlat, bIn); // flatten B (rectangular allowed)

    // Prepare flattened output
    std::vector<amrex::Real> xFlat(N * nrhs);

    solve_system(xFlat, aFlat, bFlat, N, nrhs);
    // Unflatten X_flat into X_out
    xOut.resize(N, std::vector<amrex::Real>(nrhs));
    unflatten_rm(xOut, xFlat, N);
}

template <typename T>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE T minmod (T a, T b)
{
    static_assert(std::is_arithmetic<T>::value, "minmod requires numeric types");

    // If a and b have opposite signs or either is zero, return zero
    if (a * b <= T(0)) return T(0);

    // Otherwise, return the one with smaller magnitude (preserving sign)
    return (std::abs(a) < std::abs(b)) ? a : b;
}

} //namespace Gempic

#endif

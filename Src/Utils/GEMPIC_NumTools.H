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
#ifdef USE_MKL
#include <mkl.h>
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

// Define a macro to switch between precision types
// Comment or uncomment the following line to toggle precision
//#define MY_HIGHER_PRECISION

#ifdef MY_HIGHER_PRECISION
using my_precision = long double;
//#define PRECISION_FORMAT "%.36Qf"
//#define PRINT_PRECISION quadmath_snprintf
#else
using my_precision = amrex::Real;
//#define PRECISION_FORMAT "%.16f"
//#define PRINT_PRECISION snprintf
#endif

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
 * @tparam dataStruct a data structure having a fill(amrex::Real) function, and an associated
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
    constexpr unsigned int size = get_array_size<DataStruct>::s_myValue;
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
    constexpr unsigned int size = get_array_size<DataStruct>::s_myValue;
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
template <typename FAB, std::enable_if_t<IsBaseFab<FAB>::value, int> FOO = 0>
typename FAB::value_type Dot (FabArray<FAB> const& x,
                              int xcomp,
                              FabArray<FAB> const& y,
                              int ycomp,
                              int ncomp,
                              IntVect const& nghost,
                              bool local = false)
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
typename FAB::value_type wgt_dot(amrex::FabArray<FAB> const& wgt,
                                 amrex::FabArray<FAB> const& x,
                                 int xcomp,
                                 amrex::FabArray<FAB> const& y,
                                 int ycomp,
                                 int ncomp,
                                 amrex::IntVect const& nghost,
                                 bool local = false);

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

bool is_inverse_matrix_correct(std::vector<std::vector<double>>& iA,
                               std::vector<std::vector<double>> const& A,
                               std::vector<std::vector<double>> const& expectedInverse);

void matrix_inverse(std::vector<std::vector<double>>& iA,
                    std::vector<std::vector<double>> const& A,
                    int const N);

void matrix_inverse_ld(std::vector<std::vector<my_precision>>& iA,
                       std::vector<std::vector<my_precision>> const& A,
                       int const N);

void matrix_inverse_mkl(std::vector<std::vector<double>>& iA,
                        std::vector<std::vector<double>> const& A,
                        int const N);

void solve_system_mkl(std::vector<std::vector<double>> const& M,
                      std::vector<std::vector<double>> const& K,
                      std::vector<std::vector<double>>& A,
                      int N);

double condition_number(std::vector<std::vector<double>> const& A, int N);

AMREX_GPU_HOST_DEVICE amrex::Real minmod(amrex::Real const a, amrex::Real const b);

#endif
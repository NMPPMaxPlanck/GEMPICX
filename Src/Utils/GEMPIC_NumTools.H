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

/**
 * @brief computes n Gauss-Legendre quadrature nodes x in [x1,x2] and the corresponding quadrature
 * weights w. The algorithm comes from numerical recipies.
 * @tparam n int, number of quadrature points
 * @param x array, quadrature nodes
 * @param w array, quadrature weights
 * @param x1 amrex::Real, left endpoint
 * @param x2 amrex::Real, right endpoint
 */
template <int n>
void gauleg (std::array<amrex::Real, n>& x,
             std::array<amrex::Real, n>& w,
             amrex::Real const x1,
             amrex::Real const x2)
{
    // Ensure valid inputs
    static_assert(n > 0);                             // n must be greater than zero.
    AMREX_ASSERT(x.size() == static_cast<size_t>(n)); // x must have size n.
    AMREX_ASSERT(w.size() == static_cast<size_t>(n)); // w must have size n.
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
void gauleg(std::vector<amrex::Real>& x,
            std::vector<amrex::Real>& w,
            amrex::Real const x1,
            amrex::Real const x2,
            int n);

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
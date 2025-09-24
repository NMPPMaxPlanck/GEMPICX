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
 * Numerical Scheme handler
 */
class NumericalScheme
{
private:
public:
    virtual void set_initial_condition() = 0;

    virtual void compute_norms_reduce() = 0;

    virtual void one_step_solve_dqdt_explicit() = 0;

    virtual void one_step_eval_dqdt_implicit(amrex::Real dtLoc) = 0;

    virtual void one_step_solve_q_implicit(amrex::Real dtLoc) = 0;

    virtual void one_step_solve_dqdt_semiimplicit(amrex::Real dtLoc) = 0;

    virtual bool is_finaltime_reached() = 0;

    virtual void init_new_timestep() = 0;

    virtual void finalize_new_timestep() = 0;

    virtual void init_first_register() = 0;

    virtual void cumulative_mult_and_add() = 0;

    virtual void prepare_registers_type() = 0;

    virtual void prepare_registers() = 0;

    virtual void print_now() = 0;

    virtual void check_and_print(int timeStep) = 0;

    virtual int get_nfields() = 0;

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

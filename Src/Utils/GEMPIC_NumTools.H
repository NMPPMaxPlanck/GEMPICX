#ifndef GEMPIC_NUMTOOLS_H_
#define GEMPIC_NUMTOOLS_H_

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <vector>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_FabArray.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

//#include "GEMPIC_FDDeRhamComplex.H"

std::unique_ptr<amrex::MultiFab> get_shared_bnd_mask(amrex::MultiFab& thisMF,
                                                     amrex::Periodicity const& period);

void sum_boundary_sync(amrex::MultiFab& thisMF, amrex::Periodicity const& period);

/**
 * a modified sum_boundary
 * the FabArray may have multi components but also local dof...
 * **/
template <typename MFA, int locDof>
void sum_boundary_sync_fab (MFA& thisMF, amrex::Periodicity const& period)
{
    if (thisMF.ixType().cellCentered())
    {
        return;
    }

    int const ncomp = thisMF.nComp();

    MFA tmpmf(thisMF.boxArray(), thisMF.DistributionMap(), ncomp, 0, amrex::MFInfo(),
              thisMF.Factory());

    amrex::GpuArray<amrex::Real, locDof> locdof0;
    locdof0.fill(0.0);
    tmpmf.setVal(locdof0);
    tmpmf.ParallelCopy(thisMF, period, amrex::FabArrayBase::ADD);
    thisMF.ParallelCopy(tmpmf, period);

    return;
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
 * @brief Compute dot products of two FabArrays
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

#endif
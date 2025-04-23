#ifndef GEMPIC_POISSONSOLVER_H_
#define GEMPIC_POISSONSOLVER_H_

#include <memory>

#include <AMReX.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_Print.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Solvers.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{
using namespace Forms;

// Strategy pattern
using PoissonSolverMethod =
    SolverMethod<DeRhamField<Grid::dual, Space::cell>, DeRhamField<Grid::primal, Space::node>>;

namespace Impl
{
/// @brief The function used to select a specific poisson solver method
std::unique_ptr<PoissonSolverMethod> make_specific_poisson_solver(
    std::shared_ptr<DeRhamComplex> deRham,
    const ComputationalDomain& infra,
    const std::string& solver,
    const amrex::Real relTol = 1e-10,
    const amrex::Real absTol = 1e-12,
    const bool tolGiven = false);
} // namespace Impl

/**
 * @brief Construct a new Poisson Solver object
 *
 * @param deRham
 * @param infra
 */
inline std::unique_ptr<PoissonSolverMethod> make_poisson_solver (
    std::shared_ptr<DeRhamComplex> deRham, const Gempic::ComputationalDomain& infra)
{
    Gempic::Io::Parameters params("PoissonSolver", "function make_poisson_solver");
    // Solver defaults:
    // If periodic and FFT was compiled: FFT
    // Else if hypre was compiled:       Hypre
    // Else                              ConjugateGradient
    std::string solver{"ConjugateGradient"};
#ifdef AMREX_USE_HYPRE
    solver = "Hypre";
#endif
#ifdef AMREX_USE_FFT
    if (infra.geometry().isAllPeriodic())
    {
        solver = "FFT";
    }
#endif
    params.get_or_set("solver", solver);

    amrex::Real relTol = 1e-11;
    amrex::Real absTol = 1e-12;

    // we want to warn if tolerances are supplied to solvers that don't use them
    bool tolGiven{params.is_in_input_file("relTol") || params.is_in_input_file("absTol")};
    params.get_or_set("relTol", relTol);
    params.get_or_set("absTol", absTol);
    return Impl::make_specific_poisson_solver(deRham, infra, solver, relTol, absTol, tolGiven);
}

/**
 * Solves the Poisson equation for the given dual and primal fields using the second order AMReX
 * nodal Poisson solver
 *
 * @param[out] phi The primal field to store the solution of the equation.
 * @param rho The dual field representing the right-hand side of the equation.
 */
class AmrexSolver : public PoissonSolverMethod
{
    std::unique_ptr<amrex::MLNodeLaplacian> m_linop;
    std::unique_ptr<amrex::MLMG> m_mlmg;
    ComputationalDomain m_compDom;
    amrex::Real m_relTol;
    amrex::Real m_absTol;

public:
    AmrexSolver(const ComputationalDomain& compDom,
                int maxCoarseningLevel,
                int maxIter,
                int maxFmgIter,
                int mgBottomMaxIter,
                amrex::Real relTol = 1e-10,
                amrex::Real absTol = 1e-12,
                int verbose = 0,
                int bottomVerbose = 0);

    void solve(DeRhamField<Grid::primal, Space::node>& phi,
               DeRhamField<Grid::dual, Space::cell>& rho) override;
};

void subtract_constant_part(DeRhamField<Grid::dual, Space::cell>& rho,
                            const ComputationalDomain& compDom);

} // namespace Gempic::FieldSolvers
#endif

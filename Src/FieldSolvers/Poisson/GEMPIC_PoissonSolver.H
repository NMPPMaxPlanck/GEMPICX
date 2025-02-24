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

/// @brief The function used to select a specific poisson solver method
std::unique_ptr<PoissonSolverMethod> make_poisson_solver(std::shared_ptr<DeRhamComplex> deRham,
                                                         const ComputationalDomain& infra);

/**
 * Solves the Poisson equation for the given dual and primal fields using the second order AMReX
 * nodal Poisson solver
 *
 * @param[out] phi : The primal field to store the solution of the equation.
 * @param rho : The dual field representing the right-hand side of the equation.
 *
 * @throws None
 */
class AmrexSolver : public PoissonSolverMethod
{
    std::unique_ptr<amrex::MLNodeLaplacian> m_linop;
    std::unique_ptr<amrex::MLMG> m_mlmg;
    ComputationalDomain m_compDom;
    amrex::Real m_relTol;
    amrex::Real m_absTol;
    amrex::Real m_cellVolume;

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

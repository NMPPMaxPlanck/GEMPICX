#include <memory>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic::FieldSolvers;

/**
 * @brief Construct a new Poisson Solver:: Poisson Solver object
 *
 * @param deRham
 */
PoissonSolver::PoissonSolver(std::shared_ptr<Forms::DeRhamComplex> deRham,
                             const ComputationalDomain& infra) :
    m_deRham{deRham}, m_infra{infra}, m_primalEdge(deRham), m_dualFace(deRham)
{
    m_maxCoarseningLevel = 10;
    m_maxIter = 100;
    m_mgBottomMaxIter = 100;
    m_maxFmgIter = 0;
    m_verbose = Gempic::Utils::Verbosity::level();
    m_bottomVerbose = m_verbose;
}

PoissonSolver::~PoissonSolver() {}

/**
 * Solves the Poisson equation for the given dual and primal fields using the second order AMReX
 * nodal Poisson solver
 *
 * @param rho The dual field representing the right-hand side of the equation.
 * @param phi The primal field to store the solution of the equation.
 *
 * @throws None
 */
void PoissonSolver::solve_amrex (Forms::DeRhamField<Grid::dual, Space::cell>& rho,
                                Forms::DeRhamField<Grid::primal, Space::node>& phi)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::solve()");

    amrex::LPInfo lpInfo;
    lpInfo.setMaxCoarseningLevel(m_maxCoarseningLevel);

    amrex::Real sigma = -1.0;
    amrex::MLNodeLaplacian linop({m_infra.m_geom}, {m_infra.m_grid}, {m_infra.m_distriMap}, lpInfo,
                                 {}, sigma);

    // Set boundary conditions on linear operator for lower end and higher end
    linop.setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic,
                                    amrex::LinOpBCType::Periodic)},
                      {AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic,
                                    amrex::LinOpBCType::Periodic)});

    // Sum of rhs needs to be 0 if domain is periodic in all directions
    subtract_constant_part(rho);

    // Initialize solver class
    amrex::MLMG mlmg(linop);

    // Configure solver class
    mlmg.setMaxIter(m_maxIter);
    mlmg.setMaxFmgIter(m_maxFmgIter);
    mlmg.setBottomMaxIter(m_mgBottomMaxIter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(m_bottomVerbose);
    mlmg.setBottomSolver(amrex::BottomSolver::cg);
    amrex::Real relTol = 1.e-10;
    amrex::Real absTol = 1.e-12;
    // Solve Poisson equation
    mlmg.solve({&phi.m_data}, {&rho.m_data}, relTol, absTol);
    // AMReX Poisson solver does not use Hodge. Need to rescale phi
    phi *= m_infra.m_dxi[GEMPIC_SPACEDIM];

    phi.average_sync();
    phi.fill_boundary();
}

/**
 * Applies the Poisson operator to the given primal and dual fields.
 * The Poisson operator applies successively the grad, the hodge operator and the divergence
 * The order of the solver is hodgedegree
 *
 * @param phi The primal field to apply the operator to.
 * @param rho The dual field to store the result of the operator application.
 *
 * @throws None
 */
void PoissonSolver::apply_poisson_operator (DeRhamField<Grid::primal, Space::node>& phi,
                                           DeRhamField<Grid::dual, Space::cell>& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::apply_poisson_operator()");
    m_deRham->grad(phi, m_primalEdge);
    m_primalEdge *= -1;  // E = -grad phi
    m_deRham->hodge(m_primalEdge, m_dualFace);
    m_deRham->div(m_dualFace, rho);
    // add penalty term to avoid nullspace
    // amrex::Real penalty = 1.0e-8;
    // rho += penalty;
    // Sum of rhs needs to be 0 if domain is periodic in all directions
    // subtract_constant_part(rho);
}

/**
 * Applies the Poisson operator with inverse Hodge transformation to the given primal and dual
 * fields. The order of the solver is hodgedegree - 2 (for hodgedegree = 4 and 6) and 2 for
 * hodgedegree = 2, for which Hodge is diagonal
 *
 * @param phi The primal field to apply the operator to.
 * @param rho The dual field to store the result of the operator application after inverse Hodge
 * transformation.
 *
 * @return None
 */
void PoissonSolver::apply_poisson_operator_inverse_hodge (
    DeRhamField<Grid::primal, Space::node>& phi, DeRhamField<Grid::dual, Space::cell>& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::apply_poisson_operator_inverse_hodge()");
    // Conjugate gradient to compute inverse Hodge
    if (!m_cgHodge)
    {
        m_cgHodge = std::make_unique<
            ConjugateGradient<DeRhamField<Grid::primal, Space::edge>,
                              DeRhamField<Grid::dual, Space::face>, Operator::hodge>>(m_deRham);
    }

    m_deRham->grad(phi, m_primalEdge);
    m_primalEdge *= -1;  // E = -grad phi
    m_cgHodge->solve(m_primalEdge, m_dualFace);
    m_deRham->div(m_dualFace, rho);

    // Sum of rhs needs to be 0 if domain is periodic in all directions
    subtract_constant_part(rho);
}

/// Ensures that the average value of the field is 0 if domain is periodic in all directions
void PoissonSolver::subtract_constant_part (DeRhamField<Grid::dual, Space::cell>& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::subtract_constant_part()");
    if (m_deRham->get_periodicity().isAllPeriodic())
    {
        amrex::Real rhoSum = rho.m_data.sum_unique(0, false, m_infra.m_geom.periodicity());
        amrex::Real ninv = 1.0 / GEMPIC_D_MULT(m_infra.m_nCell[xDir], m_infra.m_nCell[yDir],
                                               m_infra.m_nCell[zDir]);
        amrex::Real rhoSumNinv = rhoSum * ninv;
        rho -= rhoSumNinv;
    }
}

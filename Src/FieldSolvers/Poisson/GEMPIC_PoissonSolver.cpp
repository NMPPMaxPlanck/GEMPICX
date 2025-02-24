#include <memory>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_Solvers.H"
#ifdef AMREX_USE_FFT
#include "GEMPIC_FFT.H"
#endif
#ifdef AMREX_USE_HYPRE
#include "GEMPIC_Hypre.H"
#endif

using namespace Gempic::FieldSolvers;

/// @brief Applies the Poisson equation
class PoissonApply
{
private:
    std::shared_ptr<DeRhamComplex> m_deRham;

    // Auxiliary multifabs for Poisson operator
    DeRhamField<Grid::primal, Space::edge> m_primalEdge;
    DeRhamField<Grid::dual, Space::face> m_dualFace;

public:
    PoissonApply(std::shared_ptr<DeRhamComplex> deRham);

    void operator()(DeRhamField<Grid::dual, Space::cell>& rho,
                    DeRhamField<Grid::primal, Space::node>& phi);
};

/// @brief Applies the Poisson equation, using a ConjugateGradient solver instead of the Hodge
class PoissonApplyInverseHodge
{
private:
    using CGRhs = DeRhamField<Grid::primal, Space::edge>;
    using CGSol = DeRhamField<Grid::dual, Space::face>;

    std::shared_ptr<DeRhamComplex> m_deRham;
    const Gempic::ComputationalDomain m_infra;
    std::unique_ptr<SolverMethod<CGRhs, CGSol>> m_cgHodge;

    // Auxiliary multifabs for Poisson operator
    CGRhs m_primalEdge;
    CGSol m_dualFace;

public:
    PoissonApplyInverseHodge(std::shared_ptr<DeRhamComplex> deRham,
                             const Gempic::ComputationalDomain& infra);

    void operator()(DeRhamField<Grid::dual, Space::cell>& rho,
                    DeRhamField<Grid::primal, Space::node>& phi);
};

/**
 * @brief Construct a new Poisson Solver:: Poisson Solver object
 *
 * @param deRham
 * @param infra
 */
std::unique_ptr<PoissonSolverMethod> Gempic::FieldSolvers::make_poisson_solver (
    std::shared_ptr<DeRhamComplex> deRham, const Gempic::ComputationalDomain& infra)
{
    int maxCoarseningLevel = 10;
    int maxIter = 100;
    int mgBottomMaxIter = 100;
    int maxFmgIter = 0;
    int verbose = Gempic::Utils::Verbosity::level();
    int bottomVerbose = verbose;

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

    if (solver == "Amrex")
    {
        return std::make_unique<AmrexSolver> (AmrexSolver(infra, maxCoarseningLevel, maxIter,
                                                         maxFmgIter, mgBottomMaxIter, 1e-10, 1e-12,
                                                         verbose, bottomVerbose));
    }
    else if (solver == "ConjugateGradient")
    {
        using Rhs = DeRhamField<Grid::dual, Space::cell>;
        using Sol = DeRhamField<Grid::primal, Space::node>;
        return make_conjugate_gradient_unique_ptr<Rhs, Sol>(
            deRham, PoissonApply{deRham}, // make the operator the poisson operator
            [amrexSolver = AmrexSolver{infra, maxCoarseningLevel, maxIter, maxFmgIter,
                                       mgBottomMaxIter, 1e-10, 1e-12, verbose, bottomVerbose}] (
                Sol& z, Rhs& r) mutable
            {
                amrexSolver.solve(z, r); // precondition using the amrex solver
            },
            [=] (Rhs& b)
            {
                subtract_constant_part(b, infra); // average b to 0 in the usual way
            },
            1.e-11, verbose);
    }
    else if (solver == "ConjugateGradientInverseHodge")
    {
        using Rhs = DeRhamField<Grid::dual, Space::cell>;
        using Sol = DeRhamField<Grid::primal, Space::node>;
        return make_conjugate_gradient_unique_ptr<Rhs, Sol>(
            deRham,
            // make the operator the poisson inverse hodge operator
            PoissonApplyInverseHodge{deRham, infra}, nullptr,
            [=] (Rhs& b)
            {
                subtract_constant_part(b, infra); // average b to 0 in the usual way
            },
            1.e-11, verbose);
    }
    else if (solver == "FFT")
    {
#ifdef AMREX_USE_FFT
        if (infra.geometry().isAllPeriodic())
        {
            return std::make_unique<FFTSolver> (
                FFTSolver(infra, deRham->get_hodge_degree(), HodgeScheme::FDHodge));
        }
        else
        {
            amrex::Assert("Non-periodic boundary conditions not compatible with the FFT solver",
                          __FILE__, __LINE__);
        }
#else
        amrex::Assert("FFT was not compiled with GEMPICX and is thus not available as a solver",
                      __FILE__, __LINE__);
#endif
    }
    else if (solver == "Hypre")
    {
#ifdef AMREX_USE_HYPRE
        switch (deRham->get_hodge_degree())
        {
            case 2:
                return std::make_unique<HypreLinearSystem<2>>(infra, deRham, 1.e-11, verbose);
            case 4:
                return std::make_unique<HypreLinearSystem<4>>(infra, deRham, 1.e-11, verbose);
            case 6:
                return std::make_unique<HypreLinearSystem<6>>(infra, deRham, 1.e-11, verbose);
            default:
            {
                std::string msg = "Hodge degree " + std::to_string(deRham->get_hodge_degree()) +
                                  " not implemented for Hypre solver";
                amrex::Assert(msg.c_str(), __FILE__, __LINE__);
            }
            break;
        }
#else
        amrex::Assert("Hypre was not compiled with GEMPICX and is thus not available as a solver",
                      __FILE__, __LINE__);
#endif
    }
    else
    {
        std::string msg{solver + " not a recognised PoissonSolver"};
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }
    exit(1); // Calms the compiler even though we don't technically return anything
}

AmrexSolver::AmrexSolver(const ComputationalDomain& compDom,
                         int maxCoarseningLevel,
                         int maxIter,
                         int maxFmgIter,
                         int mgBottomMaxIter,
                         amrex::Real relTol,
                         amrex::Real absTol,
                         int verbose,
                         int bottomVerbose) :
    m_compDom{compDom}, m_relTol{relTol}, m_absTol{absTol}, m_cellVolume{compDom.cell_volume()}
{
    BL_PROFILE("Gempic::FieldSolvers::AmrexSolver::AmrexSolver()");
    amrex::LPInfo lpInfo;
    lpInfo.setMaxCoarseningLevel(maxCoarseningLevel);

    amrex::Real sigma = -1.0;
    m_linop = std::make_unique<amrex::MLNodeLaplacian>(
        amrex::Vector<amrex::Geometry>{compDom.geometry()},
        amrex::Vector<amrex::BoxArray>{compDom.m_grid},
        amrex::Vector<amrex::DistributionMapping>{compDom.m_distriMap}, lpInfo,
        amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, sigma);

    // Set boundary conditions on linear operator for lower end and higher end
    m_linop->setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic,
                                       amrex::LinOpBCType::Periodic)},
                         {AMREX_D_DECL(amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic,
                                       amrex::LinOpBCType::Periodic)});

    // Initialize solver class
    m_mlmg = std::make_unique<amrex::MLMG>(*m_linop);

    // Configure solver class
    m_mlmg->setMaxIter(maxIter);
    m_mlmg->setMaxFmgIter(maxFmgIter);
    m_mlmg->setBottomMaxIter(mgBottomMaxIter);
    m_mlmg->setVerbose(verbose);
    m_mlmg->setBottomVerbose(bottomVerbose);
    m_mlmg->setBottomSolver(amrex::BottomSolver::cg);
}

/**
 * Solves the Poisson equation for the given dual and primal fields using the second order AMReX
 * nodal Poisson solver
 *
 * @param[out] phi : The primal field to store the solution of the equation.
 * @param rho : The dual field representing the right-hand side of the equation.
 *
 * @throws None
 */
void AmrexSolver::solve (Forms::DeRhamField<Grid::primal, Space::node>& phi,
                        Forms::DeRhamField<Grid::dual, Space::cell>& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::AmrexSolver::solve()");

    // Sum of rhs needs to be 0 if domain is periodic in all directions
    subtract_constant_part(rho, m_compDom);

    // Solve Poisson equation
    m_mlmg->solve({&phi.m_data}, {&rho.m_data}, m_relTol, m_absTol);
    // AMReX Poisson solver does not use Hodge. Need to rescale phi
    phi *= 1.0 / m_cellVolume;

    phi.average_sync();
    phi.fill_boundary();
}

PoissonApply::PoissonApply(std::shared_ptr<DeRhamComplex> deRham) :
    m_deRham{deRham}, m_primalEdge(deRham), m_dualFace(deRham)
{
}

/**
 * Applies the Poisson operator to the given primal and dual fields.
 * The Poisson operator applies successively the grad, the hodge operator and the divergence
 * The order of the solver is hodgedegree
 *
 * @param[out] rho : The dual field to store the result of the operator application.
 * @param phi : The primal field to apply the operator to.
 *
 * @throws None
 */
void PoissonApply::operator ()(DeRhamField<Grid::dual, Space::cell>& rho,
                              DeRhamField<Grid::primal, Space::node>& phi)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonApply::operator()()");
    m_deRham->grad(m_primalEdge, phi);
    m_primalEdge *= -1; // E = -grad phi
    m_deRham->hodge(m_dualFace, m_primalEdge);
    m_deRham->div(rho, m_dualFace);
    // add penalty term to avoid nullspace
    // amrex::Real penalty = 1.0e-8;
    // rho += penalty;
    // Sum of rhs needs to be 0 if domain is periodic in all directions
    // subtract_constant_part(rho);
}

PoissonApplyInverseHodge::PoissonApplyInverseHodge(std::shared_ptr<DeRhamComplex> deRham,
                                                   const Gempic::ComputationalDomain& infra) :
    m_deRham{deRham},
    m_infra{infra},
    m_cgHodge{make_conjugate_gradient_unique_ptr<CGRhs, CGSol>(
        deRham, [=] (CGRhs& primalEdge, CGSol& dualFace) { deRham->hodge(primalEdge, dualFace); })},
    m_primalEdge(deRham),
    m_dualFace(deRham)
{
}

/**
 * Applies the Poisson operator with inverse Hodge transformation to the given primal and dual
 * fields. The order of the solver is hodgedegree - 2 (for hodgedegree = 4 and 6) and 2 for
 * hodgedegree = 2, for which Hodge is diagonal
 *
 * @param[out] rho : The dual field to store the result of the operator application after inverse
 * Hodge transformation.
 * @param phi : The primal field to apply the operator to.
 *
 * @return None
 */
void PoissonApplyInverseHodge::operator ()(DeRhamField<Grid::dual, Space::cell>& rho,
                                          DeRhamField<Grid::primal, Space::node>& phi)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonApplyInverseHodge::operator()()");
    // Conjugate gradient to compute inverse Hodge
    m_deRham->grad(m_primalEdge, phi);
    m_primalEdge *= -1; // E = -grad phi
    m_cgHodge->solve(m_dualFace, m_primalEdge);
    m_deRham->div(rho, m_dualFace);

    // Sum of rhs needs to be 0 if domain is periodic in all directions
    subtract_constant_part(rho, m_infra);
}

/// Ensures that the average value of the field is 0 if domain is periodic in all directions
void Gempic::FieldSolvers::subtract_constant_part (DeRhamField<Grid::dual, Space::cell>& rho,
                                                  const ComputationalDomain& compDom)
{
    BL_PROFILE("Gempic::FieldSolvers::subtract_constant_part()");
    if (compDom.geometry().isAllPeriodic())
    {
        amrex::Real rhoSum = rho.m_data.sum_unique(0, false, compDom.geometry().periodicity());
        amrex::Real ninv = 1.0 / GEMPIC_D_MULT(compDom.m_nCell[xDir], compDom.m_nCell[yDir],
                                               compDom.m_nCell[zDir]);
        amrex::Real rhoSumNinv = rhoSum * ninv;
        rho -= rhoSumNinv;
    }
}

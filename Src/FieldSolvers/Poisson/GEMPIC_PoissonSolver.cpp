#include <memory>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_NumericalIntegrationDifferentiation.H"
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
    Gempic::ComputationalDomain const m_infra;
    std::unique_ptr<SolverMethod<CGRhs, CGSol>> m_cgHodge;

    // Auxiliary multifabs for Poisson operator
    CGRhs m_primalEdge;
    CGSol m_dualFace;

public:
    PoissonApplyInverseHodge(std::shared_ptr<DeRhamComplex> deRham,
                             Gempic::ComputationalDomain const& infra);

    void operator()(DeRhamField<Grid::dual, Space::cell>& rho,
                    DeRhamField<Grid::primal, Space::node>& phi);
};

std::unique_ptr<PoissonSolverMethod> Gempic::FieldSolvers::Impl::make_specific_poisson_solver (
    std::shared_ptr<DeRhamComplex> deRham,
    Gempic::ComputationalDomain const& infra,
    std::string const& solver,
    amrex::Real const relTol,
    amrex::Real const absTol,
    bool const tolerancesGiven)
{
    int maxCoarseningLevel = 10;
    int maxIter = 100;
    int mgBottomMaxIter = 100;
    int maxFmgIter = 0;
    int verbose = Gempic::Utils::Verbosity::level();
    int bottomVerbose = verbose;

    if (solver == "FFT")
    {
        if (tolerancesGiven)
        {
            amrex::Warning("FFT Poisson solver does not use relTol and absTol.\n");
        }
#ifdef AMREX_USE_FFT
        if (infra.geometry().isAllPeriodic())
        {
            return std::make_unique<FFTSolver> (
                FFTSolver(infra, deRham->get_hodge_degree(), HodgeScheme::FDHodge));
        }
        else
        {
            GEMPIC_ERROR("Non-periodic boundary conditions not compatible with the FFT solver");
        }
#else
        GEMPIC_ERROR("FFT was not compiled with GEMPICX and is thus not available as a solver");
#endif
    }
    else if (solver == "Amrex")
    {
        if (infra.geometry().isAllPeriodic())
        {
            return std::make_unique<AmrexSolver> (infra, maxCoarseningLevel, maxIter, maxFmgIter,
                                                 mgBottomMaxIter, relTol, absTol, verbose,
                                                 bottomVerbose);
        }
        else
        {
            GEMPIC_ERROR(
                "Non-periodic boundary conditions not compatible with the AMReX Poisson solver");
        }
    }
    else if (solver == "ConjugateGradient")
    {
        using Rhs = DeRhamField<Grid::dual, Space::cell>;
        using Sol = DeRhamField<Grid::primal, Space::node>;
        return make_conjugate_gradient_unique_ptr<Rhs, Sol>(
            deRham, PoissonApply{deRham}, // make the operator the poisson operator
            [amrexSolver = AmrexSolver{infra, maxCoarseningLevel, maxIter, maxFmgIter,
                                       mgBottomMaxIter, relTol, absTol, verbose, bottomVerbose}] (
                Sol& z, Rhs& r) mutable
            {
                amrexSolver.solve(z, r); // precondition using the amrex solver
            },
            [=] (Rhs& b)
            {
                subtract_constant_part(b, infra); // average b to 0 in the usual way
            },
            relTol, absTol, verbose);
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
            relTol, absTol, verbose);
    }
    else if (solver == "Hypre")
    {
#ifdef AMREX_USE_HYPRE
        switch (deRham->get_hodge_degree())
        {
            case 2:
                return std::make_unique<HypreLinearSystem<2>>(infra, deRham, relTol, absTol,
                                                              verbose);
            case 4:
                return std::make_unique<HypreLinearSystem<4>>(infra, deRham, relTol, absTol,
                                                              verbose);
            case 6:
                return std::make_unique<HypreLinearSystem<6>>(infra, deRham, relTol, absTol,
                                                              verbose);
            default:
            {
                GEMPIC_ERROR("Hodge degree " + std::to_string(deRham->get_hodge_degree()) +
                             " not implemented for Hypre solver");
            }
            break;
        }
#else
        GEMPIC_ERROR("Hypre was not compiled with GEMPICX and is thus not available as a solver");
#endif
    }
    else
    {
        GEMPIC_ERROR(solver + " not a recognised PoissonSolver");
    }
    exit(1); // Calms the compiler even though we don't technically return anything
}

AmrexSolver::AmrexSolver(ComputationalDomain const& compDom,
                         int maxCoarseningLevel,
                         int maxIter,
                         int maxFmgIter,
                         int mgBottomMaxIter,
                         amrex::Real relTol,
                         amrex::Real absTol,
                         int verbose,
                         int bottomVerbose) :
    m_compDom{compDom}, m_relTol{relTol}, m_absTol{absTol}
{
    BL_PROFILE("Gempic::FieldSolvers::AmrexSolver::AmrexSolver()");
    // Check for periodic boundaries, required for this solver
    if (!compDom.geometry().periodicity().isAllPeriodic())
    {
        GEMPIC_ERROR("Amrex Poisson solver requires periodic boundary conditions");
    }
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

void AmrexSolver::solve (Forms::DeRhamField<Grid::primal, Space::node>& phi,
                        Forms::DeRhamField<Grid::dual, Space::cell>& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::AmrexSolver::solve()");

    // Sum of rhs needs to be 0 if domain is periodic in all directions
    subtract_constant_part(rho, m_compDom);

    // Solve Poisson equation
    m_mlmg->solve({&phi.m_data}, {&rho.m_data}, m_relTol, m_absTol);
    // AMReX Poisson solver does not use Hodge. Need to rescale phi
    phi *= 1.0 / m_compDom.cell_volume();

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
 * @param[out] rho The dual field to store the result of the operator application.
 * @param phi The primal field to apply the operator to.
 */
void PoissonApply::operator ()(DeRhamField<Grid::dual, Space::cell>& rho,
                              DeRhamField<Grid::primal, Space::node>& phi)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonApply::operator()()");
    grad(m_primalEdge, phi);
    m_primalEdge *= -1; // E = -grad phi
    hodge(m_dualFace, m_primalEdge);
    div(rho, m_dualFace);
    // add penalty term to avoid nullspace
    // amrex::Real penalty = 1.0e-8;
    // rho += penalty;
    // Sum of rhs needs to be 0 if domain is periodic in all directions
    // subtract_constant_part(rho);
}

PoissonApplyInverseHodge::PoissonApplyInverseHodge(std::shared_ptr<DeRhamComplex> deRham,
                                                   Gempic::ComputationalDomain const& infra) :
    m_deRham{deRham},
    m_infra{infra},
    m_cgHodge{make_conjugate_gradient_unique_ptr<CGRhs, CGSol>(
        deRham, [=] (CGRhs& primalEdge, CGSol& dualFace) { hodge(primalEdge, dualFace); })},
    m_primalEdge(deRham),
    m_dualFace(deRham)
{
}

/**
 * Applies the Poisson operator with inverse Hodge transformation to the given primal and dual
 * fields. The order of the solver is hodgedegree - 2 (for hodgedegree = 4 and 6) and 2 for
 * hodgedegree = 2, for which Hodge is diagonal
 *
 * @param[out] rho The dual field to store the result of the operator application after inverse
 * Hodge transformation.
 * @param phi The primal field to apply the operator to.
 */
void PoissonApplyInverseHodge::operator ()(DeRhamField<Grid::dual, Space::cell>& rho,
                                          DeRhamField<Grid::primal, Space::node>& phi)
{
    BL_PROFILE("Gempic::FieldSolvers::PoissonApplyInverseHodge::operator()()");
    // Conjugate gradient to compute inverse Hodge
    grad(m_primalEdge, phi);
    m_primalEdge *= -1; // E = -grad phi
    m_cgHodge->solve(m_dualFace, m_primalEdge);
    div(rho, m_dualFace);

    // Sum of rhs needs to be 0 if domain is periodic in all directions
    subtract_constant_part(rho, m_infra);
}

/// Ensures that the average value of the field is 0 if domain is periodic in all directions
void Gempic::FieldSolvers::subtract_constant_part (DeRhamField<Grid::dual, Space::cell>& rho,
                                                  ComputationalDomain const& compDom)
{
    BL_PROFILE("Gempic::FieldSolvers::subtract_constant_part()");
    if (compDom.geometry().isAllPeriodic())
    {
        amrex::Real rhoInt = compute_rho_integral(rho);
        rho -= rhoInt;
    }
}

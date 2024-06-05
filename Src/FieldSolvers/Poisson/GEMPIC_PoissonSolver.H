#ifndef GEMPIC_POISSONSOLVER_H_
#define GEMPIC_POISSONSOLVER_H_

#include <type_traits>

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_Print.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{
using namespace Forms;

enum class Operator
{
    poisson,
    poissonInverseHodge,
    hodge
};

class PoissonSolver;

template <class rhs, class sol, Operator op>
class ConjugateGradient
{
private:
    std::shared_ptr<DeRhamComplex> m_deRham;
    std::shared_ptr<PoissonSolver> m_poissonSolver;
    rhs m_residual, m_ap;
    sol m_p, m_z;
    amrex::Real m_relTol, m_absTol;
    amrex::Real m_maxiter;
    int m_verbose;

public:
    ConjugateGradient(std::shared_ptr<DeRhamComplex> drc,
                      std::shared_ptr<PoissonSolver> pslv,
                      amrex::Real tol = 1.e-11,
                      int verbose = 0) :
        m_deRham{drc},
        m_poissonSolver{pslv},
        m_residual(drc),
        m_ap(drc),
        m_p(drc),
        m_z(drc),
        m_relTol{tol},
        m_absTol{1.e-15},
        m_maxiter{1000}
    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    ConjugateGradient(std::shared_ptr<DeRhamComplex> drc,
                      amrex::Real tol = 1.e-11,
                      int verbose = 0) :
        ConjugateGradient{drc, nullptr, tol, verbose}
    {
        static_assert(op == Operator::hodge,
                      "ConjugateGradient: Operator::poisson and Operator::poissonInverseHodge "
                      "require PoissonSolver object!");
    }

    /**
     * Solves the linear system using Conjugate Gradient method.
     *
     * @param b The right-hand side vector
     * @param x The solution vector
     *
     * @return void
     *
     * @throws None
     */
    void solve (rhs& b, sol& x)
    {
        amrex::Real pAp{0}, normResSquareOld{0}, normResSquareNew{0};
        amrex::Real alpha{0}, beta{0};

        // average rhs to zero in periodic case
        average_b_zero(b);
        // Compute norm of b for relative error
        amrex::Real normbSquare = dot_product(b, b);

        // Compute A*x (x contains initial guess for solution)
        cg_mat_mul(x, m_residual);
        // compute b - A*x
        linear_combination(m_residual, -1., m_residual, 1., b);
        normResSquareOld = dot_product(m_residual, m_residual);
        if ((sqrt(normResSquareOld) < m_relTol * sqrt(normbSquare)) ||
            sqrt(normResSquareOld) < m_absTol)
        {
            return;
        }
        copy(m_p, m_residual);  // copy residual into p

        if (m_verbose > 0)
        {
            amrex::Print() << "CG initialization " << sqrt(normResSquareOld) << " "
                           << sqrt(normbSquare) << '\n';
        }

        bool hasConverged{false};
        for (int iter = 0; iter < m_maxiter; iter++)
        {
            cg_mat_mul(m_p, m_ap);
            pAp = dot_product(m_p, m_ap);
            alpha = normResSquareOld / pAp;
            linear_combination(x, 1., x, alpha, m_p);
            linear_combination(m_residual, 1., m_residual, -alpha, m_ap);
            normResSquareNew = dot_product(m_residual, m_residual);

            if ((sqrt(normResSquareNew) < m_relTol * sqrt(normbSquare)) ||
                sqrt(normResSquareNew) < m_absTol)
            {
                if (m_verbose > 0)
                {
                    amrex::Print() << "CG final iteration residual " << iter << " "
                                   << sqrt(normResSquareNew / normbSquare) << '\n';
                }
                hasConverged = true;
                break;
            }
            beta = normResSquareNew / normResSquareOld;
            normResSquareOld = normResSquareNew;
            linear_combination(m_p, beta, m_p, 1., m_residual);

            if (m_verbose > 1)
            {
                amrex::Print() << "CG " << iter << " " << sqrt(normResSquareOld) << '\n';
            }
        }
        if (m_verbose > 0 && !hasConverged)
        {
            amrex::Print(amrex::ErrorStream())
                << "WARNING: CG has not converged after " << m_maxiter << " iterations "
                << sqrt(normResSquareOld / normbSquare) << std::endl;
        }
        x.average_sync();
        x.fill_boundary();
    }

    /**
     * Solves the linear system using Preconditioned Conjugate Gradient method.
     * Does not work well with the AMReX solver as a preconditioner (at least in periodic domain).
     * Can be useful later with a better preconditioner (from HYPRE for example)
     *
     * @param b The right-hand side vector
     * @param x The solution vector
     *
     * @return void
     *
     * @throws None
     */
    void precond_solve (rhs& b, sol& x)
    {
        amrex::Real pAp{0}, rdotzNew{0};
        amrex::Real alpha{0}, beta{0};

        // average rhs to zero in periodic case
        average_b_zero(b);
        // Compute norm of b for relative error
        amrex::Real normbSquare = dot_product(b, b);

        // Compute A*x (x contains initial guess for solution)
        cg_mat_mul(x, m_residual);
        // compute b - A*x
        linear_combination(m_residual, -1., m_residual, 1., b);
        amrex::Real normResSquare = dot_product(m_residual, m_residual);
        if ((sqrt(normResSquare) < m_relTol * sqrt(normbSquare)) || sqrt(normResSquare) < m_absTol)
        {
            return;
        }

        // possibly precondition system
        precond(m_z, m_residual);
        amrex::Real rdotzOld = dot_product(m_residual, m_z);
        copy(m_p, m_z);  // copy z into p

        if (m_verbose > 0)
        {
            amrex::Print() << "CG initialization " << sqrt(normResSquare) << " "
                           << sqrt(normbSquare) << '\n';
        }

        bool hasConverged{false};
        for (int iter = 0; iter < m_maxiter; iter++)
        {
            cg_mat_mul(m_p, m_ap);
            pAp = dot_product(m_p, m_ap);
            alpha = rdotzOld / pAp;
            linear_combination(x, 1., x, alpha, m_p);
            linear_combination(m_residual, 1., m_residual, -alpha, m_ap);
            normResSquare = dot_product(m_residual, m_residual);

            if ((sqrt(normResSquare) < m_relTol * sqrt(normbSquare)) ||
                sqrt(normResSquare) < m_absTol)
            {
                if (m_verbose > 0)
                {
                    amrex::Print() << "CG final iteration residual " << iter << " "
                                   << sqrt(normResSquare / normbSquare) << '\n';
                }
                hasConverged = true;
                break;
            }
            precond(m_z, m_residual);
            rdotzNew = dot_product(m_residual, m_z);
            beta = rdotzNew / rdotzOld;
            rdotzOld = rdotzNew;
            linear_combination(m_p, beta, m_p, 1., m_z);

            if (m_verbose > 1)
            {
                amrex::Print() << "CG " << iter << " " << sqrt(normResSquare) << '\n';
            }
        }
        if (m_verbose > 0 && !hasConverged)
        {
            amrex::Print(amrex::ErrorStream())
                << "WARNING: CG has not converged after " << m_maxiter << " iterations "
                << sqrt(normResSquare / normbSquare) << std::endl;
        }
        x.average_sync();
        x.fill_boundary();
    }

    inline void cg_mat_mul(sol& x, rhs& mResidual);
    inline void precond(sol& z, rhs& r);
    inline void average_b_zero(rhs& b);
};

class PoissonSolver
{
private:
    int m_maxIter;
    int m_maxCoarseningLevel;
    int m_maxFmgIter;
    int m_mgBottomMaxIter;
    int m_verbose;
    int m_bottomVerbose;

    std::shared_ptr<Forms::DeRhamComplex> m_deRham;
    const ComputationalDomain m_infra;
    //DeRhamField<Grid::primal, Space::edge> m_residual;

    // Auxiliary multifabs for Poisson operator
    DeRhamField<Grid::primal, Space::edge> m_primalEdge;
    DeRhamField<Grid::dual, Space::face> m_dualFace;

    std::unique_ptr<ConjugateGradient<DeRhamField<Grid::primal, Space::edge>,
                                      DeRhamField<Grid::dual, Space::face>,
                                      Operator::hodge>>
        m_cgHodge;

public:
    PoissonSolver(std::shared_ptr<Forms::DeRhamComplex> deRham, const ComputationalDomain& infra);

    ~PoissonSolver();

    void solve_amrex(DeRhamField<Grid::dual, Space::cell>& rho,
                     DeRhamField<Grid::primal, Space::node>& phi);

    void apply_poisson_operator(DeRhamField<Grid::primal, Space::node>& phi,
                                DeRhamField<Grid::dual, Space::cell>& rho);

    void apply_poisson_operator_inverse_hodge(DeRhamField<Grid::primal, Space::node>& phi,
                                              DeRhamField<Grid::dual, Space::cell>& rho);

    void subtract_constant_part(DeRhamField<Grid::dual, Space::cell>& rho);
};

template <class rhs, class sol, Operator op>
inline void ConjugateGradient<rhs, sol, op>::cg_mat_mul(sol& x, rhs& Ax)
{
    if constexpr (op == Operator::hodge)
    {
        m_deRham->hodge(x, Ax);
    }
    else if constexpr (op == Operator::poisson)
    {
        m_poissonSolver->apply_poisson_operator(x, Ax);
    }
    else if constexpr (op == Operator::poissonInverseHodge)
    {
        m_poissonSolver->apply_poisson_operator_inverse_hodge(x, Ax);
    }
    else
    {
        static_assert(
            op == Operator::hodge || op == Operator::poisson || op == Operator::poissonInverseHodge,
            "Conjugate Gradient: operator not implemented");
    }
}

template <class rhs, class sol, Operator op>
inline void ConjugateGradient<rhs, sol, op>::precond(sol& z, rhs& r)
{
    if constexpr (op == Operator::hodge)
    {
        copy(z, r);  // not preconditioned
    }
    else if constexpr (op == Operator::poisson)
    {
        m_poissonSolver->solve_amrex(r, z);
    }
    else if constexpr (op == Operator::poissonInverseHodge)  // not preconditioned
    {
        //m_poissonSolver->solve_amrex(r, z);
        copy(z, r);  // not preconditioned
    }
    else
    {
        static_assert(
            op == Operator::hodge || op == Operator::poisson || op == Operator::poissonInverseHodge,
            "Conjugate Gradient: operator not implemented");
    }
}

/// average rhs to zero in periodic case
template <class rhs, class sol, Operator op>
inline void ConjugateGradient<rhs, sol, op>::average_b_zero(rhs& b)
{
    if constexpr (op == Operator::poisson || op == Operator::poissonInverseHodge)
    {
        m_poissonSolver->subtract_constant_part(b);
    }
}
}  // namespace Gempic::FieldSolvers

#endif

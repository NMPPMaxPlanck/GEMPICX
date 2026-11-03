/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_SOLVERS_H_
#define GEMPIC_SOLVERS_H_

#include <memory>
#include <type_traits>

#include <AMReX.H>
#include <AMReX_Print.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{
// Strategy pattern
template <typename Rhs, typename Sol>
class SolverMethod
{
public:
    virtual void solve(Sol& out, Rhs const& in) = 0;
    virtual ~SolverMethod() {};
};

/** @brief Conjugate gradient solver class to solve equations of the type
 * A * x = b
 *
 * User must provide an operation that takes x and yields b (e.g. a matrix multiplication)
 * Additionally, one can provide
 * - a precondition operation that applies z = P^{-1} r to a residual r (used only by
 *   precond_solve(), solve() ignores it)
 * - a function that checks whether the system is solvable for a given b
 * - relative tolerance for the solver
 * - absolute tolerance for the solver
 * - verbosity level
 *
 * Rhs and Sol may be different types. They must support copy, linear_combination and dot_product
 * between each other. The solver does not fill ghost cells of the solution; callers that read
 * ghosts of x afterwards must call fill_boundary() themselves.
 */
template <class Rhs,
          class Sol,
          class Operation,
          class PreconditionOp = std::nullptr_t,
          class SolvableOp = std::nullptr_t>
class ConjugateGradient : public SolverMethod<Rhs, Sol>
{
private:
    PreconditionOp m_precondOp;
    SolvableOp m_solvableOp;
    Rhs m_residual, m_ap;
    Sol m_p, m_z;
    amrex::Real m_relTol, m_absTol;
    amrex::Real m_maxiter{1000};
    int m_verbose;
    int m_convergedAtIter;
    amrex::Real m_convergedWithTol;

    static_assert(std::is_invocable_r_v<void, Operation, Rhs&, Sol const&>,
                  "Given operator must compute A*x (signature void(Rhs&, Sol const&), meaning "
                  "void(A*x, x))");
    static_assert(std::is_null_pointer_v<PreconditionOp> ||
                      std::is_invocable_r_v<void, PreconditionOp, Sol&, Rhs&>,
                  "Given precondition operation must follow the precondition signature "
                  "(void(Sol&, Rhs&))");
    static_assert(std::is_null_pointer_v<SolvableOp> ||
                      std::is_invocable_r_v<void, SolvableOp, Rhs const&>,
                  "Given solvability check must follow the signature void(Rhs const&)");

public:
    Operation m_operator;

    ConjugateGradient(std::shared_ptr<Forms::DeRhamComplex> drc,
                      Operation&& matMultOp,
                      PreconditionOp&& precondOp = nullptr,
                      SolvableOp&& solvableOp = nullptr,
                      amrex::Real relTol = 1e-10,
                      amrex::Real absTol = 1e-12,
                      int verbose = 0) :
        ConjugateGradient(Rhs(drc),
                          Rhs(drc),
                          Sol(drc),
                          Sol(drc),
                          std::move(matMultOp),
                          std::move(precondOp),
                          std::move(solvableOp),
                          relTol,
                          absTol,
                          verbose)
    {
    }

    /**
     * Constructor for field types that are not built from a DeRhamComplex. The work fields are
     * allocated by the caller and must not share memory with each other or with the fields passed
     * to solve() (copies of DiscreteField are shallow).
     *
     * @param residual, ap work fields of the right-hand side type
     * @param p, z work fields of the solution type
     * @param matMultOp operation computing A x, signature void(Rhs& ax, Sol const& x)
     * @param precondOp preconditioner z = P^{-1} r, signature void(Sol& z, Rhs& r) (only used by
     * precond_solve())
     * @param solvableOp solvability check, signature void(Rhs const& b)
     * @param relTol relative tolerance (to the norm of b)
     * @param absTol absolute tolerance
     * @param verbose verbosity level (at least the global GEMPIC verbosity)
     */
    ConjugateGradient(Rhs residual,
                      Rhs ap,
                      Sol p,
                      Sol z,
                      Operation&& matMultOp,
                      PreconditionOp&& precondOp = nullptr,
                      SolvableOp&& solvableOp = nullptr,
                      amrex::Real relTol = 1e-10,
                      amrex::Real absTol = 1e-12,
                      int verbose = 0) :
        m_precondOp{std::move(precondOp)},
        m_solvableOp{std::move(solvableOp)},
        m_residual(std::move(residual)),
        m_ap(std::move(ap)),
        m_p(std::move(p)),
        m_z(std::move(z)),
        m_relTol{relTol},
        m_absTol{absTol},
        m_operator{std::move(matMultOp)}
    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    /**
     * Solves the linear system using Conjugate Gradient method. PreconditionOp is not used.
     *
     * @param[inout] x The solution vector (on input: the initial guess)
     * @param b The right-hand side vector
     */
    void solve (Sol& x, Rhs const& b) override
    {
        amrex::Real pAp{0}, normResSquareOld{0}, normResSquareNew{0};
        amrex::Real alpha{0}, beta{0};

        check_solvability(b);
        // Compute norm of b for relative error
        amrex::Real normbSquare = dot_product(b, b);

        // Compute A*x (x contains initial guess for solution)
        cg_mat_mul(m_residual, x);
        // compute b - A*x
        linear_combination(m_residual, -1., m_residual, 1., b);
        normResSquareOld = dot_product(m_residual, m_residual);
        if ((sqrt(normResSquareOld) < m_relTol * sqrt(normbSquare)) ||
            sqrt(normResSquareOld) < m_absTol)
        {
            m_convergedAtIter = 0;
            m_convergedWithTol = sqrt(normResSquareOld);
            return;
        }
        copy(m_p, m_residual); // copy residual into p

        if (m_verbose > 0)
        {
            amrex::Print() << "CG initialization " << sqrt(normResSquareOld) << " "
                           << sqrt(normbSquare) << '\n';
        }

        bool hasConverged{false};
        m_convergedAtIter = m_maxiter;
        for (int iter = 0; iter < m_maxiter; iter++)
        {
            cg_mat_mul(m_ap, m_p);
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
                m_convergedAtIter = iter;
                m_convergedWithTol = sqrt(normResSquareNew);
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
            amrex::Warning("Warning: CG has not converged after " + std::to_string(m_maxiter) +
                           " iterations " + std::to_string(sqrt(normResSquareOld / normbSquare)));
        }
        if (!hasConverged)
        {
            m_convergedAtIter = m_maxiter;
            m_convergedWithTol = sqrt(normResSquareNew);
        }
    }

    /**
     * Solves the linear system using Preconditioned Conjugate Gradient method.
     * Does not work well with the AMReX solver as a preconditioner (at least in periodic domain).
     * Can be useful later with a better preconditioner (from HYPRE for example)
     *
     * @param[out] x The solution vector (on input: the initial guess)
     * @param b The right-hand side vector
     */
    void precond_solve (Sol& x, Rhs const& b)
    {
        amrex::Real pAp{0}, rdotzNew{0};
        amrex::Real alpha{0}, beta{0};

        check_solvability(b);
        // Compute norm of b for relative error
        amrex::Real normbSquare = dot_product(b, b);

        // Compute A*x (x contains initial guess for solution)
        cg_mat_mul(m_residual, x);
        // compute b - A*x
        linear_combination(m_residual, -1., m_residual, 1., b);
        amrex::Real normResSquare = dot_product(m_residual, m_residual);
        if ((sqrt(normResSquare) < m_relTol * sqrt(normbSquare)) || sqrt(normResSquare) < m_absTol)
        {
            m_convergedAtIter = 0;
            m_convergedWithTol = sqrt(normResSquare);
            return;
        }

        // possibly precondition system
        precond(m_z, m_residual);
        amrex::Real rdotzOld = dot_product(m_residual, m_z);
        copy(m_p, m_z); // copy z into p

        if (m_verbose > 0)
        {
            amrex::Print() << "CG initialization " << sqrt(normResSquare) << " "
                           << sqrt(normbSquare) << '\n';
        }

        bool hasConverged{false};
        for (int iter = 0; iter < m_maxiter; iter++)
        {
            cg_mat_mul(m_ap, m_p);
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
                m_convergedAtIter = iter;
                m_convergedWithTol = sqrt(normResSquare);
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
            amrex::Warning("Warning: CG has not converged after " + std::to_string(m_maxiter) +
                           " iterations " + std::to_string(sqrt(normResSquare / normbSquare)));
        }
        if (!hasConverged)
        {
            m_convergedAtIter = m_maxiter;
            m_convergedWithTol = sqrt(normResSquare);
        }
    }

    inline void cg_mat_mul (Rhs& mResidual, Sol const& x) { m_operator(mResidual, x); }

    inline void precond (Sol& z, Rhs& r)
    {
        if constexpr (!std::is_null_pointer_v<PreconditionOp>)
        {
            m_precondOp(z, r);
        }
        else
        {
            copy(z, r);
        }
    }

    inline void check_solvability (Rhs const& b)
    {
        if constexpr (!std::is_null_pointer_v<SolvableOp>)
        {
            m_solvableOp(b);
        }
    }

    inline amrex::Real get_tol () { return m_convergedWithTol; }

    inline amrex::Real get_iter () { return m_convergedAtIter; }
};

/// Helper function for template deduction. Allows you to specify Rhs and Sol explicitly, but have
/// the operator types automatically deduced.
template <class Rhs,
          class Sol,
          class Operation,
          class PreconditionOp = std::nullptr_t,
          class SolvableOp = std::nullptr_t>
inline auto make_conjugate_gradient (std::shared_ptr<Forms::DeRhamComplex> drc,
                                     Operation&& matMultOp,
                                     PreconditionOp&& precondOp = nullptr,
                                     SolvableOp&& solvableOp = nullptr,
                                     amrex::Real relTol = 1e-10,
                                     amrex::Real absTol = 1e-12,
                                     int verbose = 0)
{
    return ConjugateGradient<Rhs, Sol, Operation, PreconditionOp, SolvableOp>(
        drc, std::forward<Operation> (matMultOp), std::forward<PreconditionOp>(precondOp),
        std::forward<SolvableOp>(solvableOp), relTol, absTol, verbose);
}

/// Helper function for template deduction. Allows you to specify Rhs and Sol explicitly, but have
/// the operator types automatically deduced.
template <class Rhs,
          class Sol,
          class Operation,
          class PreconditionOp = std::nullptr_t,
          class SolvableOp = std::nullptr_t>
inline auto make_conjugate_gradient_unique_ptr (std::shared_ptr<Forms::DeRhamComplex> drc,
                                                Operation&& matMultOp,
                                                PreconditionOp&& precondOp = nullptr,
                                                SolvableOp&& solvableOp = nullptr,
                                                amrex::Real relTol = 1e-10,
                                                amrex::Real absTol = 1e-12,
                                                int verbose = 0)
{
    return std::make_unique<ConjugateGradient<Rhs, Sol, Operation, PreconditionOp, SolvableOp>>(
        drc, std::forward<Operation> (matMultOp), std::forward<PreconditionOp>(precondOp),
        std::forward<SolvableOp>(solvableOp), relTol, absTol, verbose);
}
} //namespace Gempic::FieldSolvers
#endif

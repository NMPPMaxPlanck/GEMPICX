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
    virtual void solve(Sol& out, Rhs& in) = 0;
    virtual ~SolverMethod() {};
};

template <class Rhs,
          class Sol,
          class Operation,
          class PreconditionOp = std::nullptr_t,
          class AverageB0Op = std::nullptr_t>
class ConjugateGradient : public SolverMethod<Rhs, Sol>
{
private:
    PreconditionOp m_precondOp;
    AverageB0Op m_avgB0Op;
    Rhs m_residual, m_ap;
    Sol m_p, m_z;
    amrex::Real m_relTol, m_absTol;
    amrex::Real m_maxiter{1000};
    int m_verbose;
    int m_convergedAtIter;
    amrex::Real m_convergedWithTol;
    amrex::Real m_mysqrtres;

public:
    Operation m_operator;

    ConjugateGradient(std::shared_ptr<Forms::DeRhamComplex> drc,
                      Operation&& matMultOp,
                      PreconditionOp&& precondOp = nullptr,
                      AverageB0Op&& avgB0Op = nullptr,
                      amrex::Real relTol = 1e-10,
                      amrex::Real absTol = 1e-12,
                      int verbose = 0) :
        m_precondOp{std::move(precondOp)},
        m_avgB0Op{std::move(avgB0Op)},
        m_residual(drc),
        m_ap(drc),
        m_p(drc),
        m_z(drc),
        m_relTol{relTol},
        m_absTol{absTol},
        m_operator{std::move(matMultOp)}

    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
        static_assert(
            std::is_invocable_r_v<void, Operation, Rhs&, Sol&>,
            "Given operator must compute A*x (signature void(Rhs&, Sol&), meaning void(A*x, x))");
        static_assert(std::is_null_pointer_v<PreconditionOp> ||
                          std::is_invocable_r_v<void, PreconditionOp, Sol&, Rhs&>,
                      "Given precondition operation must follow the precondition signature "
                      "(void(Sol&, Rhs&))");
        static_assert(
            std::is_null_pointer_v<AverageB0Op> || std::is_invocable_r_v<void, AverageB0Op, Rhs&>,
            "Given average_b_zero operation must follow the signature void(Rhs&)");
    }

    /**
     * Solves the linear system using Conjugate Gradient method.
     *
     * @param[out] x The solution vector
     * @param b The right-hand side vector
     */
    void solve (Sol& x, Rhs& b) override
    {
        amrex::Real pAp{0}, normResSquareOld{0}, normResSquareNew{0};
        amrex::Real alpha{0}, beta{0};

        // average rhs to zero in periodic case
        average_b_zero(b);
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
                m_mysqrtres = sqrt(normResSquareOld);
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
        if (!hasConverged)
        {
            m_convergedAtIter = m_maxiter;
            m_convergedWithTol = sqrt(normResSquareNew);
            ;
        }
        x.fill_boundary();
    }

    /**
     * Solves the linear system using Preconditioned Conjugate Gradient method.
     * Does not work well with the AMReX solver as a preconditioner (at least in periodic domain).
     * Can be useful later with a better preconditioner (from HYPRE for example)
     *
     * @param[out] x The solution vector
     * @param b The right-hand side vector
     */
    void precond_solve (Sol& x, Rhs& b)
    {
        amrex::Real pAp{0}, rdotzNew{0};
        amrex::Real alpha{0}, beta{0};

        // average rhs to zero in periodic case
        average_b_zero(b);
        // Compute norm of b for relative error
        amrex::Real normbSquare = dot_product(b, b);

        // Compute A*x (x contains initial guess for solution)
        cg_mat_mul(m_residual, x);
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
            amrex::Print(amrex::ErrorStream())
                << "WARNING: CG has not converged after " << m_maxiter << " iterations "
                << sqrt(normResSquare / normbSquare) << std::endl;
        }
        if (!hasConverged)
        {
            m_convergedAtIter = m_maxiter;
            m_convergedWithTol = sqrt(normResSquare);
        }
        x.fill_boundary();
    }

    inline void cg_mat_mul (Rhs& mResidual, Sol& x) { m_operator(mResidual, x); }

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

    inline void average_b_zero (Rhs& b)
    {
        if constexpr (!std::is_null_pointer_v<AverageB0Op>)
        {
            m_avgB0Op(b);
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
          class AverageB0Op = std::nullptr_t>
inline auto make_conjugate_gradient (std::shared_ptr<Forms::DeRhamComplex> drc,
                                     Operation&& matMultOp,
                                     PreconditionOp&& precondOp = nullptr,
                                     AverageB0Op&& avgB0Op = nullptr,
                                     amrex::Real relTol = 1e-10,
                                     amrex::Real absTol = 1e-12,
                                     int verbose = 0)
{
    return ConjugateGradient<Rhs, Sol, Operation, PreconditionOp, AverageB0Op>(
        drc, std::forward<Operation> (matMultOp), std::forward<PreconditionOp>(precondOp),
        std::forward<AverageB0Op>(avgB0Op), absTol, relTol, verbose);
}

/// Helper function for template deduction. Allows you to specify Rhs and Sol explicitly, but have
/// the operator types automatically deduced.
template <class Rhs,
          class Sol,
          class Operation,
          class PreconditionOp = std::nullptr_t,
          class AverageB0Op = std::nullptr_t>
inline auto make_conjugate_gradient_unique_ptr (std::shared_ptr<Forms::DeRhamComplex> drc,
                                                Operation&& matMultOp,
                                                PreconditionOp&& precondOp = nullptr,
                                                AverageB0Op&& avgB0Op = nullptr,
                                                amrex::Real relTol = 1e-10,
                                                amrex::Real absTol = 1e-12,
                                                int verbose = 0)
{
    return std::make_unique<ConjugateGradient<Rhs, Sol, Operation, PreconditionOp, AverageB0Op>>(
        drc, std::forward<Operation> (matMultOp), std::forward<PreconditionOp>(precondOp),
        std::forward<AverageB0Op>(avgB0Op), relTol, absTol, verbose);
}
} //namespace Gempic::FieldSolvers
#endif
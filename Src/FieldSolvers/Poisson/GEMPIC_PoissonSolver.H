#ifndef GEMPIC_POISSONSOLVER_H_
#define GEMPIC_POISSONSOLVER_H_

#include <memory>
#include <type_traits>

#include <AMReX.H>
#include <AMReX_FFT.H> // Include FFT header for FFT Poisson Solver, but linker has some issues
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <AMReX_VisMF.H>
#ifdef AMREX_USE_HYPRE
#include <_hypre_IJ_mv.h>
#include <_hypre_sstruct_mv.h>

#include <AMReX_HypreSolver.H>
#endif

#include "GEMPIC_ComputationalDomain.H"
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

template <class Rhs, class Sol, Operator op>
class ConjugateGradient
{
private:
    std::shared_ptr<DeRhamComplex> m_deRham;
    std::shared_ptr<PoissonSolver> m_poissonSolver;
    Rhs m_residual, m_ap;
    Sol m_p, m_z;
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
     * @param[out] x : The solution vector
     * @param b : The right-hand side vector
     *
     * @return void
     *
     * @throws None
     */
    void solve (Sol& x, Rhs& b)
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
        copy(m_p, m_residual); // copy residual into p

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
     * @param[out] x : The solution vector
     * @param b : The right-hand side vector
     *
     * @return void
     *
     * @throws None
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
        copy(m_p, m_z); // copy z into p

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

    inline void cg_mat_mul(Sol& x, Rhs& mResidual);
    inline void precond(Sol& z, Rhs& r);
    inline void average_b_zero(Rhs& b);
};

#ifdef AMREX_USE_HYPRE
template <class rhs, class sol, Operator op, int hodgeDegree>
class HypreLinearSystem
{
private:
    ComputationalDomain* m_infra;
    std::shared_ptr<PoissonSolver> m_poissonSolver;
    rhs m_residual, m_ap;
    sol m_p, m_z;
    amrex::Real m_relTol, m_absTol{1.e-15};
    amrex::Real m_maxiter{1000};
    int m_verbose;

public:
    // Try to remove all raw pointer data management!
    HypreLinearSystem(ComputationalDomain* infr,
                      std::shared_ptr<DeRhamComplex> drc,
                      std::shared_ptr<PoissonSolver> pslv,
                      amrex::Real tol = 1.e-11,
                      int verbose = 0) :
        m_infra{infr},
        m_poissonSolver{pslv},
        m_residual(drc),
        m_ap(drc),
        m_p(drc),
        m_z(drc),
        m_relTol{tol}

    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    /**
     *
     * @param[out] x : The solution vector
     * @param b : The right-hand side vector
     *
     * @return void
     *
     * @throws None
     */
    void solve (sol& x, rhs& b)
    {
        average_b_zero(b); // average rhs to zero in periodic case

        auto const& nddom = amrex::surroundingNodes(m_infra->m_geom.Domain());
        const auto dxi = m_infra->m_geom.InvCellSizeArray();

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{AMREX_D_DECL(
            static_cast<HYPRE_Real>(dxi[0] * dxi[0]), static_cast<HYPRE_Real>(dxi[1] * dxi[1]),
            static_cast<HYPRE_Real>(dxi[2] * dxi[2]))};

        HYPRE_Real fac0;

        switch (hodgeDegree)
        {
            case 2:
                fac0 = HYPRE_Real(-2.) * (AMREX_D_TERM(fac[0], +fac[1], +fac[2]));
                break;
            case 4:
                fac0 = HYPRE_Real(-2.25) * (AMREX_D_TERM(fac[0], +fac[1], +fac[2]));
                break;
            case 6:
                fac0 = HYPRE_Real(-75.0 / 32.0) * (AMREX_D_TERM(fac[0], +fac[1], +fac[2]));
                break;
            default:
                AMREX_ASSERT("Degree not implemented");
                break;
        }

        // Is variable n at (i,j,k) in Box boxno (local index) valid?
        // (i.e., not exactly on Dirichlet boundary)
        auto marker = [=] AMREX_GPU_DEVICE(int /*boxno*/, int i, int j, int k, int /*n*/) -> bool
        { return nddom.contains(i, j, k); };

        // For variable n at (i,j,k) in Box boxno (local index), fill its row in
        // the matrix.
        // [in ] gid : gid[n] is the id for variable n at (i,j,k)
        // [out] ncols: # of columns in this row.
        // [out] cols: column indices in this row.
        // [out] mat : matrix elements in this row.
        auto filler = [=] AMREX_GPU_DEVICE(int /*boxno*/, int i, int j, int k, int n,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            switch (hodgeDegree)
            {
                case 2:
                    if ((i == 0) && (j == 0) && (k == 0))
                    {   // clang-format off
                        ncols = 1;
                        cols[0] = gid[n](i,j,k);       mat[0] = -fac0;
                    }
                    else
                    {
                        ncols = 2 * AMREX_SPACEDIM + 1;
                        cols[0] = gid[n](i,j,k);       mat[0] = -fac0;
                        cols[1] = gid[n]((i-1), j, k); mat[1] = -fac[0]; cols[2] = gid[n]((i+1), j, k); mat[2] = -fac[0];
#if (AMREX_SPACEDIM > 1)
                        cols[3] = gid[n](i, (j-1), k); mat[3] = -fac[1]; cols[4] = gid[n](i, (j+1), k); mat[4] = -fac[1];
#endif
#if (AMREX_SPACEDIM > 2)
                        cols[5] = gid[n](i, j, (k-1)); mat[5] = -fac[2]; cols[6] = gid[n](i, j, (k+1)); mat[6] = -fac[2];
#endif
                    }
                    break;
                case 4:
                    if ((i == 0) && (j == 0) && (k == 0))
                    {
                        ncols = 1;
                        cols[0] = gid[n](i,j,k);        mat[0] = -fac0;
                    }
                    else
                    {
                        ncols = 4 *AMREX_SPACEDIM + 1;
                        cols[0] = gid[n](i,j,k);        mat[0] = -fac0;
                        cols[1] = gid[n]((i-1), j, k);  mat[1] = -(7.0/6.0)*fac[0];  cols[2] = gid[n]((i+1), j, k);  mat[2] = -(7.0/6.0)*fac[0];
                        cols[3] = gid[n]((i-2), j, k);  mat[3] = (1.0/24.0)*fac[0];  cols[4] = gid[n]((i+2), j, k);  mat[4] = (1.0/24.0)*fac[0];
#if (AMREX_SPACEDIM > 1)
                        cols[5] = gid[n](i, (j-1), k);  mat[5] = -(7.0/6.0)*fac[1];  cols[6] = gid[n](i, (j+1), k);  mat[6] = -(7.0/6.0)*fac[1];
                        cols[7] = gid[n](i, (j-2), k);  mat[7] = (1.0/24.0)*fac[1];  cols[8] = gid[n](i, (j+2), k);  mat[8] = (1.0/24.0)*fac[1];
#endif
#if (AMREX_SPACEDIM > 2)
                        cols[9] = gid[n](i, j, (k-1));  mat[9] = -(7.0/6.0)*fac[2];  cols[10] = gid[n](i, j, (k+1)); mat[10] = -(7.0/6.0)*fac[2];
                        cols[11] = gid[n](i, j, (k-2)); mat[11] = (1.0/24.0)*fac[2]; cols[12] = gid[n](i, j, (k+2)); mat[12] = (1.0/24.0)*fac[2];
#endif
                    }
                    break;
                case 6:
                    if ((i == 0) && (j == 0) && (k == 0))
                    {
                        ncols = 1;
                        cols[0] = gid[n](i,j,k);        mat[0] = -fac0;
                    }
                    else
                    {
                        ncols =  6*AMREX_SPACEDIM + 1;
                        cols[0] = gid[n](i,j,k);        mat[0] = -fac0;
                        cols[1] = gid[n]((i-1), j, k);  mat[1] = -(2375.0/1920.0)*fac[0];   cols[2] = gid[n]((i+1), j, k);  mat[2] = -(2375.0/1920.0)*fac[0];
                        cols[3] = gid[n]((i-2), j, k);  mat[3] = (67.0/960.0)*fac[0];   cols[4] = gid[n]((i+2), j, k);  mat[4] = (67.0/960.0)*fac[0];
                        cols[5] = gid[n]((i-3), j, k);  mat[5] = -(3.0/640.0)*fac[0];  cols[6] = gid[n]((i+3), j, k);  mat[6] = -(3.0/640.0)*fac[0];
#if (AMREX_SPACEDIM > 1)
                        cols[7] = gid[n](i, (j-1), k);  mat[7] = -(2375.0/1920.0)*fac[1];   cols[8] = gid[n](i, (j+1), k);  mat[8] = -(2375.0/1920.0)*fac[1];
                        cols[9] = gid[n](i, (j-2), k);  mat[9] = (67.0/960.0)*fac[1];   cols[10] = gid[n](i, (j+2), k); mat[10] = (67.0/960.0)*fac[1];
                        cols[11] = gid[n](i, (j-3), k); mat[11] = -(3.0/640.0)*fac[1]; cols[12] = gid[n](i, (j+3), k); mat[12] = -(3.0/640.0)*fac[1];
#endif
#if (AMREX_SPACEDIM > 2)
                        cols[13] = gid[n](i, j, (k-1)); mat[13] = -(2375.0/1920.0)*fac[2];  cols[14] = gid[n](i, j, (k+1)); mat[14] = -(2375.0/1920.0)*fac[2];
                        cols[15] = gid[n](i, j, (k-2)); mat[15] = (67.0/960.0)*fac[2];  cols[16] = gid[n](i, j, (k+2)); mat[16] = (67.0/960.0)*fac[2];
                        cols[17] = gid[n](i, j, (k-3)); mat[17] = -(3.0/640.0)*fac[2]; cols[18] = gid[n](i, j, (k+3)); mat[18] = -(3.0/640.0)*fac[2];
                        // clang-format on
#endif
                    }
                    break;
                default:
                    AMREX_ASSERT("Hodge degree not implemented");
                    break;
            };
        };

        constexpr int maxStencilSize = hodgeDegree * AMREX_SPACEDIM + 1;

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {amrex::IndexType::TheNodeType()}, b.m_deRham->get_n_ghost(), m_infra->m_geom,
            m_infra->m_grid, m_infra->m_distriMap, marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data)},
                          amrex::Vector<amrex::MultiFab const*>{&(b.m_data)}, m_relTol, m_absTol,
                          m_maxiter);

        x *= 1.0 / m_infra->cell_volume();

        x.average_sync();
        x.fill_boundary();
    }

    inline void average_b_zero(rhs& b);
};
#endif

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

    void solve_amrex(DeRhamField<Grid::primal, Space::node>& phi,
                     DeRhamField<Grid::dual, Space::cell>& rho);

    /** @brief FFT Poisson solver
     *
     *  needs rhs dual grid cell rho and the solution we want in primal grid node
     *  Algorithm process:
     *  equation: GHG^{tilde} phi = rho
     *  1. input rho(cell value), calculate phi(node value)
     *  2. determine matrices G, H, G^{tilde}:
     *      ----dimension of equations (for G, G^{tilde}, H)
     *      ----order of Hodge matrix H (for H)
     *      ----only need the first rows since they are all circulant
     *  3.calculate eigenvalues of GHG^{tilde} for its diagonal form D
     *      ----eigenvalues_total = lamda_x + lamda_y + lamda_z
     *      ----lamda_x_k = [2-2*cos(2*M_PI*k/Nx)]*(kth eigenvalue of hodge_x)
     *  4.Apply FFT :
     *      ----phi = FFT_backward((FFT_forward(rho) / eigenvalues))
     *      ----amrex provides from real to complex and from complex to real.
     *          In our case, both are needed
     *
     *  Due to the way the eigenvalues are calculated, it is necessary to perform the Fourier
     *  transform between cell-centered MultiFabs.
     *  rho and phi are both node-centered and as such, we need to copy values back and forth.
     *
     * @param[out] phi : The solution vector
     * @param rho : The right-hand side vect
     *
     */

    inline void solve_fft (DeRhamField<Grid::primal, Space::node>& phi,
                           DeRhamField<Grid::dual, Space::cell>& rho)
    {
        Forms::HodgeScheme hodgeScheme = Gempic::Forms::HodgeScheme::FDHodge;

        // get hodge degree from deRham complex
        const int hodgeDegree = m_deRham->get_hodge_degree();
        BL_PROFILE("Gempic::FieldSolvers::PoissonSolver::solve_fft()");

        // Check for periodic boundaries, required for FFT solver
        if (!m_deRham->get_periodicity().isAllPeriodic())
        {
            amrex::Abort("FFT Poisson solver requires periodic boundary conditions");
        }
        subtract_constant_part(rho);

        // define cell-centered containers for storing data, declaring 1 ghost cell to be able to
        // copy back to the node-centered MultiFab, phi, without data loss.
        amrex::MultiFab rhoFft(m_infra.m_grid, m_infra.m_distriMap, rho.m_data.nComp(), 1);
        amrex::MultiFab phiFft(m_infra.m_grid, m_infra.m_distriMap, phi.m_data.nComp(), 1);

        // assign values to containers from rho.m_data
        for (amrex::MFIter mfi(rhoFft); mfi.isValid(); ++mfi)
        {
            auto const& rhoFftPtr = rhoFft.array(mfi);

            auto const& rhoMDataPtr = rho.m_data.array(mfi);

            const amrex::Box& bx = mfi.fabbox();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { rhoFftPtr(i, j, k) = rhoMDataPtr(i, j, k); });
        }

        //dimensions in x, y and z directions
        AMREX_D_TERM(const int Nx = m_infra.box().size()[xDir];
                     , const int Ny = m_infra.box().size()[yDir];
                     , const int Nz = m_infra.box().size()[zDir];)

        // sizes in x, y and z directions
        auto dx = m_infra.geometry().CellSizeArray();

        AMREX_D_TERM(auto eigenH0x = amrex::Vector<amrex::Real>(Nx, 1.0);
                     , auto eigenH0y = amrex::Vector<amrex::Real>(Ny, 1.0);
                     , auto eigenH0z = amrex::Vector<amrex::Real>(Nz, 1.0);)

        if (hodgeScheme == Gempic::Forms::HodgeScheme::FDHodge)
        {
            switch (hodgeDegree)
            {
                case 2:
                    for (int iter = 0; iter < Nx; iter++)
                    {
                        eigenH0x[iter] = dx.product() / (dx[xDir] * dx[xDir]);
                    }

#if AMREX_SPACEDIM > 1
                    for (int iter = 0; iter < Ny; iter++)
                    {
                        eigenH0y[iter] = dx.product() / (dx[yDir] * dx[yDir]);
                    }
#if AMREX_SPACEDIM == 3
                    for (int iter = 0; iter < Nz; iter++)
                    {
                        eigenH0z[iter] = dx.product() / (dx[zDir] * dx[zDir]);
                    }
#endif
#endif
                    break;
                case 4:
                    for (int iter = 0; iter < Nx; iter++)
                    {
                        eigenH0x[iter] = (13.0 - cos(2 * M_PI * iter / Nx)) * dx.product() /
                                         (12 * dx[xDir] * dx[xDir]);
                    }

#if AMREX_SPACEDIM > 1
                    for (int iter = 0; iter < Ny; iter++)
                    {
                        eigenH0y[iter] = (13.0 - cos(2 * M_PI * iter / Ny)) * dx.product() /
                                         (12 * dx[yDir] * dx[yDir]);
                    }
#if AMREX_SPACEDIM == 3
                    for (int iter = 0; iter < Nz; iter++)
                    {
                        eigenH0z[iter] = (13.0 - cos(2 * M_PI * iter / Nz)) * dx.product() /
                                         (12 * dx[zDir] * dx[zDir]);
                    }
#endif
#endif
                    break;
                case 6:
                    for (int iter = 0; iter < Nx; iter++)
                    {
                        eigenH0x[iter] = (1067.0 - 116.0 * cos(2 * M_PI * iter / Nx) +
                                          9.0 * cos(4 * M_PI * iter / Nx)) *
                                         dx.product() / (960 * dx[xDir] * dx[xDir]);
                    }

#if AMREX_SPACEDIM > 1
                    for (int iter = 0; iter < Ny; iter++)
                    {
                        eigenH0y[iter] = (1067.0 - 116.0 * cos(2 * M_PI * iter / Ny) +
                                          9.0 * cos(4 * M_PI * iter / Ny)) *
                                         dx.product() / (960 * dx[yDir] * dx[yDir]);
                    }
#if AMREX_SPACEDIM == 3
                    for (int iter = 0; iter < Nz; iter++)
                    {
                        eigenH0z[iter] = (1067.0 - 116.0 * cos(2 * M_PI * iter / Nz) +
                                          9.0 * cos(4 * M_PI * iter / Nz)) *
                                         dx.product() / (960 * dx[zDir] * dx[zDir]);
                    }
#endif
#endif
                    break;
                default:
                    AMREX_ASSERT("Degree not implemented for three dimensional Hodge in FFT");
                    break;
            }

            // calculate eigenvalues of the lhs matrix
            AMREX_D_TERM(auto eigenvalues0x = amrex::Vector<amrex::Real>(Nx, 1.0);
                         , auto eigenvalues0y = amrex::Vector<amrex::Real>(Ny, 1.0);
                         , auto eigenvalues0z = amrex::Vector<amrex::Real>(Nz, 1.0);)

            for (int iter = 0; iter < Nx; iter++)
            {
                eigenvalues0x[iter] = (2 - 2 * cos(2 * M_PI * iter / Nx)) * eigenH0x[iter];
            }

#if AMREX_SPACEDIM > 1
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenvalues0y[iter] = (2 - 2 * cos(2 * M_PI * iter / Ny)) * eigenH0y[iter];
            }
#if AMREX_SPACEDIM == 3
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenvalues0z[iter] = (2 - 2 * cos(2 * M_PI * iter / Nz)) * eigenH0z[iter];
            }
#endif
#endif

            // Handle the gpu parallel for for gpu running
            AMREX_D_TERM(
                amrex::Gpu::DeviceVector<amrex::Real> eigenvalues0xGpu(static_cast<size_t>(Nx));
                , amrex::Gpu::DeviceVector<amrex::Real> eigenvalues0yGpu(static_cast<size_t>(Ny));
                , amrex::Gpu::DeviceVector<amrex::Real> eigenvalues0zGpu(static_cast<size_t>(Nz));)

            AMREX_D_TERM(amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenvalues0x.begin(),
                                               eigenvalues0x.end(), eigenvalues0xGpu.begin());
                         , amrex::Gpu::copyAsync (amrex::Gpu::hostToDevice, eigenvalues0y.begin(),
                                                 eigenvalues0y.end(), eigenvalues0yGpu.begin());
                         , amrex::Gpu::copyAsync (amrex::Gpu::hostToDevice, eigenvalues0z.begin(),
                                                 eigenvalues0z.end(), eigenvalues0zGpu.begin());)

            AMREX_D_TERM(auto* const eigenvalues0xGpuPtr = eigenvalues0xGpu.dataPtr();
                         , auto* const eigenvalues0yGpuPtr = eigenvalues0yGpu.dataPtr();
                         , auto* const eigenvalues0zGpuPtr = eigenvalues0zGpu.dataPtr();)

            amrex::FFT::R2C r2c(m_infra.box());
            // Amrex FFT only fourier transforms the first component, so we need to do hacks
            for (int comp{0}; comp < rho.m_data.nComp(); ++comp)
            {
                amrex::MultiFab rhoFftComp(rhoFft, amrex::make_alias, comp, 1);
                amrex::MultiFab phiFftComp(phiFft, amrex::make_alias, comp, 1);
                r2c.forwardThenBackward(rhoFftComp, phiFftComp,
                                        [=] AMREX_GPU_DEVICE(int i, int j, int k, auto& sp)
                                        {
                                            auto eigenvalueSum{GEMPIC_D_ADD(
                                                eigenvalues0xGpuPtr[i], eigenvalues0yGpuPtr[j],
                                                eigenvalues0zGpuPtr[k])};
                                            if (eigenvalueSum == 0)
                                            {
                                                sp = 0;
                                            }
                                            else
                                            {
                                                sp /= eigenvalueSum * GEMPIC_D_MULT(Nx, Ny, Nz);
                                            }
                                        });
            }
        }
        else
        {
            amrex::Abort("The FFT Poison solver only works with Finite Difference Hodge");
        }

        //boundary handling
        phiFft.FillBoundary(m_deRham->get_periodicity());

        for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
        {
            auto const& phiFftPtr = phiFft.array(mfi);

            auto const& phiMDataPtr = phi.m_data.array(mfi);

            const amrex::Box& bx = mfi.validbox();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { phiMDataPtr(i, j, k) = phiFftPtr(i, j, k); });
        }

        phi.fill_boundary();
    };

    void apply_poisson_operator(DeRhamField<Grid::dual, Space::cell>& rho,
                                DeRhamField<Grid::primal, Space::node>& phi);

    void apply_poisson_operator_inverse_hodge(DeRhamField<Grid::dual, Space::cell>& rho,
                                              DeRhamField<Grid::primal, Space::node>& phi);

    void subtract_constant_part(DeRhamField<Grid::dual, Space::cell>& rho);
};

template <class Rhs, class Sol, Operator op>
inline void ConjugateGradient<Rhs, Sol, op>::cg_mat_mul(Sol& x, Rhs& Ax)
{
    if constexpr (op == Operator::hodge)
    {
        m_deRham->hodge(Ax, x);
    }
    else if constexpr (op == Operator::poisson)
    {
        m_poissonSolver->apply_poisson_operator(Ax, x);
    }
    else if constexpr (op == Operator::poissonInverseHodge)
    {
        m_poissonSolver->apply_poisson_operator_inverse_hodge(Ax, x);
    }
    else
    {
        static_assert(
            op == Operator::hodge || op == Operator::poisson || op == Operator::poissonInverseHodge,
            "Conjugate Gradient: operator not implemented");
    }
}

template <class Rhs, class Sol, Operator op>
inline void ConjugateGradient<Rhs, Sol, op>::precond(Sol& z, Rhs& r)
{
    if constexpr (op == Operator::hodge)
    {
        copy(z, r); // not preconditioned
    }
    else if constexpr (op == Operator::poisson)
    {
        m_poissonSolver->solve_amrex(r, z);
    }
    else if constexpr (op == Operator::poissonInverseHodge) // not preconditioned
    {
        //m_poissonSolver->solve_amrex(r, z);
        copy(z, r); // not preconditioned
    }
    else
    {
        static_assert(
            op == Operator::hodge || op == Operator::poisson || op == Operator::poissonInverseHodge,
            "Conjugate Gradient: operator not implemented");
    }
}

/// average rhs to zero in periodic case
template <class Rhs, class Sol, Operator op>
inline void ConjugateGradient<Rhs, Sol, op>::average_b_zero(Rhs& b)
{
    if constexpr (op == Operator::poisson || op == Operator::poissonInverseHodge)
    {
        m_poissonSolver->subtract_constant_part(b);
    }
}

#ifdef AMREX_USE_HYPRE
/// average rhs to zero in periodic case
template <class rhs, class sol, Operator op, int hodgeDegree>
inline void HypreLinearSystem<rhs, sol, op, hodgeDegree>::average_b_zero(rhs& b)
{
    if constexpr (op == Operator::poisson || op == Operator::poissonInverseHodge)
    {
        m_poissonSolver->subtract_constant_part(b);
    }
}
#endif
} // namespace Gempic::FieldSolvers

#endif

#include <_hypre_IJ_mv.h>
#include <_hypre_sstruct_mv.h>

#include <AMReX_HypreSolver.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"

namespace Gempic::FieldSolvers
{
/** @brief Hypre Poisson solver
 *
 *  Solves the equation \f$ -\tilde{D}HG \phi = \rho \f$ for \f$\phi\f$ using the Hypre library.
 *
 * @tparam hodgeDegree
 *
 * @param[out] phi The solution vector
 * @param rho The right-hand side vector
 */
template <int hodgeDegree>
class HypreLinearSystem : public PoissonSolverMethod
{
private:
    static constexpr int s_maxStencilSize = hodgeDegree * AMREX_SPACEDIM + 1;
    std::unique_ptr<amrex::HypreSolver<s_maxStencilSize>> m_hypreSolver;

    ComputationalDomain m_compDom;
    amrex::Real m_relTol, m_absTol;
    amrex::Real m_maxiter{1000};
    amrex::Real m_cellVolume;

public:
    HypreLinearSystem(ComputationalDomain const& compDom,
                      std::shared_ptr<Gempic::Forms::DeRhamComplex> drc,
                      amrex::Real relTol = 1.e-10,
                      amrex::Real absTol = 1.e-12,
                      int verbose = 0) :
        m_compDom{compDom}, m_relTol{relTol}, m_absTol{absTol}, m_cellVolume{compDom.cell_volume()}
    {
        verbose = std::max(verbose, Gempic::Utils::Verbosity::level());

        auto const& nddom = amrex::surroundingNodes(compDom.box());
        auto const dxi = compDom.inv_cell_size_array();

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

        m_hypreSolver = std::make_unique<amrex::HypreSolver<maxStencilSize>>(
            amrex::Vector<amrex::IndexType>{amrex::IndexType::TheNodeType()}, drc->get_n_ghost(),
            compDom.geometry(), compDom.m_grid, compDom.m_distriMap, marker, filler, verbose);
    }

    /**
     *
     * @param[out] phi The solution vector
     * @param rho The right-hand side vector
     */
    void solve (DeRhamField<Grid::primal, Space::node>& phi,
                DeRhamField<Grid::dual, Space::cell>& rho) override
    {
        subtract_constant_part(rho, m_compDom); // average rhs to zero in periodic case

        m_hypreSolver->solve(amrex::Vector<amrex::MultiFab*>{&(phi.m_data)},
                             amrex::Vector<amrex::MultiFab const*>{&(rho.m_data)}, m_relTol,
                             m_absTol, m_maxiter);

        phi *= 1.0 / m_cellVolume;

        phi.fill_boundary();
    }
};
} //namespace Gempic::FieldSolvers

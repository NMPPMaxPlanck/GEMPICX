#ifndef GEMPIC_QUASINEUTRALSOLVER_H_
#define GEMPIC_QUASINEUTRALSOLVER_H_

#include <type_traits>

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>
#ifdef AMREX_USE_HYPRE
#include <_hypre_IJ_mv.h>
#include <_hypre_sstruct_mv.h>

#include <AMReX_HypreSolver.H>
#endif

#include <map>

#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{
using namespace Forms;
using namespace Particle;

#ifdef AMREX_USE_HYPRE
template <class rhs, class sol, int hodgeDegree, unsigned int vdim, int degx, int degy, int degz>
class HypreQuasineutralLinearSystem
{
private:
    ComputationalDomain m_infra;
    rhs m_residual, m_ap;
    sol m_p, m_z;
    amrex::Real m_relTol, m_absTol{1.e-15};
    amrex::Real m_maxiter{1000};
    int m_verbose;

    static constexpr int s_offx[3] = {(degx - 1), degx,
                                      degx};  // x,y,z-field offsets in x-direction
    static constexpr int s_offy[3] = {degy, (degy - 1),
                                      degy};  // x,y,z-field offsets in y-direction
    static constexpr int s_offz[3] = {degz, degz,
                                      (degz - 1)};  // x,y,z-field offsets in z-direction

    static constexpr int s_BWx[3] = {
        (2 * s_offx[xDir] + 1), (2 * s_offx[yDir] + 1),
        (2 * s_offx[zDir] + 1)};  // x,y,z-field bandwidths in x-direction
    static constexpr int s_BWy[3] = {
        (2 * s_offy[xDir] + 1), (2 * s_offy[yDir] + 1),
        (2 * s_offy[zDir] + 1)};  // x,y,z-field bandwidths in y-direction
    static constexpr int s_BWz[3] = {
        (2 * s_offz[xDir] + 1), (2 * s_offz[yDir] + 1),
        (2 * s_offz[zDir] + 1)};  // x,y,z-field bandwidths in z-direction

public:
    HypreQuasineutralLinearSystem(ComputationalDomain infr,
                                  std::shared_ptr<DeRhamComplex> drc,
                                  amrex::Real tol = 1.e-11,
                                  int verbose = 0) :
        m_infra{infr}, m_residual(drc), m_ap(drc), m_p(drc), m_z(drc), m_relTol{tol}

    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    void assign_or_add_col (int gdtmp,
                            int& icol,
                            HYPRE_Int& ncols,
                            HYPRE_Int* cols,
                            HYPRE_Real* mat,
                            std::map<int, int>& mapColsGeneral)
    {
        if (mapColsGeneral.find(gdtmp) != mapColsGeneral.end())  // element exists already
        {
            icol = mapColsGeneral[gdtmp];
        }
        else  // element does not exist, and is newly added
        {
            icol = ncols++;
            cols[icol] = gdtmp;
            mapColsGeneral.insert(std::make_pair(cols[icol], icol));
            mat[icol] = 0.0;  // matrix element is initialized to 0, and then incremented by filler
                              // functions
        }
    }

    void filler_curlcurl_matrix (int boxno,
                                 int i,
                                 int j,
                                 int k,
                                 int ndir,
                                 amrex::Array4<HYPRE_Int const> const* gid,
                                 HYPRE_Int& ncols,
                                 HYPRE_Int* cols,
                                 HYPRE_Real* mat,
                                 const amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac,
                                 const amrex::GpuArray<HYPRE_Real, 3> edgetoface,
                                 std::map<int, int>& mapCols)
    {
        switch (hodgeDegree)
        {
            case 2:
                // This function contributes (<= 6*GEMPIC_SPACEDIM-5) columns to ncols
                // clang-format off
                int icol; int gdtmp;
                int n[GEMPIC_SPACEDIM], in[GEMPIC_SPACEDIM], jn[GEMPIC_SPACEDIM], kn[GEMPIC_SPACEDIM];
                for (int dir = 0; dir < GEMPIC_SPACEDIM; dir++)
                {
                    n[dir] = (ndir + dir)%GEMPIC_SPACEDIM;
                    in[dir] = (n[dir] == 0)?1:0; jn[dir] = (n[dir] == 1)?1:0; kn[dir] = (n[dir] == 2)?1:0;
                }

                amrex::Real temp; temp = 0.0;
                for (int dir = 1; dir < GEMPIC_SPACEDIM; dir++) {
                    temp += 2.0*(fac[n[dir]]);
}

                gdtmp = gid[n[0]](i,j,k);                                             assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += temp*edgetoface[n[0]];

                gdtmp = gid[n[0]]((i-in[1]), (j-jn[1]), (k-kn[1]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[1]]*edgetoface[n[0]];
                gdtmp = gid[n[0]]((i+in[1]), (j+jn[1]), (k+kn[1]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[1]]*edgetoface[n[0]];

#if GEMPIC_SPACEDIM == 3
                gdtmp = gid[n[0]]((i-in[2]), (j-jn[2]), (k-kn[2]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[2]]*edgetoface[n[0]];
                gdtmp = gid[n[0]]((i+in[2]), (j+jn[2]), (k+kn[2]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[2]]*edgetoface[n[0]];
#endif
                gdtmp = gid[n[1]]((i+in[0]), (j+jn[0]), (k+kn[0]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += fac[n[1]]*edgetoface[n[1]];
                gdtmp = gid[n[1]]((i-in[1]), (j-jn[1]), (k-kn[1]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += fac[n[1]]*edgetoface[n[1]];
                gdtmp = gid[n[1]](i, j, k);                                           assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[1]]*edgetoface[n[1]];
                gdtmp = gid[n[1]]((i-in[1]+in[0]), (j-jn[1]+jn[0]), (k-kn[1]+kn[0])); assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[1]]*edgetoface[n[1]];

#if GEMPIC_SPACEDIM == 3
                gdtmp = gid[n[2]]((i+in[0]), (j+jn[0]), (k+kn[0]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += fac[n[2]]*edgetoface[n[2]];
                gdtmp = gid[n[2]]((i-in[2]), (j-jn[2]), (k-kn[2]));                   assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += fac[n[2]]*edgetoface[n[2]];
                gdtmp = gid[n[2]](i, j, k);                                           assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[2]]*edgetoface[n[2]];
                gdtmp = gid[n[2]]((i-in[2]+in[0]), (j-jn[2]+jn[0]), (k-kn[2]+kn[0])); assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols); mat[icol] += -fac[n[2]]*edgetoface[n[2]];
#endif
                // clang-format on
                break;
            default:
                AMREX_ASSERT("Hodge degree not implemented");
                break;
        };
    }

    void filler_field_charge_matrix (int boxno,
                                     int i,
                                     int j,
                                     int k,
                                     int ndir,
                                     amrex::Array4<HYPRE_Int const> const* gid,
                                     HYPRE_Int& ncols,
                                     HYPRE_Int* cols,
                                     HYPRE_Real* mat,
                                     const amrex::MultiArray4<const amrex::Real>& rhoarrs,
                                     const amrex::GpuArray<HYPRE_Real, 3> edgetoface,
                                     std::map<int, int>& mapCols)
    {
        switch (hodgeDegree)
        {
            case 2:
                // This function contributes (<= 1) columns to ncols
                // clang-format off
                int n[GEMPIC_SPACEDIM], in[GEMPIC_SPACEDIM], jn[GEMPIC_SPACEDIM], kn[GEMPIC_SPACEDIM];
                for (int dir = 0; dir < GEMPIC_SPACEDIM; dir++)
                {
                    n[dir] = (ndir + dir)%GEMPIC_SPACEDIM;
                    in[dir] = (n[dir] == 0)?1:0; jn[dir] = (n[dir] == 1)?1:0; kn[dir] = (n[dir] == 2)?1:0;
                }

                int icol, gdtmp;
                gdtmp = gid[n[0]](i,j,k);

                assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                mat[icol] += 0.5*(rhoarrs[boxno](i,j,k) + rhoarrs[boxno](i+in[0],j+jn[0],k+kn[0])) / m_infra.cell_volume() * edgetoface[n[0]];
                // clang-format on
                break;
            default:
                AMREX_ASSERT("Hodge degree not implemented");
                break;
        };
    }

    void filler_curlcurl_matrix_2dz (int boxno,
                                     int i,
                                     int j,
                                     int k,
                                     amrex::Array4<HYPRE_Int const> const* gid,
                                     HYPRE_Int& ncols,
                                     HYPRE_Int* cols,
                                     HYPRE_Real* mat,
                                     const amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac,
                                     const amrex::GpuArray<HYPRE_Real, 3> edgetoface,
                                     std::map<int, int>& mapCols2dz)
    {
        switch (hodgeDegree)
        {
            case 2:
                // This function contributes (<= 5) columns to ncols
                // clang-format off
                int icol; int gdtmp;
                gdtmp = gid[0](i,j,k);       assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz); mat[icol] += 2.0*(fac[0] + fac[1])*edgetoface[2];

                gdtmp = gid[0]((i-1), j, k); assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz); mat[icol] += -fac[0]*edgetoface[2];
                gdtmp = gid[0]((i+1), j, k); assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz); mat[icol] += -fac[0]*edgetoface[2];
                gdtmp = gid[0](i, (j-1), k); assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz); mat[icol] += -fac[1]*edgetoface[2];
                gdtmp = gid[0](i, (j+1), k); assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz); mat[icol] += -fac[1]*edgetoface[2];
                // clang-format on
                break;
            default:
                AMREX_ASSERT("Hodge degree not implemented");
                break;
        };
    }

    void filler_field_charge_matrix_2dz (int boxno,
                                         int i,
                                         int j,
                                         int k,
                                         amrex::Array4<HYPRE_Int const> const* gid,
                                         HYPRE_Int& ncols,
                                         HYPRE_Int* cols,
                                         HYPRE_Real* mat,
                                         const amrex::MultiArray4<const amrex::Real>& rhoarrs,
                                         const amrex::GpuArray<HYPRE_Real, 3> edgetoface,
                                         std::map<int, int>& mapCols2dz)
    {
        switch (hodgeDegree)
        {
            case 2:
                // This function contributes (<= 1) columns to ncols
                // clang-format off
                int icol, gdtmp;
                gdtmp = gid[0](i,j,k);

                assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz);
                mat[icol] += (rhoarrs[boxno](i,j,k)) / m_infra.cell_volume() * edgetoface[2];
                // clang-format on
                break;
            default:
                AMREX_ASSERT("Hodge degree not implemented");
                break;
        };
    }

    void filler_particle_charge_matrix (
        int boxno,
        int i,
        int j,
        int k,
        int ndir,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        const std::array<amrex::MultiArray4<amrex::Real>, 3>& eCoeffsArrs,
        std::map<int, int>& mapCols)
    {
        // This function contributes (<= nMmCoeffs[ndir]) columns to ncols
        int iiTemp{0}, jjTemp{0}, kkTemp{0};
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        GEMPIC_D_LOOP_BEGIN(for (ii = -s_offx[ndir]; ii <= s_offx[ndir]; ++ii),
                            for (jj = -s_offy[ndir]; jj <= s_offy[ndir]; ++jj),
                            for (kk = -s_offz[ndir]; kk <= s_offz[ndir]; ++kk))
            // clang-format off
            AMREX_D_EXPR(iiTemp = ii + s_offx[ndir], jjTemp = jj + s_offy[ndir], kkTemp = kk + s_offz[ndir]);

            gdtmp = gid[ndir]((i+ii), (j+jj), (k+kk));

            assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += eCoeffsArrs[ndir][boxno](i, j, k, AMREX_D_TERM(iiTemp, + jjTemp*s_BWx[ndir], + kkTemp*s_BWx[ndir]*s_BWy[ndir]));
            // clang-format on
        GEMPIC_D_LOOP_END
    }

    void filler_particle_charge_matrix_2dz (
        int boxno,
        int i,
        int j,
        int k,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        const std::array<amrex::MultiArray4<amrex::Real>, 3>& eCoeffsArrs,
        std::map<int, int>& mapCols2dz)
    {
        // This function contributes (<= nMmCoeffs[zdir]) columns to ncols
        int iiTemp{0}, jjTemp{0}, kkTemp{0};
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        for (ii = -s_offx[zDir]; ii <= s_offx[zDir]; ++ii)
        {
            for (jj = -s_offy[zDir]; jj <= s_offy[zDir]; ++jj)
            {
                // clang-format off
                iiTemp = ii + s_offx[zDir]; jjTemp = jj + s_offy[zDir];

                gdtmp = gid[0]((i+ii), (j+jj), (k+kk));

                assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols2dz);
                mat[icol] += eCoeffsArrs[zDir][boxno](i, j, k, (iiTemp + jjTemp*s_BWx[zDir]));
                // clang-format on
            }
        }
    }

    template <class splineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_e_weights_at_particle_positions_and_deposit_to_face (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        splineType const& spline,
        amrex::Real chargeWeight)
    {
        constexpr int nWtsX[3] = {degx, (degx + 1),
                                  (degx + 1)};  // number of x,y,z-field weights in X-direction
        constexpr int nWtsY[3] = {(degy + 1), degy,
                                  (degy + 1)};  // number of x,y,z-field weights in Y-direction
        constexpr int nWtsZ[3] = {(degz + 1), (degz + 1),
                                  degz};  // number of x,y,z-field weights in Z-direction

        constexpr int nWts[3] = {
            GEMPIC_D_MULT(nWtsX[xDir], nWtsY[xDir], nWtsZ[xDir]),   // number of x-field weights
            GEMPIC_D_MULT(nWtsX[yDir], nWtsY[yDir], nWtsZ[yDir]),   // number of y-field weights
            GEMPIC_D_MULT(nWtsX[zDir], nWtsY[zDir], nWtsZ[zDir])};  // number of z-field weights
        constexpr int nWtsStart[3] = {0, nWts[xDir], (nWts[xDir] + nWts[yDir])};

        amrex::GpuArray<amrex::Real, (nWts[xDir] + nWts[yDir] + nWts[zDir])> eWts;
        amrex::GpuArray<amrex::IntVect, (nWts[xDir] + nWts[yDir] + nWts[zDir])> eCells;

        spline.eval_e_field_spline_weights(eWts, eCells);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int iiTemp, jjTemp, kkTemp;
        int component;

        for (int ndir = 0; ndir < 3; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsX[ndir]; ++ii),
                                for (int jj = 0; jj < nWtsY[ndir]; ++jj),
                                for (int kk = 0; kk < nWtsZ[ndir]; ++kk))
                const amrex::Real sCoeff =
                    ((ndir == 0)
                         ? (spline.template eval_b_spline<Field::DualTwoForm, xDir>(ii, jj, kk))
                         : ((ndir == 1) ? (spline.template eval_b_spline<Field::DualTwoForm, yDir>(
                                              ii, jj, kk))
                                        : (spline.template eval_b_spline<Field::DualTwoForm, zDir>(
                                              ii, jj, kk))));

                amrex::Dim3 cell = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                                 spline.m_firstIndex[yDir] + jj,
                                                 spline.m_firstIndex[zDir] + kk)};

                for (int i = 0; i < nWts[ndir]; i++)
                {
                    amrex::Dim3 wCell = (eCells[nWtsStart[ndir] + i]).dim3();

                    AMREX_D_EXPR(iiTemp = wCell.x - cell.x + s_offx[ndir],
                                 jjTemp = wCell.y - cell.y + s_offy[ndir],
                                 kkTemp = wCell.z - cell.z + s_offz[ndir]);
                    component = GEMPIC_D_ADD(iiTemp, jjTemp * s_BWx[ndir],
                                             kkTemp * s_BWx[ndir] * s_BWy[ndir]);

                    amrex::HostDevice::Atomic::Add(
                        eCoeffsA[ndir].ptr(cell.x, cell.y, cell.z, component),
                        sCoeff * chargeWeight * eWts[nWtsStart[ndir] + i]);
                }
            GEMPIC_D_LOOP_END
        }
    }

    void solve_curlcurl_plus_fieldcharge_e (rhs& b,
                                            sol& x,
                                            const DeRhamField<Grid::dual, Space::cell>& rho)
    {
        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        auto const& rhoarrs = (rho.m_data).arrays();

        const auto dxi = m_infra.inv_cell_size_array();
        const auto dx = m_infra.cell_size_array();

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        amrex::GpuArray<HYPRE_Real, 3> edgetoface{
            static_cast<HYPRE_Real>(GEMPIC_D_MULT(dxi[xDir], dx[yDir], dx[zDir])),
            static_cast<HYPRE_Real>(GEMPIC_D_MULT(dx[xDir], dxi[yDir], dx[zDir])),
            static_cast<HYPRE_Real>(GEMPIC_D_MULT(dx[xDir], dx[yDir], dxi[zDir]))};

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
        auto filler = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int ndir,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0;  // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
                                       edgetoface, mapCols);
            filler_curlcurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, fac, edgetoface,
                                   mapCols);
            //filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
            //edgetoface, map_cols);
        };

        constexpr int maxStencilSize = 6 * GEMPIC_SPACEDIM - 5;

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {AMREX_D_DECL(x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
                          x.m_data[zDir].boxArray().ixType())},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{AMREX_D_DECL(
                              &(x.m_data[xDir]), &(x.m_data[yDir]), &(x.m_data[zDir]))},
                          amrex::Vector<amrex::MultiFab const*>{AMREX_D_DECL(
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir]))},
                          m_relTol, m_absTol, m_maxiter);

        // In 2D, the Z E-field is decoupled from the XY E-field and is solved separately
#if GEMPIC_SPACEDIM == 2
        auto filler2Dz = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int /*ndir*/,
                                              amrex::Array4<HYPRE_Int const> const* gid,
                                              HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> map_cols_2dz;
            ncols = 0;  // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            filler_field_charge_matrix_2dz(boxno, i, j, k, gid, ncols, cols, mat, rhoarrs,
                                           edgetoface, map_cols_2dz);
            filler_curlcurl_matrix_2dz(boxno, i, j, k, gid, ncols, cols, mat, fac, edgetoface,
                                       map_cols_2dz);
            //filler_field_charge_matrix_2dz(boxno, i, j, k, gid, ncols, cols, mat, rhoarrs,
            //edgetoface, map_cols_2dz);
        };

        constexpr int maxStencilSize2Dz = 5;

        amrex::HypreSolver<maxStencilSize2Dz> hypreSolver2Dz(
            {x.m_data[zDir].boxArray().ixType()}, b.m_deRham->get_n_ghost(), m_infra.geometry(),
            m_infra.m_grid, m_infra.m_distriMap, marker, filler2Dz, amrex::system::verbose);

        hypreSolver2Dz.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[zDir])},
                             amrex::Vector<amrex::MultiFab const*>{&(b.m_data[zDir])}, m_relTol,
                             m_absTol, m_maxiter);
#endif

        x.average_sync();
        x.fill_boundary();
    }

    void solve_particle_charge_e (rhs& b,
                                  sol& x,
                                  std::vector<std::shared_ptr<ParticleGroups<vdim>>>& partGr)
    {
        constexpr int nMmCoeffs[3] = {
            GEMPIC_D_MULT(s_BWx[xDir], s_BWy[xDir],
                          s_BWz[xDir]),  // total number of x-field coefficients
            GEMPIC_D_MULT(s_BWx[yDir], s_BWy[yDir],
                          s_BWz[yDir]),  // total number of y-field coefficients
            GEMPIC_D_MULT(s_BWx[zDir], s_BWy[zDir],
                          s_BWz[zDir])};  // total number of z-field coefficients

        std::array<int, 3> nMmCoeffsArray{nMmCoeffs[xDir], nMmCoeffs[yDir], nMmCoeffs[zDir]};
        DeRhamField<Grid::dual, Space::face> eCoeffs(b.m_deRham, nMmCoeffsArray);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();

            for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*particleSpecies, 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                auto* const particles = pti.GetArrayOfStructs()().data();
                auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsA;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsA[cc] = (eCoeffs.m_data[cc])[pti].array();
                }

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, m_infra.m_plo, m_infra.inv_cell_size_array());

                        get_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsA, spline, charge * weight[pp]);
                    });
            }
        }

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffs.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffs.m_data[0].arrays()), vdim> eCoeffsArrs;
        for (int ndir = 0; ndir < 3; ++ndir)
        {
            eCoeffsArrs[ndir] = (eCoeffs.m_data[ndir]).arrays();
        }

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

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
        auto filler = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int ndir,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0;  // initialize to 0 and then increment it through filler functions

            filler_particle_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, eCoeffsArrs,
                                          mapCols);
        };

        constexpr int maxStencilSize =
            std::max(std::max(nMmCoeffs[xDir], nMmCoeffs[yDir]), nMmCoeffs[zDir]);

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {AMREX_D_DECL(x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
                          x.m_data[zDir].boxArray().ixType())},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{AMREX_D_DECL(
                              &(x.m_data[xDir]), &(x.m_data[yDir]), &(x.m_data[zDir]))},
                          amrex::Vector<amrex::MultiFab const*>{AMREX_D_DECL(
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir]))},
                          m_relTol, m_absTol, m_maxiter);

        // In 2D, the Z E-field is decoupled from the XY E-field and is solved separately
#if GEMPIC_SPACEDIM == 2
        auto filler2Dz = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int,
                                              amrex::Array4<HYPRE_Int const> const* gid,
                                              HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> map_cols_2dz;
            ncols = 0;  // initialize to 0 and then increment it through filler functions

            filler_particle_charge_matrix_2dz(boxno, i, j, k, gid, ncols, cols, mat, eCoeffsArrs,
                                              map_cols_2dz);
        };

        constexpr int maxStencilSize2Dz = nMmCoeffs[zDir];

        amrex::HypreSolver<maxStencilSize2Dz> hypreSolver2Dz(
            {x.m_data[zDir].boxArray().ixType()}, b.m_deRham->get_n_ghost(), m_infra.geometry(),
            m_infra.m_grid, m_infra.m_distriMap, marker, filler2Dz, amrex::system::verbose);

        hypreSolver2Dz.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[zDir])},
                             amrex::Vector<amrex::MultiFab const*>{&(b.m_data[zDir])}, m_relTol,
                             m_absTol, m_maxiter);
#endif

        x.average_sync();
        x.fill_boundary();
    }

    void solve_curlcurl_plus_particle_charge_e (
        rhs& b, sol& x, std::vector<std::shared_ptr<ParticleGroups<vdim>>>& partGr)
    {
        constexpr int nMmCoeffs[3] = {
            GEMPIC_D_MULT(s_BWx[xDir], s_BWy[xDir],
                          s_BWz[xDir]),  // total number of x-field coefficients
            GEMPIC_D_MULT(s_BWx[yDir], s_BWy[yDir],
                          s_BWz[yDir]),  // total number of y-field coefficients
            GEMPIC_D_MULT(s_BWx[zDir], s_BWy[zDir],
                          s_BWz[zDir])};  // total number of z-field coefficients

        std::array<int, 3> nMmCoeffsArray{nMmCoeffs[xDir], nMmCoeffs[yDir], nMmCoeffs[zDir]};
        DeRhamField<Grid::dual, Space::face> eCoeffs(b.m_deRham, nMmCoeffsArray);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();

            for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*particleSpecies, 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                auto* const particles = pti.GetArrayOfStructs()().data();
                auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsA;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsA[cc] = (eCoeffs.m_data[cc])[pti].array();
                }

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, m_infra.m_plo, m_infra.inv_cell_size_array());

                        get_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsA, spline, charge * weight[pp]);
                    });
            }
        }

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffs.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffs.m_data[0].arrays()), vdim> eCoeffsArrs;
        for (int ndir = 0; ndir < 3; ++ndir)
        {
            eCoeffsArrs[ndir] = (eCoeffs.m_data[ndir]).arrays();
        }

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        const auto dxi = m_infra.inv_cell_size_array();
        const auto dx = m_infra.cell_size_array();

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        amrex::GpuArray<HYPRE_Real, 3> edgetoface{
            static_cast<HYPRE_Real>(GEMPIC_D_MULT(dxi[xDir], dx[yDir], dx[zDir])),
            static_cast<HYPRE_Real>(GEMPIC_D_MULT(dx[xDir], dxi[yDir], dx[zDir])),
            static_cast<HYPRE_Real>(GEMPIC_D_MULT(dx[xDir], dx[yDir], dxi[zDir]))};

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
        auto filler = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int ndir,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0;  // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            filler_curlcurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, fac, edgetoface,
                                   mapCols);
            filler_particle_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, eCoeffsArrs,
                                          mapCols);
        };

        constexpr int maxStencilSize =
            6 * GEMPIC_SPACEDIM - 5 +
            std::max(std::max(nMmCoeffs[xDir], nMmCoeffs[yDir]), nMmCoeffs[zDir]);

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {AMREX_D_DECL(x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
                          x.m_data[zDir].boxArray().ixType())},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{AMREX_D_DECL(
                              &(x.m_data[xDir]), &(x.m_data[yDir]), &(x.m_data[zDir]))},
                          amrex::Vector<amrex::MultiFab const*>{AMREX_D_DECL(
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir]))},
                          m_relTol, m_absTol, m_maxiter);

        // In 2D, the Z E-field is decoupled from the XY E-field and is solved separately
#if GEMPIC_SPACEDIM == 2
        auto filler2Dz = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int /*ndir*/,
                                              amrex::Array4<HYPRE_Int const> const* gid,
                                              HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> map_cols_2dz;
            ncols = 0;  // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            filler_curlcurl_matrix_2dz(boxno, i, j, k, gid, ncols, cols, mat, fac, edgetoface,
                                       map_cols_2dz);
            filler_particle_charge_matrix_2dz(boxno, i, j, k, gid, ncols, cols, mat, eCoeffsArrs,
                                              map_cols_2dz);
        };

        constexpr int maxStencilSize2Dz = 5 + nMmCoeffs[zDir];

        amrex::HypreSolver<maxStencilSize2Dz> hypreSolver2Dz(
            {x.m_data[zDir].boxArray().ixType()}, b.m_deRham->get_n_ghost(), m_infra.geometry(),
            m_infra.m_grid, m_infra.m_distriMap, marker, filler2Dz, amrex::system::verbose);

        hypreSolver2Dz.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[zDir])},
                             amrex::Vector<amrex::MultiFab const*>{&(b.m_data[zDir])}, m_relTol,
                             m_absTol, m_maxiter);
#endif

        x.average_sync();
        x.fill_boundary();
    }
};
#endif
}  // namespace Gempic::FieldSolvers

#endif

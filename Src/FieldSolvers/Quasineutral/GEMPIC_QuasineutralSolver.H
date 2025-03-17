#ifndef GEMPIC_QUASINEUTRALSOLVER_H_
#define GEMPIC_QUASINEUTRALSOLVER_H_

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

#include <_hypre_IJ_mv.h>
#include <_hypre_sstruct_mv.h>
#include <map>
#include <type_traits>

#include <AMReX.H>
#include <AMReX_HypreSolver.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

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

template <int hodgeDegree, unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
class HypreQuasineutralLinearSystem
{
private:
    ComputationalDomain m_infra;
    amrex::Real m_relTol, m_absTol{1.e-15};
    amrex::Real m_maxiter{1000};
    int m_verbose;

    // Stencil offsets and bandwidths for the 'rho' matrix
    static constexpr int s_rho_offx[3] = {(degx - 1), degx,
                                          degx}; // x,y,z-field offsets in x-direction
    static constexpr int s_rho_offy[3] = {degy, (degy - 1),
                                          degy}; // x,y,z-field offsets in y-direction
    static constexpr int s_rho_offz[3] = {degz, degz,
                                          (degz - 1)}; // x,y,z-field offsets in z-direction

    static constexpr int s_rho_BWx[3] = {
        (2 * s_rho_offx[xDir] + 1), (2 * s_rho_offx[yDir] + 1),
        (2 * s_rho_offx[zDir] + 1)}; // x,y,z-field bandwidths in x-direction
    static constexpr int s_rho_BWy[3] = {
        (2 * s_rho_offy[xDir] + 1), (2 * s_rho_offy[yDir] + 1),
        (2 * s_rho_offy[zDir] + 1)}; // x,y,z-field bandwidths in y-direction
    static constexpr int s_rho_BWz[3] = {
        (2 * s_rho_offz[xDir] + 1), (2 * s_rho_offz[yDir] + 1),
        (2 * s_rho_offz[zDir] + 1)}; // x,y,z-field bandwidths in z-direction

    static constexpr int s_nRhoCoeffs[3] = {
        GEMPIC_D_MULT(s_rho_BWx[xDir],
                      s_rho_BWy[xDir],
                      s_rho_BWz[xDir]), // total number of x-field coefficients
        GEMPIC_D_MULT(s_rho_BWx[yDir],
                      s_rho_BWy[yDir],
                      s_rho_BWz[yDir]), // total number of y-field coefficients
        GEMPIC_D_MULT(s_rho_BWx[zDir],
                      s_rho_BWy[zDir],
                      s_rho_BWz[zDir])}; // total number of z-field coefficients

    static constexpr std::array<int, 3> s_nRhoCoeffsArray{s_nRhoCoeffs[xDir], s_nRhoCoeffs[yDir],
                                                          s_nRhoCoeffs[zDir]};

    // Stencil offsets, bandwidths and number of coefficients for the 'J x curl' matrix
    // jcrosscurlE x,y,z-component (first bracket) dependency offsets on Ex,Ey,Ez-fields (second
    // bracket) in x-direction
    static constexpr int s_jcrosscurl_offx[3][3] = {
        {(degx - 1), (degx - 1), (degx - 1)}, {degx, degx, degx}, {degx, degx, degx}};
    // jcrosscurlE x,y,z-component (first bracket) dependency offsets on Ex,Ey,Ez-fields (second
    // bracket) in y-direction
    static constexpr int s_jcrosscurl_offy[3][3] = {
        {degy, degy, degy}, {(degy - 1), (degy - 1), (degy - 1)}, {degy, degy, degy}};
    // jcrosscurlE x,y,z-component (first bracket) dependency offsets on Ex,Ey,Ez-fields (second
    // bracket) in z-direction
    static constexpr int s_jcrosscurl_offz[3][3] = {
        {degz, degz, degz}, {degz, degz, degz}, {(degz - 1), (degz - 1), (degz - 1)}};

    // jcrosscurlE x,y,z-component (first bracket) dependency bandwidths on Ex,Ey,Ez-fields (second
    // bracket) in x-direction
    static constexpr int s_jcrosscurl_BWx[3][3] = {{(2 * degx - 1), (2 * degx), (2 * degx)},
                                                   {(2 * degx), (2 * degx + 1), (2 * degx + 1)},
                                                   {(2 * degx), (2 * degx + 1), (2 * degx + 1)}};

    // jcrosscurlE x,y,z-component (first bracket) dependency bandwidths on Ex,Ey,Ez-fields (second
    // bracket) in y-direction
    static constexpr int s_jcrosscurl_BWy[3][3] = {{(2 * degy + 1), (2 * degy), (2 * degy + 1)},
                                                   {(2 * degy), (2 * degy - 1), (2 * degy)},
                                                   {(2 * degy + 1), (2 * degy), (2 * degy + 1)}};

    // jcrosscurlE x,y,z-component (first bracket) dependency bandwidths on Ex,Ey,Ez-fields (second
    // bracket) in z-direction
    static constexpr int s_jcrosscurl_BWz[3][3] = {{(2 * degz + 1), (2 * degz + 1), (2 * degz)},
                                                   {(2 * degz + 1), (2 * degz + 1), (2 * degz)},
                                                   {(2 * degz), (2 * degz), (2 * degz - 1)}};

    static constexpr int s_nJcrosscurlCoeffs[3][3] = {
        // Number of x, y and z E-field coefficients required for (J x curl)[xDir]
        {GEMPIC_D_MULT(s_jcrosscurl_BWx[xDir][xDir],
                       s_jcrosscurl_BWy[xDir][xDir],
                       s_jcrosscurl_BWz[xDir][xDir]),
         GEMPIC_D_MULT(s_jcrosscurl_BWx[xDir][yDir],
                       s_jcrosscurl_BWy[xDir][yDir],
                       s_jcrosscurl_BWz[xDir][yDir]),
         GEMPIC_D_MULT(s_jcrosscurl_BWx[xDir][zDir],
                       s_jcrosscurl_BWy[xDir][zDir],
                       s_jcrosscurl_BWz[xDir][zDir])},
        // Number of x, y and z E-field coefficients required for (J x curl)[yDir]
        {GEMPIC_D_MULT(s_jcrosscurl_BWx[yDir][xDir],
                       s_jcrosscurl_BWy[yDir][xDir],
                       s_jcrosscurl_BWz[yDir][xDir]),
         GEMPIC_D_MULT(s_jcrosscurl_BWx[yDir][yDir],
                       s_jcrosscurl_BWy[yDir][yDir],
                       s_jcrosscurl_BWz[yDir][yDir]),
         GEMPIC_D_MULT(s_jcrosscurl_BWx[yDir][zDir],
                       s_jcrosscurl_BWy[yDir][zDir],
                       s_jcrosscurl_BWz[yDir][zDir])},
        // Number of x, y and z E-field coefficients required for (J x curl)[zDir]
        {GEMPIC_D_MULT(s_jcrosscurl_BWx[zDir][xDir],
                       s_jcrosscurl_BWy[zDir][xDir],
                       s_jcrosscurl_BWz[zDir][xDir]),
         GEMPIC_D_MULT(s_jcrosscurl_BWx[zDir][yDir],
                       s_jcrosscurl_BWy[zDir][yDir],
                       s_jcrosscurl_BWz[zDir][yDir]),
         GEMPIC_D_MULT(s_jcrosscurl_BWx[zDir][zDir],
                       s_jcrosscurl_BWy[zDir][zDir],
                       s_jcrosscurl_BWz[zDir][zDir])}};

    static constexpr int s_nJcrosscurlCoeffsTotal[3] = {
        (s_nJcrosscurlCoeffs[xDir][xDir] + s_nJcrosscurlCoeffs[xDir][yDir] +
         s_nJcrosscurlCoeffs[xDir][zDir]), // Total number of x,y,z E-field coefficients required
                                           // for (J x curl)[xDir]
        (s_nJcrosscurlCoeffs[yDir][xDir] + s_nJcrosscurlCoeffs[yDir][yDir] +
         s_nJcrosscurlCoeffs[yDir][zDir]), // Total number of x,y,z E-field coefficients required
                                           // for (J x curl)[yDir]
        (s_nJcrosscurlCoeffs[zDir][xDir] + s_nJcrosscurlCoeffs[zDir][yDir] +
         s_nJcrosscurlCoeffs[zDir][zDir]) // Total number of x,y,z E-field coefficients required
                                          // for (J x curl)[zDir]
    };

    static constexpr std::array<int, 3> s_nJcrosscurlCoeffsArray{
        (s_nJcrosscurlCoeffsTotal[xDir]), // Total number of x,y,z E-field coefficients
                                          // required for (J x curl)[xDir]
        (s_nJcrosscurlCoeffsTotal[yDir]), // Total number of x,y,z E-field coefficients
                                          // required for (J x curl)[yDir]
        (s_nJcrosscurlCoeffsTotal[zDir])  // Total number of x,y,z E-field coefficients required
                                          // for (J x curl)[zDir]
    };

    static constexpr int s_nJcrosscurlCoeffsArrayStart[3][3] = {
        {0, s_nJcrosscurlCoeffs[xDir][xDir],
         (s_nJcrosscurlCoeffs[xDir][xDir] + s_nJcrosscurlCoeffs[xDir][yDir])},
        {0, s_nJcrosscurlCoeffs[yDir][xDir],
         (s_nJcrosscurlCoeffs[yDir][xDir] + s_nJcrosscurlCoeffs[yDir][yDir])},
        {0, s_nJcrosscurlCoeffs[zDir][xDir],
         (s_nJcrosscurlCoeffs[zDir][xDir] + s_nJcrosscurlCoeffs[zDir][yDir])}};

public:
    HypreQuasineutralLinearSystem(ComputationalDomain infr,
                                  std::shared_ptr<DeRhamComplex> drc,
                                  amrex::Real tol = 1.e-11,
                                  int verbose = 0) :
        m_infra{infr}, m_relTol{tol}

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
        if (mapColsGeneral.find(gdtmp) != mapColsGeneral.end()) // element exists already
        {
            icol = mapColsGeneral[gdtmp];
        }
        else // element does not exist, and is newly added
        {
            icol = ncols++;
            cols[icol] = gdtmp;
            mapColsGeneral.insert(std::make_pair(cols[icol], icol));
            mat[icol] = 0.0; // matrix element is initialized to 0, and then incremented by filler
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
                if ((ndir == zDir) && (AMREX_SPACEDIM == 2))
                {
                    // This function contributes (<= 5) columns to ncols
                    int icol;
                    int gdtmp;
                    gdtmp = gid[2](i, j, k);
                    assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                    mat[icol] += 2.0 * (fac[0] + fac[1]) * edgetoface[2];

                    gdtmp = gid[2]((i - 1), j, k);
                    assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                    mat[icol] += -fac[0] * edgetoface[2];
                    gdtmp = gid[2]((i + 1), j, k);
                    assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                    mat[icol] += -fac[0] * edgetoface[2];
                    gdtmp = gid[2](i, (j - 1), k);
                    assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                    mat[icol] += -fac[1] * edgetoface[2];
                    gdtmp = gid[2](i, (j + 1), k);
                    assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                    mat[icol] += -fac[1] * edgetoface[2];
                }
                else
                {
                    // This function contributes (<= 6*AMREX_SPACEDIM-5) columns to ncols
                    int icol;
                    int gdtmp;
                    int n[AMREX_SPACEDIM], in[AMREX_SPACEDIM], jn[AMREX_SPACEDIM],
                        kn[AMREX_SPACEDIM];
                    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                    {
                        n[dir] = (ndir + dir) % AMREX_SPACEDIM;
                        in[dir] = (n[dir] == 0);
                        jn[dir] = (n[dir] == 1);
                        kn[dir] = (n[dir] == 2);
                    }

                    amrex::Real temp;
                    temp = 0.0;
                    for (int dir = 1; dir < AMREX_SPACEDIM; dir++)
                    {
                        temp += 2.0 * (fac[n[dir]]);
                    }

                    gdtmp = gid[n[0]](i, j, k);
                    assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                    mat[icol] += temp * edgetoface[n[0]];

                    for (int dir = 1; dir < AMREX_SPACEDIM; dir++)
                    {
                        gdtmp = gid[n[0]]((i - in[dir]), (j - jn[dir]), (k - kn[dir]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += -fac[n[dir]] * edgetoface[n[0]];

                        gdtmp = gid[n[0]]((i + in[dir]), (j + jn[dir]), (k + kn[dir]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += -fac[n[dir]] * edgetoface[n[0]];
                    }

                    for (int dir = 1; dir < AMREX_SPACEDIM; dir++)
                    {
                        gdtmp = gid[n[dir]]((i + in[0]), (j + jn[0]), (k + kn[0]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += fac[n[dir]] * edgetoface[n[dir]];

                        gdtmp = gid[n[dir]]((i - in[dir]), (j - jn[dir]), (k - kn[dir]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += fac[n[dir]] * edgetoface[n[dir]];

                        gdtmp = gid[n[dir]](i, j, k);
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += -fac[n[dir]] * edgetoface[n[dir]];

                        gdtmp = gid[n[dir]]((i - in[dir] + in[0]), (j - jn[dir] + jn[0]),
                                            (k - kn[dir] + kn[0]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += -fac[n[dir]] * edgetoface[n[dir]];
                    }
                }

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
                int nc[3]; // = {0,0,0};
                nc[0] = 0;
                nc[1] = 0;
                nc[2] = 0;

                for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                {
                    nc[dir] = (ndir == dir);
                }

                int icol, gdtmp;
                gdtmp = gid[ndir](i, j, k);

                assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);

                mat[icol] += 0.5 *
                             (rhoarrs[boxno](i, j, k) +
                              rhoarrs[boxno](i + nc[xDir], j + nc[yDir], k + nc[zDir])) *
                             (1.0 / m_infra.cell_volume()) * edgetoface[ndir];
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
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        GEMPIC_D_LOOP_BEGIN(
            for (ii = -s_rho_offx[ndir]; ii < (-s_rho_offx[ndir] + s_rho_BWx[ndir]); ++ii),
            for (jj = -s_rho_offy[ndir]; jj < (-s_rho_offy[ndir] + s_rho_BWy[ndir]); ++jj),
            for (kk = -s_rho_offz[ndir]; kk < (-s_rho_offz[ndir] + s_rho_BWz[ndir]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_rho_offx[ndir], jjTemp = jj + s_rho_offy[ndir],
                         kkTemp = kk + s_rho_offz[ndir]);

            gdtmp = gid[ndir]((i + ii), (j + jj), (k + kk));

            assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] +=
                eCoeffsArrs[ndir][boxno](i, j, k,
                                         AMREX_D_TERM(iiTemp, +jjTemp * s_rho_BWx[ndir],
                                                      +kkTemp * s_rho_BWx[ndir] * s_rho_BWy[ndir]));
        GEMPIC_D_LOOP_END
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_rho_e_weights_at_particle_positions_and_deposit_to_face (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        SplineType const& spline,
        amrex::Real chargeWeight)
    {
        constexpr int nWtsXYZ[3][3] = {
            {degx, degx + 1, degx + 1},  // number of 1-form x,y,z-field weights in X-direction
            {degy + 1, degy, degy + 1},  // number of 1-form x,y,z-field weights in Y-direction
            {degz + 1, degz + 1, degz}}; // number of 1-form x,y,z-field weights in Z-direction

        constexpr int nWts[3] = {
            // number of 1-form x-field weights
            GEMPIC_D_MULT(nWtsXYZ[xDir][xDir], nWtsXYZ[yDir][xDir], nWtsXYZ[zDir][xDir]),
            // number of 1-form y-field weights
            GEMPIC_D_MULT(nWtsXYZ[xDir][yDir], nWtsXYZ[yDir][yDir], nWtsXYZ[zDir][yDir]),
            // number of 1-form z-field weights
            GEMPIC_D_MULT(nWtsXYZ[xDir][zDir], nWtsXYZ[yDir][zDir], nWtsXYZ[zDir][zDir])};
        constexpr int nWtsStart[3] = {0, nWts[xDir], (nWts[xDir] + nWts[yDir])};

        amrex::GpuArray<amrex::Real, (nWts[xDir] + nWts[yDir] + nWts[zDir])> eWts;
        amrex::GpuArray<amrex::IntVect, (nWts[xDir] + nWts[yDir] + nWts[zDir])> eCells;

        spline.eval_e_field_spline_weights(eWts, eCells);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ[xDir][ndir]; ++ii),
                                for (int jj = 0; jj < nWtsXYZ[yDir][ndir]; ++jj),
                                for (int kk = 0; kk < nWtsXYZ[zDir][ndir]; ++kk))
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

                    AMREX_D_EXPR(iiTemp = wCell.x - cell.x + s_rho_offx[ndir],
                                 jjTemp = wCell.y - cell.y + s_rho_offy[ndir],
                                 kkTemp = wCell.z - cell.z + s_rho_offz[ndir]);

                    component = GEMPIC_D_ADD(iiTemp, jjTemp * s_rho_BWx[ndir],
                                             kkTemp * s_rho_BWx[ndir] * s_rho_BWy[ndir]);

                    amrex::HostDevice::Atomic::Add(
                        eCoeffsA[ndir].ptr(cell.x, cell.y, cell.z, component),
                        sCoeff * chargeWeight * eWts[nWtsStart[ndir] + i]);
                }
            GEMPIC_D_LOOP_END
        }
    }

    void filler_jcrosscurl_matrix (
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
        // This function contributes (<= nJcrosscurlCoeffsTotal[ndir]) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};

        for (int ncDir = 0; ncDir < 3; ncDir++)
        {
            GEMPIC_D_LOOP_BEGIN(
                for (ii = -s_jcrosscurl_offx[ndir][ncDir];
                     ii < (-s_jcrosscurl_offx[ndir][ncDir] + s_jcrosscurl_BWx[ndir][ncDir]); ++ii),
                for (jj = -s_jcrosscurl_offy[ndir][ncDir];
                     jj < (-s_jcrosscurl_offy[ndir][ncDir] + s_jcrosscurl_BWy[ndir][ncDir]); ++jj),
                for (kk = -s_jcrosscurl_offz[ndir][ncDir];
                     kk < (-s_jcrosscurl_offz[ndir][ncDir] + s_jcrosscurl_BWz[ndir][ncDir]); ++kk))
                AMREX_D_EXPR(iiTemp = ii + s_jcrosscurl_offx[ndir][ncDir],
                             jjTemp = jj + s_jcrosscurl_offy[ndir][ncDir],
                             kkTemp = kk + s_jcrosscurl_offz[ndir][ncDir]);

                gdtmp = gid[ncDir]((i + ii), (j + jj), (k + kk));

                assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                mat[icol] += eCoeffsArrs[ndir][boxno](
                    i, j, k,
                    s_nJcrosscurlCoeffsArrayStart[ndir][ncDir] +
                        AMREX_D_TERM(iiTemp, +jjTemp * s_jcrosscurl_BWx[ndir][ncDir],
                                     +kkTemp * s_jcrosscurl_BWx[ndir][ncDir] *
                                         s_jcrosscurl_BWy[ndir][ncDir]));
            GEMPIC_D_LOOP_END
        }
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_jcrosscurl_e_weights_at_particle_positions_and_deposit_to_face (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        SplineType const& spline,
        amrex::GpuArray<amrex::Real, vdim> const& vel,
        amrex::Real chargeWeight)
    {
        constexpr int nWtsXYZ2f[3][3] = {
            {(degx + 1), degx, degx},  // number of 2-form x,y,z-field weights in X-direction
            {degy, (degy + 1), degy},  // number of 2-form x,y,z-field weights in Y-direction
            {degz, degz, (degz + 1)}}; // number of 2-form x,y,z-field weights in Z-direction

        constexpr int nWts2f[3] = {
            // number of 2-form X-field weights in x,y,z-directions
            GEMPIC_D_MULT(nWtsXYZ2f[xDir][xDir], nWtsXYZ2f[yDir][xDir], nWtsXYZ2f[zDir][xDir]),
            // number of 2-form Y-field weights in x,y,z-directions
            GEMPIC_D_MULT(nWtsXYZ2f[xDir][yDir], nWtsXYZ2f[yDir][yDir], nWtsXYZ2f[zDir][yDir]),
            // number of 2-form Z-field weights in x,y,z-directions
            GEMPIC_D_MULT(nWtsXYZ2f[xDir][zDir], nWtsXYZ2f[yDir][zDir], nWtsXYZ2f[zDir][zDir])};
        constexpr int nWtsStart2f[3] = {0, nWts2f[xDir], (nWts2f[xDir] + nWts2f[yDir])};

        amrex::GpuArray<amrex::Real, (nWts2f[xDir] + nWts2f[yDir] + nWts2f[zDir])> bWts;
        amrex::GpuArray<amrex::IntVect, (nWts2f[xDir] + nWts2f[yDir] + nWts2f[zDir])> bCells;

        spline.eval_b_field_spline_weights(bWts, bCells);

        constexpr int nWtsXYZ1f[3][3] = {
            {degx, (degx + 1), (degx + 1)},  // number of 1-form x,y,z-field weights in X-direction
            {(degy + 1), degy, (degy + 1)},  // number of 1-form x,y,z-field weights in Y-direction
            {(degz + 1), (degz + 1), degz}}; // number of 1-form x,y,z-field weights in Z-direction

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ1f[xDir][ndir]; ++ii),
                                for (int jj = 0; jj < nWtsXYZ1f[yDir][ndir]; ++jj),
                                for (int kk = 0; kk < nWtsXYZ1f[zDir][ndir]; ++kk))
                const amrex::Real sCoeff =
                    ((ndir == 0)
                         ? (spline.template eval_b_spline<Field::DualTwoForm, xDir>(ii, jj, kk))
                         : ((ndir == 1) ? (spline.template eval_b_spline<Field::DualTwoForm, yDir>(
                                              ii, jj, kk))
                                        : (spline.template eval_b_spline<Field::DualTwoForm, zDir>(
                                              ii, jj, kk))));

                amrex::Dim3 cell = {GEMPIC_D_PAD(
                    spline.m_firstIndex[xDir] + ii, spline.m_firstIndex[yDir] + jj,
                    spline.m_firstIndex[zDir] + kk)}; // this index is of a primal edge with E[ndir]

                amrex::Dim3 wCell[4];

                int ncDir[4];
                amrex::Real sign[4];

                int n[3], in[3], jn[3], kn[3];
                for (int dir = 0; dir < 3; dir++)
                {
                    n[dir] = (ndir + dir) % 3;
                    in[dir] = (n[dir] == 0);
                    jn[dir] = (n[dir] == 1);
                    kn[dir] = (n[dir] == 2);
                }

#if AMREX_SPACEDIM == 2
                kn[0] = 0;
                kn[1] = 0;
                kn[2] = 0;
#endif

                // vcrossb[n[0]] = (v[n[1]] * B[n[2]] - v[n[2]] * B[n[1]])
                for (int i = 0; i < nWts2f[n[1]]; i++)
                { // vcrossb[n[0]] = (... - v[n[2]] * B[n[1]])
                    amrex::Dim3 bCell =
                        (bCells[nWtsStart2f[n[1]] + i]).dim3(); // this index is of a primal face

                    // These indices are of primal edges corresponding to the primal face bCell
                    wCell[0] = {GEMPIC_D_PAD(bCell.x + in[2], bCell.y + jn[2], bCell.z + kn[2])};
                    ncDir[0] = n[0];
                    sign[0] = 1.0; // -vel[n[2]] * (E[n[0]](i+in2,j+jn2,k+kn2))
                    wCell[1] = {GEMPIC_D_PAD(bCell.x, bCell.y, bCell.z)};
                    ncDir[1] = n[0];
                    sign[1] = -1.0; // -vel[n[2]] * (-E[n[0]](i,j,k))
                    wCell[2] = {GEMPIC_D_PAD(bCell.x + in[0], bCell.y + jn[0], bCell.z + kn[0])};
                    ncDir[2] = n[2];
                    sign[2] = -1.0; // -vel[n[2]] * (-E[n[2]](i+in0,j+jn0,k+kn0))
                    wCell[3] = {GEMPIC_D_PAD(bCell.x, bCell.y, bCell.z)};
                    ncDir[3] = n[2];
                    sign[3] = 1.0; // -vel[n[2]] * (E[n[2]](i,j,k))

                    for (int nc = 0; nc < 4; nc++)
                    {
                        AMREX_D_EXPR(
                            iiTemp = wCell[nc].x - cell.x + s_jcrosscurl_offx[ndir][ncDir[nc]],
                            jjTemp = wCell[nc].y - cell.y + s_jcrosscurl_offy[ndir][ncDir[nc]],
                            kkTemp = wCell[nc].z - cell.z + s_jcrosscurl_offz[ndir][ncDir[nc]]);
                        component = GEMPIC_D_ADD(iiTemp, jjTemp * s_jcrosscurl_BWx[ndir][ncDir[nc]],
                                                 kkTemp * s_jcrosscurl_BWx[ndir][ncDir[nc]] *
                                                     s_jcrosscurl_BWy[ndir][ncDir[nc]]);

                        amrex::HostDevice::Atomic::Add(
                            eCoeffsA[ndir].ptr(
                                cell.x, cell.y, cell.z,
                                s_nJcrosscurlCoeffsArrayStart[ndir][ncDir[nc]] + component),
                            sCoeff * chargeWeight * (-vel[n[2]]) * (sign[nc]) *
                                bWts[nWtsStart2f[n[1]] + i]);
                    }
                }

                for (int i = 0; i < nWts2f[n[2]]; i++)
                { // vcrossb[n[0]] = (v[n[1]] * B[n[2]] - ...)
                    amrex::Dim3 bCell =
                        (bCells[nWtsStart2f[n[2]] + i]).dim3(); // this index is of a primal face

                    // These indices are of primal edges corresponding to the primal face bCell
                    wCell[0] = {GEMPIC_D_PAD(bCell.x + in[0], bCell.y + jn[0], bCell.z + kn[0])};
                    ncDir[0] = n[1];
                    sign[0] = 1.0; // vel[n[1]] * (E[n[1]](i+in0,j+jn0,k+kn0))
                    wCell[1] = {GEMPIC_D_PAD(bCell.x, bCell.y, bCell.z)};
                    ncDir[1] = n[1];
                    sign[1] = -1.0; // vel[n[1]] * (-E[n[1]](i,j,k))
                    wCell[2] = {GEMPIC_D_PAD(bCell.x + in[1], bCell.y + jn[1], bCell.z + kn[1])};
                    ncDir[2] = n[0];
                    sign[2] = -1.0; // vel[n[1]] * (-E[n[0]](i+in1,j+jn1,k+kn1))
                    wCell[3] = {GEMPIC_D_PAD(bCell.x, bCell.y, bCell.z)};
                    ncDir[3] = n[0];
                    sign[3] = 1.0; // vel[n[1]] * (E[n[0]](i,j,k))

                    for (int nc = 0; nc < 4; nc++)
                    {
                        AMREX_D_EXPR(
                            iiTemp = wCell[nc].x - cell.x + s_jcrosscurl_offx[ndir][ncDir[nc]],
                            jjTemp = wCell[nc].y - cell.y + s_jcrosscurl_offy[ndir][ncDir[nc]],
                            kkTemp = wCell[nc].z - cell.z + s_jcrosscurl_offz[ndir][ncDir[nc]]);
                        component = GEMPIC_D_ADD(iiTemp, jjTemp * s_jcrosscurl_BWx[ndir][ncDir[nc]],
                                                 kkTemp * s_jcrosscurl_BWx[ndir][ncDir[nc]] *
                                                     s_jcrosscurl_BWy[ndir][ncDir[nc]]);

                        amrex::HostDevice::Atomic::Add(
                            eCoeffsA[ndir].ptr(
                                cell.x, cell.y, cell.z,
                                s_nJcrosscurlCoeffsArrayStart[ndir][ncDir[nc]] + component),
                            sCoeff * chargeWeight * (vel[n[1]]) * (sign[nc]) *
                                bWts[nWtsStart2f[n[2]] + i]);
                    }
                }
            GEMPIC_D_LOOP_END
        }
    }

    void solve_curlcurl_plus_field_charge_e (DeRhamField<Grid::dual, Space::face>& b,
                                             DeRhamField<Grid::primal, Space::edge>& x,
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
            ncols = 0; // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
                                       edgetoface, mapCols);
            filler_curlcurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, fac, edgetoface,
                                   mapCols);
            //filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
            //edgetoface, mapCols);
        };

        constexpr int maxStencilSize = 6 * AMREX_SPACEDIM - 5;

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);

        x.fill_boundary();
    }

    void solve_particle_charge_e (DeRhamField<Grid::dual, Space::face>& b,
                                  DeRhamField<Grid::primal, Space::edge>& x,
                                  std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsRho(b.m_deRham, s_nRhoCoeffsArray);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                 ++pti)
            {
                const long np = pti.numParticles();
                auto* const particles = pti.GetArrayOfStructs()().data();
                auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsARho;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsARho[cc] = (eCoeffsRho.m_data[cc])[pti].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        get_rho_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsARho, spline, chargeSqrByMass * weight[pp]);
                    });
            }
        }

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsRho.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffsRho.m_data[0].arrays()), vdim> eCoeffsArrsRho;
        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsArrsRho[ndir] = (eCoeffsRho.m_data[ndir]).arrays();
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
            ncols = 0; // initialize to 0 and then increment it through filler functions

            filler_particle_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                          eCoeffsArrsRho, mapCols);
        };

        constexpr int maxStencilSize =
            std::max(std::max(s_nRhoCoeffs[xDir], s_nRhoCoeffs[yDir]), s_nRhoCoeffs[zDir]);

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);
        x.fill_boundary();
    }

    void solve_curlcurl_plus_particle_charge_e (
        DeRhamField<Grid::dual, Space::face>& b,
        DeRhamField<Grid::primal, Space::edge>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsRho(b.m_deRham, s_nRhoCoeffsArray);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                 ++pti)
            {
                const long np = pti.numParticles();
                auto* const particles = pti.GetArrayOfStructs()().data();
                auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsARho;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsARho[cc] = (eCoeffsRho.m_data[cc])[pti].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        get_rho_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsARho, spline, chargeSqrByMass * weight[pp]);
                    });
            }
        }

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsRho.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffsRho.m_data[0].arrays()), vdim> eCoeffsArrsRho;
        for (int ndir = 0; ndir < 3; ++ndir)
        {
            eCoeffsArrsRho[ndir] = (eCoeffsRho.m_data[ndir]).arrays();
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
            ncols = 0; // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            filler_curlcurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, fac, edgetoface,
                                   mapCols);
            filler_particle_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                          eCoeffsArrsRho, mapCols);
        };

        constexpr int maxStencilSize =
            6 * AMREX_SPACEDIM - 5 +
            std::max(std::max(s_nRhoCoeffs[xDir], s_nRhoCoeffs[yDir]), s_nRhoCoeffs[zDir]);

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);

        x.fill_boundary();
    }

    void solve_jcrosscurl_plus_field_charge_e (
        DeRhamField<Grid::dual, Space::face>& b,
        DeRhamField<Grid::primal, Space::edge>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        const DeRhamField<Grid::dual, Space::cell>& rho)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsJcrosscurl(b.m_deRham,
                                                               s_nJcrosscurlCoeffsArray);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                 ++pti)
            {
                const long np = pti.numParticles();
                auto* const particles = pti.GetArrayOfStructs()().data();
                auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                auto* const velx = pti.GetStructOfArrays().GetRealData(0).data();
                auto* const vely = pti.GetStructOfArrays().GetRealData(1).data();
                auto* const velz = pti.GetStructOfArrays().GetRealData(2).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsAJcrosscurl;

                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsAJcrosscurl[cc] = (eCoeffsJcrosscurl.m_data[cc])[pti].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }

                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};

                        get_jcrosscurl_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsAJcrosscurl, spline, vel, chargeSqrByMass * weight[pp]);
                    });
            }
        }

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsJcrosscurl.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
            //E_coeffs.m_data[ndir].FillBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffsJcrosscurl.m_data[0].arrays()), vdim> eCoeffsArrsJcrosscurl;
        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsArrsJcrosscurl[ndir] = (eCoeffsJcrosscurl.m_data[ndir]).arrays();
        }

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        auto const& rhoarrs = (rho.m_data).arrays();

        const auto dxi = m_infra.inv_cell_size_array();
        const auto dx = m_infra.cell_size_array();

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
            ncols = 0; // initialize to 0 and then increment it through filler functions

            // The two filler functions called below can be called in any order
            //filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
            //edgetoface, mapCols);
            filler_jcrosscurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                     eCoeffsArrsJcrosscurl, mapCols);
            filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
                                       edgetoface, mapCols);
        };

        constexpr int maxStencilSize =
            std::max(std::max(s_nJcrosscurlCoeffsTotal[xDir], s_nJcrosscurlCoeffsTotal[yDir]),
                     s_nJcrosscurlCoeffsTotal[zDir]);

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);

        x.fill_boundary();
    }

    void solve_curlcurl_plus_particle_charge_plus_jcrosscurl_e (
        DeRhamField<Grid::dual, Space::face>& b,
        DeRhamField<Grid::primal, Space::edge>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        amrex::Real jcrosscurlCoeff)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsRho(b.m_deRham, s_nRhoCoeffsArray);
        DeRhamField<Grid::dual, Space::face> eCoeffsJcrosscurl(b.m_deRham,
                                                               s_nJcrosscurlCoeffsArray);
        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*particleSpecies, 0); pti.isValid();
                 ++pti)
            {
                const long np = pti.numParticles();
                auto* const particles = pti.GetArrayOfStructs()().data();
                auto* const weight = pti.GetStructOfArrays().GetRealData(vdim).data();
                auto* const velx = pti.GetStructOfArrays().GetRealData(0).data();
                auto* const vely = pti.GetStructOfArrays().GetRealData(1).data();
                auto* const velz = pti.GetStructOfArrays().GetRealData(2).data();
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsARho;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsAJcrosscurl;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsARho[cc] = (eCoeffsRho.m_data[cc])[pti].array();
                    eCoeffsAJcrosscurl[cc] = (eCoeffsJcrosscurl.m_data[cc])[pti].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};
                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());
                        amrex::GpuArray<amrex::Real, vdim> vel{velx[pp], vely[pp], velz[pp]};
                        get_rho_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsARho, spline, chargeSqrByMass * weight[pp]);
                        get_jcrosscurl_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsAJcrosscurl, spline, vel,
                            jcrosscurlCoeff * chargeSqrByMass * weight[pp]);
                    });
            }
        }
        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsRho.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
            eCoeffsJcrosscurl.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
            //E_coeffs.m_data[ndir].FillBoundary(m_infra.geometry().periodicity());
        }
        std::array<decltype(eCoeffsRho.m_data[0].arrays()), vdim> eCoeffsArrsRho;
        std::array<decltype(eCoeffsJcrosscurl.m_data[0].arrays()), vdim> eCoeffsArrsJcrosscurl;
        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsArrsRho[ndir] = (eCoeffsRho.m_data[ndir]).arrays();
            eCoeffsArrsJcrosscurl[ndir] = (eCoeffsJcrosscurl.m_data[ndir]).arrays();
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
            ncols = 0; // initialize to 0 and then increment it through filler functions

            // The filler functions called below can be called in any order
            filler_curlcurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, fac, edgetoface,
                                   mapCols);
            filler_particle_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                          eCoeffsArrsRho, mapCols);
            filler_jcrosscurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                     eCoeffsArrsJcrosscurl, mapCols);
        };
        constexpr int maxStencilSize =
            6 * AMREX_SPACEDIM - 5 +
            std::max(std::max(s_nRhoCoeffs[xDir], s_nRhoCoeffs[yDir]), s_nRhoCoeffs[zDir]) +
            std::max(std::max(s_nJcrosscurlCoeffsTotal[xDir], s_nJcrosscurlCoeffsTotal[yDir]),
                     s_nJcrosscurlCoeffsTotal[zDir]);
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            b.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, amrex::system::verbose);
        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);
        x.fill_boundary();
    }
};

template <int hodgeDegree, unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
class QuasineutralTimeStepping
{
private:
    ComputationalDomain m_infra;
    amrex::Real m_relTol, m_absTol{1.e-15};
    amrex::Real m_maxiter{1000};
    int m_verbose;

public:
    QuasineutralTimeStepping(ComputationalDomain infr,
                             std::shared_ptr<DeRhamComplex> drc,
                             amrex::Real tol = 1.e-11,
                             int verbose = 0) :
        m_infra{infr}, m_relTol{tol}

    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    void push_x_using_v (amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& pos,
                         amrex::GpuArray<amrex::Real, vdim> const& vel,
                         amrex::Real dt) const
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            pos[i] += vel[i] * dt;
        }
    }

    void push_v_using_e (amrex::GpuArray<amrex::Real, vdim>& vel,
                         amrex::GpuArray<amrex::Real, vdim> const& efield,
                         amrex::Real dt,
                         amrex::Real chargeByMass) const
    {
        for (int i = 0; i < vdim; i++)
        {
            vel[i] += chargeByMass * efield[i] * dt;
        }
    }

    void rotate_v_using_b (amrex::GpuArray<amrex::Real, vdim>& vel,
                           amrex::GpuArray<amrex::Real, vdim> const& bfield,
                           amrex::Real dt,
                           amrex::Real chargeByMass) const
    {
        amrex::Real bmag = std::sqrt(bfield[xDir] * bfield[xDir] + bfield[yDir] * bfield[yDir] +
                                     bfield[zDir] * bfield[zDir]);
        if (bmag != 0.0)
        {
            amrex::GpuArray<amrex::Real, vdim> b{bfield[xDir] / bmag, bfield[yDir] / bmag,
                                                 bfield[zDir] / bmag};
            amrex::Real a = chargeByMass * bmag * dt;
            amrex::Real ca = cos(a);
            amrex::Real sa = sin(a);
            amrex::GpuArray<amrex::Real, vdim> vtemp{vel[xDir], vel[yDir], vel[zDir]};

            vel[xDir] =
                (b[xDir] * b[xDir] + (b[yDir] * b[yDir] + b[zDir] * b[zDir]) * ca) * vtemp[xDir] +
                (b[zDir] * sa + b[yDir] * b[xDir] * (1.0 - ca)) * vtemp[yDir] +
                (-b[yDir] * sa + b[zDir] * b[xDir] * (1.0 - ca)) * vtemp[zDir];

            vel[yDir] =
                (-b[zDir] * sa + b[yDir] * b[xDir] * (1.0 - ca)) * vtemp[xDir] +
                (b[yDir] * b[yDir] + (b[xDir] * b[xDir] + b[zDir] * b[zDir]) * ca) * vtemp[yDir] +
                (b[xDir] * sa + b[zDir] * b[yDir] * (1.0 - ca)) * vtemp[zDir];

            vel[zDir] =
                (b[yDir] * sa + b[zDir] * b[xDir] * (1.0 - ca)) * vtemp[xDir] +
                (-b[xDir] * sa + b[zDir] * b[yDir] * (1.0 - ca)) * vtemp[yDir] +
                (b[zDir] * b[zDir] + (b[xDir] * b[xDir] + b[yDir] * b[yDir]) * ca) * vtemp[zDir];
        }
    }
};

} // namespace Gempic::FieldSolvers

#endif

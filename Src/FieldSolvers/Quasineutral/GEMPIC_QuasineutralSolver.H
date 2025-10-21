#ifndef GEMPIC_QUASINEUTRALSOLVER_H_
#define GEMPIC_QUASINEUTRALSOLVER_H_

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

#include <_hypre_IJ_mv.h>
#include <_hypre_sstruct_mv.h>
#include <map>

#include <AMReX.H>
#include <AMReX_HypreSolver.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_QuasineutralDebug.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{
using namespace Forms;
using namespace Particle;

enum class RotatePushOrder
{
    PushvThenRotatev,
    RotatevThenPushv,
};

template <int hodgeDegree, unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
class QuasineutralSolver
{
private:
    ComputationalDomain m_infra;
    amrex::Real m_relTol, m_absTol{1.e-15};
    amrex::Real m_maxiter{1000};
    int m_verbose;

    // Stencil offsets and bandwidths for the 'rho' matrix
    static constexpr std::array<int, 3> s_rhoOffx{(degx - 1), degx,
                                                  degx}; // x,y,z-field offsets in x-direction
    static constexpr std::array<int, 3> s_rhoOffy{degy, (degy - 1),
                                                  degy}; // x,y,z-field offsets in y-direction
    static constexpr std::array<int, 3> s_rhoOffz{degz, degz,
                                                  (degz - 1)}; // x,y,z-field offsets in z-direction

    static constexpr std::array<int, 3> s_rhoBWx{
        (2 * s_rhoOffx[xDir] + 1), (2 * s_rhoOffx[yDir] + 1),
        (2 * s_rhoOffx[zDir] + 1)}; // x,y,z-field bandwidths in x-direction
    static constexpr std::array<int, 3> s_rhoBWy{
        (2 * s_rhoOffy[xDir] + 1), (2 * s_rhoOffy[yDir] + 1),
        (2 * s_rhoOffy[zDir] + 1)}; // x,y,z-field bandwidths in y-direction
    static constexpr std::array<int, 3> s_rhoBWz{
        (2 * s_rhoOffz[xDir] + 1), (2 * s_rhoOffz[yDir] + 1),
        (2 * s_rhoOffz[zDir] + 1)}; // x,y,z-field bandwidths in z-direction

    static constexpr std::array<int, 3> s_nRhoCoeffs{
        GEMPIC_D_MULT(s_rhoBWx[xDir],
                      s_rhoBWy[xDir],
                      s_rhoBWz[xDir]), // total number of x-field coefficients
        GEMPIC_D_MULT(s_rhoBWx[yDir],
                      s_rhoBWy[yDir],
                      s_rhoBWz[yDir]), // total number of y-field coefficients
        GEMPIC_D_MULT(s_rhoBWx[zDir],
                      s_rhoBWy[zDir],
                      s_rhoBWz[zDir])}; // total number of z-field coefficients

    // Stencil offsets, bandwidths and number of coefficients for the 'J x curl' matrix
    // jcrosscurlE x,y,z-component (first bracket) dependency offsets on Ex,Ey,Ez-fields (second
    // bracket) in x-direction
    static constexpr std::array<std::array<int, 3>, 3> s_jcrosscurlOffx = {
        {{{(degx - 1), (degx - 1), (degx - 1)}}, {{degx, degx, degx}}, {{degx, degx, degx}}}};
    // jcrosscurlE x,y,z-component (first bracket) dependency offsets on Ex,Ey,Ez-fields (second
    // bracket) in y-direction
    static constexpr std::array<std::array<int, 3>, 3> s_jcrosscurlOffy = {
        {{{degy, degy, degy}}, {{(degy - 1), (degy - 1), (degy - 1)}}, {{degy, degy, degy}}}};
    // jcrosscurlE x,y,z-component (first bracket) dependency offsets on Ex,Ey,Ez-fields (second
    // bracket) in z-direction
    static constexpr std::array<std::array<int, 3>, 3> s_jcrosscurlOffz = {
        {{{degz, degz, degz}}, {{degz, degz, degz}}, {{(degz - 1), (degz - 1), (degz - 1)}}}};

    // jcrosscurlE x,y,z-component (first bracket) dependency bandwidths on Ex,Ey,Ez-fields (second
    // bracket) in x-direction
    static constexpr std::array<std::array<int, 3>, 3> s_jcrosscurlBWx = {
        {{{(2 * degx - 1), (2 * degx), (2 * degx)}},
         {{(2 * degx), (2 * degx + 1), (2 * degx + 1)}},
         {{(2 * degx), (2 * degx + 1), (2 * degx + 1)}}}};

    // jcrosscurlE x,y,z-component (first bracket) dependency bandwidths on Ex,Ey,Ez-fields (second
    // bracket) in y-direction
    static constexpr std::array<std::array<int, 3>, 3> s_jcrosscurlBWy = {
        {{{(2 * degy + 1), (2 * degy), (2 * degy + 1)}},
         {{(2 * degy), (2 * degy - 1), (2 * degy)}},
         {{(2 * degy + 1), (2 * degy), (2 * degy + 1)}}}};

    // jcrosscurlE x,y,z-component (first bracket) dependency bandwidths on Ex,Ey,Ez-fields (second
    // bracket) in z-direction
    static constexpr std::array<std::array<int, 3>, 3> s_jcrosscurlBWz = {
        {{{(2 * degz + 1), (2 * degz + 1), (2 * degz)}},
         {{(2 * degz + 1), (2 * degz + 1), (2 * degz)}},
         {{(2 * degz), (2 * degz), (2 * degz - 1)}}}};

    static constexpr std::array<std::array<int, 3>, 3> s_nJcrosscurlCoeffs = {
        {// Number of x, y and z E-field coefficients required for (J x curl)[xDir]
         {{GEMPIC_D_MULT(s_jcrosscurlBWx[xDir][xDir],
                         s_jcrosscurlBWy[xDir][xDir],
                         s_jcrosscurlBWz[xDir][xDir]),
           GEMPIC_D_MULT(s_jcrosscurlBWx[xDir][yDir],
                         s_jcrosscurlBWy[xDir][yDir],
                         s_jcrosscurlBWz[xDir][yDir]),
           GEMPIC_D_MULT(s_jcrosscurlBWx[xDir][zDir],
                         s_jcrosscurlBWy[xDir][zDir],
                         s_jcrosscurlBWz[xDir][zDir])}},
         // Number of x, y and z E-field coefficients required for (J x curl)[yDir]
         {{GEMPIC_D_MULT(s_jcrosscurlBWx[yDir][xDir],
                         s_jcrosscurlBWy[yDir][xDir],
                         s_jcrosscurlBWz[yDir][xDir]),
           GEMPIC_D_MULT(s_jcrosscurlBWx[yDir][yDir],
                         s_jcrosscurlBWy[yDir][yDir],
                         s_jcrosscurlBWz[yDir][yDir]),
           GEMPIC_D_MULT(s_jcrosscurlBWx[yDir][zDir],
                         s_jcrosscurlBWy[yDir][zDir],
                         s_jcrosscurlBWz[yDir][zDir])}},
         // Number of x, y and z E-field coefficients required for (J x curl)[zDir]
         {{GEMPIC_D_MULT(s_jcrosscurlBWx[zDir][xDir],
                         s_jcrosscurlBWy[zDir][xDir],
                         s_jcrosscurlBWz[zDir][xDir]),
           GEMPIC_D_MULT(s_jcrosscurlBWx[zDir][yDir],
                         s_jcrosscurlBWy[zDir][yDir],
                         s_jcrosscurlBWz[zDir][yDir]),
           GEMPIC_D_MULT(s_jcrosscurlBWx[zDir][zDir],
                         s_jcrosscurlBWy[zDir][zDir],
                         s_jcrosscurlBWz[zDir][zDir])}}}};

    static constexpr std::array<int, 3> s_nJcrosscurlCoeffsTotal{
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

    static constexpr std::array<std::array<int, 3>, 3> s_nJcrosscurlCoeffsArrayStart = {
        {{{0, s_nJcrosscurlCoeffs[xDir][xDir],
           s_nJcrosscurlCoeffs[xDir][xDir] + s_nJcrosscurlCoeffs[xDir][yDir]}},
         {{0, s_nJcrosscurlCoeffs[yDir][xDir],
           s_nJcrosscurlCoeffs[yDir][xDir] + s_nJcrosscurlCoeffs[yDir][yDir]}},
         {{0, s_nJcrosscurlCoeffs[zDir][xDir],
           s_nJcrosscurlCoeffs[zDir][xDir] + s_nJcrosscurlCoeffs[zDir][yDir]}}}};

    // Stencil offsets, bandwidths and number of coefficients for the 'div grad Phi' matrix
    // divgradphi dependency offsets on phi(primal node) in x,y,z-directions (bracket)
    static constexpr std::array<int, 3> s_divgradphiOff{degx, degy, degz};

    // divgradphi dependency bandwidths on phi(primal node) in x,y,z-directions (bracket)
    static constexpr std::array<int, 3> s_divgradphiBw{(2 * degx + 1), (2 * degy + 1),
                                                       (2 * degz + 1)};

    // Number of phi(primal node) coefficients required for divgradphi[]
    static constexpr int s_ndivgradphiCoeffs =
        GEMPIC_D_MULT(s_divgradphiBw[xDir], s_divgradphiBw[yDir], s_divgradphiBw[zDir]);

public:
    QuasineutralSolver(ComputationalDomain infr,
                       std::shared_ptr<DeRhamComplex> drc,
                       amrex::Real tol = 1.e-15,
                       int verbose = 0) :
        m_infra{infr}, m_relTol{tol}
    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    // Generic function to build and solve linear systems with HYPRE
    template <int maxStencilSize, Grid grid, Space space, typename Tmarker, typename Tfiller>
    void create_and_use_hypre_solver (DeRhamField<grid, space>& x,
                                      DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& rhs,
                                      Tmarker&& marker,
                                      Tfiller&& filler)
    {
        amrex::Vector<amrex::IndexType> idxTypeVector;
        amrex::Vector<amrex::MultiFab*> xPointerVector;
        amrex::Vector<amrex::MultiFab const*> rhsPointerVector;
        if constexpr (isOneOrTwoForm<space>)
        {
            idxTypeVector = {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
                             x.m_data[zDir].boxArray().ixType()};
            xPointerVector = {&(x.m_data[xDir]), &(x.m_data[yDir]), &(x.m_data[zDir])};
            rhsPointerVector = {&(rhs.m_data[xDir]), &(rhs.m_data[yDir]), &(rhs.m_data[zDir])};
        }
        else
        {
            idxTypeVector = {x.m_data.boxArray().ixType()};
            xPointerVector = {&(x.m_data)};
            rhsPointerVector = {&(rhs.m_data)};
        }
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            idxTypeVector, rhs.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid,
            m_infra.m_distriMap, std::forward<Tmarker> (marker), std::forward<Tfiller>(filler),
            m_verbose);

        hypreSolver.solve(xPointerVector, rhsPointerVector, m_relTol, m_absTol, m_maxiter);

        x.fill_boundary();
    }

    // Function needed for building all HYPRE matrices
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

    // Function needed for adding diagonal dominance to singular matrices for testing
    void filler_field_charge_matrix (int boxno,
                                     int i,
                                     int j,
                                     int k,
                                     int ndir,
                                     amrex::Array4<HYPRE_Int const> const* gid,
                                     HYPRE_Int& ncols,
                                     HYPRE_Int* cols,
                                     HYPRE_Real* mat,
                                     amrex::MultiArray4<amrex::Real const> const& rhoarrs,
                                     amrex::GpuArray<HYPRE_Real, 3> const edgetoface,
                                     std::map<int, int>& mapCols)
    {
        std::array<int, 3> nc = {0, 0, 0};
        switch (hodgeDegree)
        {
            case 2:
                // This function contributes (<= 1) columns to ncols
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

    // Functions relating to the curlcurl matrix begin
    void solve_curlcurl_plus_field_charge_e (DeRhamField<Grid::dual, Space::face>& b,
                                             DeRhamField<Grid::primal, Space::edge>& x,
                                             DeRhamField<Grid::dual, Space::cell> const& rho)
    {
        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        auto const& rhoarrs = (rho.m_data).arrays();

        auto const dxi = m_infra.inv_cell_size_array();
        auto const dx = m_infra.cell_size_array();

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
        };

        constexpr int maxStencilSize = 6 * AMREX_SPACEDIM - 5;

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::edge>(x, b, marker,
                                                                               filler);
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
                                 amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> const fac,
                                 amrex::GpuArray<HYPRE_Real, 3> const edgetoface,
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
                        mat[icol] += fac[n[dir]] * edgetoface[n[0]];

                        gdtmp = gid[n[dir]]((i - in[dir]), (j - jn[dir]), (k - kn[dir]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += fac[n[dir]] * edgetoface[n[0]];

                        gdtmp = gid[n[dir]](i, j, k);
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += -fac[n[dir]] * edgetoface[n[0]];

                        gdtmp = gid[n[dir]]((i - in[dir] + in[0]), (j - jn[dir] + jn[0]),
                                            (k - kn[dir] + kn[0]));
                        assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                        mat[icol] += -fac[n[dir]] * edgetoface[n[0]];
                    }
                }

                break;
            default:
                AMREX_ASSERT("Hodge degree not implemented");
                break;
        };
    }
    // Functions relating to the curlcurl matrix end

    // Functions relating to the particle charge matrix begin
    void solve_particle_charge_e (DeRhamField<Grid::dual, Space::face>& b,
                                  DeRhamField<Grid::primal, Space::edge>& x,
                                  std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsRho(b.m_deRham, s_nRhoCoeffs);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real const charge = particleSpecies->get_charge();
            amrex::Real const mass = particleSpecies->get_mass();
            amrex::Real const chargeSqrByMass = charge * charge / mass;

            for (auto& particleGrid : *particleSpecies)
            {
                long const np = particleGrid.numParticles();
                auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsARho;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsARho[cc] = (eCoeffsRho.m_data[cc])[particleGrid].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        get_rho_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsARho, spline, chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
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

        constexpr int maxStencilSize = *std::max_element(s_nRhoCoeffs.begin(), s_nRhoCoeffs.end());

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::edge>(x, b, marker,
                                                                               filler);
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_rho_e_weights_at_particle_positions_and_deposit_to_face (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        SplineType const& spline,
        amrex::Real chargeSqrByMassWeight)
    {
        constexpr int nWtsXYZ1f[3][3] = {
            {degx, (degx + 1), (degx + 1)},  // number of 1-form x,y,z-field weights in X-direction
            {(degy + 1), degy, (degy + 1)},  // number of 1-form x,y,z-field weights in Y-direction
            {(degz + 1), (degz + 1), degz}}; // number of 1-form x,y,z-field weights in Z-direction

        constexpr int nWts1f[3] = {
            // number of 1-form x-field weights
            GEMPIC_D_MULT(nWtsXYZ1f[xDir][xDir], nWtsXYZ1f[yDir][xDir], nWtsXYZ1f[zDir][xDir]),
            // number of 1-form y-field weights
            GEMPIC_D_MULT(nWtsXYZ1f[xDir][yDir], nWtsXYZ1f[yDir][yDir], nWtsXYZ1f[zDir][yDir]),
            // number of 1-form z-field weights
            GEMPIC_D_MULT(nWtsXYZ1f[xDir][zDir], nWtsXYZ1f[yDir][zDir], nWtsXYZ1f[zDir][zDir])};
        constexpr int nWts1fStart[3] = {0, nWts1f[xDir], (nWts1f[xDir] + nWts1f[yDir])};

        amrex::GpuArray<amrex::Real, (nWts1f[xDir] + nWts1f[yDir] + nWts1f[zDir])> eWts;
        amrex::GpuArray<amrex::IntVect, (nWts1f[xDir] + nWts1f[yDir] + nWts1f[zDir])> eEdges;

        spline.eval_e_field_spline_weights(eWts, eEdges);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ1f[xDir][ndir]; ++ii),
                                for (int jj = 0; jj < nWtsXYZ1f[yDir][ndir]; ++jj),
                                for (int kk = 0; kk < nWtsXYZ1f[zDir][ndir]; ++kk))
                amrex::Real const sCoeff =
                    ((ndir == 0)
                         ? (spline.template eval_b_spline<Field::DualTwoForm, xDir>(ii, jj, kk))
                         : ((ndir == 1) ? (spline.template eval_b_spline<Field::DualTwoForm, yDir>(
                                              ii, jj, kk))
                                        : (spline.template eval_b_spline<Field::DualTwoForm, zDir>(
                                              ii, jj, kk))));

                // this index is of a primal edge and dual face where deposit is happening
                amrex::Dim3 edge = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                                 spline.m_firstIndex[yDir] + jj,
                                                 spline.m_firstIndex[zDir] + kk)};

                for (int i = 0; i < nWts1f[ndir]; i++)
                {
                    // this index is of primal edge contributing to E-field at particle
                    amrex::Dim3 wEdge = (eEdges[nWts1fStart[ndir] + i]).dim3();

                    AMREX_D_EXPR(iiTemp = wEdge.x - edge.x + s_rhoOffx[ndir],
                                 jjTemp = wEdge.y - edge.y + s_rhoOffy[ndir],
                                 kkTemp = wEdge.z - edge.z + s_rhoOffz[ndir]);

                    component = GEMPIC_D_ADD(iiTemp, jjTemp * s_rhoBWx[ndir],
                                             kkTemp * s_rhoBWx[ndir] * s_rhoBWy[ndir]);

                    amrex::HostDevice::Atomic::Add(
                        eCoeffsA[ndir].ptr(edge.x, edge.y, edge.z, component),
                        sCoeff * chargeSqrByMassWeight * eWts[nWts1fStart[ndir] + i]);
                }
            GEMPIC_D_LOOP_END
        }
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
        std::array<amrex::MultiArray4<amrex::Real>, 3> const& eCoeffsArrs,
        std::map<int, int>& mapCols)
    {
        // This function contributes (<= nMmCoeffs[ndir]) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        GEMPIC_D_LOOP_BEGIN(
            for (ii = -s_rhoOffx[ndir]; ii < (-s_rhoOffx[ndir] + s_rhoBWx[ndir]); ++ii),
            for (jj = -s_rhoOffy[ndir]; jj < (-s_rhoOffy[ndir] + s_rhoBWy[ndir]); ++jj),
            for (kk = -s_rhoOffz[ndir]; kk < (-s_rhoOffz[ndir] + s_rhoBWz[ndir]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_rhoOffx[ndir], jjTemp = jj + s_rhoOffy[ndir],
                         kkTemp = kk + s_rhoOffz[ndir]);

            gdtmp = gid[ndir]((i + ii), (j + jj), (k + kk));

            assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] +=
                eCoeffsArrs[ndir][boxno](i, j, k,
                                         GEMPIC_D_ADD(iiTemp, jjTemp * s_rhoBWx[ndir],
                                                      kkTemp * s_rhoBWx[ndir] * s_rhoBWy[ndir]));
        GEMPIC_D_LOOP_END
    }
    // Functions relating to the particle charge matrix end

    // Functions relating to the J cross curl matrix begin
    void solve_jcrosscurl_plus_field_charge_e (
        DeRhamField<Grid::dual, Space::face>& b,
        DeRhamField<Grid::primal, Space::edge>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        DeRhamField<Grid::dual, Space::cell> const& rho)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsJcrosscurl(b.m_deRham,
                                                               s_nJcrosscurlCoeffsTotal);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (auto& particleGrid : *particleSpecies)
            {
                long const np = particleGrid.numParticles();
                auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                //amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsAJcrosscurl;

                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsAJcrosscurl[cc] = (eCoeffsJcrosscurl.m_data[cc])[particleGrid].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                               ptd.rdata(ii.m_ively)[pp],
                                                               ptd.rdata(ii.m_ivelz)[pp]};

                        get_jcrosscurl_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsAJcrosscurl, spline, vel,
                            chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
                    });
            }
        }

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsJcrosscurl.m_data[ndir].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffsJcrosscurl.m_data[0].arrays()), vdim> eCoeffsArrsJcrosscurl;
        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eCoeffsArrsJcrosscurl[ndir] = (eCoeffsJcrosscurl.m_data[ndir]).arrays();
        }

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        auto const& rhoarrs = (rho.m_data).arrays();

        auto const dxi = m_infra.inv_cell_size_array();
        auto const dx = m_infra.cell_size_array();

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
            filler_jcrosscurl_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                     eCoeffsArrsJcrosscurl, mapCols);
            filler_field_charge_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, rhoarrs,
                                       edgetoface, mapCols);
        };

        constexpr int maxStencilSize =
            *std::max_element(s_nJcrosscurlCoeffsTotal.begin(), s_nJcrosscurlCoeffsTotal.end());

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::edge>(x, b, marker,
                                                                               filler);
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_jcrosscurl_e_weights_at_particle_positions_and_deposit_to_face (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        SplineType const& spline,
        amrex::GpuArray<amrex::Real, vdim> const& vel,
        amrex::Real chargeSqrByMassWeight)
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
        constexpr int nWts2fStart[3] = {0, nWts2f[xDir], (nWts2f[xDir] + nWts2f[yDir])};

        amrex::GpuArray<amrex::Real, (nWts2f[xDir] + nWts2f[yDir] + nWts2f[zDir])> bWts;
        amrex::GpuArray<amrex::IntVect, (nWts2f[xDir] + nWts2f[yDir] + nWts2f[zDir])> bFaces;

        spline.eval_b_field_spline_weights(bWts, bFaces);

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
                amrex::Real const sCoeff =
                    ((ndir == 0)
                         ? (spline.template eval_b_spline<Field::DualTwoForm, xDir>(ii, jj, kk))
                         : ((ndir == 1) ? (spline.template eval_b_spline<Field::DualTwoForm, yDir>(
                                              ii, jj, kk))
                                        : (spline.template eval_b_spline<Field::DualTwoForm, zDir>(
                                              ii, jj, kk))));

                // this index is of primal edge and dual face where deposit is happening
                amrex::Dim3 edge = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                                 spline.m_firstIndex[yDir] + jj,
                                                 spline.m_firstIndex[zDir] + kk)};

                amrex::Dim3 wEdge[4];

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
                    amrex::Dim3 bFace = (bFaces[nWts2fStart[n[1]] + i])
                                            .dim3(); // this index is of primal face contributing to
                                                     // B-field at particle

                    // These indices are of primal edges corresponding to the primal face bFace
                    wEdge[0] = {GEMPIC_D_PAD(bFace.x + in[2], bFace.y + jn[2], bFace.z + kn[2])};
                    ncDir[0] = n[0];
                    sign[0] = 1.0; // -vel[n[2]] * (E[n[0]](i+in2,j+jn2,k+kn2))
                    wEdge[1] = {GEMPIC_D_PAD(bFace.x, bFace.y, bFace.z)};
                    ncDir[1] = n[0];
                    sign[1] = -1.0; // -vel[n[2]] * (-E[n[0]](i,j,k))
                    wEdge[2] = {GEMPIC_D_PAD(bFace.x + in[0], bFace.y + jn[0], bFace.z + kn[0])};
                    ncDir[2] = n[2];
                    sign[2] = -1.0; // -vel[n[2]] * (-E[n[2]](i+in0,j+jn0,k+kn0))
                    wEdge[3] = {GEMPIC_D_PAD(bFace.x, bFace.y, bFace.z)};
                    ncDir[3] = n[2];
                    sign[3] = 1.0; // -vel[n[2]] * (E[n[2]](i,j,k))

                    for (int nc = 0; nc < 4; nc++)
                    {
                        AMREX_D_EXPR(
                            iiTemp = wEdge[nc].x - edge.x + s_jcrosscurlOffx[ndir][ncDir[nc]],
                            jjTemp = wEdge[nc].y - edge.y + s_jcrosscurlOffy[ndir][ncDir[nc]],
                            kkTemp = wEdge[nc].z - edge.z + s_jcrosscurlOffz[ndir][ncDir[nc]]);
                        component = GEMPIC_D_ADD(iiTemp, jjTemp * s_jcrosscurlBWx[ndir][ncDir[nc]],
                                                 kkTemp * s_jcrosscurlBWx[ndir][ncDir[nc]] *
                                                     s_jcrosscurlBWy[ndir][ncDir[nc]]);

                        amrex::HostDevice::Atomic::Add(
                            eCoeffsA[ndir].ptr(
                                edge.x, edge.y, edge.z,
                                s_nJcrosscurlCoeffsArrayStart[ndir][ncDir[nc]] + component),
                            sCoeff * chargeSqrByMassWeight * (-vel[n[2]]) * (sign[nc]) *
                                bWts[nWts2fStart[n[1]] + i]);
                    }
                }

                for (int i = 0; i < nWts2f[n[2]]; i++)
                { // vcrossb[n[0]] = (v[n[1]] * B[n[2]] - ...)
                    amrex::Dim3 bFace = (bFaces[nWts2fStart[n[2]] + i])
                                            .dim3(); // this index is of primal face contributing to
                                                     // B-field at particle

                    // These indices are of primal edges corresponding to the primal face bFace
                    wEdge[0] = {GEMPIC_D_PAD(bFace.x + in[0], bFace.y + jn[0], bFace.z + kn[0])};
                    ncDir[0] = n[1];
                    sign[0] = 1.0; // vel[n[1]] * (E[n[1]](i+in0,j+jn0,k+kn0))
                    wEdge[1] = {GEMPIC_D_PAD(bFace.x, bFace.y, bFace.z)};
                    ncDir[1] = n[1];
                    sign[1] = -1.0; // vel[n[1]] * (-E[n[1]](i,j,k))
                    wEdge[2] = {GEMPIC_D_PAD(bFace.x + in[1], bFace.y + jn[1], bFace.z + kn[1])};
                    ncDir[2] = n[0];
                    sign[2] = -1.0; // vel[n[1]] * (-E[n[0]](i+in1,j+jn1,k+kn1))
                    wEdge[3] = {GEMPIC_D_PAD(bFace.x, bFace.y, bFace.z)};
                    ncDir[3] = n[0];
                    sign[3] = 1.0; // vel[n[1]] * (E[n[0]](i,j,k))

                    for (int nc = 0; nc < 4; nc++)
                    {
                        AMREX_D_EXPR(
                            iiTemp = wEdge[nc].x - edge.x + s_jcrosscurlOffx[ndir][ncDir[nc]],
                            jjTemp = wEdge[nc].y - edge.y + s_jcrosscurlOffy[ndir][ncDir[nc]],
                            kkTemp = wEdge[nc].z - edge.z + s_jcrosscurlOffz[ndir][ncDir[nc]]);
                        component = GEMPIC_D_ADD(iiTemp, jjTemp * s_jcrosscurlBWx[ndir][ncDir[nc]],
                                                 kkTemp * s_jcrosscurlBWx[ndir][ncDir[nc]] *
                                                     s_jcrosscurlBWy[ndir][ncDir[nc]]);

                        amrex::HostDevice::Atomic::Add(
                            eCoeffsA[ndir].ptr(
                                edge.x, edge.y, edge.z,
                                s_nJcrosscurlCoeffsArrayStart[ndir][ncDir[nc]] + component),
                            sCoeff * chargeSqrByMassWeight * (vel[n[1]]) * (sign[nc]) *
                                bWts[nWts2fStart[n[2]] + i]);
                    }
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
        std::array<amrex::MultiArray4<amrex::Real>, 3> const& eCoeffsArrs,
        std::map<int, int>& mapCols)
    {
        // This function contributes (<= s_nJcrosscurlCoeffsTotal[ndir]) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};

        for (int ncDir = 0; ncDir < 3; ncDir++)
        {
            GEMPIC_D_LOOP_BEGIN(
                for (ii = -s_jcrosscurlOffx[ndir][ncDir];
                     ii < (-s_jcrosscurlOffx[ndir][ncDir] + s_jcrosscurlBWx[ndir][ncDir]); ++ii),
                for (jj = -s_jcrosscurlOffy[ndir][ncDir];
                     jj < (-s_jcrosscurlOffy[ndir][ncDir] + s_jcrosscurlBWy[ndir][ncDir]); ++jj),
                for (kk = -s_jcrosscurlOffz[ndir][ncDir];
                     kk < (-s_jcrosscurlOffz[ndir][ncDir] + s_jcrosscurlBWz[ndir][ncDir]); ++kk))
                AMREX_D_EXPR(iiTemp = ii + s_jcrosscurlOffx[ndir][ncDir],
                             jjTemp = jj + s_jcrosscurlOffy[ndir][ncDir],
                             kkTemp = kk + s_jcrosscurlOffz[ndir][ncDir]);

                gdtmp = gid[ncDir]((i + ii), (j + jj), (k + kk));

                assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                mat[icol] += eCoeffsArrs[ndir][boxno](
                    i, j, k,
                    s_nJcrosscurlCoeffsArrayStart[ndir][ncDir] +
                        GEMPIC_D_ADD(
                            iiTemp, jjTemp * s_jcrosscurlBWx[ndir][ncDir],
                            kkTemp * s_jcrosscurlBWx[ndir][ncDir] * s_jcrosscurlBWy[ndir][ncDir]));
            GEMPIC_D_LOOP_END
        }
    }
    // Functions relating to the J cross curl matrix end

    // Functions using combinations of the above matrices begin
    void solve_curlcurl_plus_particle_charge_e (
        DeRhamField<Grid::dual, Space::face>& b,
        DeRhamField<Grid::primal, Space::edge>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr)
    {
        BL_PROFILE("solve_curlcurl_plus_particle_charge_e()");
        DeRhamField<Grid::dual, Space::face> eCoeffsRho(b.m_deRham, s_nRhoCoeffs);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (auto& particleGrid : *particleSpecies)
            {
                long const np = particleGrid.numParticles();
                auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsARho;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsARho[cc] = (eCoeffsRho.m_data[cc])[particleGrid].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        get_rho_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsARho, spline, chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
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

        auto const dxi = m_infra.inv_cell_size_array();
        auto const dx = m_infra.cell_size_array();

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
            6 * AMREX_SPACEDIM - 5 + *std::max_element(s_nRhoCoeffs.begin(), s_nRhoCoeffs.end());

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::edge>(x, b, marker,
                                                                               filler);
    }

    void solve_curlcurl_plus_particle_charge_plus_jcrosscurl_e (
        DeRhamField<Grid::dual, Space::face>& b,
        DeRhamField<Grid::primal, Space::edge>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        amrex::Real jcrosscurlCoeff)
    {
        DeRhamField<Grid::dual, Space::face> eCoeffsRho(b.m_deRham, s_nRhoCoeffs);
        DeRhamField<Grid::dual, Space::face> eCoeffsJcrosscurl(b.m_deRham,
                                                               s_nJcrosscurlCoeffsTotal);
        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (auto& particleGrid : *particleSpecies)
            {
                long const np = particleGrid.numParticles();
                auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsARho;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsAJcrosscurl;
                for (int cc = 0; cc < vdim; cc++)
                {
                    eCoeffsARho[cc] = (eCoeffsRho.m_data[cc])[particleGrid].array();
                    eCoeffsAJcrosscurl[cc] = (eCoeffsJcrosscurl.m_data[cc])[particleGrid].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};
                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());
                        amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                               ptd.rdata(ii.m_ively)[pp],
                                                               ptd.rdata(ii.m_ivelz)[pp]};
                        get_rho_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsARho, spline, chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
                        get_jcrosscurl_e_weights_at_particle_positions_and_deposit_to_face(
                            eCoeffsAJcrosscurl, spline, vel,
                            jcrosscurlCoeff * chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
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

        auto const dxi = m_infra.inv_cell_size_array();
        auto const dx = m_infra.cell_size_array();

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
            6 * AMREX_SPACEDIM - 5 + *std::max_element(s_nRhoCoeffs.begin(), s_nRhoCoeffs.end()) +
            *std::max_element(s_nJcrosscurlCoeffsTotal.begin(), s_nJcrosscurlCoeffsTotal.end());

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::edge>(x, b, marker,
                                                                               filler);
    }
    // Functions using combinations of the above matrices end

    // Function for calculating vector potential A begins
    template <typename DeRhamFieldType>
    void solve_negative_poisson_equation (DeRhamFieldType& rhs, DeRhamFieldType& x)
    {
        BL_PROFILE("solve_negative_poisson_equation()");
        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(m_infra.geometry().InvCellSize(xDir),
                                                             m_infra.geometry().InvCellSize(yDir),
                                                             m_infra.geometry().InvCellSize(zDir))};

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        HYPRE_Real fac0 = HYPRE_Real(-2.) * (GEMPIC_D_ADD(fac[xDir], fac[yDir], fac[zDir]));

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
            //
            if (i == 0 && j == 0 && k == 0)
            {
                ncols = 1;
                cols[0] = gid[ndir](i, j, k);
                mat[0] = -fac0;
            }
            else
            { //*/
                ncols = 2 * AMREX_SPACEDIM + 1;
                cols[0] = gid[ndir](i, j, k);
                mat[0] = -fac0;
                cols[1] = gid[ndir]((i - 1), j, k);
                mat[1] = -fac[xDir];
                cols[2] = gid[ndir]((i + 1), j, k);
                mat[2] = -fac[xDir];
#if (AMREX_SPACEDIM > 1)
                cols[3] = gid[ndir](i, (j - 1), k);
                mat[3] = -fac[yDir];
                cols[4] = gid[ndir](i, (j + 1), k);
                mat[4] = -fac[yDir];
#endif
#if (AMREX_SPACEDIM > 2)
                cols[5] = gid[ndir](i, j, (k - 1));
                mat[5] = -fac[zDir];
                cols[6] = gid[ndir](i, j, (k + 1));
                mat[6] = -fac[zDir];
#endif
            }
        };

        constexpr int maxStencilSize = 2 * AMREX_SPACEDIM + 1;

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            rhs.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, m_verbose);

        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(rhs.m_data[xDir]), &(rhs.m_data[yDir]), &(rhs.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);

        x.fill_boundary();

        if (m_infra.geometry().isAllPeriodic())
        {
            auto nGhost = x.m_deRham->get_n_ghost();
            for (int ndir = 0; ndir < 3; ndir++)
            {
                amrex::Real xSum =
                    x.m_data[ndir].sum_unique(0, false, m_infra.geometry().periodicity());
                amrex::Real ninv = 1.0 / GEMPIC_D_MULT(m_infra.m_nCell[xDir], m_infra.m_nCell[yDir],
                                                       m_infra.m_nCell[zDir]);
                amrex::Real xSumNinv = xSum * ninv;
                x.m_data[ndir].plus(-xSumNinv, nGhost[xDir]);
            }
        }
    }
    // Function for calculating vector potential A ends

    // Function for calculating non-periodic phi begins
    void solve_nonperiodic_poisson_equation (DeRhamField<Grid::dual, Space::cell>& rhs,
                                             DeRhamField<Grid::primal, Space::node>& x)
    {
        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(m_infra.geometry().InvCellSize(xDir),
                                                             m_infra.geometry().InvCellSize(yDir),
                                                             m_infra.geometry().InvCellSize(zDir))};

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        HYPRE_Real fac0 = HYPRE_Real(-2.) * (GEMPIC_D_ADD(fac[xDir], fac[yDir], fac[zDir]));

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
        auto filler = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int n,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            ncols = 0;

            if (AMREX_D_TERM((((i == nddom.smallEnd(0)) || (i == nddom.bigEnd(0))) &&
                              !(m_infra.geometry().periodicity().isPeriodic(0))),
                             || (((j == nddom.smallEnd(1)) || (j == nddom.bigEnd(1))) &&
                                 !(m_infra.geometry().periodicity().isPeriodic(1))),
                             || (((k == nddom.smallEnd(2)) || (k == nddom.bigEnd(2))) &&
                                 !(m_infra.geometry().periodicity().isPeriodic(2)))))
            {
                cols[ncols] = gid[n](i, j, k);
                mat[ncols] = 1.0;
                ++ncols;
            }
            else
            {
                cols[ncols] = gid[n](i, j, k);
                mat[ncols] = -fac0;
                ++ncols;

                cols[ncols] = gid[n](i - 1, j, k);
                mat[ncols] = -fac[0];
                ++ncols;
                cols[ncols] = gid[n](i + 1, j, k);
                mat[ncols] = -fac[0];
                ++ncols;
                cols[ncols] = gid[n](i, j - 1, k);
                mat[ncols] = -fac[1];
                ++ncols;
                cols[ncols] = gid[n](i, j + 1, k);
                mat[ncols] = -fac[1];
                ++ncols;
#if (AMREX_SPACEDIM > 2)
                cols[ncols] = gid[n](i, j, k - 1);
                mat[ncols] = -fac[2];
                ++ncols;
                cols[ncols] = gid[n](i, j, k + 1);
                mat[ncols] = -fac[2];
                ++ncols;
#endif
            }
        };

        constexpr int maxStencilSize = 2 * AMREX_SPACEDIM + 1;

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::node>(x, rhs, marker,
                                                                               filler);
    }
    // Function for calculating non-periodic phi ends

    // Functions for correcting divergence of J begin
    void push_particles_and_correct_div_j (
        DeRhamField<Grid::dual, Space::face>& J,
        DeRhamField<Grid::primal, Space::node>& phi,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        amrex::Real dt)
    {
        BL_PROFILE("push_particles_and_correct_div_j()");
        DeRhamField<Grid::primal, Space::edge> eCorr(phi.m_deRham);
        DeRhamField<Grid::dual, Space::cell> divJ(phi.m_deRham);

        // Calculate original J and divJ
        Gempic::ParticleMeshCoupling::deposit_current_density<degx, degy, degz, vdim, ndata>(
            J, partGr, m_infra);
        phi.m_deRham->div(divJ, J);

        // Calculate phi
        solve_particle_div_grad_phi(divJ, phi, partGr, dt);

        phi.m_deRham->grad(eCorr, phi);
        eCorr *= -1.0;

        // push particles using E and calculate final J
        update_velocities_using_e_and_calculate_deposit_j(J, eCorr, partGr, dt);
    }

    void solve_particle_div_grad_phi (
        DeRhamField<Grid::dual, Space::cell>& b,
        DeRhamField<Grid::primal, Space::node>& x,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        amrex::Real dt)
    {
        DeRhamField<Grid::dual, Space::cell> phiCoeffsDivGrad(b.m_deRham, s_ndivgradphiCoeffs);

        // Fill up phi_coeffs
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeSqrByMass = charge * charge / mass;

            for (auto& pti : *particleSpecies)
            {
                long const np = pti.numParticles();
                auto const ptd = pti.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::Array4<amrex::Real> phiCoeffsAdivgradphi;
                phiCoeffsAdivgradphi = (phiCoeffsDivGrad.m_data)[pti].array();
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        get_divgrad_phi_weights_at_particle_positions_and_deposit_to_cell(
                            phiCoeffsAdivgradphi, spline,
                            -dt * chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp]);
                    });
            }
        }
        phiCoeffsDivGrad.m_data.SumBoundary(m_infra.geometry().periodicity());

        auto const& phiCoeffsArrsDivGrad = (phiCoeffsDivGrad.m_data).arrays();

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
        auto filler = [=] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int n,
                                           amrex::Array4<HYPRE_Int const> const* gid,
                                           HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0; // initialize to 0 and then increment it through filler functions
            filler_divgradphi_matrix(boxno, i, j, k, n, gid, ncols, cols, mat, phiCoeffsArrsDivGrad,
                                     mapCols);
        };

        constexpr int maxStencilSize = s_ndivgradphiCoeffs;

        create_and_use_hypre_solver<maxStencilSize, Grid::primal, Space::node>(x, b, marker,
                                                                               filler);

        if (m_infra.geometry().isAllPeriodic())
        {
            auto nGhost = x.m_deRham->get_n_ghost();
            amrex::Real xSum = x.m_data.sum_unique(0, false, m_infra.geometry().periodicity());
            amrex::Real ninv = 1.0 / GEMPIC_D_MULT(m_infra.m_nCell[xDir], m_infra.m_nCell[yDir],
                                                   m_infra.m_nCell[zDir]);
            amrex::Real xSumNinv = xSum * ninv;
            x.m_data.plus(-xSumNinv, nGhost[xDir]);
        }
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_divgrad_phi_weights_at_particle_positions_and_deposit_to_cell (
        amrex::Array4<amrex::Real> const& phiCoeffsA,
        SplineType const& spline,
        amrex::Real minusdtChargeSqrByMassWeight)
    {
        constexpr int nWtsXYZ1f[3][3] = {
            {degx, (degx + 1), (degx + 1)},  // number of 1-form x,y,z-field weights in X-direction
            {(degy + 1), degy, (degy + 1)},  // number of 1-form x,y,z-field weights in Y-direction
            {(degz + 1), (degz + 1), degz}}; // number of 1-form x,y,z-field weights in Z-direction

        constexpr int nWts1f[3] = {
            // number of 1-form x-field weights
            GEMPIC_D_MULT(nWtsXYZ1f[xDir][xDir], nWtsXYZ1f[yDir][xDir], nWtsXYZ1f[zDir][xDir]),
            // number of 1-form y-field weights
            GEMPIC_D_MULT(nWtsXYZ1f[xDir][yDir], nWtsXYZ1f[yDir][yDir], nWtsXYZ1f[zDir][yDir]),
            // number of 1-form z-field weights
            GEMPIC_D_MULT(nWtsXYZ1f[xDir][zDir], nWtsXYZ1f[yDir][zDir], nWtsXYZ1f[zDir][zDir])};
        constexpr int nWts1fStart[3] = {0, nWts1f[xDir], (nWts1f[xDir] + nWts1f[yDir])};

        amrex::GpuArray<amrex::Real, (nWts1f[xDir] + nWts1f[yDir] + nWts1f[zDir])> eWts;
        amrex::GpuArray<amrex::IntVect, (nWts1f[xDir] + nWts1f[yDir] + nWts1f[zDir])> eEdges;

        spline.eval_e_field_spline_weights(eWts, eEdges);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            int in = (ndir == xDir);
            int jn = (ndir == yDir);
            int kn = (ndir == zDir);

#if AMREX_SPACEDIM == 2
            kn = 0;
#endif

            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ1f[xDir][ndir]; ++ii),
                                for (int jj = 0; jj < nWtsXYZ1f[yDir][ndir]; ++jj),
                                for (int kk = 0; kk < nWtsXYZ1f[zDir][ndir]; ++kk))

                amrex::Real const sCoeff =
                    ((ndir == 0)
                         ? (spline.template eval_b_spline<Field::DualTwoForm, xDir>(ii, jj, kk))
                         : ((ndir == 1) ? (spline.template eval_b_spline<Field::DualTwoForm, yDir>(
                                              ii, jj, kk))
                                        : (spline.template eval_b_spline<Field::DualTwoForm, zDir>(
                                              ii, jj, kk))));

                // this index is of primal edge and dual face where deposit is happening
                // this face contributes to the divergences in the two cells sharing it
                amrex::Dim3 dualFace = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                                     spline.m_firstIndex[yDir] + jj,
                                                     spline.m_firstIndex[zDir] + kk)};

                // these indices are for the dual cell corresponding to edge
                amrex::Dim3 wDualCell[2];
                wDualCell[0] = {GEMPIC_D_PAD(dualFace.x, dualFace.y, dualFace.z)};
                wDualCell[1] = {GEMPIC_D_PAD(dualFace.x + in, dualFace.y + jn, dualFace.z + kn)};

                for (int i = 0; i < nWts1f[ndir]; i++)
                {
                    // this index is of primal edge contributing to E-field at particle
                    amrex::Dim3 wEdge = (eEdges[nWts1fStart[ndir] + i]).dim3();

                    // these indices are for the primal nodes corresponding to wEdge
                    amrex::Dim3 wNode[2];
                    wNode[0] = {GEMPIC_D_PAD(wEdge.x, wEdge.y, wEdge.z)};
                    wNode[1] = {GEMPIC_D_PAD(wEdge.x + in, wEdge.y + jn, wEdge.z + kn)};

                    for (int nc = 0; nc < 2; nc++)
                    {
                        for (int nn = 0; nn < 2; nn++)
                        {
                            AMREX_D_EXPR(
                                iiTemp = wNode[nn].x - wDualCell[nc].x + s_divgradphiOff[xDir],
                                jjTemp = wNode[nn].y - wDualCell[nc].y + s_divgradphiOff[yDir],
                                kkTemp = wNode[nn].z - wDualCell[nc].z + s_divgradphiOff[zDir]);
                            component =
                                GEMPIC_D_ADD(iiTemp, jjTemp * s_divgradphiBw[xDir],
                                             kkTemp * s_divgradphiBw[xDir] * s_divgradphiBw[yDir]);

                            amrex::HostDevice::Atomic::Add(
                                phiCoeffsA.ptr(wDualCell[nc].x, wDualCell[nc].y, wDualCell[nc].z,
                                               component),
                                sCoeff * minusdtChargeSqrByMassWeight * ((nn == nc) ? 1.0 : -1.0) *
                                    eWts[nWts1fStart[ndir] + i]);
                        }
                    }
                }
            GEMPIC_D_LOOP_END
        }
    }

    void filler_divgradphi_matrix (int boxno,
                                   int i,
                                   int j,
                                   int k,
                                   int n,
                                   amrex::Array4<HYPRE_Int const> const* gid,
                                   HYPRE_Int& ncols,
                                   HYPRE_Int* cols,
                                   HYPRE_Real* mat,
                                   amrex::MultiArray4<amrex::Real> const& phiCoeffsArrs,
                                   std::map<int, int>& mapCols)
    {
        // This function contributes (<= s_ndivgradphi_Coeffs) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        GEMPIC_D_LOOP_BEGIN(for (ii = -s_divgradphiOff[xDir];
                                 ii < (-s_divgradphiOff[xDir] + s_divgradphiBw[xDir]); ++ii),
                            for (jj = -s_divgradphiOff[yDir];
                                 jj < (-s_divgradphiOff[yDir] + s_divgradphiBw[yDir]); ++jj),
                            for (kk = -s_divgradphiOff[zDir];
                                 kk < (-s_divgradphiOff[zDir] + s_divgradphiBw[zDir]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_divgradphiOff[xDir], jjTemp = jj + s_divgradphiOff[yDir],
                         kkTemp = kk + s_divgradphiOff[zDir]);

            gdtmp = gid[n]((i + ii), (j + jj), (k + kk));

            assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += phiCoeffsArrs[boxno](
                i, j, k,
                GEMPIC_D_ADD(iiTemp, jjTemp * s_divgradphiBw[xDir],
                             kkTemp * s_divgradphiBw[xDir] * s_divgradphiBw[yDir]));
        GEMPIC_D_LOOP_END
    }

    void update_velocities_using_e_and_calculate_deposit_j (
        DeRhamField<Grid::dual, Space::face>& J,
        DeRhamField<Grid::primal, Space::edge>& E,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        amrex::Real dt)
    {
        for (int comp = 0; comp < vdim; ++comp)
        {
            J.m_data[comp].setVal(0.0);
        }
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeByMass = charge / mass;

            for (auto& pti : *particleSpecies)
            {
                long const np = pti.numParticles();
                auto const ptd = pti.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

                for (int cc = 0; cc < vdim; cc++)
                {
                    jA[cc] = (J.m_data[cc])[pti].array();
                    eA[cc] = (E.m_data[cc])[pti].array();
                }

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                            AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                         ptd.rdata(ii.m_iposz)[pp])};
                        amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                               ptd.rdata(ii.m_ively)[pp],
                                                               ptd.rdata(ii.m_ivelz)[pp]};

                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, vdim> efield =
                            spline.template eval_spline_field<Field::PrimalOneForm>(eA);

                        Gempic::ParticleMeshCoupling::push_v_using_e<vdim>(vel, efield, dt,
                                                                           chargeByMass);

                        // Update global particle velocities arrays
                        ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                        ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                        ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];

                        deposit_twoform(jA, spline, vel, charge * ptd.rdata(ii.m_iweight)[pp]);
                    });
            }
        }
        J.post_particle_loop_sync();
    }
    // Functions for correcting divergence of J end

    // Functions for correcting average of J begin
    void push_particles_and_correct_javg (
        DeRhamField<Grid::dual, Space::face>& J,
        DeRhamField<Grid::dual, Space::cell>& rho,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr)
    {
        BL_PROFILE("push_particles_and_correct_javg()");
        std::vector<DeRhamField<Grid::dual, Space::cell>> rhoVec;
        std::vector<DeRhamField<Grid::dual, Space::cell>> rhoMassVec;

        for (int i = 0; i < partGr.size(); i++)
        {
            rhoVec.emplace_back(rho.m_deRham);
            rhoMassVec.emplace_back(rho.m_deRham);
        }

        Gempic::ParticleMeshCoupling::deposit_individual_particle_density<degx, degy, degz, vdim,
                                                                          ndata>(rhoVec, rhoMassVec,
                                                                                 partGr, m_infra);

        std::vector<amrex::GpuArray<amrex::Real, vdim>> vAvg;
        vAvg.resize(partGr.size());

        amrex::GpuArray<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(m_infra.geometry().CellSize(xDir),
                                                            m_infra.geometry().CellSize(yDir),
                                                            m_infra.geometry().CellSize(zDir))};

        if (m_infra.geometry().isAllPeriodic())
        {
            std::vector<amrex::Real> rhoSum, rhoMassSum;
            rhoSum.resize(partGr.size());
            rhoMassSum.resize(partGr.size());

            amrex::Real ninv = 1.0 / GEMPIC_D_MULT(m_infra.m_nCell[xDir], m_infra.m_nCell[yDir],
                                                   m_infra.m_nCell[zDir]);

            for (int i = 0; i < partGr.size(); i++)
            {
                rhoSum[i] = rhoVec[i].m_data.sum_unique(0, false, m_infra.geometry().periodicity());
                rhoMassSum[i] =
                    rhoMassVec[i].m_data.sum_unique(0, false, m_infra.geometry().periodicity());
            }

            auto nGhost = J.m_deRham->get_n_ghost();
            for (int ndir = 0; ndir < 3; ndir++)
            {
                amrex::Real jSum =
                    J.m_data[ndir].sum_unique(0, false, m_infra.geometry().periodicity());
                amrex::Real jSumNinv = jSum * ninv;
                J.m_data[ndir].plus(-jSumNinv, nGhost[xDir]);

                if (partGr.size() == 1)
                {
                    vAvg[0][ndir] = jSum * dx[ndir] / rhoSum[0];
                }
                else if (partGr.size() == 2)
                {
                    vAvg[0][ndir] = jSum * dx[ndir] * rhoMassSum[1] /
                                    (rhoSum[0] * rhoMassSum[1] - rhoSum[1] * rhoMassSum[0]);
                    vAvg[1][ndir] = jSum * dx[ndir] * rhoMassSum[0] /
                                    (rhoSum[1] * rhoMassSum[0] - rhoSum[0] * rhoMassSum[1]);
                }
                else
                {
                    GEMPIC_ERROR("Unknown particle group size: " + std::to_string(partGr.size()));
                }
            }
        }
        else
        {
            GEMPIC_ERROR(
                "Instructions for push_particles_and_correct_javg() in non-periodic case not "
                "implemented yet.");
        }

        push_back_by_v_avg(vAvg, partGr);
    }

    void push_back_by_v_avg (std::vector<amrex::GpuArray<amrex::Real, vdim>> const& vAvg,
                             std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr)
    {
        int index = 0;
        for (auto& particleSpecies : partGr)
        {
            for (auto& pti : *particleSpecies)
            {
                long const np = pti.numParticles();
                auto const ptd = pti.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(np,
                                   [=] AMREX_GPU_DEVICE(long pp)
                                   {
                                       amrex::GpuArray<amrex::Real, vdim> vel{
                                           ptd.rdata(ii.m_ivelx)[pp], ptd.rdata(ii.m_ively)[pp],
                                           ptd.rdata(ii.m_ivelz)[pp]};

                                       for (int i = 0; i < vdim; i++)
                                       {
                                           vel[i] -= vAvg[index][i];
                                       }

                                       // Update global particle velocities arrays
                                       ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                                       ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                                       ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];
                                   });
            }
            index++;
        }
    }
    // Functions for correcting average of J end

    // Function for updating X using V and V using E,B begin
    void update_particle_positions_and_velocities (
        DeRhamField<Grid::primal, Space::edge>& E,
        DeRhamField<Grid::primal, Space::face>& B,
        std::vector<std::shared_ptr<ParticleGroups<vdim, ndata>>>& partGr,
        amrex::Real dt,
        RotatePushOrder order)
    {
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();
            amrex::Real chargeByMass = charge / mass;

            for (auto& pti : *particleSpecies)
            {
                long const np = pti.numParticles();
                auto const ptd = pti.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;

                for (int cc = 0; cc < vdim; cc++)
                {
                    eA[cc] = (E.m_data[cc])[pti].array();
                    bA[cc] = (B.m_data[cc])[pti].array();
                }

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{
                            AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                         ptd.rdata(ii.m_iposz)[pp])};
                        amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                               ptd.rdata(ii.m_ively)[pp],
                                                               ptd.rdata(ii.m_ivelz)[pp]};

                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            pos, plo, m_infra.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, vdim> efield =
                            spline.template eval_spline_field<Field::PrimalOneForm>(eA);
                        amrex::GpuArray<amrex::Real, vdim> bfield =
                            spline.template eval_spline_field<Field::PrimalTwoForm>(bA);

                        if (order == RotatePushOrder::PushvThenRotatev)
                        {
                            Gempic::ParticleMeshCoupling::push_v_using_e<vdim>(
                                vel, efield, 0.5 * dt, chargeByMass);
                            Gempic::ParticleMeshCoupling::rotate_v_using_b<vdim>(
                                vel, bfield, 0.5 * dt, chargeByMass);

                            Gempic::ParticleMeshCoupling::push_x_using_v<vdim>(pos, vel, dt);

                            AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] = pos[xDir],
                                         ptd.rdata(ii.m_iposy)[pp] = pos[yDir],
                                         ptd.rdata(ii.m_iposz)[pp] = pos[zDir]);
                        }
                        else if (order == RotatePushOrder::RotatevThenPushv)
                        {
                            Gempic::ParticleMeshCoupling::rotate_v_using_b<vdim>(
                                vel, bfield, 0.5 * dt, chargeByMass);
                            Gempic::ParticleMeshCoupling::push_v_using_e<vdim>(
                                vel, efield, 0.5 * dt, chargeByMass);
                        }

                        // Update global particle velocities arrays
                        ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                        ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                        ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];
                    });
            }

            if (order == RotatePushOrder::PushvThenRotatev)
            {
                particleSpecies->Redistribute();
            }
        }
    }
    // Function for updating X using V and V using E,B end
};

} // namespace Gempic::FieldSolvers

#endif

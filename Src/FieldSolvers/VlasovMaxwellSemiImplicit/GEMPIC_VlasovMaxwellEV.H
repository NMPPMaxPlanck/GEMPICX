/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_VLASOVMAXWELLEV_H_
#define GEMPIC_VLASOVMAXWELLEV_H_

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
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{
template <int hodgeDegree, int degx, int degy, int degz, unsigned int vdim>
class VlasovMaxwellEV
{
private:
    ComputationalDomain m_infra;
    amrex::Real m_relTol, m_absTol{1.e-15};
    amrex::Real m_maxiter{1000};
    int m_verbose;

    // Stencil offsets and bandwidths for the 'rho' matrix
    static constexpr std::array<int, 3> s_sGdecOffx{
        (hodgeDegree - 2), (hodgeDegree - 1),
        (hodgeDegree - 1)}; // x,y,z-field offsets in x-direction
    static constexpr std::array<int, 3> s_sGdecOffy{
        (hodgeDegree - 1), (hodgeDegree - 2),
        (hodgeDegree - 1)}; // x,y,z-field offsets in y-direction
    static constexpr std::array<int, 3> s_sGdecOffz{
        (hodgeDegree - 1), (hodgeDegree - 1),
        (hodgeDegree - 2)}; // x,y,z-field offsets in z-direction

    static constexpr std::array<int, 3> s_sGdecBWx{
        (2 * s_sGdecOffx[xDir] + 1), (2 * s_sGdecOffx[yDir] + 1),
        (2 * s_sGdecOffx[zDir] + 1)}; // x,y,z-field bandwidths in x-direction
    static constexpr std::array<int, 3> s_sGdecBWy{
        (2 * s_sGdecOffy[xDir] + 1), (2 * s_sGdecOffy[yDir] + 1),
        (2 * s_sGdecOffy[zDir] + 1)}; // x,y,z-field bandwidths in y-direction
    static constexpr std::array<int, 3> s_sGdecBWz{
        (2 * s_sGdecOffz[xDir] + 1), (2 * s_sGdecOffz[yDir] + 1),
        (2 * s_sGdecOffz[zDir] + 1)}; // x,y,z-field bandwidths in z-direction

    static constexpr std::array<int, 3> s_sNGdecCoeffs{
        GEMPIC_D_MULT(s_sGdecBWx[xDir],
                      s_sGdecBWy[xDir],
                      s_sGdecBWz[xDir]), // total number of x-field coefficients
        GEMPIC_D_MULT(s_sGdecBWx[yDir],
                      s_sGdecBWy[yDir],
                      s_sGdecBWz[yDir]), // total number of y-field coefficients
        GEMPIC_D_MULT(s_sGdecBWx[zDir],
                      s_sGdecBWy[zDir],
                      s_sGdecBWz[zDir])}; // total number of z-field coefficients

    // Stencil offsets and bandwidths for the Hodge matrix
    static constexpr std::array<int, 3> s_sHodgeOffx{
        (hodgeDegree - 2), (hodgeDegree - 1),
        (hodgeDegree - 1)}; // x,y,z-field offsets in x-direction
    static constexpr std::array<int, 3> s_sHodgeOffy{
        (hodgeDegree - 1), (hodgeDegree - 2),
        (hodgeDegree - 1)}; // x,y,z-field offsets in y-direction
    static constexpr std::array<int, 3> s_sHodgeOffz{
        (hodgeDegree - 1), (hodgeDegree - 1),
        (hodgeDegree - 2)}; // x,y,z-field offsets in z-direction

    static constexpr std::array<int, 3> s_sHodgeBWx{
        (2 * s_sHodgeOffx[xDir] + 1), (2 * s_sHodgeOffx[yDir] + 1),
        (2 * s_sHodgeOffx[zDir] + 1)}; // x,y,z-field bandwidths in x-direction
    static constexpr std::array<int, 3> s_sHodgeBWy{
        (2 * s_sHodgeOffy[xDir] + 1), (2 * s_sHodgeOffy[yDir] + 1),
        (2 * s_sHodgeOffy[zDir] + 1)}; // x,y,z-field bandwidths in y-direction
    static constexpr std::array<int, 3> s_sHodgeBWz{
        (2 * s_sHodgeOffz[xDir] + 1), (2 * s_sHodgeOffz[yDir] + 1),
        (2 * s_sHodgeOffz[zDir] + 1)}; // x,y,z-field bandwidths in z-direction

    static constexpr std::array<int, 3> s_sNhodgeCoeffs{
        GEMPIC_D_MULT(s_sHodgeBWx[xDir],
                      s_sHodgeBWy[xDir],
                      s_sHodgeBWz[xDir]), // total number of x-field coefficients
        GEMPIC_D_MULT(s_sHodgeBWx[yDir],
                      s_sHodgeBWy[yDir],
                      s_sHodgeBWz[yDir]), // total number of y-field coefficients
        GEMPIC_D_MULT(s_sHodgeBWx[zDir],
                      s_sHodgeBWy[zDir],
                      s_sHodgeBWz[zDir])}; // total number of z-field coefficients

    // V0 : Hodge degree 2,4,6-weights (first bracket) for stencil cells (second bracket)
    static constexpr std::array<std::array<amrex::Real, 6>, 3> s_hodgeV0 = {
        {{{2.0 / 3.0, 1.0 / 6.0, 0, 0, 0, 0}},
         {{733.0 / 945.0, 257.0 / 1680.0, -3.0 / 70.0, 31.0 / 15120.0, 0, 0}},
         {{455963.0 / 554400.0, 906919.0 / 6652800.0, -12421.0 / 207900.0, 59053.0 / 4435200.0,
           -3937.0 / 3326400.0, 313.0 / 4435200.0}}}};

    // V1 : Hodge degree 2,4,6-weights (first bracket) for stencil cells (second bracket)
    static constexpr std::array<std::array<amrex::Real, 5>, 3> s_hodgeV1 = {
        {{{1.0, 0, 0, 0, 0}},
         {{21.0 / 20.0, -1.0 / 180.0, -7.0 / 360.0, 0, 0}},
         {{99307.0 / 90720.0, -9691.0 / 453600.0, -4003.0 / 113400.0, 4547.0 / 453600.0,
           -89.0 / 129600.0}}}};

public:
    VlasovMaxwellEV(ComputationalDomain infr,
                    std::shared_ptr<Forms::DeRhamComplex> drc,
                    amrex::Real tol = 1.e-15,
                    int verbose = 0) :
        m_infra{infr}, m_relTol{tol}
    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
    }

    /**
     * @brief The hypre solver for the operator 4(b) on page 9 in
     * [Energy conserving PIC, Kormann & Sonnendruecker, 2021, JCP].
     * @param[out] x Array of the electric field.
     * @param b right hand side of the linear system in operator 4(b).
     * @param eGDECCoeffsRho non-zero components of the particle mass matrix
     * @param infra ComputationalDomain.
     */
    void solve_hodge_plus_particle_matrix_e_gdec (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& b,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& eGDECCoeffsRho,
        Gempic::ComputationalDomain const& infra)
    {
        BL_PROFILE(
            "Gempic::FieldSolvers::VlasovMaxwellEV::solve_hodge_plus_particle_matrix_e_gdec()");

        auto dx{infra.cell_size_3darray()};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> scalingDF{
            AMREX_D_DECL(dx[yDir] * dx[zDir] / dx[xDir], dx[xDir] * dx[zDir] / dx[yDir],
                         dx[xDir] * dx[yDir] / dx[zDir])};

        std::array<amrex::MultiArray4<amrex::Real>, vdim> eGDECCoeffsArrsRho;
        for (int ndir = 0; ndir < vdim; ++ndir)
        {
            eGDECCoeffsArrsRho[ndir] = (eGDECCoeffsRho.m_data[ndir]).arrays();
        }

        auto const& nddom = amrex::surroundingNodes(infra.box());

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
        auto filler = [=, this] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int ndir,
                                                 amrex::Array4<HYPRE_Int const> const* gid,
                                                 HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0; // initialize to 0 and then increment it through filler functions
            // Filler of the particle_mass_matrix based on GDEC of S4 in operator 4(b) on page 9.
            filler_gdec_particle_mass_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                             eGDECCoeffsArrsRho, mapCols);
            // Filler of the mass_matrix based on GDEC in of S4 in operator 4(b) on page 9.
            filler_gdec_mass_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat, mapCols,
                                    scalingDF);
        };

        constexpr int maxStencilSize =
            *std::max_element(s_sNhodgeCoeffs.begin(), s_sNhodgeCoeffs.end());

        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[xDir].boxArray().ixType(), x.m_data[yDir].boxArray().ixType(),
             x.m_data[zDir].boxArray().ixType()},
            b.m_deRham->get_n_ghost(), infra.geometry(), infra.m_grid, infra.m_distriMap, marker,
            filler, m_verbose);
        // Use Hypre solver to solve the linear system,
        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[xDir]), &(x.m_data[yDir]),
                                                          &(x.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(b.m_data[xDir]), &(b.m_data[yDir]), &(b.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);
        x.fill_boundary();
    }

    void filler_gdec_particle_mass_matrix (
        int boxno,
        int i,
        int j,
        int k,
        int ndir,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        std::array<amrex::MultiArray4<amrex::Real>, 3> const& eGDECCoeffsArrs,
        std::map<int, int>& mapCols)
    {
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        GEMPIC_D_LOOP_BEGIN(
            for (ii = -s_sGdecOffx[ndir]; ii < (-s_sGdecOffx[ndir] + s_sGdecBWx[ndir]); ++ii),
            for (jj = -s_sGdecOffy[ndir]; jj < (-s_sGdecOffy[ndir] + s_sGdecBWy[ndir]); ++jj),
            for (kk = -s_sGdecOffz[ndir]; kk < (-s_sGdecOffz[ndir] + s_sGdecBWz[ndir]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_sGdecOffx[ndir], jjTemp = jj + s_sGdecOffy[ndir],
                         kkTemp = kk + s_sGdecOffz[ndir]);

            gdtmp = gid[ndir]((i + ii), (j + jj), (k + kk));

            assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += eGDECCoeffsArrs[ndir][boxno](
                i, j, k,
                GEMPIC_D_ADD(iiTemp, jjTemp * s_sGdecBWx[ndir],
                             kkTemp * s_sGdecBWx[ndir] * s_sGdecBWy[ndir]));
        GEMPIC_D_LOOP_END
    }

    void filler_gdec_mass_matrix (int boxno,
                                  int i,
                                  int j,
                                  int k,
                                  int ndir,
                                  amrex::Array4<HYPRE_Int const> const* gid,
                                  HYPRE_Int& ncols,
                                  HYPRE_Int* cols,
                                  HYPRE_Real* mat,
                                  std::map<int, int>& mapCols,
                                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> scalingDF)
    {
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};
        GEMPIC_D_LOOP_BEGIN(
            for (ii = -s_sHodgeOffx[ndir]; ii < (-s_sHodgeOffx[ndir] + s_sHodgeBWx[ndir]); ++ii),
            for (jj = -s_sHodgeOffy[ndir]; jj < (-s_sHodgeOffy[ndir] + s_sHodgeBWy[ndir]); ++jj),
            for (kk = -s_sHodgeOffz[ndir]; kk < (-s_sHodgeOffz[ndir] + s_sHodgeBWz[ndir]); ++kk))

            gdtmp = gid[ndir]((i + ii), (j + jj), (k + kk));

            assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] +=
                scalingDF[ndir] *
                ((ndir == 0)
                     ? (GEMPIC_D_MULT(s_hodgeV1[hodgeDegree / 2 - 1][std::abs(ii)],
                                      s_hodgeV0[hodgeDegree / 2 - 1][std::abs(jj)],
                                      s_hodgeV0[hodgeDegree / 2 - 1][std::abs(kk)]))
                     : ((ndir == 1)
                            ? (GEMPIC_D_MULT(s_hodgeV0[hodgeDegree / 2 - 1][std::abs(ii)],
                                             s_hodgeV1[hodgeDegree / 2 - 1][std::abs(jj)],
                                             s_hodgeV0[hodgeDegree / 2 - 1][std::abs(kk)]))
                            : (GEMPIC_D_MULT(s_hodgeV0[hodgeDegree / 2 - 1][std::abs(ii)],
                                             s_hodgeV0[hodgeDegree / 2 - 1][std::abs(jj)],
                                             s_hodgeV1[hodgeDegree / 2 - 1][std::abs(kk)]))));

        GEMPIC_D_LOOP_END
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
};

} // namespace Gempic::FieldSolvers

#endif

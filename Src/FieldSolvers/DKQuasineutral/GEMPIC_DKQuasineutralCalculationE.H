/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_DKQUASINEUTRALECALCULATOR_H_
#define GEMPIC_DKQUASINEUTRALECALCULATOR_H_

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

#include <_hypre_IJ_mv.h>
#include <_hypre_sstruct_mv.h>
#include <map>

#include <Kokkos_Profiling_ProfileSection.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <AMReX.H>
#include <AMReX_HypreSolver.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_DKQuasineutralDebug.H"
#include "GEMPIC_DKQuasineutralImpl.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_HypreUtils.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_ParticleUtils.H"
#include "GEMPIC_RungeKutta.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::FieldSolvers
{

template <int hodgeDegree, unsigned int vdim, unsigned int ndata, int degx, int degy, int degz>
class DKQuasineutralCalculationE
{
private:
    ComputationalDomain m_infra;
    std::shared_ptr<Forms::FDDeRhamComplex> m_deRham;
    amrex::Real m_relTol, m_absTol{1.e-14}, m_accTol;
    amrex::Real m_maxiter{1000};
    int m_verbose;

    std::string m_chargeMatrixTypeEpar{"particle"};
    std::string m_chargeMatrixTypeEperp{"particle"};

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

    // Stencil offsets and bandwidths for the 'ExB' matrix
    //                 Offx   Offy   Offz      BWx     BWy     BWz
    // ndir = 0
    // Ez -> Y-face    degX   degY-1 degZ      2degX+1 2degY   2degZ
    // Ey -> Z-face    degX   degY   degZ-1    2degX+1 2degY   2degZ
    // ndir = 1
    // Ex -> Z-face    degX   degY   degZ-1    2degX   2degY+1 2degZ
    // Ez -> X-face    degX-1 degY   degZ      2degX   2degY+1 2degZ
    // ndir = 2
    // Ey -> X-face    degX-1 degY   degZ      2degX   2degY   2degZ+1
    // Ex -> Y-face    degX   degY-1 degZ      2degX   2degY   2degZ+1

    static constexpr std::array<std::array<int, 3>, 3> s_exBOff{{
        {(degx - 1), degy, degz}, // y/z-field offsets in x,y,z-directions when depositing on x-face
        {degx, (degy - 1), degz}, // z/x-field offsets in x,y,z-directions when depositing on y-face
        {degx, degy, (degz - 1)}  // x/y-field offsets in x,y,z-directions when depositing on z-face
    }};

    static constexpr std::array<std::array<int, 3>, 3> s_exBBw{{
        // s_ExB_BW[xDir]
        {(2 * degx + 1), (2 * degx), (2 * degx)}, // z/y, x/z, y/x -field bandwidths in x-direction
                                                  // when Bmean is in x,y,z-directions
        {(2 * degy), (2 * degy + 1), (2 * degy)}, // z/y, x/z, y/x -field bandwidths in y-direction
                                                  // when Bmean is in x,y,z-directions
        {(2 * degz), (2 * degz),
         (2 * degz +
          1)} // z/y, x/z, y/x -field bandwidths in z-direction when Bmean is in x,y,z-directions
    }};

    static constexpr std::array<int, 3> s_nExBCoeffs{
        GEMPIC_D_MULT(s_exBBw[xDir][xDir], // total number of z/y-field coefficients when depositing
                                           // on y/z faces, when Bmean is in x-direction
                      s_exBBw[yDir][xDir],
                      s_exBBw[zDir][xDir]),
        GEMPIC_D_MULT(s_exBBw[xDir][yDir], // total number of x/z-field coefficients when depositing
                                           // on z/x faces, when Bmean is in y-direction
                      s_exBBw[yDir][yDir],
                      s_exBBw[zDir][yDir]),
        GEMPIC_D_MULT(s_exBBw[xDir][zDir], // total number of y/x-field coefficients when depositing
                                           // on x/y faces, when Bmean is in z-direction
                      s_exBBw[yDir][zDir],
                      s_exBBw[zDir][zDir])};

    static constexpr int s_nExBCoeffsMax =
        *std::max_element(s_nExBCoeffs.begin(), s_nExBCoeffs.end());

public:
    DKQuasineutralCalculationE(ComputationalDomain infr,
                               std::shared_ptr<Forms::FDDeRhamComplex> drc,
                               amrex::Real relTol = 1.e-12,
                               amrex::Real accTol = 1.e-9,
                               int verbose = 0) :
        m_infra{infr}, m_deRham{drc}, m_relTol{relTol}, m_accTol{accTol}
    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
        static_assert(vdim == 3, "vdim should be 3");

        Io::Parameters params("ECalculation");
        params.get("chargeMatrixTypeEperp", m_chargeMatrixTypeEperp);
        params.get("chargeMatrixTypeEpar", m_chargeMatrixTypeEpar);
    }

    // Functions for calculating E_perp by projecting fields on same direction face for a symmetric
    // matrix begin
    void solve_de_fi_model_eperp (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec)
    {
        if (m_chargeMatrixTypeEperp == "particle")
        {
            solve_de_fi_model_eperp_particle_symm(x, rhs, B, partGr, bdir, pdir);
        }
        else if (m_chargeMatrixTypeEperp == "field_qbym")
        {
            solve_de_fi_model_eperp_qbym_field(x, rhs, B, partGr, bdir, pdir, rhoVec);
        }
    }

    void solve_de_fi_model_eperp_qbym_field (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::solve_de_fi_model_Eperp_qbym_field");

        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> rhsDummy(rhs.m_deRham);

        auto dxi{m_infra.inv_cell_size_3darray()};
        auto dx{m_infra.cell_size_3darray()};

        amrex::GpuArray<amrex::Real, 3> edgetoface{
            static_cast<amrex::Real>(GEMPIC_D_MULT(dxi[xDir], dx[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dxi[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dx[yDir], dxi[zDir]))};

        Gempic::FieldSolvers::Impl::interpolate_from_rhs_and_swap_directions(rhsDummy, rhs, dxi,
                                                                             pdir);

        amrex::IntVect ng{rhs.m_deRham->get_n_ghost()};

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

        int index = 0;
        int indexE = 0;
        for (auto& particleSpecies : partGr)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                indexE = index;
            }
            index++;
        }

        std::array<int, 3> nc = {0, 0, 0};

        for (int n = 0; n < 2; n++)
        {
            for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
            {
                nc[dir] = (pdir[n] == dir);
            }

            for (amrex::MFIter mfi(rhs.m_data[pdir[n]]); mfi.isValid(); ++mfi)
            {
                amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bA;
                for (unsigned int cc = 0; cc < vdim; cc++)
                {
                    bA[cc] = (B.m_data[cc])[mfi].array();
                }

                amrex::Array4<amrex::Real> const& rhsDummyArrPdir =
                    (rhsDummy.m_data[pdir[n]])[mfi].array();
                amrex::Array4<amrex::Real> const& xArrPdir = (x.m_data[pdir[n]])[mfi].array();

                amrex::Array4<amrex::Real> const& rhoEArr = (rhoVec[indexE].m_data)[mfi].array();

                auto bx{(rhsDummy.m_data[pdir[n]])[mfi].box()};
                bx.grow(-ng + 1);
                amrex::Real inverseVolume{1.0 / m_infra.cell_volume()};

                amrex::ParallelFor(
                    bx,
                    [=, compDom = m_infra] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position = {
                            AMREX_D_DECL(plo[xDir] + (i + 0.5 * nc[xDir]) * dx[xDir],
                                         plo[yDir] + (j + 0.5 * nc[yDir]) * dx[yDir],
                                         plo[zDir] + (k + 0.5 * nc[zDir]) * dx[zDir])};

                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            position, plo, compDom.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, 3> bp =
                            spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

                        xArrPdir(AMREX_D_DECL(i, j, k), 0) =
                            -(1.0 - 2.0 * n) * rhsDummyArrPdir(AMREX_D_DECL(i, j, k), 0) *
                            bp[bdir] /
                            (0.5 *
                             (rhoEArr(AMREX_D_DECL(i, j, k), 0) +
                              rhoEArr(AMREX_D_DECL(i + nc[xDir], j + nc[yDir], k + nc[zDir]), 0)) *
                             inverseVolume * edgetoface[pdir[n]]);
                    });
            }
        }
    }

    void solve_de_fi_model_eperp_particle_symm (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::solve_de_fi_model_Eperp_particle_symm()");
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> eCoeffsExB(rhs.m_deRham,
                                                                             s_nRhoCoeffs);

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                amrex::Real charge = particleSpecies->get_charge();

                for (auto& particleGrid : *particleSpecies)
                {
                    long const np = particleGrid.numParticles();
                    auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                    auto const ii = particleSpecies->get_data_indices();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsAExB;
                    for (unsigned int cc = 0; cc < vdim; cc++)
                    {
                        eCoeffsAExB[cc] = (eCoeffsExB.m_data[cc])[particleGrid].array();
                    }

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;
                    for (unsigned int cc = 0; cc < vdim; cc++)
                    {
                        bA[cc] = (B.m_data[cc])[particleGrid].array();
                    }

                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                        m_infra.geometry().ProbLoArray()};

                    amrex::ParallelFor(
                        np,
                        [=, this] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                             ptd.rdata(ii.m_iposz)[pp])};
                            Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                positionParticle, plo, m_infra.inv_cell_size_array());

                            get_exb_e_weights_at_particle_positions_and_deposit_to_face_symm(
                                eCoeffsAExB, bA, spline, charge * ptd.rdata(ii.m_iweight)[pp], bdir,
                                pdir);
                        });
                }
            }
        }

        for (int n = 0; n < 2; n++)
        {
            eCoeffsExB.m_data[pdir[n]].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffsExB.m_data[0].arrays()), vdim> eCoeffsArrsExB;
        for (int n = 0; n < 2; n++)
        {
            eCoeffsArrsExB[pdir[n]] = (eCoeffsExB.m_data[pdir[n]]).arrays();
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
        auto filler = [=, this] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int n,
                                                 amrex::Array4<HYPRE_Int const> const* gid,
                                                 HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0; // initialize to 0 and then increment it through filler functions

            filler_particle_ex_b_matrix_symm(boxno, i, j, k, n, gid, ncols, cols, mat,
                                             eCoeffsArrsExB, mapCols, pdir);
        };

        constexpr int maxStencilSize = *std::max_element(s_nRhoCoeffs.begin(), s_nRhoCoeffs.end());

        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> rhsDummy(rhs.m_deRham);

        auto dxi{m_infra.inv_cell_size_3darray()};

        Gempic::FieldSolvers::Impl::interpolate_from_rhs_and_swap_directions(rhsDummy, rhs, dxi,
                                                                             pdir);

        Kokkos::Profiling::ProfilingSection p1(
            "GEMPIC::DKQuasineutralSolver::CreateHypre::Eperp_particle_symm()");
        p1.start();
        amrex::Print() << " CreateHypre::Eperp_particle_symm " << std::endl;
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[pdir[0]].boxArray().ixType(), x.m_data[pdir[1]].boxArray().ixType()},
            rhs.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, m_verbose, "Eperp_particle_symm");
        p1.stop();

        Kokkos::Profiling::ProfilingSection p2(
            "GEMPIC::DKQuasineutralSolver::SolveHypre::Eperp_particle_symm()");
        p2.start();
        amrex::Print() << " SolveHypre::Eperp_particle_symm " << std::endl;
        hypreSolver.solve(
            amrex::Vector<amrex::MultiFab*>{&(x.m_data[pdir[0]]), &(x.m_data[pdir[1]])},
            amrex::Vector<amrex::MultiFab const*>{&(rhsDummy.m_data[pdir[0]]),
                                                  &(rhsDummy.m_data[pdir[1]])},
            m_relTol, m_absTol, m_maxiter);
        Gempic::Utils::check_solver_convergence(hypreSolver, m_maxiter, m_accTol);

        p2.stop();

        x.fill_boundary();
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_exb_e_weights_at_particle_positions_and_deposit_to_face_symm (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& bA,
        SplineType const& spline,
        amrex::Real chargeWeight,
        int bdir,
        std::array<int, 2> const& pdir)
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

        spline.eval_e_field_spline_weights_all_directions(eWts, eEdges);

        // Evaluate the magnetic field at the particle's position
        amrex::GpuArray<amrex::Real, 3> bp =
            spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        for (int n = 0; n < 2; n++)
        {
            // Depositing E[pdir[n]] on pdir[n]-face
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ1f[xDir][pdir[n]]; ++ii),
                                for (int jj = 0; jj < nWtsXYZ1f[yDir][pdir[n]]; ++jj),
                                for (int kk = 0; kk < nWtsXYZ1f[zDir][pdir[n]]; ++kk))
                amrex::Real const sCoeff =
                    ((pdir[n] == xDir)
                         ? (spline.template eval_b_spline<Forms::Field::DualTwoForm, xDir>(ii, jj,
                                                                                           kk))
                         : ((pdir[n] == yDir)
                                ? (spline.template eval_b_spline<Forms::Field::DualTwoForm, yDir>(
                                      ii, jj, kk))
                                : (spline.template eval_b_spline<Forms::Field::DualTwoForm, zDir>(
                                      ii, jj, kk))));

                // this index is of a primal X-edge and dual X-face where deposit is happening
                amrex::Dim3 edge = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                                 spline.m_firstIndex[yDir] + jj,
                                                 spline.m_firstIndex[zDir] + kk)};

                for (int i = 0; i < nWts1f[pdir[n]]; i++)
                {
                    // this index is of primal Y-edge contributing to Ey-field at particle
                    amrex::Dim3 wEdge = (eEdges[nWts1fStart[pdir[n]] + i]).dim3();

                    AMREX_D_EXPR(iiTemp = wEdge.x - edge.x + s_rhoOffx[pdir[n]],
                                 jjTemp = wEdge.y - edge.y + s_rhoOffy[pdir[n]],
                                 kkTemp = wEdge.z - edge.z + s_rhoOffz[pdir[n]]);

                    component = GEMPIC_D_ADD(iiTemp, jjTemp * s_rhoBWx[pdir[n]],
                                             kkTemp * s_rhoBWx[pdir[n]] * s_rhoBWy[pdir[n]]);

                    amrex::HostDevice::Atomic::Add(
                        eCoeffsA[pdir[n]].ptr(edge.x, edge.y, edge.z, component),
                        -(1.0 - 2.0 * n) * sCoeff * chargeWeight * eWts[nWts1fStart[pdir[n]] + i] /
                            (bp[bdir]));
                }
            GEMPIC_D_LOOP_END
        }
    }

    void filler_particle_ex_b_matrix_symm (
        int boxno,
        int i,
        int j,
        int k,
        int n,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        std::array<amrex::MultiArray4<amrex::Real>, 3> const& eCoeffsArrs,
        std::map<int, int>& mapCols,
        std::array<int, 2> const& pdir)
    {
        // This function contributes (<= nMmCoeffs[ndir]) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};

        GEMPIC_D_LOOP_BEGIN(
            for (ii = -s_rhoOffx[pdir[n]]; ii < (-s_rhoOffx[pdir[n]] + s_rhoBWx[pdir[n]]); ++ii),
            for (jj = -s_rhoOffy[pdir[n]]; jj < (-s_rhoOffy[pdir[n]] + s_rhoBWy[pdir[n]]); ++jj),
            for (kk = -s_rhoOffz[pdir[n]]; kk < (-s_rhoOffz[pdir[n]] + s_rhoBWz[pdir[n]]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_rhoOffx[pdir[n]], jjTemp = jj + s_rhoOffy[pdir[n]],
                         kkTemp = kk + s_rhoOffz[pdir[n]]);

            gdtmp = gid[n]((i + ii), (j + jj), (k + kk));

            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += eCoeffsArrs[pdir[n]][boxno](
                i, j, k,
                GEMPIC_D_ADD(iiTemp, jjTemp * s_rhoBWx[pdir[n]],
                             kkTemp * s_rhoBWx[pdir[n]] * s_rhoBWy[pdir[n]]));
        GEMPIC_D_LOOP_END
    }
    // Functions for calculating E_perp by projecting fields on same direction face for a symmetric
    // matrix end

    // Functions for calculating E_perp by projecting fields on the correct direction for a combined
    // symmetric matrix begin
    void solve_de_fi_model_eperp_particle_combo (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::solve_de_fi_model_eperp_particle_combo()");
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> eCoeffsExB(
            rhs.m_deRham, std::array<int, 3>{s_nExBCoeffsMax, s_nExBCoeffsMax, s_nExBCoeffsMax});

        // Fill up E_coeffs
        for (auto& particleSpecies : partGr)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                amrex::Real charge = particleSpecies->get_charge();

                for (auto& particleGrid : *particleSpecies)
                {
                    long const np = particleGrid.numParticles();
                    auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
                    auto const ii = particleSpecies->get_data_indices();

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eCoeffsAExB;
                    for (unsigned int cc = 0; cc < vdim; cc++)
                    {
                        eCoeffsAExB[cc] = (eCoeffsExB.m_data[cc])[particleGrid].array();
                    }

                    amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;
                    for (unsigned int cc = 0; cc < vdim; cc++)
                    {
                        bA[cc] = (B.m_data[cc])[particleGrid].array();
                    }

                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{
                        m_infra.geometry().ProbLoArray()};

                    amrex::ParallelFor(
                        np,
                        [=, this] AMREX_GPU_DEVICE(long pp)
                        {
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                                AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                             ptd.rdata(ii.m_iposz)[pp])};
                            Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                                positionParticle, plo, m_infra.inv_cell_size_array());

                            get_exb_e_weights_at_particle_positions_and_deposit_to_face_combo(
                                eCoeffsAExB, bA, spline, charge * ptd.rdata(ii.m_iweight)[pp], bdir,
                                pdir);
                        });
                }
            }
        }

        for (int n = 0; n < 2; n++)
        {
            eCoeffsExB.m_data[pdir[n]].SumBoundary(m_infra.geometry().periodicity());
        }

        std::array<decltype(eCoeffsExB.m_data[0].arrays()), vdim> eCoeffsArrsExB;
        for (int n = 0; n < 2; n++)
        {
            eCoeffsArrsExB[pdir[n]] = (eCoeffsExB.m_data[pdir[n]]).arrays();
        }

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        // Is variable n at (i,j,k) in Box boxno (local index) valid?
        // (i.e., not exactly on Dirichlet boundary)
        auto marker = [=] AMREX_GPU_DEVICE(int, int i, int j, int k, int) -> bool
        { return nddom.contains(i, j, k); };

        // For variable n at (i,j,k) in Box boxno (local index), fill its row in
        // the matrix.
        // [in ] gid : gid[n] is the id for variable n at (i,j,k)
        // [out] ncols: # of columns in this row.
        // [out] cols: column indices in this row.
        // [out] mat : matrix elements in this row.
        auto filler = [=, this] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int n,
                                                 amrex::Array4<HYPRE_Int const> const* gid,
                                                 HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0; // initialize to 0 and then increment it through filler functions

            filler_particle_ex_b_matrix_combo(boxno, i, j, k, n, gid, ncols, cols, mat,
                                              eCoeffsArrsExB, mapCols, bdir, pdir);
        };

        constexpr int maxStencilSize = s_nExBCoeffsMax;

        Kokkos::Profiling::ProfilingSection p3(
            "GEMPIC::DKQuasineutralSolver::CreateHypre::Eperp_particle_asymm()");
        p3.start();
        amrex::Print() << " CreateHypre::Eperp_particle_asymm " << std::endl;
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[pdir[0]].boxArray().ixType(), x.m_data[pdir[1]].boxArray().ixType()},
            rhs.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, m_verbose, "Eperp_particle_asymm");
        p3.stop();

        Kokkos::Profiling::ProfilingSection p4(
            "GEMPIC::DKQuasineutralSolver::SolveHypre::Eperp_particle_asymm()");
        p4.start();
        amrex::Print() << " SolveHypre::Eperp_particle_asymm " << std::endl;
        hypreSolver.solve(
            amrex::Vector<amrex::MultiFab*>{&(x.m_data[pdir[0]]), &(x.m_data[pdir[1]])},
            amrex::Vector<amrex::MultiFab const*>{&(rhs.m_data[pdir[0]]), &(rhs.m_data[pdir[1]])},
            m_relTol, m_absTol, m_maxiter);
        p4.stop();
        Gempic::Utils::check_solver_convergence(hypreSolver, m_maxiter, m_accTol);

        x.fill_boundary();
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_exb_e_weights_at_particle_positions_and_deposit_to_face_combo (
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const& eCoeffsA,
        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& bA,
        SplineType const& spline,
        amrex::Real chargeWeight,
        int bdir,
        std::array<int, 2> const& pdir)
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

        spline.eval_e_field_spline_weights_all_directions(eWts, eEdges);

        // Evaluate the magnetic field at the particle's position
        amrex::GpuArray<amrex::Real, 3> bp =
            spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        for (int n = 0; n < 2; n++)
        {
            // Depositing E[pdir[1]] on pdir[0]-face
            GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ1f[xDir][pdir[n]]; ++ii),
                                for (int jj = 0; jj < nWtsXYZ1f[yDir][pdir[n]]; ++jj),
                                for (int kk = 0; kk < nWtsXYZ1f[zDir][pdir[n]]; ++kk))
                amrex::Real const sCoeff =
                    ((pdir[n] == xDir)
                         ? (spline.template eval_b_spline<Forms::Field::DualTwoForm, xDir>(ii, jj,
                                                                                           kk))
                         : ((pdir[n] == yDir)
                                ? (spline.template eval_b_spline<Forms::Field::DualTwoForm, yDir>(
                                      ii, jj, kk))
                                : (spline.template eval_b_spline<Forms::Field::DualTwoForm, zDir>(
                                      ii, jj, kk))));

                // this index is of a primal pdir[n]-edge and dual pdir[n]-face where deposit is
                // happening
                amrex::Dim3 edge = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                                 spline.m_firstIndex[yDir] + jj,
                                                 spline.m_firstIndex[zDir] + kk)};

                for (int i = 0; i < nWts1f[pdir[1 - n]]; i++)
                {
                    // this index is of primal pdir[1-n]-edge contributing to E[pdir[1-n]]-field at
                    // particle
                    amrex::Dim3 wEdge = (eEdges[nWts1fStart[pdir[1 - n]] + i]).dim3();

                    AMREX_D_EXPR(iiTemp = wEdge.x - edge.x + s_exBOff[pdir[n]][xDir],
                                 jjTemp = wEdge.y - edge.y + s_exBOff[pdir[n]][yDir],
                                 kkTemp = wEdge.z - edge.z + s_exBOff[pdir[n]][zDir]);

                    component = GEMPIC_D_ADD(iiTemp, jjTemp * s_exBBw[xDir][bdir],
                                             kkTemp * s_exBBw[xDir][bdir] * s_exBBw[yDir][bdir]);

                    amrex::HostDevice::Atomic::Add(
                        eCoeffsA[pdir[n]].ptr(edge.x, edge.y, edge.z, component),
                        (1.0 - 2.0 * n) * sCoeff * chargeWeight *
                            eWts[nWts1fStart[pdir[1 - n]] + i] / (bp[bdir]));
                }
            GEMPIC_D_LOOP_END
        }
    }

    void filler_particle_ex_b_matrix_combo (
        int boxno,
        int i,
        int j,
        int k,
        int n,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        std::array<amrex::MultiArray4<amrex::Real>, 3> const& eCoeffsArrs,
        std::map<int, int>& mapCols,
        int bdir,
        std::array<int, 2> const& pdir)
    {
        // This function contributes (<= nMmCoeffs[ndir]) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};

        GEMPIC_D_LOOP_BEGIN(for (ii = -s_exBOff[pdir[n]][xDir];
                                 ii < (-s_exBOff[pdir[n]][xDir] + s_exBBw[xDir][bdir]); ++ii),
                            for (jj = -s_exBOff[pdir[n]][yDir];
                                 jj < (-s_exBOff[pdir[n]][yDir] + s_exBBw[yDir][bdir]); ++jj),
                            for (kk = -s_exBOff[pdir[n]][zDir];
                                 kk < (-s_exBOff[pdir[n]][zDir] + s_exBBw[zDir][bdir]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_exBOff[pdir[n]][xDir],
                         jjTemp = jj + s_exBOff[pdir[n]][yDir],
                         kkTemp = kk + s_exBOff[pdir[n]][zDir]);

            gdtmp = gid[1 - n]((i + ii), (j + jj), (k + kk));

            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += eCoeffsArrs[pdir[n]][boxno](
                i, j, k, // pdir[n] is direction of the RHS
                GEMPIC_D_ADD(iiTemp, jjTemp * s_exBBw[xDir][bdir],
                             kkTemp * s_exBBw[xDir][bdir] * s_exBBw[yDir][bdir]));
        GEMPIC_D_LOOP_END
    }

    // Functions for calculating E_perp by projecting fields on the correct direction for a combined
    // symmetric matrix end

    // Functions for calculating E_par begin
    void solve_de_fi_model_epar (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec)
    {
        if (m_chargeMatrixTypeEpar == "particle")
        {
            solve_de_fi_model_epar_particle(x, rhs, partGr, bdir);
        }
        else if (m_chargeMatrixTypeEpar == "field_qbym")
        {
            solve_de_fi_model_epar_qbym_field(x, rhs, partGr, bdir, rhoVec);
        }
    }

    void solve_de_fi_model_epar_qbym_field (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::vector<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>>& rhoVec)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::solve_de_fi_model_epar_qbym_field()");

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        std::vector<amrex::MultiArray4<amrex::Real>> rhoarrsVec;
        rhoarrsVec.resize(rhoVec.size());
        for (int i = 0; i < partGr.size(); i++)
        {
            rhoarrsVec[i] = rhoVec[i].m_data.arrays();
        }

        auto const dxi = m_infra.inv_cell_size_array();
        auto const dx = m_infra.cell_size_array();

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        amrex::GpuArray<amrex::Real, 3> edgetoface{
            static_cast<amrex::Real>(GEMPIC_D_MULT(dxi[xDir], dx[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dxi[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dx[yDir], dxi[zDir]))};

        // Is variable n at (i,j,k) in Box boxno (local index) valid?
        // (i.e., not exactly on Dirichlet boundary)
        auto marker = [=] AMREX_GPU_DEVICE(int /*boxno*/, int i, int j, int k, int /*n*/) -> bool
        { return nddom.contains(i, j, k); };

        std::vector<amrex::Real> qbym(partGr.size());

        int index = 0;
        for (auto& particleSpecies : partGr)
        {
            amrex::Real charge = particleSpecies->get_charge();
            amrex::Real mass = particleSpecies->get_mass();

            qbym[index] = charge / mass;
            index++;
        }

        // For variable n at (i,j,k) in Box boxno (local index), fill its row in
        // the matrix.
        // [in ] gid : gid[n] is the id for variable n at (i,j,k)
        // [out] ncols: # of columns in this row.
        // [out] cols: column indices in this row.
        // [out] mat : matrix elements in this row.
        auto filler = [=, this] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int n,
                                                 amrex::Array4<HYPRE_Int const> const* gid,
                                                 HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0; // initialize to 0 and then increment it through filler functions

            filler_poisson_epar_matrix(boxno, i, j, k, n, gid, ncols, cols, mat, fac, edgetoface,
                                       mapCols, bdir);
            filler_qbym_field_epar_charge_matrix(boxno, i, j, k, n, gid, ncols, cols, mat, qbym,
                                                 rhoarrsVec, edgetoface, mapCols, bdir);
        };

        constexpr int maxStencilSize = (2 * AMREX_SPACEDIM + 1);

        Kokkos::Profiling::ProfilingSection p5(
            "GEMPIC::DKQuasineutralSolver::CreateHypre::Epar_field()");
        p5.start();
        amrex::Print() << " CreateHypre::Epar_field " << std::endl;
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[bdir].boxArray().ixType()}, rhs.m_deRham->get_n_ghost(), m_infra.geometry(),
            m_infra.m_grid, m_infra.m_distriMap, marker, filler, m_verbose, "Epar_field");
        p5.stop();

        Kokkos::Profiling::ProfilingSection p6(
            "GEMPIC::DKQuasineutralSolver::SolveHypre::Epar_field()");
        p6.start();
        amrex::Print() << " SolveHypre::Epar_field " << std::endl;
        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[bdir])},
                          amrex::Vector<amrex::MultiFab const*>{&(rhs.m_data[bdir])}, m_relTol,
                          m_absTol, m_maxiter);
        p6.stop();
        Gempic::Utils::check_solver_convergence(hypreSolver, m_maxiter, m_accTol);

        x.fill_boundary();
    }

    void filler_qbym_field_epar_charge_matrix (
        int boxno,
        int i,
        int j,
        int k,
        int ndir,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        std::vector<amrex::Real> qbym,
        std::vector<amrex::MultiArray4<amrex::Real>> const& rhoarrsVec,
        amrex::GpuArray<HYPRE_Real, 3> const edgetoface,
        std::map<int, int>& mapCols,
        int bdir)
    {
        UNUSED(ndir);
        std::array<int, 3> nc = {0, 0, 0};
        switch (hodgeDegree)
        {
            case 2:
                for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
                {
                    nc[dir] = (bdir == dir);
                }

                // This function contributes (<= 1) columns to ncols
                int icol, gdtmp;
                gdtmp = gid[0](i, j, k);

                Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);

                for (int index = 0; index < qbym.size(); index++)
                {
                    mat[icol] +=
                        0.5 * qbym[index] *
                        (rhoarrsVec[index][boxno](i, j, k) +
                         rhoarrsVec[index][boxno](i + nc[xDir], j + nc[yDir], k + nc[zDir])) *
                        (1.0 / m_infra.cell_volume()) * edgetoface[bdir];
                }

                break;
            default:
                AMREX_ASSERT("Hodge degree not implemented");
                break;
        };
    }

    void solve_de_fi_model_epar_particle (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& x,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::solve_de_fi_model_Epar_particle()");
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> eparCoeffsRho(rhs.m_deRham,
                                                                                s_nRhoCoeffs);

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

                amrex::Array4<amrex::Real> eparCoeffsARho;

                eparCoeffsARho = (eparCoeffsRho.m_data[bdir])[particleGrid].array();

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=, this] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                            AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                         ptd.rdata(ii.m_iposz)[pp])};
                        Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, plo, m_infra.inv_cell_size_array());

                        get_rho_epar_weights_at_particle_positions_and_deposit_to_face(
                            eparCoeffsARho, spline, chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp],
                            bdir);
                    });
            }
        }

        eparCoeffsRho.m_data[bdir].SumBoundary(m_infra.geometry().periodicity());

        decltype(eparCoeffsRho.m_data[0].arrays()) eparCoeffsArrsRho;
        eparCoeffsArrsRho = (eparCoeffsRho.m_data[bdir]).arrays();

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        auto const dxi = m_infra.inv_cell_size_array();
        auto const dx = m_infra.cell_size_array();

        amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> fac{
            AMREX_D_DECL(static_cast<HYPRE_Real>(dxi[xDir] * dxi[xDir]),
                         static_cast<HYPRE_Real>(dxi[yDir] * dxi[yDir]),
                         static_cast<HYPRE_Real>(dxi[zDir] * dxi[zDir]))};

        amrex::GpuArray<amrex::Real, 3> edgetoface{
            static_cast<amrex::Real>(GEMPIC_D_MULT(dxi[xDir], dx[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dxi[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dx[yDir], dxi[zDir]))};

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
        auto filler = [=, this] AMREX_GPU_DEVICE(int boxno, int i, int j, int k, int n,
                                                 amrex::Array4<HYPRE_Int const> const* gid,
                                                 HYPRE_Int& ncols, HYPRE_Int* cols, HYPRE_Real* mat)
        {
            std::map<int, int> mapCols;
            ncols = 0; // initialize to 0 and then increment it through filler functions

            filler_poisson_epar_matrix(boxno, i, j, k, n, gid, ncols, cols, mat, fac, edgetoface,
                                       mapCols, bdir);
            filler_particle_epar_charge_matrix(boxno, i, j, k, n, gid, ncols, cols, mat,
                                               eparCoeffsArrsRho, mapCols, bdir);
        };

        constexpr int maxStencilSize =
            *std::max_element(s_nRhoCoeffs.begin(), s_nRhoCoeffs.end()) + (2 * AMREX_SPACEDIM + 1);

        Kokkos::Profiling::ProfilingSection p7(
            "GEMPIC::DKQuasineutralSolver::CreateHypre::Epar_particle()");
        p7.start();
        amrex::Print() << " CreateHypre::Epar_particle " << std::endl;
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {x.m_data[bdir].boxArray().ixType()}, rhs.m_deRham->get_n_ghost(), m_infra.geometry(),
            m_infra.m_grid, m_infra.m_distriMap, marker, filler, m_verbose, "Epar_particle");
        p7.stop();

        Kokkos::Profiling::ProfilingSection p8(
            "GEMPIC::DKQuasineutralSolver::SolveHypre::Epar_particle()");
        p8.start();
        amrex::Print() << " SolveHypre::Epar_particle " << std::endl;
        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(x.m_data[bdir])},
                          amrex::Vector<amrex::MultiFab const*>{&(rhs.m_data[bdir])}, m_relTol,
                          m_absTol, m_maxiter);
        p8.stop();
        Gempic::Utils::check_solver_convergence(hypreSolver, m_maxiter, m_accTol);

        x.fill_boundary();
    }

    template <class SplineType = ParticleMeshCoupling::SplineBase<degx, degy, degz>>
    void get_rho_epar_weights_at_particle_positions_and_deposit_to_face (
        amrex::Array4<amrex::Real> const& eparCoeffsA,
        SplineType const& spline,
        amrex::Real chargeSqrByMassWeight,
        int bdir)
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

        spline.eval_e_field_spline_weights_all_directions(eWts, eEdges);

        GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

        int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);
        int component;

        GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nWtsXYZ1f[xDir][bdir]; ++ii),
                            for (int jj = 0; jj < nWtsXYZ1f[yDir][bdir]; ++jj),
                            for (int kk = 0; kk < nWtsXYZ1f[zDir][bdir]; ++kk))
            amrex::Real const sCoeff =
                ((bdir == xDir)
                     ? (spline.template eval_b_spline<Forms::Field::DualTwoForm, xDir>(ii, jj, kk))
                     : ((bdir == yDir)
                            ? (spline.template eval_b_spline<Forms::Field::DualTwoForm, yDir>(
                                  ii, jj, kk))
                            : (spline.template eval_b_spline<Forms::Field::DualTwoForm, zDir>(
                                  ii, jj, kk))));

            // this index is of a primal edge and dual face where deposit is happening
            amrex::Dim3 edge = {GEMPIC_D_PAD(spline.m_firstIndex[xDir] + ii,
                                             spline.m_firstIndex[yDir] + jj,
                                             spline.m_firstIndex[zDir] + kk)};

            for (int i = 0; i < nWts1f[bdir]; i++)
            {
                // this index is of primal edge contributing to E-field at particle
                amrex::Dim3 wEdge = (eEdges[nWts1fStart[bdir] + i]).dim3();

                AMREX_D_EXPR(iiTemp = wEdge.x - edge.x + s_rhoOffx[bdir],
                             jjTemp = wEdge.y - edge.y + s_rhoOffy[bdir],
                             kkTemp = wEdge.z - edge.z + s_rhoOffz[bdir]);

                component = GEMPIC_D_ADD(iiTemp, jjTemp * s_rhoBWx[bdir],
                                         kkTemp * s_rhoBWx[bdir] * s_rhoBWy[bdir]);

                amrex::HostDevice::Atomic::Add(
                    eparCoeffsA.ptr(edge.x, edge.y, edge.z, component),
                    sCoeff * chargeSqrByMassWeight * eWts[nWts1fStart[bdir] + i]);
            }
        GEMPIC_D_LOOP_END
    }

    void filler_particle_epar_charge_matrix (int boxno,
                                             int i,
                                             int j,
                                             int k,
                                             int n,
                                             amrex::Array4<HYPRE_Int const> const* gid,
                                             HYPRE_Int& ncols,
                                             HYPRE_Int* cols,
                                             HYPRE_Real* mat,
                                             amrex::MultiArray4<amrex::Real> const& eparCoeffsArrs,
                                             std::map<int, int>& mapCols,
                                             int bdir)
    {
        UNUSED(n);
        // This function contributes (<= nMmCoeffs[ndir]) columns to ncols
        int AMREX_D_DECL(iiTemp{0}, jjTemp{0}, kkTemp{0});
        int icol, gdtmp;

        int ii{0}, jj{0}, kk{0};

        GEMPIC_D_LOOP_BEGIN(
            for (ii = -s_rhoOffx[bdir]; ii < (-s_rhoOffx[bdir] + s_rhoBWx[bdir]); ++ii),
            for (jj = -s_rhoOffy[bdir]; jj < (-s_rhoOffy[bdir] + s_rhoBWy[bdir]); ++jj),
            for (kk = -s_rhoOffz[bdir]; kk < (-s_rhoOffz[bdir] + s_rhoBWz[bdir]); ++kk))
            AMREX_D_EXPR(iiTemp = ii + s_rhoOffx[bdir], jjTemp = jj + s_rhoOffy[bdir],
                         kkTemp = kk + s_rhoOffz[bdir]);

            gdtmp = gid[0]((i + ii), (j + jj), (k + kk));

            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] +=
                eparCoeffsArrs[boxno](i, j, k,
                                      GEMPIC_D_ADD(iiTemp, jjTemp * s_rhoBWx[bdir],
                                                   kkTemp * s_rhoBWx[bdir] * s_rhoBWy[bdir]));
        GEMPIC_D_LOOP_END
    }

    void filler_poisson_epar_matrix (int boxno,
                                     int i,
                                     int j,
                                     int k,
                                     int n,
                                     amrex::Array4<HYPRE_Int const> const* gid,
                                     HYPRE_Int& ncols,
                                     HYPRE_Int* cols,
                                     HYPRE_Real* mat,
                                     amrex::GpuArray<HYPRE_Real, AMREX_SPACEDIM> const fac,
                                     amrex::GpuArray<HYPRE_Real, 3> const edgetoface,
                                     std::map<int, int>& mapCols,
                                     int bdir)
    {
        UNUSED(n);
        UNUSED(boxno);

        if (AMREX_SPACEDIM == 2 && bdir == zDir)
        {
            // This function contributes (<= 5) columns to ncols
            int icol;
            int gdtmp;
            gdtmp = gid[0](i, j, k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += 2.0 * (fac[0] + fac[1]) * edgetoface[2];

            gdtmp = gid[0]((i - 1), j, k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += -fac[0] * edgetoface[2];
            gdtmp = gid[0]((i + 1), j, k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += -fac[0] * edgetoface[2];
            gdtmp = gid[0](i, (j - 1), k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += -fac[1] * edgetoface[2];
            gdtmp = gid[0](i, (j + 1), k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += -fac[1] * edgetoface[2];
        }
        else
        {
            // This function contributes (<= 6*AMREX_SPACEDIM-5) columns to ncols
            int icol;
            int gdtmp;
            int n[AMREX_SPACEDIM], in[AMREX_SPACEDIM], jn[AMREX_SPACEDIM], kn[AMREX_SPACEDIM];
            for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
            {
                n[dir] = (bdir + dir) % AMREX_SPACEDIM;
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

            gdtmp = gid[0](i, j, k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += temp * edgetoface[n[0]];

            for (int dir = 1; dir < AMREX_SPACEDIM; dir++)
            {
                gdtmp = gid[0]((i - in[dir]), (j - jn[dir]), (k - kn[dir]));
                Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                mat[icol] += -fac[n[dir]] * edgetoface[n[0]];

                gdtmp = gid[0]((i + in[dir]), (j + jn[dir]), (k + kn[dir]));
                Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
                mat[icol] += -fac[n[dir]] * edgetoface[n[0]];
            }
        }
    }
    // Functions for calculating E_par end

    // Functions for calculating rhs of the E_perp equation begin
    void de_ki_calculate_rhs_eperp_eqn (
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhsEperpEqn,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& curlH,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::de_ki_calculate_rhs_Eperp_eqn()");

        Gempic::ParticleMeshCoupling::deposit_current_density_drift_kinetic<
            degx, degy, degz, Gempic::ParticleMeshCoupling::ElectricFieldUse::ExcludeElectricField>(
            J, partGr, m_infra, E, B, bdir);

        // Calculate curlB on dual face
        hodge(H, B, m_deRham->scaling_bto_h());
        curl(curlH, H);

        // RHS = curlB - J
        linear_combination(rhsEperpEqn, 1.0, curlH, -1.0, J);
    }
    // Functions for calculating rhs of the E_perp equation end

    // Functions for calculating rhs of the E_parallel equation begin
    void de_ki_calculate_rhs_epar_eqn (
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhsEparallelEqn,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        int bdir,
        std::array<int, 2> const& pdir)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::de_ki_calculate_rhs_Epar_eqn()");
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> eperpDotBstarElectrons(m_deRham);
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> delDotSElectrons(m_deRham);
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> jcrossBIons(m_deRham);
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> delDotSIons(m_deRham);

        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> curlE(m_deRham);
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge> curlEProj(m_deRham);
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> curlCurlE(m_deRham);

        for (unsigned int i = 0; i < vdim; i++)
        {
            rhsEparallelEqn.m_data[i].setVal(0.0);
        }
        // particle loop
        for (auto& particleSpecies : partGr)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                de_deposit_eperpdotbstar_deldotsdrift(eperpDotBstarElectrons, delDotSElectrons,
                                                      particleSpecies, E, B, bdir, pdir);
            }
            else
            { // "ions"
                ki_deposit_jcrossb_deldots(jcrossBIons, delDotSIons, B, particleSpecies);
            }
        } //end of loop over species

        curl(curlE, E);
        hodge(curlEProj, curlE, m_deRham->scaling_bto_h());
        curl(curlCurlE, curlEProj);

        rhsEparallelEqn -= curlCurlE;
        rhsEparallelEqn -= jcrossBIons;
        rhsEparallelEqn += delDotSIons;
        rhsEparallelEqn -= eperpDotBstarElectrons;
        rhsEparallelEqn += delDotSElectrons;

        rhsEparallelEqn.post_particle_loop_sync();
    }

    void de_deposit_eperpdotbstar_deldotsdrift (
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& eperpDotBstar,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& delDotSdrift,
        std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>& particleSpecies,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        int bdir,
        std::array<int, 2> const& pdir)
    {
        std::array<amrex::Real, vdim> b = {0.0};
        b[bdir] = 1.0;

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
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eperpdotbstarA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> deldotsdriftA;

            for (unsigned int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (E.m_data[cc])[pti].array();
                bA[cc] = (B.m_data[cc])[pti].array();
                eperpdotbstarA[cc] = (eperpDotBstar.m_data[cc])[pti].array();
                deldotsdriftA[cc] = (delDotSdrift.m_data[cc])[pti].array();
            }

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};
            auto dxi = m_infra.inv_cell_size_array();

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    auto const weight = ptd.rdata(ii.m_iweight)[pp];
                    amrex::Real weightEff = weight;

                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};

                    std::array<amrex::Real const, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                            ptd.rdata(ii.m_ively)[pp],
                                                            ptd.rdata(ii.m_ivelz)[pp]};

                    Gempic::ParticleMeshCoupling::SplineWithFirstDerivative<degx, degy, degz>
                        splineWithDeriv(positionParticle, plo, dxi);

                    amrex::GpuArray<amrex::Real, vdim> ep =
                        splineWithDeriv.template eval_spline_field<Forms::Field::PrimalOneForm>(eA);

                    amrex::GpuArray<amrex::Real, vdim> bp =
                        splineWithDeriv.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

                    amrex::GpuArray<amrex::Real, vdim> velGC;
                    velGC[0] = (vel[bdir] * bp[0] + ep[1] * b[2] - ep[2] * b[1]) / bp[bdir];
                    velGC[1] = (vel[bdir] * bp[1] + ep[2] * b[0] - ep[0] * b[2]) / bp[bdir];
                    velGC[2] = (vel[bdir] * bp[2] + ep[0] * b[1] - ep[1] * b[0]) / bp[bdir];

                    amrex::GpuArray<amrex::Real, vdim> aGC{0.0, 0.0, 0.0};
                    aGC[bdir] = chargeByMass *
                                (ep[pdir[0]] * bp[pdir[0]] + ep[pdir[1]] * bp[pdir[1]]) / bp[bdir];

                    deposit_twoform(eperpdotbstarA, splineWithDeriv, aGC, charge * weightEff);
                    deposit_deldot_s(deldotsdriftA, splineWithDeriv, velGC, charge * weightEff);
                });
        }
    }

    void ki_deposit_jcrossb_deldots (
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& jcrossB,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& delDotS,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>& particleSpecies)
    {
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeByMass = charge / mass;

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jcrossbA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> deldotsA;

            for (unsigned int cc = 0; cc < vdim; cc++)
            {
                bA[cc] = (B.m_data[cc])[pti].array();
                jcrossbA[cc] = (jcrossB.m_data[cc])[pti].array();
                deldotsA[cc] = (delDotS.m_data[cc])[pti].array();
            }

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{m_infra.geometry().ProbLoArray()};
            auto dxi = m_infra.inv_cell_size_array();

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    auto const weight = ptd.rdata(ii.m_iweight)[pp];
                    amrex::Real weightEff = weight;

                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};

                    amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                           ptd.rdata(ii.m_ively)[pp],
                                                           ptd.rdata(ii.m_ivelz)[pp]};

                    Gempic::ParticleMeshCoupling::SplineWithFirstDerivative<degx, degy, degz>
                        splineWithDeriv(positionParticle, plo, dxi);

                    amrex::GpuArray<amrex::Real, vdim> bp =
                        splineWithDeriv.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

                    amrex::GpuArray<amrex::Real, vdim> vcrossb{
                        (vel[yDir] * bp[zDir] - vel[zDir] * bp[yDir]),
                        (vel[zDir] * bp[xDir] - vel[xDir] * bp[zDir]),
                        (vel[xDir] * bp[yDir] - vel[yDir] * bp[xDir])};

                    deposit_twoform(jcrossbA, splineWithDeriv, vcrossb,
                                    chargeByMass * charge * weightEff);
                    deposit_deldot_s(deldotsA, splineWithDeriv, vel, charge * weightEff);
                });
        }
    }
    // Functions for calculating rhs of the E_parallel equation end
};

} // namespace Gempic::FieldSolvers

#endif

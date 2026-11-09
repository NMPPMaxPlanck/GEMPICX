/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_DKQUASINEUTRALTIMESTEPPER_H_
#define GEMPIC_DKQUASINEUTRALTIMESTEPPER_H_

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
class DKQuasineutralTimestepping
{
private:
    ComputationalDomain m_infra;
    std::shared_ptr<Forms::FDDeRhamComplex> m_deRham;
    amrex::Real m_relTol, m_absTol{1.e-14}, m_accTol{1.e-9};
    amrex::Real m_maxiter{1000};
    int m_verbose;

    Gempic::TimeLoop::RungeKutta m_lsrk;

    Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> m_s2B;

public:
    DKQuasineutralTimestepping(ComputationalDomain infr,
                               std::shared_ptr<Forms::FDDeRhamComplex> drc,
                               amrex::Real tol = 1.e-12,
                               int verbose = 0) :
        m_infra{infr}, m_deRham{drc}, m_relTol{tol}, m_lsrk{infr, drc}, m_s2B(drc)
    {
        m_verbose = std::max(verbose, Gempic::Utils::Verbosity::level());
        static_assert(vdim == 3, "vdim should be 3");
    }

    // Function for pushing particle velocities and positions starts
    void lsrk_de_fi_push_particles (
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>>& partGr,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        amrex::Real dt,
        int bdir,
        int stageIndex)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::lsrk_de_fi_push_particles()");

        for (auto& particleSpecies : partGr)
        {
            if (particleSpecies->get_name() == "electrons")
            {
                lsrk_de_push_particles(particleSpecies, E, B, dt, bdir, stageIndex);
            }
            else
            {
                lsrk_fi_push_particles(particleSpecies, E, B, dt, stageIndex);
            }
        }
    }

    void lsrk_de_push_particles (
        std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>& particleSpecies,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        amrex::Real dt,
        int bdir,
        int stageIndex)
    {
        std::array<amrex::Real, vdim> b = {0.0};
        b[bdir] = 1.0;

        amrex::Real const coeffA = m_lsrk.get_coeff_a(stageIndex);
        amrex::Real const coeffB = m_lsrk.get_coeff_b(stageIndex);

        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeByMass = charge / mass;

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bA;

            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (E.m_data[cc])[pti].array();
                bA[cc] = (B.m_data[cc])[pti].array();
            }

            // loop over particles: add contribution of old particle position to J and push
            auto plo = m_infra.geometry().ProbLoArray();
            auto dxInverse = m_infra.inv_cell_size_array();
            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};

                    amrex::GpuArray<amrex::Real, vdim> V{ptd.rdata(ii.m_ivelx)[pp],
                                                         ptd.rdata(ii.m_ively)[pp],
                                                         ptd.rdata(ii.m_ivelz)[pp]};
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{
                        AMREX_D_DECL(ptd.rdata(ii.m_is2X)[pp], ptd.rdata(ii.m_is2Y)[pp],
                                     ptd.rdata(ii.m_is2Z)[pp])};
                    amrex::GpuArray<amrex::Real, vdim> s2V{ptd.rdata(ii.m_is2Vx)[pp],
                                                           ptd.rdata(ii.m_is2Vy)[pp],
                                                           ptd.rdata(ii.m_is2Vz)[pp]};
                    // evaluate the splines at particle position R
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(R, plo,
                                                                                      dxInverse);
                    amrex::GpuArray<amrex::Real, vdim> ep =
                        spline.template eval_spline_field<Forms::Field::PrimalOneForm>(eA);
                    // evaluate the magnetic field at the particle position
                    amrex::GpuArray<amrex::Real, vdim> bp =
                        spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

                    // compute guiding center velocity and acceleration
                    amrex::GpuArray<amrex::Real, vdim> velGC;
                    velGC[0] = (V[bdir] * bp[0] + ep[1] * b[2] - ep[2] * b[1]) / bp[bdir];
                    velGC[1] = (V[bdir] * bp[1] + ep[2] * b[0] - ep[0] * b[2]) / bp[bdir];
                    velGC[2] = (V[bdir] * bp[2] + ep[0] * b[1] - ep[1] * b[0]) / bp[bdir];
                    amrex::Real accGC;

                    // compute guiding center velocity and acceleration
                    accGC =
                        chargeByMass * (bp[0] * ep[0] + bp[1] * ep[1] + bp[2] * ep[2]) / bp[bdir];

                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        s2R[d] = coeffA * s2R[d] + dt * velGC[d];
                    }
                    s2V[bdir] = coeffA * s2V[bdir] + dt * accGC;
                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        R[d] += coeffB * s2R[d];
                    }
                    V[bdir] += coeffB * s2V[bdir];

                    // Write back particle position and velocity
                    AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp] = R[0], ptd.rdata(ii.m_iposy)[pp] = R[1],
                                 ptd.rdata(ii.m_iposz)[pp] = R[2]);
                    ptd.rdata(ii.m_ivelx)[pp] = V[0];
                    ptd.rdata(ii.m_ively)[pp] = V[1];
                    ptd.rdata(ii.m_ivelz)[pp] = V[2];
                    AMREX_D_DECL(ptd.rdata(ii.m_is2X)[pp] = s2R[0],
                                 ptd.rdata(ii.m_is2Y)[pp] = s2R[1],
                                 ptd.rdata(ii.m_is2Z)[pp] = s2R[2]);
                    ptd.rdata(ii.m_is2Vx)[pp] = s2V[0];
                    ptd.rdata(ii.m_is2Vy)[pp] = s2V[1];
                    ptd.rdata(ii.m_is2Vz)[pp] = s2V[2];
                }); // end of parallel_for
        } //end of loop over particles
        //redistribute particles
        particleSpecies->Redistribute();
    }

    void lsrk_fi_push_particles (
        std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>& particleSpecies,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        amrex::Real dt,
        int stageIndex)
    {
        amrex::Real const coeffA = m_lsrk.get_coeff_a(stageIndex);
        amrex::Real const coeffB = m_lsrk.get_coeff_b(stageIndex);

        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeByMass = charge / mass;

        for (auto& pti : *particleSpecies)
        {
            long const np = pti.numParticles();
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bA;
            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (E.m_data[cc])[pti].array();
                bA[cc] = (B.m_data[cc])[pti].array();
            }

            // loop over particles: add contribution of old particle position to J and push
            auto plo = m_infra.geometry().ProbLoArray();
            auto dxInverse = m_infra.inv_cell_size_array();

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Read out particle position
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> R{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};

                    amrex::GpuArray<amrex::Real, vdim> V{ptd.rdata(ii.m_ivelx)[pp],
                                                         ptd.rdata(ii.m_ively)[pp],
                                                         ptd.rdata(ii.m_ivelz)[pp]};

                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> s2R{
                        AMREX_D_DECL(ptd.rdata(ii.m_is2X)[pp], ptd.rdata(ii.m_is2Y)[pp],
                                     ptd.rdata(ii.m_is2Z)[pp])};
                    amrex::GpuArray<amrex::Real, vdim> s2V{ptd.rdata(ii.m_is2Vx)[pp],
                                                           ptd.rdata(ii.m_is2Vy)[pp],
                                                           ptd.rdata(ii.m_is2Vz)[pp]};
                    // evaluate the splines at particle position R
                    Gempic::ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(R, plo,
                                                                                      dxInverse);
                    amrex::GpuArray<amrex::Real, vdim> ep =
                        spline.template eval_spline_field<Forms::Field::PrimalOneForm>(eA);
                    // evaluate the magnetic field at the particle position
                    amrex::GpuArray<amrex::Real, vdim> bp =
                        spline.template eval_spline_field<Forms::Field::PrimalTwoForm>(bA);

                    for (int d = 0; d < AMREX_SPACEDIM; d++)
                    {
                        s2R[d] = coeffA * s2R[d] + dt * V[d];
                        R[d] += coeffB * s2R[d];
                    }

                    s2V[0] =
                        coeffA * s2V[0] + dt * chargeByMass * (ep[0] + V[1] * bp[2] - V[2] * bp[1]);
                    s2V[1] =
                        coeffA * s2V[1] + dt * chargeByMass * (ep[1] + V[2] * bp[0] - V[0] * bp[2]);
                    s2V[2] =
                        coeffA * s2V[2] + dt * chargeByMass * (ep[2] + V[0] * bp[1] - V[1] * bp[0]);
                    V[0] += coeffB * s2V[0];
                    V[1] += coeffB * s2V[1];
                    V[2] += coeffB * s2V[2];

                    // Write back particle position and velocity
                    AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp] = R[0], ptd.rdata(ii.m_iposy)[pp] = R[1],
                                 ptd.rdata(ii.m_iposz)[pp] = R[2]);
                    ptd.rdata(ii.m_ivelx)[pp] = V[0];
                    ptd.rdata(ii.m_ively)[pp] = V[1];
                    ptd.rdata(ii.m_ivelz)[pp] = V[2];
                    AMREX_D_DECL(ptd.rdata(ii.m_is2X)[pp] = s2R[0],
                                 ptd.rdata(ii.m_is2Y)[pp] = s2R[1],
                                 ptd.rdata(ii.m_is2Z)[pp] = s2R[2]);
                    ptd.rdata(ii.m_is2Vx)[pp] = s2V[0];
                    ptd.rdata(ii.m_is2Vy)[pp] = s2V[1];
                    ptd.rdata(ii.m_is2Vz)[pp] = s2V[2];
                }); // end of parallel_for
        } //end of loop over particles
        particleSpecies->Redistribute();
    }
    // Function for pushing particle velocities and positions ends

    // Function to update B from curlE starts
    void lsrk_update_b_from_curl_e (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge> const& E,
        amrex::Real dt,
        int stageIndex)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::lsrk_update_B_from_curlE()");
        amrex::Real const coeffA = m_lsrk.get_coeff_a(stageIndex);
        amrex::Real const coeffB = m_lsrk.get_coeff_b(stageIndex);

        // Component-0 of Faraday's equation
        for (amrex::MFIter mfi(B.m_data[xDir]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real> const& s2BxArray = (m_s2B.m_data[xDir])[mfi].array();
            AMREX_D_TERM(amrex::Array4<amrex::Real> const& bxArray = (B.m_data[xDir])[mfi].array();
                         , amrex::Array4<amrex::Real const> const& ezArray =
                               (E.m_data[zDir])[mfi].const_array();
                         , amrex::Array4<amrex::Real const> const& eyArray =
                               (E.m_data[yDir])[mfi].const_array();)
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            s2BxArray(i, j, k) =
                                coeffA * s2BxArray(i, j, k) -
                                dt * (GEMPIC_D_ADD(0., ezArray(i, j + 1, k) - ezArray(i, j, k),
                                                   -eyArray(i, j, k + 1) + eyArray(i, j, k)));
                            bxArray(i, j, k) += coeffB * s2BxArray(i, j, k);
                        });
        }
        B.m_data[xDir].FillBoundary_nowait(m_infra.geometry().periodicity());
        m_s2B.m_data[xDir].FillBoundary_nowait(m_infra.geometry().periodicity());

        // Component-1 of Faraday's equation
        for (amrex::MFIter mfi(B.m_data[yDir]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real> const& s2ByArray = (m_s2B.m_data[yDir])[mfi].array();
            amrex::Array4<amrex::Real> const& byArray = (B.m_data[yDir])[mfi].array();
            amrex::Array4<amrex::Real const> const& ezArray = (E.m_data[zDir])[mfi].const_array();
#if (AMREX_SPACEDIM > 2)
            amrex::Array4<amrex::Real const> const& exArray = (E.m_data[xDir])[mfi].const_array();
#endif
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            s2ByArray(i, j, k) =
                                coeffA * s2ByArray(i, j, k) -
                                dt * (GEMPIC_D_ADD(-ezArray(i + 1, j, k) + ezArray(i, j, k), 0.,
                                                   exArray(i, j, k + 1) - exArray(i, j, k)));
                            byArray(i, j, k) += coeffB * s2ByArray(i, j, k);
                        });
        }
        B.m_data[yDir].FillBoundary_nowait(m_infra.geometry().periodicity());
        m_s2B.m_data[yDir].FillBoundary_nowait(m_infra.geometry().periodicity());

        // Component-2 of Faraday's equation
        for (amrex::MFIter mfi(B.m_data[zDir]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();

            amrex::Array4<amrex::Real> const& s2BzArray = (m_s2B.m_data[zDir])[mfi].array();
            amrex::Array4<amrex::Real> const& bzArray = (B.m_data[zDir])[mfi].array();
#if (AMREX_SPACEDIM > 1)
            amrex::Array4<amrex::Real const> const& exArray = (E.m_data[xDir])[mfi].const_array();
#endif
            amrex::Array4<amrex::Real const> const& eyArray = (E.m_data[yDir])[mfi].const_array();

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            s2BzArray(i, j, k) =
                                coeffA * s2BzArray(i, j, k) -
                                dt * (GEMPIC_D_ADD(eyArray(i + 1, j, k) - eyArray(i, j, k),
                                                   -exArray(i, j + 1, k) + exArray(i, j, k), 0.));
                            bzArray(i, j, k) += coeffB * s2BzArray(i, j, k);
                        });
        }
        B.m_data[zDir].FillBoundary_nowait(m_infra.geometry().periodicity());
        m_s2B.m_data[zDir].FillBoundary_nowait(m_infra.geometry().periodicity());

        // Wait for completed communication of guard data
        B.m_data[xDir].FillBoundary_finish();
        B.m_data[yDir].FillBoundary_finish();
        B.m_data[zDir].FillBoundary_finish();
        m_s2B.m_data[xDir].FillBoundary_finish();
        m_s2B.m_data[yDir].FillBoundary_finish();
        m_s2B.m_data[zDir].FillBoundary_finish();
    }
    // Function to update B from curlE ends

    // Functions for Faraday equation Crank-Nicolson update start
    void de_ki_faraday_crank_nicolson_step (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& curlH,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& rhs,
        int bdir,
        std::array<int, 2> const& pdir,
        amrex::Real dt,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& rhoElectron)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::de_ki_faraday_crank_nicolson_step()");

        de_ki_calculate_rhs_faraday_crank_nicolson(rhs, B, H, curlH, bdir, pdir, dt, rhoElectron);
        de_ki_solve_faraday_crank_nicolson_b(B, rhs, bdir, dt, rhoElectron);
    }

    void de_ki_calculate_rhs_faraday_crank_nicolson (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& rhs,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>& H,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& curlH,
        int bdir,
        std::array<int, 2> const& pdir,
        amrex::Real dt,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& rhoElectron)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::de_ki_calculate_rhs_faraday_crank_nicolson()");

        // Calculate curlB on dual face
        hodge(H, B, m_deRham->scaling_bto_h());
        curl(curlH, H);

        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face> bextXCurlH(m_deRham);
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge> bextXCurlHPrimal(m_deRham);
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> curlBextXCurlHPrimal(m_deRham);

        auto dxi{m_infra.inv_cell_size_3darray()};

        Gempic::FieldSolvers::Impl::interpolate_from_rhs_and_swap_directions(bextXCurlH, curlH, dxi,
                                                                             pdir);
        bextXCurlH.m_data[bdir].setVal(0.0);
        bextXCurlH.m_data[pdir[0]].mult(-1.0);

        amrex::IntVect ng{rhs.m_deRham->get_n_ghost()};

        // Divide by rhoe
        std::array<int, 3> nc = {0, 0, 0};

        for (int n = 0; n < 2; n++)
        {
            for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
            {
                nc[dir] = (pdir[n] == dir);
            }

            for (amrex::MFIter mfi(rhs.m_data[pdir[n]]); mfi.isValid(); ++mfi)
            {
                amrex::Array4<amrex::Real> const& bextXCurlHArrPdir =
                    (bextXCurlH.m_data[pdir[n]])[mfi].array();
                amrex::Array4<amrex::Real const> const& rhoEArr = (rhoElectron.m_data)[mfi].array();

                amrex::Dim3 lo = lbound(bextXCurlHArrPdir);
                amrex::Dim3 hi = ubound(bextXCurlHArrPdir);

                GEMPIC_D_LOOP_BEGIN(
                    for (int j = lo.y + (ng[xDir] - 1); j <= hi.y - (ng[xDir] - 1); j++),
                    for (int i = lo.x + (ng[yDir] - 1); i <= hi.x - (ng[yDir] - 1); i++),
                    for (int k = lo.z + (ng[zDir] - 1); k <= hi.z - (ng[zDir] - 1); k++))

                    bextXCurlHArrPdir(AMREX_D_DECL(i, j, k), 0) *=
                        2.0 * m_infra.cell_volume() /
                        (rhoEArr(AMREX_D_DECL(i, j, k), 0) +
                         rhoEArr(AMREX_D_DECL(i + nc[xDir], j + nc[yDir], k + nc[zDir]), 0));

                GEMPIC_D_LOOP_END
            }
        }
        // Divide by rhoe done
        bextXCurlH.fill_boundary();

        hodge(bextXCurlHPrimal, bextXCurlH, m_deRham->scaling_dto_e());
        curl(curlBextXCurlHPrimal, bextXCurlHPrimal);

        // RHS = B - dt*curl(bext x curl(Bn/2))
        linear_combination(rhs, 1.0, B, -0.5 * dt, curlBextXCurlHPrimal);
    }

    void de_ki_solve_faraday_crank_nicolson_b (
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
        Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face> const& rhs,
        int bdir,
        amrex::Real dt,
        Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& rhoElectron)
    {
        Kokkos::Profiling::ScopedRegion region(
            "GEMPIC::DKQuasineutralSolver::de_ki_solve_faraday_crank_nicolson_b()");

        auto const& nddom = amrex::surroundingNodes(m_infra.box());

        auto const& rhoearrs = (rhoElectron.m_data).arrays();

        auto dxi{m_infra.inv_cell_size_3darray()};
        auto dx{m_infra.cell_size_3darray()};

        amrex::GpuArray<amrex::Real, 3> edgetoface{
            static_cast<amrex::Real>(GEMPIC_D_MULT(dxi[xDir], dx[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dxi[yDir], dx[zDir])),
            static_cast<amrex::Real>(GEMPIC_D_MULT(dx[xDir], dx[yDir], dxi[zDir]))};

        amrex::GpuArray<HYPRE_Real, 3> bext{0.0, 0.0, 0.0};
        bext[bdir] = 1.0;

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

            // The two filler functions called below can be called in any order
            filler_de_ki_faraday_crank_nicolson_matrix(boxno, i, j, k, ndir, gid, ncols, cols, mat,
                                                       rhoearrs, dx, edgetoface, bext, dt, mapCols);
        };

        // this is the size of the linear system as calculated from the stencil
        constexpr int maxStencilSize = 33;

        Kokkos::Profiling::ProfilingSection p10(
            "GEMPIC::DKQuasineutralSolver::CreateHypre::CrankNicolson()");
        p10.start();
        amrex::Print() << " CreateHypre::CrankNicolson " << std::endl;
        amrex::HypreSolver<maxStencilSize> hypreSolver(
            {B.m_data[xDir].boxArray().ixType(), B.m_data[yDir].boxArray().ixType(),
             B.m_data[zDir].boxArray().ixType()},
            rhs.m_deRham->get_n_ghost(), m_infra.geometry(), m_infra.m_grid, m_infra.m_distriMap,
            marker, filler, m_verbose, "faraday_crank_nicolson");
        p10.stop();

        Kokkos::Profiling::ProfilingSection p11(
            "GEMPIC::DKQuasineutralSolver::SolveHypre::CrankNicolson()");
        p11.start();
        amrex::Print() << " SolveHypre::CrankNicolson " << std::endl;
        hypreSolver.solve(amrex::Vector<amrex::MultiFab*>{&(B.m_data[xDir]), &(B.m_data[yDir]),
                                                          &(B.m_data[zDir])},
                          amrex::Vector<amrex::MultiFab const*>{
                              &(rhs.m_data[xDir]), &(rhs.m_data[yDir]), &(rhs.m_data[zDir])},
                          m_relTol, m_absTol, m_maxiter);
        p11.stop();
        Gempic::Utils::check_solver_convergence(hypreSolver, m_maxiter, m_accTol);

        B.fill_boundary();
    }

    // This function updates the B-field using the Faraday equation
    // dB/dt + curl(E) = 0. The implicit Crank Nicolson time-stepping
    // scheme is used. The E-field is written in terms of B using the
    // drift-kinetic approximation.
    void filler_de_ki_faraday_crank_nicolson_matrix (
        int boxno,
        int i,
        int j,
        int k,
        int ndir,
        amrex::Array4<HYPRE_Int const> const* gid,
        HYPRE_Int& ncols,
        HYPRE_Int* cols,
        HYPRE_Real* mat,
        amrex::MultiArray4<amrex::Real const> const& rhoearrs,
        amrex::GpuArray<amrex::Real, 3> dx,
        amrex::GpuArray<HYPRE_Real, 3> const edgetoface,
        amrex::GpuArray<HYPRE_Real, 3> const bext,
        amrex::Real dt,
        std::map<int, int>& mapCols)
    {
        std::array<int, 3> n = {0, 0, 0};
        std::array<int, 3> in = {0, 0, 0};
        std::array<int, 3> jn = {0, 0, 0};
        std::array<int, 3> kn = {0, 0, 0};

        for (int dir = 0; dir < 3; dir++)
        {
            n[dir] = (ndir + dir) % 3;
            AMREX_D_EXPR(in[dir] = (n[dir] == 0), jn[dir] = (n[dir] == 1), kn[dir] = (n[dir] == 2));
        }

        amrex::Real rhoe00 = rhoearrs[boxno](i, j, k);
        amrex::Real rhoe01 = rhoearrs[boxno](i + in[2], j + jn[2], k + kn[2]);
        amrex::Real rhoe10 = rhoearrs[boxno](i + in[1], j + jn[1], k + kn[1]);
        amrex::Real rhoe11 =
            rhoearrs[boxno](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
        amrex::Real rhoe0001 = rhoe00 + rhoe01;
        amrex::Real rhoe0010 = rhoe00 + rhoe10;
        amrex::Real rhoe1011 = rhoe10 + rhoe11;
        amrex::Real rhoe0111 = rhoe01 + rhoe11;

        int gdtmp = gid[n[0]](i, j, k);
        int icol;

        Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
        mat[icol] += 1.0;

        if (bext[n[0]] != 0)
        {
            // (I) b[n[0]] * d2B[n[1]]/d[n[0]]d[n[2]] * dt/2
            amrex::Real coeff = dt * m_infra.cell_volume() / (4.0 * dx[n[0]] * edgetoface[n[1]]);

            gdtmp = gid[n[1]](i - in[2], j - jn[2], k - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0010);
            gdtmp = gid[n[1]](i + in[1] - in[2], j + jn[1] - jn[2], k + kn[1] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0010);
            gdtmp = gid[n[1]](i - in[0] + in[2], j - jn[0] + jn[2], k - kn[0] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0111);
            gdtmp = gid[n[1]](i - in[0] + in[1] + in[2], j - jn[0] + jn[1] + jn[2],
                              k - kn[0] + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0111);
            gdtmp = gid[n[1]](i - in[0] - in[2], j - jn[0] - jn[2], k - kn[0] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
            gdtmp = gid[n[1]](i - in[0] + in[1] - in[2], j - jn[0] + jn[1] - jn[2],
                              k - kn[0] + kn[1] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
            gdtmp = gid[n[1]](i + in[2], j + jn[2], k + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);
            gdtmp = gid[n[1]](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);

            // (II) -b[n[0]] * d2B[n[2]]/d[n[0]]d[n[1]] * dt/2
            coeff = dt * m_infra.cell_volume() / (4.0 * dx[n[0]] * edgetoface[n[2]]);

            gdtmp = gid[n[2]](i - in[1], j - jn[1], k - kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0001);
            gdtmp = gid[n[2]](i - in[1] + in[2], j - jn[1] + jn[2], k - kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0001);
            gdtmp = gid[n[2]](i - in[0] + in[1], j - jn[0] + jn[1], k - kn[0] + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe1011);
            gdtmp = gid[n[2]](i - in[0] + in[1] + in[2], j - jn[0] + jn[1] + jn[2],
                              k - kn[0] + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe1011);
            gdtmp = gid[n[2]](i - in[0] - in[1], j - jn[0] - jn[1], k - kn[0] - kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
            gdtmp = gid[n[2]](i - in[0] - in[1] + in[2], j - jn[0] - jn[1] + jn[2],
                              k - kn[0] - kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
            gdtmp = gid[n[2]](i + in[1], j + jn[1], k + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
            gdtmp = gid[n[2]](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
        }
        else if (bext[n[1]] != 0)
        {
            // (III) b[n[1]] * d2B[n[1]]/d[n[1]]d[n[2]] * dt/2
            amrex::Real coeff = dt * m_infra.cell_volume() / (4.0 * dx[n[1]] * edgetoface[n[1]]);

            gdtmp = gid[n[1]](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe1011);
            gdtmp = gid[n[1]](i - in[0] + in[1] + in[2], j - jn[0] + jn[1] + jn[2],
                              k - kn[0] + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe1011);
            gdtmp = gid[n[1]](i - in[2], j - jn[2], k - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0001);
            gdtmp = gid[n[1]](i - in[0] - in[2], j - jn[0] - jn[2], k - kn[0] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0001);
            gdtmp = gid[n[1]](i + in[1] - in[2], j + jn[1] - jn[2], k + kn[1] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
            gdtmp = gid[n[1]](i - in[0] + in[1] - in[2], j - jn[0] + jn[1] - jn[2],
                              k - kn[0] + kn[1] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
            gdtmp = gid[n[1]](i + in[2], j + jn[2], k + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
            gdtmp = gid[n[1]](i - in[0] + in[2], j - jn[0] + jn[2], k - kn[0] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);

            // (IV) -bn[1] * d2Bn[2]/dn[1]2 * dt/2
            coeff = dt * m_infra.cell_volume() / (4.0 * dx[n[1]] * edgetoface[n[2]]);

            gdtmp = gid[n[2]](i, j, k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff * (1.0 / rhoe1011 + 1.0 / rhoe0001));
            gdtmp = gid[n[2]](i + in[2], j + jn[2], k + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff * (1.0 / rhoe1011 + 1.0 / rhoe0001));
            gdtmp = gid[n[2]](i - in[0], j - jn[0], k - kn[0]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff * (1.0 / rhoe1011 + 1.0 / rhoe0001));
            gdtmp = gid[n[2]](i - in[0] + in[2], j - jn[0] + jn[2], k - kn[0] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff * (1.0 / rhoe1011 + 1.0 / rhoe0001));

            gdtmp = gid[n[2]](i + in[1], j + jn[1], k + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
            gdtmp = gid[n[2]](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
            gdtmp = gid[n[2]](i + in[1] - in[0], j - jn[0] + jn[1], k - kn[0] + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);
            gdtmp = gid[n[2]](i + in[1] - in[0] + in[2], j - jn[0] + jn[1] + jn[2],
                              k - kn[0] + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe1011);

            gdtmp = gid[n[2]](i - in[1], j - jn[1], k - kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
            gdtmp = gid[n[2]](i - in[1] + in[2], j - jn[1] + jn[2], k - kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
            gdtmp = gid[n[2]](i - in[0] - in[1], j - jn[0] - jn[1], k - kn[0] - kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
            gdtmp = gid[n[2]](i - in[0] - in[1] + in[2], j - jn[0] - jn[1] + jn[2],
                              k - kn[0] - kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0001);
        }
        else if (bext[n[2]] != 0)
        {
            // (V) -b[n[2]] * d2B[n[2]]/d[n[1]]d[n[2]] * dt/2
            amrex::Real coeff = dt * m_infra.cell_volume() / (4.0 * dx[n[2]] * edgetoface[n[2]]);

            gdtmp = gid[n[2]](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0111);
            gdtmp = gid[n[2]](i - in[0] + in[1] + in[2], j - jn[0] + jn[1] + jn[2],
                              k - kn[0] + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0111);
            gdtmp = gid[n[2]](i - in[1], j - jn[1], k - kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0010);
            gdtmp = gid[n[2]](i - in[0] - in[1], j - jn[0] - jn[1], k - kn[0] - kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff / rhoe0010);
            gdtmp = gid[n[2]](i - in[1] + in[2], j - jn[1] + jn[2], k - kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);
            gdtmp = gid[n[2]](i - in[0] - in[1] + in[2], j - jn[0] - jn[1] + jn[2],
                              k - kn[0] - kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);
            gdtmp = gid[n[2]](i + in[1], j + jn[1], k + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
            gdtmp = gid[n[2]](i - in[0] + in[1], j - jn[0] + jn[1], k - kn[0] + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);

            // (VI) bn[2] * d2Bn[1]/dn[2]2 * dt/2
            coeff = dt * m_infra.cell_volume() / (4.0 * dx[n[2]] * edgetoface[n[1]]);

            gdtmp = gid[n[1]](i, j, k);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff * (1.0 / rhoe0111 + 1.0 / rhoe0010));
            gdtmp = gid[n[1]](i + in[1], j + jn[1], k + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff * (1.0 / rhoe0111 + 1.0 / rhoe0010));
            gdtmp = gid[n[1]](i - in[0], j - jn[0], k - kn[0]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff * (1.0 / rhoe0111 + 1.0 / rhoe0010));
            gdtmp = gid[n[1]](i + in[1] - in[0], j - jn[0] + jn[1], k - kn[0] + kn[1]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (-coeff * (1.0 / rhoe0111 + 1.0 / rhoe0010));

            gdtmp = gid[n[1]](i + in[2], j + jn[2], k + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);
            gdtmp = gid[n[1]](i + in[1] + in[2], j + jn[1] + jn[2], k + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);
            gdtmp = gid[n[1]](i - in[0] + in[2], j - jn[0] + jn[2], k - kn[0] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);
            gdtmp = gid[n[1]](i + in[1] - in[0] + in[2], j - jn[0] + jn[1] + jn[2],
                              k - kn[0] + kn[1] + kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0111);

            gdtmp = gid[n[1]](i - in[2], j - jn[2], k - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
            gdtmp = gid[n[1]](i + in[1] - in[2], j + jn[1] - jn[2], k + kn[1] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
            gdtmp = gid[n[1]](i - in[0] - in[2], j - jn[0] - jn[2], k - kn[0] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
            gdtmp = gid[n[1]](i + in[1] - in[0] - in[2], j - jn[0] + jn[1] - jn[2],
                              k - kn[0] + kn[1] - kn[2]);
            Gempic::Utils::assign_or_add_col(gdtmp, icol, ncols, cols, mat, mapCols);
            mat[icol] += (coeff / rhoe0010);
        }
    }
    // Functions for Faraday equation Crank-Nicolson update end
};

} // namespace Gempic::FieldSolvers

#endif

/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_VLASOVMAXWELLSEMIIMPLICITOPERATORS_H_
#define GEMPIC_VLASOVMAXWELLSEMIIMPLICITOPERATORS_H_

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_VlasovMaxwellEV.H"

namespace Gempic::TimeLoop
{

/**
 * @brief update all particle positions with their velocities.
 *
 * @param[out] particlesgroup Particle container.
 * @param dt The time step size.
 */

template <int degX, int degY, int degZ, unsigned int vdim, unsigned int nData>
void push_all_x_using_v_gdec (
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim, nData>>> const&
        particlesgroup,
    amrex::Real const dt)
{
    BL_PROFILE(
        "Gempic::TimeLoop::GEMPIC_VlasovMaxwellSemiImplicitOperators::push_all_x_using_v_gdec()");

    for (auto const& particleSpecies : particlesgroup)
    {
        for (auto& particleGrid : *particleSpecies)
        {
            long const np = particleGrid.numParticles();
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::ParallelFor(np,
                               [=] AMREX_GPU_DEVICE(long pp)
                               {
                                   // update particle position
                                   AMREX_D_EXPR(
                                       ptd.rdata(ii.m_iposx)[pp] += dt * ptd.rdata(ii.m_ivelx)[pp],
                                       ptd.rdata(ii.m_iposy)[pp] += dt * ptd.rdata(ii.m_ively)[pp],
                                       ptd.rdata(ii.m_iposz)[pp] += dt * ptd.rdata(ii.m_ivelz)[pp]);
                               });
        }
        particleSpecies->Redistribute();
    }
}

/**
 * @brief Processing of the non-vanishing basis function values and their corresponding indices,
 * used for computing the non-zero components of the particle mass matrix.
 *
 * @param[out] eWts The non-vanishing basis functions' values at the particle position in a 1D array
 * @param[out] eCells The global indices of the non-vanishing basis functions.
 * @param cellParticle The index of the cell where the particle is located.
 * @param plo The left boundary of the domain.
 * @param dx The cell size array.
 * @param basisEdgeX The non-vanishing basis values of the x component of the primal 1 form.
 * @param basisEdgeY The non-vanishing basis values of the y component of the primal 1 form.
 * @param basisEdgeZ The non-vanishing basis values of the z component of the primal 1 form.
 */

template <int hodgeDegree, unsigned int nWtsSum>
AMREX_GPU_HOST_DEVICE void eval_e_field_gdec_weights (
    amrex::GpuArray<amrex::Real, nWtsSum>& eWts,
    amrex::GpuArray<amrex::IntVect, nWtsSum>& eCells,
    amrex::IntVect& cellParticle,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0, hodgeDegree - 1>&
        basisEdgeX,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0, hodgeDegree - 1>&
        basisEdgeY,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 1, 0, hodgeDegree - 2>&
        basisEdgeZ)
{
    BL_PROFILE(
        "Gempic::TimeLoop::GEMPIC_VlasovMaxwellSemiImplicitOperators::eval_e_field_gdec_weights()");

    constexpr std::array<std::array<int, 3>, 3> nGDECWtsXYZ = {{
        {{(hodgeDegree - 1), hodgeDegree,
          hodgeDegree}}, // number of 1-form x,y,z-field weights in X-direction
        {{hodgeDegree, (hodgeDegree - 1),
          hodgeDegree}}, // number of 1-form x,y,z-field weights in Y-direction
        {{hodgeDegree, hodgeDegree,
          (hodgeDegree - 1)}} // number of 1-form x,y,z-field weights in Z-direction
    }};

    constexpr std::array<int, 3> nWts = {
        // number of x-field weights
        GEMPIC_D_MULT(nGDECWtsXYZ[xDir][xDir], nGDECWtsXYZ[yDir][xDir], nGDECWtsXYZ[zDir][xDir]),
        // number of y-field weights
        GEMPIC_D_MULT(nGDECWtsXYZ[xDir][yDir], nGDECWtsXYZ[yDir][yDir], nGDECWtsXYZ[zDir][yDir]),
        // number of z-field weights
        GEMPIC_D_MULT(nGDECWtsXYZ[xDir][zDir], nGDECWtsXYZ[yDir][zDir], nGDECWtsXYZ[zDir][zDir])};

    constexpr std::array<int, 3> nWtsStart = {0, nWts[xDir], (nWts[xDir] + nWts[yDir])};

    GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

    // Weights for E at particle location
    for (int ndir = 0; ndir < 3; ++ndir)
    {
        GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nGDECWtsXYZ[xDir][ndir]; ++ii),
                            for (int jj = 0; jj < nGDECWtsXYZ[yDir][ndir]; ++jj),
                            for (int kk = 0; kk < nGDECWtsXYZ[zDir][ndir]; ++kk))

            int AMREX_D_DECL(a, b, c);

            AMREX_D_EXPR(a = ii + 1 - hodgeDegree / 2, b = jj + 1 - hodgeDegree / 2,
                         c = kk + 1 - hodgeDegree / 2);

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr;
            rr = {AMREX_D_DECL(plo[xDir] + cellParticle[xDir] * dx[xDir],
                               plo[yDir] + cellParticle[yDir] * dx[yDir],
                               plo[zDir] + cellParticle[zDir] * dx[zDir])};

            amrex::Real const sCoeff =
                ((ndir == 0) ? (basisEdgeX(ii, jj, kk))
                             : ((ndir == 1) ? (basisEdgeY(ii, jj, kk)) : (basisEdgeZ(ii, jj, kk))));

            eWts[nWtsStart[ndir] + kk * nGDECWtsXYZ[yDir][ndir] * nGDECWtsXYZ[xDir][ndir] +
                 jj * nGDECWtsXYZ[xDir][ndir] + ii] = sCoeff;
            eCells[nWtsStart[ndir] + kk * nGDECWtsXYZ[yDir][ndir] * nGDECWtsXYZ[xDir][ndir] +
                   jj * nGDECWtsXYZ[xDir][ndir] + ii] =
                amrex::IntVect{AMREX_D_DECL(cellParticle[xDir] + a, cellParticle[yDir] + b,
                                            cellParticle[zDir] + c)};
        GEMPIC_D_LOOP_END
    }
}

/**
 * @brief Deposition of the particle mass matrix.
 *
 * @param[out] eGDECCoeffsA The array of particle mass matrix.
 * @param cellParticle The index of the cell where the particle is located.
 * @param plo The left boundary of the domain.
 * @param dx The cell size array.
 * @param dt The time step size.
 * @param chargeSqrByMassWeight Charge square times particle weight and mass.
 * @param scalingDF Scaling factors.
 * @param basisEdgeX The non-vanishing basis values of the x component of the primal 1 form.
 * @param basisEdgeY The non-vanishing basis values of the y component of the primal 1 form.
 * @param basisEdgeZ The non-vanishing basis values of the z component of the primal 1 form.
 */

template <int hodgeDegree, int vdim>
void get_gdec_rho_e_weights_at_particle_positions_and_deposit_to_face (
    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& eGDECCoeffsA,
    amrex::IntVect& cellParticle,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
    amrex::Real dt,
    amrex::Real chargeSqrByMassWeight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> scalingDF,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0, hodgeDegree - 1>&
        basisEdgeX,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0, hodgeDegree - 1>&
        basisEdgeY,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 1, 0, hodgeDegree - 2>&
        basisEdgeZ)
{
    BL_PROFILE(
        "Gempic::TimeLoop::GEMPIC_VlasovMaxwellSemiImplicitOperators::get_gdec_rho_e_weights_at_"
        "particle_positions_and_deposit_to_face()");
    constexpr std::array<std::array<int, 3>, 3> nGDECWtsXYZ = {{
        {{(hodgeDegree - 1), hodgeDegree,
          hodgeDegree}}, // number of 1-form x,y,z-field weights in X-direction
        {{hodgeDegree, (hodgeDegree - 1),
          hodgeDegree}}, // number of 1-form x,y,z-field weights in Y-direction
        {{hodgeDegree, hodgeDegree,
          (hodgeDegree - 1)}} // number of 1-form x,y,z-field weights in Z-direction
    }};

    // Stencil offsets and bandwidths for the 'rho' matrix
    static constexpr std::array<int, 3> sSGdecOffx{
        (hodgeDegree - 2), (hodgeDegree - 1),
        (hodgeDegree - 1)}; // x,y,z-field offsets in x-direction
    static constexpr std::array<int, 3> sSGdecOffy{
        (hodgeDegree - 1), (hodgeDegree - 2),
        (hodgeDegree - 1)}; // x,y,z-field offsets in y-direction
    static constexpr std::array<int, 3> sSGdecOffz{
        (hodgeDegree - 1), (hodgeDegree - 1),
        (hodgeDegree - 2)}; // x,y,z-field offsets in z-direction

    static constexpr std::array<int, 3> sSGdecBWx{
        (2 * sSGdecOffx[xDir] + 1), (2 * sSGdecOffx[yDir] + 1),
        (2 * sSGdecOffx[zDir] + 1)}; // x,y,z-field bandwidths in x-direction
    static constexpr std::array<int, 3> sSGdecBWy{
        (2 * sSGdecOffy[xDir] + 1), (2 * sSGdecOffy[yDir] + 1),
        (2 * sSGdecOffy[zDir] + 1)}; // x,y,z-field bandwidths in y-direction

    constexpr std::array<int, 3> nWts = {
        // number of 1-form x-field weights
        GEMPIC_D_MULT(nGDECWtsXYZ[xDir][xDir], nGDECWtsXYZ[yDir][xDir], nGDECWtsXYZ[zDir][xDir]),
        // number of 1-form y-field weights
        GEMPIC_D_MULT(nGDECWtsXYZ[xDir][yDir], nGDECWtsXYZ[yDir][yDir], nGDECWtsXYZ[zDir][yDir]),
        // number of 1-form z-field weights
        GEMPIC_D_MULT(nGDECWtsXYZ[xDir][zDir], nGDECWtsXYZ[yDir][zDir], nGDECWtsXYZ[zDir][zDir])};
    constexpr std::array<int, 3> nWtsStart = {0, nWts[xDir], (nWts[xDir] + nWts[yDir])};

    amrex::GpuArray<amrex::Real, (nWts[xDir] + nWts[yDir] + nWts[zDir])> eWts;
    amrex::GpuArray<amrex::IntVect, (nWts[xDir] + nWts[yDir] + nWts[zDir])> eEdges;

    eval_e_field_gdec_weights<hodgeDegree>(eWts, eEdges, cellParticle, plo, dx, basisEdgeX,
                                           basisEdgeY, basisEdgeZ);

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> scaling{
        AMREX_D_DECL(dx[yDir] * dx[zDir] / dx[xDir], dx[xDir] * dx[zDir] / dx[yDir],
                     dx[xDir] * dx[yDir] / dx[zDir])};

    GEMPIC_D_EXCL(int jj{0};, int kk{0};, )

    int AMREX_D_DECL(iiTemp, jjTemp, kkTemp);

    for (int ndir = 0; ndir < vdim; ++ndir)
    {
        GEMPIC_D_LOOP_BEGIN(for (int ii = 0; ii < nGDECWtsXYZ[xDir][ndir]; ++ii),
                            for (int jj = 0; jj < nGDECWtsXYZ[yDir][ndir]; ++jj),
                            for (int kk = 0; kk < nGDECWtsXYZ[zDir][ndir]; ++kk))

            int AMREX_D_DECL(a, b, c);

            AMREX_D_EXPR(a = ii + 1 - hodgeDegree / 2, b = jj + 1 - hodgeDegree / 2,
                         c = kk + 1 - hodgeDegree / 2);

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> rr;
            rr = {AMREX_D_DECL(plo[xDir] + cellParticle[xDir] * dx[xDir],
                               plo[yDir] + cellParticle[yDir] * dx[yDir],
                               plo[zDir] + cellParticle[zDir] * dx[zDir])};

            amrex::Real const sCoeff =
                ((ndir == 0) ? (basisEdgeX(ii, jj, kk))
                             : ((ndir == 1) ? (basisEdgeY(ii, jj, kk)) : (basisEdgeZ(ii, jj, kk))));

            amrex::Dim3 edge = {GEMPIC_D_PAD(cellParticle[xDir] + a, cellParticle[yDir] + b,
                                             cellParticle[zDir] + c)};

            for (int i = 0; i < nWts[ndir]; i++)
            {
                // this index is of primal edge contributing to E-field at particle
                amrex::Dim3 wEdge = (eEdges[nWtsStart[ndir] + i]).dim3();

                AMREX_D_EXPR(iiTemp = wEdge.x - edge.x + sSGdecOffx[ndir],
                             jjTemp = wEdge.y - edge.y + sSGdecOffy[ndir],
                             kkTemp = wEdge.z - edge.z + sSGdecOffz[ndir]);

                int component = GEMPIC_D_ADD(iiTemp, jjTemp * sSGdecBWx[ndir],
                                             kkTemp * sSGdecBWx[ndir] * sSGdecBWy[ndir]);

                amrex::HostDevice::Atomic::Add(
                    eGDECCoeffsA[ndir].ptr(edge.x, edge.y, edge.z, component),
                    0.25 * dt * dt * sCoeff * chargeSqrByMassWeight * eWts[nWtsStart[ndir] + i] *
                        scalingDF[ndir] / scaling[ndir]);
            }
        GEMPIC_D_LOOP_END
    }
}

/**
 * @brief Deposition of current J and estar with GDEC Basis values at particle positions.
 *
 * @param[out] jA The array of J.
 * @param[out] esA The array of estar.
 * @param cellParticle The index of the cell where the particle is located.
 * @param vel Particle velocity.
 * @param eField The value of the electric field at particle position.
 * @param dt The time step size.
 * @param chargeWeight Charge times particle weight.
 * @param chargeOverMass Charge divided by mass.
 * @param basisEdgeX The non-vanishing basis values of the x component of the primal 1 form.
 * @param basisEdgeY The non-vanishing basis values of the y component of the primal 1 form.
 * @param basisEdgeZ The non-vanishing basis values of the z component of the primal 1 form.
 */

template <int hodgeDegree>
void deposit_j_and_estar (
    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& jA,
    amrex::GpuArray<amrex::Array4<amrex::Real>, 3> const& esA,
    amrex::IntVect& cellParticle,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> eField,
    amrex::Real dt,
    amrex::Real chargeWeight,
    amrex::Real chargeOverMass,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0, hodgeDegree - 1>&
        basisEdgeX,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0, hodgeDegree - 1>&
        basisEdgeY,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 1, 0, hodgeDegree - 2>&
        basisEdgeZ)
{
    BL_PROFILE(
        "Gempic::TimeLoop::GEMPIC_VlasovMaxwellSemiImplicitOperators::deposit_j_and_estar()");

    // deposition of J
    GEMPIC_D_EXCL(int b{0};, int c{0};, )

    GEMPIC_D_LOOP_BEGIN(for (int a = 1 - hodgeDegree / 2; a < hodgeDegree / 2; ++a),
                        for (int b = 1 - hodgeDegree / 2; b < hodgeDegree / 2 + 1; ++b),
                        for (int c = 1 - hodgeDegree / 2; c < hodgeDegree / 2 + 1; ++c))

        // x component
        jA[xDir](cellParticle[xDir] + a, cellParticle[yDir] + b, cellParticle[zDir] + c) +=
            dt * chargeWeight * vel[xDir] *
            basisEdgeX(a - 1 + hodgeDegree / 2, b - 1 + hodgeDegree / 2, c - 1 + hodgeDegree / 2);

        esA[xDir](cellParticle[xDir] + a, cellParticle[yDir] + b, cellParticle[zDir] + c) -=
            0.25 * dt * dt * chargeOverMass * chargeWeight * eField[xDir] *
            basisEdgeX(a - 1 + hodgeDegree / 2, b - 1 + hodgeDegree / 2, c - 1 + hodgeDegree / 2);

    GEMPIC_D_LOOP_END

    // y component
    GEMPIC_D_LOOP_BEGIN(for (int a = 1 - hodgeDegree / 2; a < hodgeDegree / 2 + 1; ++a),
                        for (int b = 1 - hodgeDegree / 2; b < hodgeDegree / 2; ++b),
                        for (int c = 1 - hodgeDegree / 2; c < hodgeDegree / 2 + 1; ++c))

        jA[yDir](cellParticle[xDir] + a, cellParticle[yDir] + b, cellParticle[zDir] + c) +=
            dt * chargeWeight * vel[yDir] *
            basisEdgeY(a - 1 + hodgeDegree / 2, b - 1 + hodgeDegree / 2, c - 1 + hodgeDegree / 2);

        esA[yDir](cellParticle[xDir] + a, cellParticle[yDir] + b, cellParticle[zDir] + c) -=
            0.25 * dt * dt * chargeOverMass * chargeWeight * eField[yDir] *
            basisEdgeY(a - 1 + hodgeDegree / 2, b - 1 + hodgeDegree / 2, c - 1 + hodgeDegree / 2);
    GEMPIC_D_LOOP_END
    // z component
    GEMPIC_D_LOOP_BEGIN(for (int a = 1 - hodgeDegree / 2; a < hodgeDegree / 2 + 1; ++a),
                        for (int b = 1 - hodgeDegree / 2; b < hodgeDegree / 2 + 1; ++b),
                        for (int c = 1 - hodgeDegree / 2; c < hodgeDegree / 2; ++c))

        jA[zDir](cellParticle[xDir] + a, cellParticle[yDir] + b, cellParticle[zDir] + c) +=
            dt * chargeWeight * vel[zDir] *
            basisEdgeZ(a - 1 + hodgeDegree / 2, b - 1 + hodgeDegree / 2, c - 1 + hodgeDegree / 2);
        esA[zDir](cellParticle[xDir] + a, cellParticle[yDir] + b, cellParticle[zDir] + c) -=
            0.25 * dt * dt * chargeOverMass * chargeWeight * eField[zDir] *
            basisEdgeZ(a - 1 + hodgeDegree / 2, b - 1 + hodgeDegree / 2, c - 1 + hodgeDegree / 2);
    GEMPIC_D_LOOP_END
}

/**
 * @brief GDEC Basis evaluations at particle positions.
 *
 * @param[out] basisEdgeX The non-vanishing basis values of the x component of the primal 1 form.
 * @param[out] basisEdgeY The non-vanishing basis values of the y component of the primal 1 form.
 * @param[out] basisEdgeZ The non-vanishing basis values of the z component of the primal 1 form.
 * @param[out] basisFaceX The non-vanishing basis values of the x component of the primal 2 form.
 * @param[out] basisFaceY The non-vanishing basis values of the y component of the primal 2 form.
 * @param[out] basisFaceZ The non-vanishing basis values of the z component of the primal 2 form.
 * @param[out] basisNX The non-vanishing basis values of the GDEC-N basis about x.
 * @param[out] basisDX The non-vanishing basis values of the GDEC-D basis about x.
 * @param[out] basisNY The non-vanishing basis values of the GDEC-N basis about y.
 * @param[out] basisDY The non-vanishing basis values of the GDEC-D basis about y.
 * @param[out] basisNZ The non-vanishing basis values of the GDEC-N basis about z.
 * @param[out] basisDZ The non-vanishing basis values of the GDEC-D basis about z.
 * @param cellParticle The index of the primal cell where the particle is located.
 * @param pos Particle position.
 * @param plo Left boundary of the domain.
 * @param dx Cell size array.
 */
template <int hodgeDegree>
void basis_eval_at_position_gdec (
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0, hodgeDegree - 1>&
        basisEdgeX,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0, hodgeDegree - 1>&
        basisEdgeY,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 1, 0, hodgeDegree - 2>&
        basisEdgeZ,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0, hodgeDegree - 2>&
        basisFaceX,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0, hodgeDegree - 2>&
        basisFaceY,
    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 2, 0, hodgeDegree - 1>&
        basisFaceZ,
    amrex::GpuArray<amrex::Real, hodgeDegree>& basisNX,
    amrex::GpuArray<amrex::Real, hodgeDegree - 1>& basisDX,
    amrex::GpuArray<amrex::Real, hodgeDegree>& basisNY,
    amrex::GpuArray<amrex::Real, hodgeDegree - 1>& basisDY,
    amrex::GpuArray<amrex::Real, hodgeDegree>& basisNZ,
    amrex::GpuArray<amrex::Real, hodgeDegree - 1>& basisDZ,
    amrex::IntVect& cellParticle,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{
    BL_PROFILE(
        "Gempic::TimeLoop::GEMPIC_VlasovMaxwellSemiImplicitOperators::basis_eval_at_position_gdec("
        ")");
    using namespace Forms;

    using PrimalEdgeField = DeRhamField<Grid::primal, Space::edge>;

    for (int ii = 0; ii < hodgeDegree; ++ii)
    {
        int a = ii + 1 - hodgeDegree / 2;

        amrex::Real rr = plo[xDir] + cellParticle[xDir] * dx[xDir];

        basisNX[ii] =
            choose_eval<hodgeDegree, PrimalEdgeField, xDir, yDir>(a, rr, dx[xDir], pos[xDir]);
        if (ii < (hodgeDegree - 1))
        {
            basisDX[ii] =
                choose_eval<hodgeDegree, PrimalEdgeField, xDir, xDir>(a, rr, dx[xDir], pos[xDir]);
        }
    }

    for (int jj = 0; jj < hodgeDegree; ++jj)
    {
        int b = jj + 1 - hodgeDegree / 2;

        amrex::Real rr = plo[yDir] + cellParticle[yDir] * dx[yDir];

        basisNY[jj] =
            choose_eval<hodgeDegree, PrimalEdgeField, yDir, xDir>(b, rr, dx[yDir], pos[yDir]);
        if (jj < (hodgeDegree - 1))
        {
            basisDY[jj] =
                choose_eval<hodgeDegree, PrimalEdgeField, yDir, yDir>(b, rr, dx[yDir], pos[yDir]);
        }
    }

    for (int kk = 0; kk < hodgeDegree; ++kk)
    {
        int c = kk + 1 - hodgeDegree / 2;

        amrex::Real rr = plo[zDir] + cellParticle[zDir] * dx[zDir];

        basisNZ[kk] =
            choose_eval<hodgeDegree, PrimalEdgeField, zDir, xDir>(c, rr, dx[zDir], pos[zDir]);
        if (kk < (hodgeDegree - 1))
        {
            basisDZ[kk] =
                choose_eval<hodgeDegree, PrimalEdgeField, zDir, zDir>(c, rr, dx[zDir], pos[zDir]);
        }
    }

    GEMPIC_D_LOOP_BEGIN(for (int a = 0; a < hodgeDegree - 1; ++a),
                        for (int b = 0; b < hodgeDegree; ++b),
                        for (int c = 0; c < hodgeDegree; ++c))

        basisEdgeX(a, b, c) = basisDX[a] * basisNY[b] * basisNZ[c];
    GEMPIC_D_LOOP_END

    GEMPIC_D_LOOP_BEGIN(for (int a = 0; a < hodgeDegree; ++a),
                        for (int b = 0; b < hodgeDegree - 1; ++b),
                        for (int c = 0; c < hodgeDegree; ++c))

        basisEdgeY(a, b, c) = basisNX[a] * basisDY[b] * basisNZ[c];
    GEMPIC_D_LOOP_END

    GEMPIC_D_LOOP_BEGIN(for (int a = 0; a < hodgeDegree; ++a),
                        for (int b = 0; b < hodgeDegree; ++b),
                        for (int c = 0; c < hodgeDegree - 1; ++c))

        basisEdgeZ(a, b, c) = basisNX[a] * basisNY[b] * basisDZ[c];
    GEMPIC_D_LOOP_END

    GEMPIC_D_LOOP_BEGIN(for (int a = 0; a < hodgeDegree; ++a),
                        for (int b = 0; b < hodgeDegree - 1; ++b),
                        for (int c = 0; c < hodgeDegree - 1; ++c))

        basisFaceX(a, b, c) = basisNX[a] * basisDY[b] * basisDZ[c];
    GEMPIC_D_LOOP_END

    GEMPIC_D_LOOP_BEGIN(for (int a = 0; a < hodgeDegree - 1; ++a),
                        for (int b = 0; b < hodgeDegree; ++b),
                        for (int c = 0; c < hodgeDegree - 1; ++c))

        basisFaceY(a, b, c) = basisDX[a] * basisNY[b] * basisDZ[c];
    GEMPIC_D_LOOP_END

    GEMPIC_D_LOOP_BEGIN(for (int a = 0; a < hodgeDegree - 1; ++a),
                        for (int b = 0; b < hodgeDegree - 1; ++b),
                        for (int c = 0; c < hodgeDegree; ++c))

        basisFaceZ(a, b, c) = basisDX[a] * basisDY[b] * basisNZ[c];
    GEMPIC_D_LOOP_END
}

/**
 * @brief The Particle loop and the linear solver for the operator 4 on page 9 in
 * [Energy conserving PIC, Kormann & Sonnendruecker, 2021, JCP].
 * @param[out] E Electric field.
 * @param b right hand side of the linear system in operator 4(b).
 * @param dt Time step size.
 * @param infra ComputationalDomain.
 * @param partGr Particle grid.
 * @param B Magnetic field.
 * @param J Current.
 * @param eSStar The particle related part (except J) of the rhs of the linear system.
 * @param particleMassMatrix Nonzero components of the particle mass matrix.
 * @param hypreVlasovMaxwellEv
 * @param doSync Whether to synchronize after particle loop
 */
template <int hodgeDegree, int degx, int degy, int degz, unsigned int vdim>
void semi_implicit_gdec_particle_loop (
    Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
    Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& b,
    amrex::Real dt,
    Gempic::ComputationalDomain const& infra,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vdim>>>& partGr,
    Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
    Forms::DeRhamField<Gempic::Forms::Grid::dual, Gempic::Forms::Space::face>& J,
    Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& eSStar,
    Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& particleMassMatrix,
    FieldSolvers::VlasovMaxwellEV<hodgeDegree, degx, degy, degz, vdim>& hypreVlasovMaxwellEv,
    bool doSync = true)
{
    BL_PROFILE(
        "Gempic::TimeLoop::GEMPIC_VlasovMaxwellSemiImplicitOperators::semi_implicit_gdec_particle_"
        "loop()");
    using namespace Forms;

    auto dx{infra.cell_size_3darray()};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> scalingDF{
        AMREX_D_DECL(dx[yDir] * dx[zDir] / dx[xDir], dx[xDir] * dx[zDir] / dx[yDir],
                     dx[xDir] * dx[yDir] / dx[zDir])};

    // update particle position 0.5 dt
    push_all_x_using_v_gdec<degx, degy, degz>(partGr, 0.5 * dt);

    // A loop of particles for computing particle mass matrix, rotation, deposition of the particle
    // part of E_star, and push with E.
    for (auto& particleSpecies : partGr)
    {
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeSqrByMass = charge * charge / mass;
        amrex::Real chargeOverMass = charge / mass;

        for (auto& particleGrid : *particleSpecies)
        {
            long const np = particleGrid.numParticles();
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> esA;

            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (E.m_data[cc])[particleGrid].array();
                esA[cc] = (eSStar.m_data[cc])[particleGrid].array();
            }

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;

            for (int cc = 0; cc < vdim; cc++)
            {
                bA[cc] = (B.m_data[cc])[particleGrid].array();
            }

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> jA;

            for (int cc = 0; cc < vdim; cc++)
            {
                jA[cc] = (J.m_data[cc])[particleGrid].array();
            }

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eGDECCoeffsARho;
            for (int cc = 0; cc < vdim; cc++)
            {
                eGDECCoeffsARho[cc] = (particleMassMatrix.m_data[cc])[particleGrid].array();
            }
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi{infra.inv_cell_size_array()};

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};

                    amrex::IntVect cellParticle{
                        AMREX_D_DECL(static_cast<int>(std::floor((pos[0] - plo[0]) * dxi[0])),
                                     static_cast<int>(std::floor((pos[1] - plo[1]) * dxi[1])),
                                     static_cast<int>(std::floor((pos[2] - plo[2]) * dxi[2])))};
                    // non-vanishing basis functions
                    amrex::GpuArray<amrex::Real, hodgeDegree> basisNX;
                    amrex::GpuArray<amrex::Real, hodgeDegree - 1> basisDX;

                    amrex::GpuArray<amrex::Real, hodgeDegree> basisNY;
                    amrex::GpuArray<amrex::Real, hodgeDegree - 1> basisDY;

                    amrex::GpuArray<amrex::Real, hodgeDegree> basisNZ;
                    amrex::GpuArray<amrex::Real, hodgeDegree - 1> basisDZ;

                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0,
                                   hodgeDegree - 1>
                        basisEdgeX;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0,
                                   hodgeDegree - 1>
                        basisEdgeY;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 1, 0,
                                   hodgeDegree - 2>
                        basisEdgeZ;

                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0,
                                   hodgeDegree - 2>
                        basisFaceX;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0,
                                   hodgeDegree - 2>
                        basisFaceY;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 2, 0,
                                   hodgeDegree - 1>
                        basisFaceZ;

                    basis_eval_at_position_gdec<hodgeDegree>(
                        basisEdgeX, basisEdgeY, basisEdgeZ, basisFaceX, basisFaceY, basisFaceZ,
                        basisNX, basisDX, basisNY, basisDY, basisNZ, basisDZ, cellParticle, pos,
                        plo, dx);

                    // rotation
                    amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                           ptd.rdata(ii.m_ively)[pp],
                                                           ptd.rdata(ii.m_ivelz)[pp]};

                    amrex::GpuArray<amrex::Real, 3> bField =
                        eval_field_lagrange<hodgeDegree, Grid::primal, Space::face>(
                            pos, bA, dx, plo, basisEdgeX, basisEdgeY, basisEdgeZ, basisFaceX,
                            basisFaceY, basisFaceZ);

                    Gempic::ParticleMeshCoupling::rotate_v_using_b(vel, bField, 0.5 * dt,
                                                                   chargeOverMass);

                    ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                    ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                    ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];

                    // push V to Vstar
                    amrex::GpuArray<amrex::Real, 3> eField =
                        eval_field_lagrange<hodgeDegree, Grid::primal, Space::edge>(
                            pos, eA, dx, plo, basisEdgeX, basisEdgeY, basisEdgeZ, basisFaceX,
                            basisFaceY, basisFaceZ);

                    ptd.rdata(ii.m_ivelx)[pp] += 0.5 * dt * chargeOverMass * eField[xDir];
                    ptd.rdata(ii.m_ively)[pp] += 0.5 * dt * chargeOverMass * eField[yDir];
                    ptd.rdata(ii.m_ivelz)[pp] += 0.5 * dt * chargeOverMass * eField[zDir];

                    amrex::Real chargeWeight = charge * ptd.rdata(ii.m_iweight)[pp];

                    // deposit J and E_star
                    deposit_j_and_estar<hodgeDegree>(jA, esA, cellParticle, vel, eField, dt,
                                                     chargeWeight, chargeOverMass, basisEdgeX,
                                                     basisEdgeY, basisEdgeZ);

                    // particle mass matrix
                    get_gdec_rho_e_weights_at_particle_positions_and_deposit_to_face<hodgeDegree,
                                                                                     vdim>(
                        eGDECCoeffsARho, cellParticle, plo, dx, dt,
                        chargeSqrByMass * ptd.rdata(ii.m_iweight)[pp], scalingDF, basisEdgeX,
                        basisEdgeY, basisEdgeZ);
                });
        }
    }

    if (doSync)
    {
        J.post_particle_loop_sync();
        eSStar.post_particle_loop_sync();
    }

    // e star
    hodge(b, E);
    b += eSStar;
    b -= J;

    for (int ndir = 0; ndir < vdim; ++ndir)
    {
        particleMassMatrix.m_data[ndir].SumBoundary(infra.geometry().periodicity());
    }

    // here to use the hypre solver to get new E
    hypreVlasovMaxwellEv.solve_hodge_plus_particle_matrix_e_gdec(E, b, particleMassMatrix, infra);

    // The second part of the Strang splitting.
    for (auto& particleSpecies : partGr)
    {
        amrex::Real charge = particleSpecies->get_charge();
        amrex::Real mass = particleSpecies->get_mass();
        amrex::Real chargeOverMass = charge / mass;

        for (auto& particleGrid : *particleSpecies)
        {
            long const np = particleGrid.numParticles();
            auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
            auto const ii = particleSpecies->get_data_indices();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;

            for (int cc = 0; cc < vdim; cc++)
            {
                eA[cc] = (E.m_data[cc])[particleGrid].array();
            }

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> bA;

            for (int cc = 0; cc < vdim; cc++)
            {
                bA[cc] = (B.m_data[cc])[particleGrid].array();
            }

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi{
                AMREX_D_DECL(infra.geometry().InvCellSize(xDir), infra.geometry().InvCellSize(yDir),
                             infra.geometry().InvCellSize(zDir))};

            amrex::ParallelFor(
                np,
                [=] AMREX_GPU_DEVICE(long pp)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> pos{
                        AMREX_D_DECL(ptd.rdata(ii.m_iposx)[pp], ptd.rdata(ii.m_iposy)[pp],
                                     ptd.rdata(ii.m_iposz)[pp])};

                    amrex::IntVect cellParticle{
                        AMREX_D_DECL(static_cast<int>(std::floor((pos[0] - plo[0]) * dxi[0])),
                                     static_cast<int>(std::floor((pos[1] - plo[1]) * dxi[1])),
                                     static_cast<int>(std::floor((pos[2] - plo[2]) * dxi[2])))};

                    // non-vanishing basis functoins
                    amrex::GpuArray<amrex::Real, hodgeDegree> basisNX;
                    amrex::GpuArray<amrex::Real, hodgeDegree - 1> basisDX;

                    amrex::GpuArray<amrex::Real, hodgeDegree> basisNY;
                    amrex::GpuArray<amrex::Real, hodgeDegree - 1> basisDY;

                    amrex::GpuArray<amrex::Real, hodgeDegree> basisNZ;
                    amrex::GpuArray<amrex::Real, hodgeDegree - 1> basisDZ;

                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0,
                                   hodgeDegree - 1>
                        basisEdgeX;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0,
                                   hodgeDegree - 1>
                        basisEdgeY;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 1, 0,
                                   hodgeDegree - 2>
                        basisEdgeZ;

                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 1, 0, hodgeDegree - 2, 0,
                                   hodgeDegree - 2>
                        basisFaceX;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 1, 0,
                                   hodgeDegree - 2>
                        basisFaceY;
                    amrex::Array3D<amrex::Real, 0, hodgeDegree - 2, 0, hodgeDegree - 2, 0,
                                   hodgeDegree - 1>
                        basisFaceZ;

                    basis_eval_at_position_gdec<hodgeDegree>(
                        basisEdgeX, basisEdgeY, basisEdgeZ, basisFaceX, basisFaceY, basisFaceZ,
                        basisNX, basisDX, basisNY, basisDY, basisNZ, basisDZ, cellParticle, pos,
                        plo, dx);

                    // push Vstar to V
                    amrex::GpuArray<amrex::Real, 3> eField =
                        eval_field_lagrange<hodgeDegree, Grid::primal, Space::edge>(
                            pos, eA, dx, plo, basisEdgeX, basisEdgeY, basisEdgeZ, basisFaceX,
                            basisFaceY, basisFaceZ);

                    ptd.rdata(ii.m_ivelx)[pp] += 0.5 * dt * chargeOverMass * eField[xDir];
                    ptd.rdata(ii.m_ively)[pp] += 0.5 * dt * chargeOverMass * eField[yDir];
                    ptd.rdata(ii.m_ivelz)[pp] += 0.5 * dt * chargeOverMass * eField[zDir];

                    // rotation
                    amrex::GpuArray<amrex::Real, vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                           ptd.rdata(ii.m_ively)[pp],
                                                           ptd.rdata(ii.m_ivelz)[pp]};

                    amrex::GpuArray<amrex::Real, 3> bField =
                        eval_field_lagrange<hodgeDegree, Grid::primal, Space::face>(
                            pos, bA, dx, plo, basisEdgeX, basisEdgeY, basisEdgeZ, basisFaceX,
                            basisFaceY, basisFaceZ);

                    Gempic::ParticleMeshCoupling::rotate_v_using_b(vel, bField, 0.5 * dt,
                                                                   chargeOverMass);
                    ptd.rdata(ii.m_ivelx)[pp] = vel[xDir];
                    ptd.rdata(ii.m_ively)[pp] = vel[yDir];
                    ptd.rdata(ii.m_ivelz)[pp] = vel[zDir];

                    // update particle position
                    AMREX_D_EXPR(ptd.rdata(ii.m_iposx)[pp] += 0.5 * dt * ptd.rdata(ii.m_ivelx)[pp],
                                 ptd.rdata(ii.m_iposy)[pp] += 0.5 * dt * ptd.rdata(ii.m_ively)[pp],
                                 ptd.rdata(ii.m_iposz)[pp] += 0.5 * dt * ptd.rdata(ii.m_ivelz)[pp]);
                });
        }

        particleSpecies->Redistribute();
    }
}

} // namespace Gempic::TimeLoop

#endif // GEMPIC_VLASOVMAXWELLSEMIIMPLICITOPERATORS_H_

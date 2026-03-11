/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef MHD_PDE_H_
#define MHD_PDE_H_

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <AMReX_Periodicity.H>
#include <AMReX_VisMF.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FV.H"
#include "GEMPIC_MHDStructDefinitions.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_PDE.H"
#include "GEMPIC_Parameters.H"

// MHD PDE specialization:

/**
 * auxiliary PDE parameter struct;
 */
struct MhdPdeParameters
{
    amrex::Real m_gamma, m_cv, m_mu, m_nu;

    MhdPdeParameters()
    {
        Gempic::Io::Parameters paramsPDE("PDE");
        paramsPDE.get("gamma", m_gamma);
        paramsPDE.get("cv", m_cv);
        paramsPDE.get("mu", m_mu);
        paramsPDE.get("nu", m_nu);
    }
};

struct MhdPdeTag
{
    std::string m_myPdeTag = "MHD";
};

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
pde_cons_2_prim<nVarMHD, MhdPdeParameters, MhdPdeTag>(varArray<nVarMHD> const& Q,
                                                      varArray<nVarMHD>& V,
                                                      MhdPdeParameters const& params)
{
    amrex::Real rho{Q[rhoIdx]}, rhok, gamma{params.m_gamma};
    amrex::Real irho{1.0 / Q[rhoIdx]};
    V[rhoIdx] = Q[rhoIdx];
    V[vxIdx] = Q[mxIdx] * irho;
    V[vyIdx] = Q[myIdx] * irho;
    V[vzIdx] = Q[mzIdx] * irho;
    amrex::Real u2 = V[vxIdx] * V[vxIdx];
    amrex::Real v2 = V[vyIdx] * V[vyIdx];
    amrex::Real w2 = V[vzIdx] * V[vzIdx];
    rhok = 0.5 * rho * (u2 + v2 + w2);
    V[pIdx] = (gamma - 1) * (Q[eIdx] - rhok);
    V[bxIdx] = Q[bxIdx];
    V[byIdx] = Q[byIdx];
    V[bzIdx] = Q[bzIdx];
}

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
pde_prim_2_cons<nVarMHD, MhdPdeParameters, MhdPdeTag>(varArray<nVarMHD> const& V,
                                                      varArray<nVarMHD>& Q,
                                                      MhdPdeParameters const& params)
{
    amrex::Real rhok, gamma{params.m_gamma};
    amrex::Real rho{V[rhoIdx]}, u{V[vxIdx]}, v{V[vyIdx]}, w{V[vzIdx]};
    Q[rhoIdx] = V[rhoIdx];
    Q[mxIdx] = V[vxIdx] * rho;
    Q[myIdx] = V[vyIdx] * rho;
    Q[mzIdx] = V[vzIdx] * rho;

    rhok = 0.5 * rho * (u * u + v * v + w * w);
    Q[eIdx] = V[pIdx] / (gamma - 1.0) + rhok;
    Q[bxIdx] = V[bxIdx];
    Q[byIdx] = V[byIdx];
    Q[bzIdx] = V[bzIdx];
}

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void pde_get_fluxes<nVarMHD, MhdPdeParameters, MhdPdeTag>(
    varArray<nVarMHD> const& Q, fluxArray<nVarMHD>& f, MhdPdeParameters const& params)
{
    amrex::Real rho{Q[rhoIdx]}, u, v, w, p, Bx, By, Bz, b2, b28P, rhok, gamma{params.m_gamma};
    rho = Q[rhoIdx];
    u = Q[mxIdx] / rho;
    v = Q[myIdx] / rho;
    w = Q[mzIdx] / rho;
    Bx = Q[bxIdx];
    By = Q[byIdx];
    Bz = Q[bzIdx];
    b2 = Bx * Bx + By * By + Bz * Bz;

    b28P = b2 / (8 * M_PI);
    rhok = 0.5 * rho * (u * u + v * v + w * w);
    p = (gamma - 1) * (Q[eIdx] - rhok - b28P);

    // x flux
    f[xDir][rhoIdx] = Q[mxIdx];
    f[xDir][mxIdx] = rho * u * u + p + b28P;
    f[xDir][myIdx] = rho * u * v;
    f[xDir][mzIdx] = rho * u * w;
    f[xDir][eIdx] = u * (Q[eIdx] + p) + b28P;
    f[xDir][bxIdx] = 0.0;
    f[xDir][byIdx] = 0.0;
    f[xDir][bzIdx] = 0.0;
#if AMREX_SPACEDIM > 1
    // y flux
    f[yDir][rhoIdx] = Q[myIdx];
    f[yDir][mxIdx] = rho * v * u;
    f[yDir][myIdx] = rho * v * v + p + b28P;
    f[yDir][mzIdx] = rho * v * w;
    f[yDir][eIdx] = v * (Q[eIdx] + p) + b28P;
    f[yDir][bxIdx] = 0.0;
    f[yDir][byIdx] = 0.0;
    f[yDir][bzIdx] = 0.0;
#endif
#if AMREX_SPACEDIM > 2
    // z flux
    f[zDir][rhoIdx] = Q[myIdx];
    f[zDir][mxIdx] = rho * w * u;
    f[zDir][myIdx] = rho * w * v;
    f[zDir][mzIdx] = rho * w * w + p + b28P;
    f[zDir][eIdx] = w * (Q[eIdx] + p + b28P);
    f[zDir][bxIdx] = 0.0;
    f[zDir][byIdx] = 0.0;
    f[zDir][bzIdx] = 0.0;
#endif
}

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
pde_get_eigenvalues<nVarMHD, MhdPdeParameters, MhdPdeTag>(varArray<nVarMHD> const& Q,
                                                          amrex::GpuArray<amrex::Real, 3> const& n,
                                                          varArray<nVarMHD>& mask,
                                                          varArray<nVarMHD>& L,
                                                          MhdPdeParameters const& params)
{
    amrex::Real gamma{params.m_gamma}, b2, b28P, p, bn, bn2, bX, bY, bZ, b, u, a, ca, cf, cs;
    b2 = (Q[bxIdx] * Q[bxIdx] + Q[byIdx] * Q[byIdx] + Q[bzIdx] * Q[bzIdx]);
    b28P = b2 / (8 * M_PI);
    p = (gamma - 1.) *
        (Q[4] -
         0.5 / Q[rhoIdx] * (Q[vxIdx] * Q[vxIdx] + Q[vyIdx] * Q[vyIdx] + Q[vzIdx] * Q[vzIdx]) -
         b28P);
    bX = Q[bxIdx] * Q[bxIdx] / (4. * M_PI * Q[rhoIdx]);
    bY = Q[byIdx] * Q[byIdx] / (4. * M_PI * Q[rhoIdx]);
    bZ = Q[bzIdx] * Q[bzIdx] / (4. * M_PI * Q[rhoIdx]);
    b = bX + bY + bZ;
    u = (Q[vxIdx] * n[xDir] + Q[vyIdx] * n[yDir] + Q[vzIdx] * n[zDir]) / Q[rhoIdx];
    bn = (n[xDir] * Q[bxIdx] + n[yDir] * Q[byIdx] + n[zDir] * Q[bzIdx]);
    bn2 = bn * bn / (4. * M_PI * Q[rhoIdx]);
    a = gamma * p / Q[rhoIdx];
    ca = sqrt(abs(bn2));

    amrex::Real ab = a + b;
    cf = sqrt(abs(0.5 * (ab + sqrt(abs(ab * ab - 4. * a * bn2)))));
    cs = sqrt(abs(0.5 * (ab - sqrt(abs(ab * ab - 4. * a * bn2)))));

    L[rhoIdx] = u - cf;
    L[vxIdx] = u - ca;
    L[vyIdx] = u - cs;
    L[vzIdx] = u;
    L[eIdx] = u;
    L[bxIdx] = u + cs;
    L[byIdx] = u + ca;
    L[bzIdx] = u + cf;

    for (int i = 0; i < nVarMHD; i++)
    {
        mask[i] = 1.0;
    }
}

/**
 TO DO FOR MHD (!!)
*/

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
pde_get_eigenvectors<nVarMHD, MhdPdeParameters, MhdPdeTag>(varArray<nVarMHD> const& Q,
                                                           amrex::GpuArray<amrex::Real, 3> const& n,
                                                           varArray2D<nVarMHD>& R,
                                                           varArray2D<nVarMHD>& L,
                                                           varArray2D<nVarMHD>& iR,
                                                           MhdPdeParameters const& params)
{
    amrex::GpuArray<amrex::Real, 3> nv{0., 0., 0.};
    amrex::GpuArray<amrex::Real, 3> sv{0., 0., 0.};
    amrex::GpuArray<amrex::Real, 3> tv{0., 0., 0.};
    // Rotation matrices
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        nv[idim] = n[idim];
    }

    //
    // sv = (/ 1., 1., 1. /) - ABS(nv)
    // sv = sv/SQRT(SUM(sv(:)**2))
    // CALL Kreuzprodukt(tv,nv,sv)
    //
    amrex::Real n1{std::abs(n[xDir])};
    if (std::abs(n1 - 1.0) < 1e-14)
    {
        sv = {0., 1., 0.};
        tv = {0., 0., 1.};
    }
#if AMREX_SPACEDIM > 1
    amrex::Real n2{std::abs(n[yDir])};
    if (std::abs(n2 - 1.0) < 1e-14)
    {
        sv = {1., 0., 0.};
        tv = {0., 0., 1.};
    }
#endif
#if AMREX_SPACEDIM > 2
    amrex::Real n3{std::abs(n[zDir])};
    if (std::abs(n3 - 1.0) < 1e-14)
    {
        sv = {1., 0., 0.};
        tv = {0., 1., 0.};
    }
#endif

    varArray2D<nVarMHD> tm;
    for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            tm(iVar, jVar) = 0.0;
        }
    }
    tm(0, 0) = 1.0;
    tm(pIdx, pIdx) = 1.0;
    for (int idim = 0; idim < 3; ++idim)
    {
        tm(idim + 1, 1) = nv[idim];
        tm(idim + 1, 2) = sv[idim];
        tm(idim + 1, 3) = tv[idim];
    }
    for (int i = 5; i < nVarMHD; ++i)
    {
        tm(i, i) = 1.0;
    }

    /*for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            RM(iVar, jVar) = 0.0;
        }
    }
    for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            iRM(iVar, jVar) = 0.0;
        }
    }*/

    // Primitive variables rotated in normal direction
    amrex::Real gamma{params.m_gamma};
    amrex::Real rho{Q[rhoIdx]};
    amrex::Real u{(nv[0] * Q[mxIdx] + nv[1] * Q[myIdx] + nv[2] * Q[mzIdx]) / Q[rhoIdx]};
    amrex::Real v{(sv[0] * Q[mxIdx] + sv[1] * Q[myIdx] + sv[2] * Q[mzIdx]) / Q[rhoIdx]};
    amrex::Real w{(tv[0] * Q[mxIdx] + tv[1] * Q[myIdx] + tv[2] * Q[mzIdx]) / Q[rhoIdx]};
    amrex::Real bx{(nv[0] * Q[bxIdx] + nv[1] * Q[byIdx] + nv[2] * Q[bzIdx])};
    amrex::Real by{(sv[0] * Q[bxIdx] + sv[1] * Q[byIdx] + sv[2] * Q[bzIdx])};
    amrex::Real bz{(tv[0] * Q[bxIdx] + tv[1] * Q[byIdx] + tv[2] * Q[bzIdx])};
    amrex::Real b2{Q[bxIdx] * Q[bxIdx] + Q[byIdx] * Q[byIdx] + Q[bzIdx] * Q[bzIdx]};
    amrex::Real b28P{b2 / (8 * M_PI)};
    amrex::Real p{(gamma - 1.) * (Q[4] - 0.5 * rho * (u * u + v * v + w * w) - b28P)};
    amrex::Real bn{bx};
    amrex::Real bn2{bn * bn / (4. * M_PI * rho)};
    amrex::Real bbX{bx * bx / (4. * M_PI * rho)};
    amrex::Real bbY{by * by / (4. * M_PI * rho)};
    amrex::Real bbZ{bz * bz / (4. * M_PI * rho)};
    amrex::Real b{bbX + bbY + bbZ};
    amrex::Real a{gamma * p / rho};
    amrex::Real ca{std::sqrt(std::abs(bn2))};
    amrex::Real ab{a + b};
    amrex::Real cf{std::sqrt(std::abs(0.5 * (ab + std::sqrt(std::abs(ab * ab - 4. * a * bn2)))))};
    amrex::Real cs{std::sqrt(std::abs(0.5 * (ab - std::sqrt(std::abs(ab * ab - 4. * a * bn2)))))};

    // Some auxiliary variables
    amrex::Real c{std::sqrt(gamma * p / rho)};
    amrex::Real H{(Q[4] + p) / rho};
    amrex::Real v2{u * u + v * v + w * w};
    amrex::Real M{std::sqrt(v2) / c};
    amrex::Real r2c{rho / 2. / c};

    for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            L(iVar, jVar) = 0.0;
        }
    }

    L(rhoIdx, rhoIdx) = u - cf;
    L(vxIdx, vxIdx) = u - ca;
    L(vyIdx, vyIdx) = u - cs;
    L(vzIdx, vzIdx) = u;
    L(eIdx, eIdx) = u;
    L(bxIdx, bxIdx) = u + cs;
    L(byIdx, byIdx) = u + ca;
    L(bzIdx, bzIdx) = u + cf;

    // TO DO: right and eigenvector matrix + rotation?
    for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            R(iVar, jVar) = 0.0;
            iR(iVar, jVar) = 0.0;
        }
        R(iVar, iVar) = 1.0;
        iR(iVar, iVar) = 1.0;
    }
    return;

    varArray2D<nVarMHD> rm;
    for (int i = 0; i < nVarMHD; ++i)
    {
        for (int j = 0; j < nVarMHD; ++j)
        {
            rm(i, j) = 0.0;
        }
        rm(i, i) = 1.0;
    }

    // Right eigenvector matrix
    rm(0, 0) = 1.;
    rm(0, 1) = 0.;
    rm(0, 2) = 0.;
    rm(0, 3) = r2c;
    rm(0, 4) = r2c;
    rm(1, 0) = u;
    rm(1, 1) = 0.;
    rm(1, 2) = 0.;
    rm(1, 3) = r2c * (u + c);
    rm(1, 4) = r2c * (u - c);
    rm(2, 0) = v;
    rm(2, 1) = 0.;
    rm(2, 2) = -rho;
    rm(2, 3) = r2c * v;
    rm(2, 4) = r2c * v;
    rm(3, 0) = w;
    rm(3, 1) = rho;
    rm(3, 2) = 0.;
    rm(3, 3) = r2c * w;
    rm(3, 4) = r2c * w;
    rm(3, 0) = w;
    rm(3, 1) = rho;
    rm(3, 2) = 0.;
    rm(3, 3) = r2c * w;
    rm(3, 4) = r2c * w;
    rm(4, 0) = 0.5 * v2;
    rm(4, 1) = rho * w;
    rm(4, 2) = -rho * v;
    rm(4, 3) = r2c * (H + c * u);
    rm(4, 4) = r2c * (H - c * u);

    varArray2D<nVarMHD> iRM;
    for (int i = 0; i < nVarMHD; ++i)
    {
        for (int j = 0; j < nVarMHD; ++j)
        {
            iRM(i, j) = 0.0;
        }
        iRM(i, i) = 1.0;
    }
    // Left eigenvector matrix (inverse of R)
    iRM(0, 0) = 1. - (gamma - 1.) / 2. * M * M;
    iRM(0, 1) = (gamma - 1.) * u / c / c;
    iRM(0, 2) = (gamma - 1.) * v / c / c;
    iRM(0, 3) = (gamma - 1.) * w / c / c;
    iRM(0, 4) = -(gamma - 1.) / c / c;
    iRM(1, 0) = -w / rho;
    iRM(1, 1) = 0.;
    iRM(1, 2) = 0.;
    iRM(1, 3) = 1. / rho;
    iRM(1, 4) = 0.;
    iRM(2, 0) = v / rho;
    iRM(2, 1) = 0.;
    iRM(2, 2) = -1. / rho;
    iRM(2, 3) = 0.;
    iRM(2, 4) = 0.;
    iRM(3, 0) = c / rho * (0.5 * (gamma - 1.) * M * M - u / c);
    iRM(3, 1) = 1. / rho * (1. - (gamma - 1.) * u / c);
    iRM(3, 2) = 1. / rho * (-(gamma - 1.) * v / c);
    iRM(3, 3) = 1. / rho * (-(gamma - 1.) * w / c);
    iRM(3, 4) = (gamma - 1.) / rho / c;
    iRM(4, 0) = c / rho * (0.5 * (gamma - 1.) * M * M + u / c);
    iRM(4, 1) = 1. / rho * (-1. - (gamma - 1.) * u / c);
    iRM(4, 2) = 1. / rho * (-(gamma - 1.) * v / c);
    iRM(4, 3) = 1. / rho * (-(gamma - 1.) * w / c);
    iRM(4, 4) = (gamma - 1.) / rho / c;

    // Final Matrices including the rotation
    for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            amrex::Real sum{0.0};
            for (int kVar = 0; kVar < nVarMHD; ++kVar)
            {
                sum += tm(iVar, kVar) * rm(kVar, jVar);
            }
            R(iVar, jVar) = sum;
        }
    }
    for (int iVar = 0; iVar < nVarMHD; ++iVar)
    {
        for (int jVar = 0; jVar < nVarMHD; ++jVar)
        {
            amrex::Real sum{0.0};
            for (int kVar = 0; kVar < nVarMHD; ++kVar)
            {
                sum += iRM(iVar, kVar) * tm(jVar, kVar);
            }
            iR(iVar, jVar) = sum;
        }
    }

    return;
}

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
pde_numerical_flux<nVarMHD, MhdPdeParameters, MhdPdeTag>(varArray<nVarMHD>& qL,
                                                         varArray<nVarMHD>& qR,
                                                         amrex::GpuArray<amrex::Real, 3> n,
                                                         varArray<nVarMHD>& flux,
                                                         varArray<nVarMHD>& flux1,
                                                         MhdPdeParameters const& params)
{
    varArray<nVarMHD> ll;
    varArray<nVarMHD> lr;
    fluxArray<nVarMHD> fl;
    fluxArray<nVarMHD> fr;
    varArray<nVarMHD> mask;

    pde_get_eigenvalues<nVarMHD, MhdPdeParameters, MhdPdeTag>(qR, n, mask, lr, params);
    pde_get_eigenvalues<nVarMHD, MhdPdeParameters, MhdPdeTag>(qL, n, mask, ll, params);

    amrex::Real maxLL = 0.0;
    amrex::Real maxLR = 0.0;

    for (int i = 0; i < nVarMHD; i++)
    {
        maxLL = std::max(maxLL, std::abs(ll[i]));
        maxLR = std::max(maxLR, std::abs(lr[i]));
    }

    amrex::Real smax = std::max(maxLL, maxLR);

    pde_get_fluxes<nVarMHD, MhdPdeParameters, MhdPdeTag>(qL, fl, params);
    pde_get_fluxes<nVarMHD, MhdPdeParameters, MhdPdeTag>(qR, fr, params);

    for (int iVar = 0; iVar < nVarMHD; iVar++)
    {
        flux[iVar] = 0.5 * (fl[0][iVar] + fr[0][iVar]) * n[0]
#if AMREX_SPACEDIM > 1
                     + 0.5 * (fl[1][iVar] + fr[1][iVar]) * n[1]
#endif
#if AMREX_SPACEDIM > 2
                     + 0.5 * (fl[2][iVar] + fr[2][iVar]) * n[2]
#endif
                     - 0.5 * smax * mask[iVar] * (qR[iVar] - qL[iVar]);
    }
    for (int iVar = 0; iVar < nVarMHD; iVar++)
    {
        flux1[iVar] = flux[iVar];
    }
}

template <int nVarPde, typename ParamsClass, typename PdeTagType>
AMREX_GPU_HOST_DEVICE void initial_condition(varArray<nVarPde>& Q,
                                             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r,
                                             amrex::Real time,
                                             ParamsClass const& params);

template <>
AMREX_GPU_HOST_DEVICE GEMPIC_FORCE_INLINE void
initial_condition<nVarMHD, MhdPdeParameters, MhdPdeTag>(
    varArray<nVarMHD>& Q,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r,
    amrex::Real time,
    MhdPdeParameters const& params)
{
    amrex::Real epsilon{5.0}, du, dv, dT, drho, dp, gamma{params.m_gamma};
    amrex::Real rxLoc{0.0};
    amrex::Real ryLoc{0.0};
    constexpr amrex::Real mPI2 = M_PI * M_PI;
    varArray<nVarMHD> V;
    rxLoc = (r[0] - time);
#if AMREX_SPACEDIM > 1
    ryLoc = (r[1] - time);
#endif
    rxLoc = rxLoc - 10.0 * std::round(rxLoc / 10);
#if AMREX_SPACEDIM > 1
    ryLoc = ryLoc - 10.0 * std::round(ryLoc / 10);
#endif

    amrex::GpuArray<amrex::Real, 3> vel{1.0, 1.0, 0.0};

    amrex::Real r2Loc = rxLoc * rxLoc + ryLoc * ryLoc;
    du = epsilon / 2. / M_PI * std::exp(0.5 * (1. - r2Loc)) * (-ryLoc);
    dv = epsilon / 2. / M_PI * std::exp(0.5 * (1. - r2Loc)) * (rxLoc);
    dT = -(gamma - 1.) * epsilon * epsilon / 8. / gamma / mPI2 * std::exp(1. - r2Loc);
    drho = std::pow(1. + dT, (1. / (gamma - 1.))) - 1.;
    dp = std::pow(1. + dT, gamma / (gamma - 1.)) - 1.;
    V[rhoIdx] = 1. + drho;
    V[vxIdx] = vel[0] + du;
    V[vyIdx] = vel[1] + dv;
    V[vzIdx] = vel[2];
    V[pIdx] = 1. + dp;
    V[bxIdx] = 0.0;
    V[byIdx] = 0.0;
    V[bzIdx] = 0.0;

    pde_prim_2_cons<nVarMHD, MhdPdeParameters, MhdPdeTag>(V, Q, params);
}

// PDE solver class
template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
class MHD
{
private:
public:
    MHDDiscretization m_disc;
    ParamsClass m_pdeparams;
    FVreconstruction<nVarPde, polyrecDegreeFv> m_fvrec; // use typename for template types.

    /*AMREX_GPU_HOST_DEVICE void initial_condition(varArray<nVarPde>& Q,
                                                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& r,
                                                 amrex::Real time,
                                                 ParamsClass const& params) const;*/

    template <ReconstructionMode userRecMode>
    void single_time_step();

    void compute_norms_reduce();
    void compute_errornorms_reduce();

    void update_qold_q();
    //void updcate_Q(Discretization_PDE& disc, ComputationalDomain& m_disc.m_compDom, IOParameters&
    //m_disc.m_iopar);
    void update_qnew();

    void init();

    // Parameterized constructor
    MHD();
};

template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
MHD<nVarPde, polyrecDegreeFv, ParamsClass, PdeTagType>::MHD() : m_disc()
{
    amrex::Print() << "hello from MHD!\n";
}

template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
void MHD<nVarPde, polyrecDegreeFv, ParamsClass, PdeTagType>::compute_norms_reduce ()
{
    // save old dt
    m_disc.m_dtOld = m_disc.m_dtNew;
    amrex::Real const idVol = m_disc.m_idVol;
    amrex::GpuTuple<AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real)> smaxLoc;

    //// GPU ParReduce:
    // SEE FabArray<FAB>::norminf for reference (AMReX) or void compute_diags (in
    // GEMPIC_ParticleDiags) to be tested with MPI or OMP
    amrex::MultiArray4<amrex::Real const> const& qLoc =
        (m_disc.m_fields->m_q.m_data).const_arrays();

    ParamsClass myparams;
    smaxLoc = amrex::ParReduce(
        amrex::TypeList<AMREX_D_DECL(amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax)>{},
        amrex::TypeList<AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real)>{},
        m_disc.m_fields->m_q.m_data,
        amrex::IntVect(0), // zero ghost cells
        // ncomp,       // number of components
        [qLoc, idVol, myparams] AMREX_GPU_DEVICE(int boxNo, int i, int j, int k) noexcept
        -> amrex::GpuTuple<AMREX_D_DECL(amrex::Real, amrex::Real, amrex::Real)>
        {
            varArray<nVarPde> qGpu;
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lmax{AMREX_D_DECL(0.0, 0.0, 0.0)};
            amrex::GpuArray<amrex::Real, 3> nx{1.0, 0.0, 0.0};

            for (int iVar = 0; iVar < nVarPde; iVar++)
            {
                qGpu[iVar] = idVol * qLoc[boxNo](i, j, k, iVar);
            }
            varArray<nVarPde> mask;
            varArray<nVarPde> lLx;
            pde_get_eigenvalues<nVarPde, ParamsClass, PdeTagType>(qGpu, nx, mask, lLx, myparams);
            for (int iVar = 0; iVar < nVarPde; iVar++)
            {
                lmax[0] = std::max(lmax[0], std::abs(lLx[iVar]));
            }
#if AMREX_SPACEDIM > 1
            amrex::GpuArray<amrex::Real, 3> ny{0.0, 1.0, 0.0};
            varArray<nVarPde> lLy;
            pde_get_eigenvalues<nVarPde, ParamsClass, PdeTagType>(qGpu, ny, mask, lLy, myparams);
            for (int iVar = 0; iVar < nVarPde; iVar++)
            {
                lmax[1] = std::max(lmax[1], std::abs(lLy[iVar]));
            }
#endif
#if AMREX_SPACEDIM > 2
            amrex::GpuArray<amrex::Real, 3> nz{0.0, 0.0, 1.0};
            varArray<nVarPde> lLz;
            pde_get_eigenvalues<nVarPde, ParamsClass, PdeTagType>(qGpu, nz, mask, lLz, myparams);
            for (int iVar = 0; iVar < nVarPde; iVar++)
            {
                lmax[2] = std::max(lmax[2], std::abs(lLz[iVar]));
            }

#endif
            return {AMREX_D_DECL(lmax[0], lmax[1], lmax[2])};
        });

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> smax{
        AMREX_D_DECL(amrex::get<0>(smaxLoc), amrex::get<1>(smaxLoc), amrex::get<2>(smaxLoc))};

    // reduced max over mpi ranks
    amrex::ParallelDescriptor::ReduceRealMax({AMREX_D_DECL(smax[0], smax[1], smax[2])});

    amrex::Real idt = GEMPIC_D_ADD(smax[0] * m_disc.m_idr[0], smax[1] * m_disc.m_idr[1],
                                   smax[2] * m_disc.m_idr[2]);
    amrex::Real dt = m_disc.m_iopar.m_cfl / std::max(1e-14, idt);
    m_disc.m_dtNew = std::min(0.5 * (m_disc.m_dtOld + dt), dt);
    m_disc.m_dtNew = std::min(m_disc.m_dtNew, m_disc.m_iopar.m_dtMax);
}

template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
void MHD<nVarPde, polyrecDegreeFv, ParamsClass, PdeTagType>::compute_errornorms_reduce ()
{
    /*
       TO DO
    */
}

template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
void MHD<nVarPde, polyrecDegreeFv, ParamsClass, PdeTagType>::init ()
{
    int ncomp = nVarPde;
    for (amrex::MFIter mfi(m_disc.m_fields->m_q.m_data, true); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.growntilebox(); // real and ghost cells.
        // creating pointer to data for GPU:
        amrex::Array4<amrex::Real> const& qnewLoc = (m_disc.m_fields->m_q.m_data)[mfi].array();

        // starts the loop inside the box,
        amrex::GpuArray<amrex::Real, 3> r0Loc = {m_disc.m_r0};
        amrex::GpuArray<amrex::Real, 3> drLoc = {m_disc.m_dr};
        amrex::Real time = m_disc.m_time;
        amrex::Real dVol = GEMPIC_D_MULT(drLoc[xDir], drLoc[yDir], drLoc[zDir]);
        ParamsClass myparams;
        ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r{
                    AMREX_D_DECL(r0Loc[xDir] + (static_cast<amrex::Real>(i) + 0.5) * drLoc[xDir],
                                 r0Loc[yDir] + (static_cast<amrex::Real>(j) + 0.5) * drLoc[yDir],
                                 r0Loc[zDir] + (static_cast<amrex::Real>(k) + 0.5) * drLoc[zDir])};
                varArray<nVarPde> qGpu;
                initial_condition<nVarPde, ParamsClass, PdeTagType>(qGpu, r, time, myparams);
                for (int iVar = 0; iVar < nVarPde; iVar++)
                {
                    qnewLoc(i, j, k, iVar) = qGpu[iVar] * dVol;
                }
            });
    }
    amrex::MultiFab::Copy(m_disc.m_fieldsTmpEx->m_q.m_data, m_disc.m_fields->m_q.m_data, 0, 0,
                          ncomp, m_disc.m_nGhosts);

    amrex::GpuArray<amrex::Real, 3> n, ndir;
    amrex::GpuArray<amrex::Real, 3> const idrLoc = {m_disc.m_idr};
    ncomp = 1;
    for (int idir = 0; idir < 3; ++idir)
    {
        for (int jdir = 0; jdir < 3; jdir++)
        {
            ndir[jdir] = 0;
            n[jdir] = 0.0;
        }
        ndir[idir] = 1;
        n[idir] = 1.0;
        amrex::Real const dVol = m_disc.m_dVol;
        amrex::Real const dS = dVol * idrLoc[idir];
        for (amrex::MFIter mfi(m_disc.m_fields->m_b.m_data[idir], true); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.growntilebox(); // real and ghost cells.
            // creating pointer to data for GPU:
            amrex::Array4<amrex::Real> const& bnewLoc =
                (m_disc.m_fields->m_b.m_data[idir])[mfi].array();

            amrex::GpuArray<amrex::Real, 3> r0Loc = {m_disc.m_r0};
            amrex::GpuArray<amrex::Real, 3> drLoc = {m_disc.m_dr};
            amrex::Real time = m_disc.m_time;
            ParamsClass myparams;
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r{AMREX_D_DECL(
                        r0Loc[xDir] +
                            (static_cast<amrex::Real>(i) + (1.0 - ndir[0]) * 0.5) * drLoc[xDir],
                        r0Loc[yDir] +
                            (static_cast<amrex::Real>(j) + (1.0 - ndir[1]) * 0.5) * drLoc[yDir],
                        r0Loc[zDir] +
                            (static_cast<amrex::Real>(k) + (1.0 - ndir[2]) * 0.5) * drLoc[zDir])};
                    varArray<nVarPde> qGpu;
                    initial_condition<nVarPde, ParamsClass, PdeTagType>(qGpu, r, time, myparams);
                    bnewLoc(i, j, k) = qGpu[bxIdx + idir] * dS;
                });
        }
        amrex::MultiFab::Copy(m_disc.m_bold.m_data[idir], m_disc.m_fields->m_b.m_data[idir], 0, 0,
                              ncomp, m_disc.m_nGhosts);
    }
}

template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
void MHD<nVarPde, polyrecDegreeFv, ParamsClass, PdeTagType>::update_qnew ()
{
    //Euler update
    amrex::MultiFab::LinComb(m_disc.m_fields->m_q.m_data, 1.0, m_disc.m_fieldsTmpEx->m_q.m_data, 0,
                             m_disc.m_dt, m_disc.m_fieldsDtEx->m_q.m_data, 0, 0, nVarPde,
                             0); // no ghost cells update
    m_disc.m_fields->m_q.m_data.FillBoundary(
        m_disc.m_compDom.m_geom.periodicity()); // this will update the mpi/patch ghost cells
}

/**
 * This is the single-time-step update of the Explicit scheme.
 * It must evaluate time derivative of the explicit step (stored in m_fieldsDtEx),
 * as a function of the explicit state (m_fieldsTmpEx)
 */
template <int nVarPde, int polyrecDegreeFv, typename ParamsClass, typename PdeTagType>
template <ReconstructionMode userRecMode>
void MHD<nVarPde, polyrecDegreeFv, ParamsClass, PdeTagType>::single_time_step ()
{
    amrex::GpuArray<amrex::Real, 3> n, ndir;
    amrex::GpuArray<amrex::Real, 3> const idrLoc = {m_disc.m_idr};
    // Fill boundary before computing the reconstruction
    m_disc.m_fieldsTmpEx->m_q.m_data.FillBoundary(
        m_disc.m_compDom.m_geom.periodicity()); // this will update the mpi/patch ghost cells
    /*************************************************************************************************************/
    // LOOP OVER BARYCENTERS
    // from Q build reconstructed left/right face data
    // assume Q ghost cells are correctly filled??? NO. DO communications only before using ghost
    // cells. intialize reconstruction to zero (needed only for the boundary terms):
    amrex::GpuArray<amrex::Real, faceDof> zeroFaceArray;
    zeroFaceArray.fill(0.0);
    //amrex::GpuArray<amrex::Real, edgeDof> zeroEdgeArray;
    for (int iDim = 0; iDim < AMREX_SPACEDIM; ++iDim)
    {
        m_disc.m_qrRecFab[iDim].setVal(zeroFaceArray, m_disc.m_nGhosts);
        m_disc.m_qlRecFab[iDim].setVal(zeroFaceArray, m_disc.m_nGhosts);
        //m_disc.W_weno[iDim].setVal(zeroEdgeArray);
        //m_disc.B_edge[iDim].setVal(zeroEdgeArray);
    }
    // Compute primitive variables also on ghost-cells to avoid a second communication
    for (amrex::MFIter mfi(m_disc.m_v.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.growntilebox();
        // creating pointer to data for GPU:
        amrex::Array4<amrex::Real> const& v0 = (m_disc.m_v.m_data)[mfi].array();
        amrex::Array4<amrex::Real const> const& q0 =
            (m_disc.m_fieldsTmpEx->m_q.m_data)[mfi].const_array();
        amrex::Real idVolLoc{m_disc.m_idVol};

        ParamsClass myparams;
        if constexpr (userRecMode == ReconstructionMode::CharacteristicVars)
        {
            // store non scaled conservative variables
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            for (int iVar = 0; iVar < nVarPde; iVar++)
                            {
                                v0(i, j, k, iVar) = q0(i, j, k, iVar) * idVolLoc;
                            }
                        });
        }
        else
        {
            // store primitive variables
            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            varArray<nVarPde> qLoc;
                            varArray<nVarPde> vLoc;
                            for (int iVar = 0; iVar < nVarPde; iVar++)
                            {
                                qLoc[iVar] = q0(i, j, k, iVar) * idVolLoc;
                            }
                            pde_cons_2_prim<nVarPde, ParamsClass, PdeTagType>(qLoc, vLoc, myparams);
                            for (int iVar = 0; iVar < nVarPde; iVar++)
                            {
                                v0(i, j, k, iVar) = vLoc[iVar];
                            }
                        });
        }
    }

    for (amrex::MFIter mfi(m_disc.m_fields->m_q.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox(); // cache blocking (for cpus) loops over valid boxes

        // creating pointer to data for GPU:
        amrex::Array4<amrex::GpuArray<
            amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const& qRx =
            (m_disc.m_qrRecFab[xDir]).array(mfi);
        amrex::Array4<amrex::GpuArray<
            amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const& qLx =
            (m_disc.m_qlRecFab[xDir]).array(mfi);
        amrex::Array4<amrex::Real const> const& bx0 =
            (m_disc.m_fields->m_b.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& by0 =
            (m_disc.m_fields->m_b.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real const> const& bz0 =
            (m_disc.m_fields->m_b.m_data[zDir])[mfi].array();
#if AMREX_SPACEDIM > 1
        amrex::Array4<amrex::GpuArray<
            amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const& qRy =
            (m_disc.m_qrRecFab[yDir]).array(mfi);
        amrex::Array4<amrex::GpuArray<
            amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const& qLy =
            (m_disc.m_qlRecFab[yDir]).array(mfi);
#endif
#if AMREX_SPACEDIM > 2
        amrex::Array4<amrex::GpuArray<
            amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const& qRz =
            (m_disc.m_qrRecFab[zDir]).array(mfi);
        amrex::Array4<amrex::GpuArray<
            amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const& qLz =
            (m_disc.m_qlRecFab[zDir]).array(mfi);
#endif
        amrex::Array4<amrex::Real const> const& w0 = (m_disc.m_v.m_data)[mfi].const_array();
        amrex::GpuArray<amrex::Real, 3> idrLoc{m_disc.m_idr};
        amrex::Real idVolLoc{m_disc.m_idVol};
        // starts the loop inside the box,
        ParamsClass myparams;
        ParallelFor(
            bx,
            [=, fvrec = m_fvrec] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                dofXYZ_Array3D<polyrecDegreeFv, nVarPde> whwenoxyz3d;
                fvrec.template compute_weno_v3_reconstruction<ParamsClass, PdeTagType, userRecMode>(
                    whwenoxyz3d, w0, myparams, i, j, k);
                // x-face V3 face-reconstructed quadrature points
                fvrec.compute_weno_v3_to_x_face_reconstruction(qLx, qRx, whwenoxyz3d, i, j, k);
#if AMREX_SPACEDIM > 1
                // y-face V3 face-reconstructed quadrature points
                fvrec.compute_weno_v3_to_y_face_reconstruction(qLy, qRy, whwenoxyz3d, i, j, k);
#endif
#if AMREX_SPACEDIM > 2
                // z-face V3 face-reconstructed quadrature points
                fvrec.compute_weno_v3_to_z_face_reconstruction(qLz, qRz, whwenoxyz3d, i, j, k);
#endif
            });

        ParallelFor(
            bx,
            [=, fvrec = m_fvrec] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                std::array<amrex::Real, 3> idFace{idVolLoc / idrLoc[0], idVolLoc / idrLoc[1],
                                                  idVolLoc / idrLoc[2]};

                dofXYZ_V2x_Array3D<polyrecDegreeFv> bxhwenoxyz3d;
                fvrec.compute_weno_v2x_reconstruction(bxhwenoxyz3d, bx0, idFace[0], i, j, k);

                dofXYZ_V2y_Array3D<polyrecDegreeFv> byhwenoxyz3d;
                fvrec.compute_weno_v2y_reconstruction(byhwenoxyz3d, by0, idFace[1], i, j, k);

                dofXYZ_V2z_Array3D<polyrecDegreeFv> bzhwenoxyz3d;
                fvrec.compute_weno_v2z_reconstruction(bzhwenoxyz3d, bz0, idFace[2], i, j, k);
                // Bx x-face V2x face-reconstructed quadrature points
                fvrec.compute_weno_v2x_to_x_face_reconstruction(qLx, qRx, bxhwenoxyz3d, bxIdx, i, j,
                                                                k);
#if AMREX_SPACEDIM > 1
                // By y-face V2y face-reconstructed quadrature points
                fvrec.compute_weno_v2y_to_y_face_reconstruction(qLy, qRy, byhwenoxyz3d, byIdx, i, j,
                                                                k);

#endif
#if AMREX_SPACEDIM > 2
                // Bz z-face V2z face-reconstructed quadrature points
                fvrec.compute_weno_v2z_to_z_face_reconstruction(qLz, qRz, bzhwenoxyz3d, bzIdx, i, j,
                                                                k);
#endif
            });
    }
#ifdef AMREX_USE_GPU
    using dataStruct = amrex::GpuArray<amrex::Real, faceDof>;
    aos2soa_fab<dataStruct, AMREX_SPACEDIM>(m_disc.m_qlRecFab2mfab, m_disc.m_qlRecFab);
    aos2soa_fab<dataStruct, AMREX_SPACEDIM>(m_disc.m_qrRecFab2mfab, m_disc.m_qrRecFab);
    sum_boundary_sync(m_disc.m_qrRecFab2mfab[xDir], m_disc.m_compDom.m_geom.periodicity());
    sum_boundary_sync(m_disc.m_qlRecFab2mfab[xDir], m_disc.m_compDom.m_geom.periodicity());
#if AMREX_SPACEDIM > 1
    sum_boundary_sync(m_disc.m_qrRecFab2mfab[yDir], m_disc.m_compDom.m_geom.periodicity());
    sum_boundary_sync(m_disc.m_qlRecFab2mfab[yDir], m_disc.m_compDom.m_geom.periodicity());
#endif
#if AMREX_SPACEDIM > 2
    sum_boundary_sync(m_disc.m_qrRecFab2mfab[zDir], m_disc.m_compDom.m_geom.periodicity());
    sum_boundary_sync(m_disc.m_qlRecFab2mfab[zDir], m_disc.m_compDom.m_geom.periodicity());
#endif
    soa2aos_fab<dataStruct, AMREX_SPACEDIM>(m_disc.m_qlRecFab, m_disc.m_qlRecFab2mfab);
    soa2aos_fab<dataStruct, AMREX_SPACEDIM>(m_disc.m_qrRecFab, m_disc.m_qrRecFab2mfab);
#else
    sum_boundary_sync(m_disc.m_qrRecFab[xDir], m_disc.m_compDom.m_geom.periodicity());
    sum_boundary_sync(m_disc.m_qlRecFab[xDir], m_disc.m_compDom.m_geom.periodicity());
#if AMREX_SPACEDIM > 1
    sum_boundary_sync(m_disc.m_qrRecFab[yDir], m_disc.m_compDom.m_geom.periodicity());
    sum_boundary_sync(m_disc.m_qlRecFab[yDir], m_disc.m_compDom.m_geom.periodicity());
#endif
#if AMREX_SPACEDIM > 2
    sum_boundary_sync(m_disc.m_qrRecFab[zDir], m_disc.m_compDom.m_geom.periodicity());
    sum_boundary_sync(m_disc.m_qlRecFab[zDir], m_disc.m_compDom.m_geom.periodicity());
#endif
#endif
    // here we communicate more than necessary! at least we don't need all the weno stencil, just
    // the dof of the first cell:
    //amrex::IntVect m_nGhost_tmp{m_disc.m_deRham_FaceDof->get_n_ghost()};
    // LOOP OVER FACES:
    // - compute x fluxes, starting from right and left states via numerical flux.
    for (int idir = 0; idir < AMREX_SPACEDIM; idir++)
    {
        for (int jdir = 0; jdir < AMREX_SPACEDIM; jdir++)
        {
            ndir[jdir] = 0;
            n[jdir] = 0.0;
        }
        ndir[idir] = 1;
        n[idir] = 1.0;
        // precompute face areas
        amrex::Real const dVol = m_disc.m_dVol;
        amrex::Real const dS = dVol * idrLoc[idir];
        for (amrex::MFIter mfi(m_disc.m_fluxes.m_data[idir]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            amrex::Array4<amrex::Real> const& F =
                (m_disc.m_fluxes.m_data[idir])[mfi].array(); // define pointer to variable e
            amrex::Array4<amrex::GpuArray<
                amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const&& qr =
                (m_disc.m_qrRecFab[idir]).array(mfi);
            amrex::Array4<amrex::GpuArray<
                amrex::Real, FVreconstruction<nVarPde, polyrecDegreeFv>::s_faceDof>> const&& ql =
                (m_disc.m_qlRecFab[idir]).array(mfi);

            ParamsClass myparams;
            ParallelFor(bx,
                        [=, fvrec = m_fvrec] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            /**
                                here we must compute the surface integral!
                                - for loop over the 2d quadrature points
                                - solve the nodal Riemann problem
                                - add quadrature contribution to the surface integral
                            */
                            fvrec.template compute_fv_fluxes<ParamsClass, PdeTagType, userRecMode>(
                                F, ql, qr, myparams, idir, n, dS, i, j, k);
                        });
        }
    }

    /*************************************************************************************************************/
    // LOOP OVER BARYCENTERS
    // - update cell averages with numerical fluxes:    Q_i - dt/dx*(F_{i+1/2} - F_{i-1/2})
    int ncomp = nVarPde;
    for (amrex::MFIter mfi(m_disc.m_fields->m_q.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& dQdt = (m_disc.m_fieldsDtEx->m_q.m_data)[mfi].array();
        amrex::Array4<amrex::Real const> const& Fx =
            (m_disc.m_fluxes.m_data[xDir])[mfi].const_array();
#if AMREX_SPACEDIM > 1
        amrex::Array4<amrex::Real const> const& Fy =
            (m_disc.m_fluxes.m_data[yDir])[mfi].const_array();
#endif
#if AMREX_SPACEDIM > 2
        amrex::Array4<amrex::Real const> const& Fz =
            (m_disc.m_fluxes.m_data[zDir])[mfi].const_array();
#endif
        ParallelFor(bx, ncomp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int iVar)
                    {
                        dQdt(i, j, k, iVar) = -((Fx(i + 1, j, k, iVar) - Fx(i, j, k, iVar))
#if AMREX_SPACEDIM > 1
                                                + (Fy(i, j + 1, k, iVar) - Fy(i, j, k, iVar))
#endif
#if AMREX_SPACEDIM > 2
                                                + (Fz(i, j, k + 1, iVar) - Fz(i, j, k, iVar))
#endif
                        );
                    });
    }
    if (Gempic::Utils::Verbosity::level() > 1)
    {
        printf("    ---  solution updated, end single_time_step!! \n");
    }
}

/**
 * MHDFV Numerical Scheme Class.
 *  - m_disc.myfields: in and out state.
 *  - m_disc.myfields_dt_ex: unknown time-derivative (to be computed).
 *  - m_disc.myfields_tmp_ex: auxiliary temporary state.
 */
class MhdfvNumericalScheme : public NumericalScheme<MHDFieldsHandlerStruct>
{
public:
    ReconstructionMode m_recmode;
    Gempic::Io::Parameters m_parameters{};

    MHD<nVarMHD, polyrecDegreeMHD, MhdPdeParameters, MhdPdeTag> m_mhd;

    int m_nmaxSteps{1000};

    /**
     * @brief get the pointer to the field used as initial condition for the explicit time-step, by
     * reference
     * @return shared pointer of FieldsHandlerStruct
     */
    std::shared_ptr<MHDFieldsHandlerStruct> const& get_pointer_initial_field_explicit () override
    {
        return m_mhd.m_disc.m_fieldsTmpEx;
    }

    std::shared_ptr<MHDFieldsHandlerStruct> const& get_pointer_initial_field_implicit () override
    {
        return m_mhd.m_disc.m_fieldsTmpIm;
    }

    std::shared_ptr<MHDFieldsHandlerStruct> const& get_pointer_time_derivative_explicit () override
    {
        return m_mhd.m_disc.m_fieldsDtEx;
    }

    std::shared_ptr<MHDFieldsHandlerStruct> const& get_pointer_time_derivative_implicit () override
    {
        return m_mhd.m_disc.m_fieldsDtIm;
    }

    /**
     * set the initial condition.
     */
    void set_initial_condition () override
    {
        // set initial conditionm_timeStep
        m_mhd.m_disc.m_time = 0.0;
        m_mhd.m_disc.m_timeStep = 0;
        m_mhd.m_disc.m_tend = m_mhd.m_disc.m_iopar.m_tEnd;
        m_mhd.m_disc.m_iopar.m_timeIo = m_mhd.m_disc.m_time + m_mhd.m_disc.m_iopar.m_dtIo;
        m_nmaxSteps = m_mhd.m_disc.m_iopar.m_nmaxSteps;
        //amrex::Print() << "Initialization step 01_00" << std::endl;
        m_mhd.init();
    }

    void compute_norms_reduce ()
    {
        m_mhd.compute_norms_reduce(); // e.g. evaluate initial time-step
    }

    void one_step_solve_dqdt_explicit (amrex::Real rkTime) override
    {
        if (m_recmode == ReconstructionMode::CharacteristicVars)
        {
            m_mhd.single_time_step<
                ReconstructionMode::CharacteristicVars>(); // compute dQ using polynomial
                                                           // reconstruction in characteristic
                                                           // variables
        }
        else
        {
            m_mhd.single_time_step<
                ReconstructionMode::PrimitiveVars>(); // compute dQ using polynomial reconstruction
                                                      // in primitive variables
        }
    }

    /**
     * compute time derivative for the implicit solver.
     *
     * @param dtloc : time step
     */
    void one_step_eval_dqdt_implicit (amrex::Real dtLoc) override { return; }

    /** TO BE IMPLEMENTED
     * solve one implicit (Euler) step
     *
     *      here we must
     *       - glue the Ex -> Im steps;
     *           i.
     *       - update the MatOp;
     *             - Q => m
     *     - m + dp^{n+1,r}
     *     - compute rhoKb(m)
     *     - restore m
     *     - Picard iterations:
     *       - compute enthalpy => (update Matop)
     *       - rhs
     *       - Newton (gen EOS) or CG
     *       - m + dp^{n+1}
     *       - compute rhoKb
     *     - (end inner acoustic PicarD)
     *     - Update Kin Energy (UpdateRhoE)
     *     - update Q(5)=E=rhoE
     *
     */
    void one_step_solve_q_implicit (amrex::Real rkTime, amrex::Real dtLoc) override { return; }

    /**
     * compute time derivative for the semi-implicit solver (stored in m_fieldsDtIm).
     *  k_i =  H(qExplicit,qStar + dt*aii*k_i
     *  dtloc = dt*aii
     * =>  qImplicit = qStar + dtloc*(HE(qExplicit) + HI(qImplicit))
     *
     * we assume:
     *  - qExplicit and qStar already computed in the RK algorithm.
     * then:
     *  i. first compute HE(qExplicit): one_step_solve_dqdt_explicit
     *  ii. solve for qImplicit
     *  iii. evaluate k_i := (qImplicit - qStar)/dtLoc.
     *  => continue with the RK algorithm.
     *
     * @param dtloc : time step
     */
    void one_step_solve_dqdt_semiimplicit (amrex::Real timeI,
                                           amrex::Real dtI,
                                           amrex::Real timeE,
                                           amrex::Real dtE) override
    {
        return;
    }

    /**
     * @brief return `true` if the current time is greater than or equal to the final time,
     * otherwise `false`.
     */
    bool is_finaltime_reached () override { return m_mhd.m_disc.m_time >= m_mhd.m_disc.m_tend; }
    /**
     * initialize new time step.
     */
    void init_new_timestep () override
    {
        m_mhd.m_disc.m_dt = m_mhd.m_disc.m_dtNew;
        if (m_mhd.m_disc.m_time + m_mhd.m_disc.m_dt > m_mhd.m_disc.m_tend + 1e-14)
        {
            m_mhd.m_disc.m_dt = m_mhd.m_disc.m_tend - m_mhd.m_disc.m_time + 1e-14;
        }
        if (m_mhd.m_disc.m_iopar.m_printOutTime &&
            m_mhd.m_disc.m_time + m_mhd.m_disc.m_dt > m_mhd.m_disc.m_iopar.m_timeIo)
        {
            m_mhd.m_disc.m_dt = m_mhd.m_disc.m_iopar.m_timeIo - m_mhd.m_disc.m_time + 1e-14;
        }
        m_mhd.m_disc.m_timeStep++;
    }

    /**
     * finalize current time step.
     */
    void finalize_new_timestep () override { m_mhd.m_disc.m_time += m_mhd.m_disc.m_dt; }

    /**
     * print now.
     */
    void print_now ()
    {
        m_mhd.m_disc.m_fields->fill_boundary();
        m_mhd.m_disc.m_diagn.compute_and_write_to_file(m_mhd.m_disc.m_timeStep, m_mhd.m_disc.m_time,
                                                       true);
    }

    /**
     * check and eventually print.
     */
    void check_and_print (int timeStep)
    {
        if (m_mhd.m_disc.m_iopar.m_printOutNSteps &&
            timeStep % m_mhd.m_disc.m_iopar.m_nStepsIo == 0)
        {
            print_now();
        }
        if (m_mhd.m_disc.m_iopar.m_printOutTime &&
            m_mhd.m_disc.m_time >= m_mhd.m_disc.m_iopar.m_timeIo)
        {
            while (m_mhd.m_disc.m_time >= m_mhd.m_disc.m_iopar.m_timeIo)
            {
                m_mhd.m_disc.m_iopar.m_timeIo += m_mhd.m_disc.m_iopar.m_dtIo;
            }
            print_now();
        }
    }

    /**
     * get the number of components.
     */
    int get_ncomp () override { return nVarMHD; }

    /**
     * get periodicity.
     */
    amrex::Periodicity get_periodicity () override
    {
        return m_mhd.m_disc.m_compDom.m_geom.periodicity();
    }

    MhdfvNumericalScheme(int myrank, int nMPIranks) : m_mhd()
    {
        Gempic::Io::Parameters params("TimeLoop");
        amrex::Real dt;
        amrex::Real tend;
        params.get("dt", dt);
        m_mhd.m_disc.m_dt = dt;
        params.get("tend", tend);
        m_mhd.m_disc.m_tend = tend;
        Gempic::Io::Parameters paramsFV("FV");
        std::basic_string<char> recmodeStr;
        paramsFV.get("ReconstructionMode", recmodeStr);
        if (recmodeStr == "CharacteristicVars")
        {
            m_recmode = ReconstructionMode::CharacteristicVars;
        }
        else
        {
            m_recmode = ReconstructionMode::PrimitiveVars; // default
        }
    }
};

#endif
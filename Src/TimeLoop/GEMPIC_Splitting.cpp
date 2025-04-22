#include "GEMPIC_Splitting.H"

using namespace Gempic::TimeLoop;

/**
 * @brief solves the ODEs corresponding to H_J part of the Hamiltonian in the cold plasma model.
 *        Rotates and integrates cold current.
 *
 * @param[out] J cold current fields as dual 2-form (modified by the function)
 * @param[out] D electric field as dual 2-form (modified by the function)
 * @param deRham object describing the discrete deRham sequence
 * @param funcBEquilibrium background magnetic field function
 * @param dt time step
 *
 */
void Gempic::TimeLoop::apply_h_j (
    DeRhamField<Grid::dual, Space::face>& J,
    DeRhamField<Grid::dual, Space::face>& D,
    std::shared_ptr<FDDeRhamComplex> deRham,
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3> funcBEquilibrium,
    amrex::Real dt)
{
    for (amrex::MFIter mfi(J.m_data[xDir], true); mfi.isValid(); ++mfi)
    {
        // Grow box to compute on relevant indices for all components
        const amrex::Box& bx = mfi.growntilebox(1);

        amrex::Array4<amrex::Real> const& Dx = (D.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Dy = (D.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Dz = (D.m_data[zDir])[mfi].array();

        amrex::Array4<amrex::Real> const& Jx = (J.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Jy = (J.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Jz = (J.m_data[zDir])[mfi].array();

        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = deRham->m_geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi = deRham->m_geom.InvCellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = deRham->get_prob_lo();

        ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
#if AMREX_SPACEDIM == 1
                amrex::GpuArray<amrex::Real, 3> jVec = {Jx(i, j, k) * dx[xDir], Jy(i, j, k),
                                                        Jz(i, j, k)};
#elif AMREX_SPACEDIM == 2
                amrex::GpuArray<amrex::Real, 3> jVec = {Jx(i, j, k) * dx[xDir],
                                                        Jy(i, j, k) * dx[yDir], Jz(i, j, k)};
#elif AMREX_SPACEDIM == 3
                amrex::GpuArray<amrex::Real, 3> jVec = {
                    Jx(i, j, k) * dx[xDir], Jy(i, j, k) * dx[yDir], Jz(i, j, k) * dx[zDir]};
#endif

                amrex::GpuArray<amrex::Real, 3> jVecHat = {0., 0., 0.};
                amrex::GpuArray<amrex::Real, 3> dVec = {0., 0., 0.};
                amrex::GpuArray<amrex::Real, 3> dVecHat = {0., 0., 0.};

                // Compute position of dual cell center/ primal node for evaluating magnetic field
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {AMREX_D_DECL(
                    r0[xDir] + i * dx[xDir], r0[yDir] + j * dx[yDir], r0[zDir] + k * dx[zDir])};
                amrex::GpuArray<amrex::Real, 3> bZero;
                bZero[xDir] = funcBEquilibrium[xDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);
                bZero[yDir] = funcBEquilibrium[yDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);
                bZero[zDir] = funcBEquilibrium[zDir](AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);

                // Build rotation matrix with local background magnetic field
                const amrex::Real bNorm =
                    sqrt(bZero[xDir] * bZero[xDir] + bZero[yDir] * bZero[yDir] +
                         bZero[zDir] * bZero[zDir]);
                amrex::Array2D<amrex::Real, 0, 2, 0, 2> R; //rotation matrix
                bool noBZero = false;

                if (bZero[xDir] == 0 && bZero[yDir] == 0 && bZero[zDir] == 0)
                {
                    noBZero = true;
                }
                else if (bZero[xDir] == 0 && bZero[yDir] == 0)
                {
                    // Field already aligned, no transformation needed
                    R(0, 0) = 1;
                    R(1, 1) = 1;
                    R(2, 2) = 1;
                    R(0, 1) = 0;
                    R(0, 2) = 0;
                    R(1, 0) = 0;
                    R(1, 2) = 0;
                    R(2, 0) = 0;
                    R(2, 1) = 0;
                }
                else
                {
                    // Using Rodrigues' Rotation Formula in matrix notation
                    amrex::Real kN =
                        1 / sqrt(bZero[xDir] * bZero[xDir] + bZero[yDir] * bZero[yDir]);
                    amrex::Real kN2 = 1 / (bZero[xDir] * bZero[xDir] + bZero[yDir] * bZero[yDir]);
                    amrex::Real theta = acos(bZero[zDir] / bNorm);
                    amrex::Real s = sin(theta);
                    amrex::Real c = 1 - cos(theta);

                    R(0, 0) = 1 - c * kN2 * bZero[xDir] * bZero[xDir];
                    R(0, 1) = -c * kN2 * bZero[xDir] * bZero[yDir];
                    R(0, 2) = -s * kN * bZero[xDir];
                    R(1, 0) = -c * kN2 * bZero[xDir] * bZero[yDir];
                    R(1, 1) = 1 - c * kN2 * bZero[yDir] * bZero[yDir];
                    R(1, 2) = -s * kN * bZero[yDir];
                    R(2, 0) = s * kN * bZero[xDir];
                    R(2, 1) = s * kN * bZero[yDir];
                    R(2, 2) = 1 - c;
                }

                if (!noBZero)
                {
                    // Rotate reference frame s.th. B is along z-direction
                    for (int m = 0; m < 3; m++)
                    {
                        for (int n = 0; n < 3; n++)
                        {
                            jVecHat[m] += R(m, n) * jVec[n];
                        }
                    }

                    // 2D rotation
                    amrex::Real s = sin(bNorm * dt);
                    amrex::Real c = cos(bNorm * dt);
                    // Integrate cold current over time step
                    dVecHat[0] = (jVecHat[yDir] * (1 - c) + jVecHat[xDir] * s) / bNorm;
                    dVecHat[1] = (jVecHat[yDir] * s - jVecHat[xDir] * (1 - c)) / bNorm;
                    dVecHat[2] = jVecHat[zDir] * dt;
                    // Rotate cold current
                    amrex::Real jTmp = jVecHat[yDir] * s + jVecHat[xDir] * c;
                    jVecHat[yDir] = jVecHat[yDir] * c - jVecHat[xDir] * s;
                    jVecHat[xDir] = jTmp;

                    // Rotate back to original frame
                    for (int m = 0; m < 3; m++)
                    {
                        jVec[m] = 0.;
                        for (int n = 0; n < 3; n++)
                        {
                            jVec[m] += R(n, m) * jVecHat[n];
                            dVec[m] += R(n, m) * dVecHat[n];
                        }
                    }

#if AMREX_SPACEDIM == 1
                    Jx(i, j, k) = jVec[xDir] * dxi[xDir];
                    Jy(i, j, k) = jVec[yDir];
                    Jz(i, j, k) = jVec[zDir];

                    Dx(i, j, k) -= dVec[xDir] * dxi[xDir];
                    Dy(i, j, k) -= dVec[yDir];
                    Dz(i, j, k) -= dVec[zDir];
#elif AMREX_SPACEDIM == 2
                    Jx(i, j, k) = jVec[xDir] * dxi[xDir];
                    Jy(i, j, k) = jVec[yDir] * dxi[yDir];
                    Jz(i, j, k) = jVec[zDir];

                    Dx(i, j, k) -= dVec[xDir] * dxi[xDir];
                    Dy(i, j, k) -= dVec[yDir] * dxi[yDir];
                    Dz(i, j, k) -= dVec[zDir];
#elif AMREX_SPACEDIM == 3
                    Jx(i, j, k) = jVec[xDir] * dxi[xDir];
                    Jy(i, j, k) = jVec[yDir] * dxi[yDir];
                    Jz(i, j, k) = jVec[zDir] * dxi[zDir];

                    Dx(i, j, k) -= dVec[xDir] * dxi[xDir];
                    Dy(i, j, k) -= dVec[yDir] * dxi[yDir];
                    Dz(i, j, k) -= dVec[zDir] * dxi[zDir];
#endif
                }
                else
                {
                    Dx(i, j, k) -= dt * Jx(i, j, k);
                    Dy(i, j, k) -= dt * Jy(i, j, k);
                    Dz(i, j, k) -= dt * Jz(i, j, k);
                }
            });
    }
    J.fill_boundary();
    D.fill_boundary();
}
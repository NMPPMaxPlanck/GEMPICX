/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX_FFT.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"

namespace Gempic::FieldSolvers
{
/** @brief FFT Poisson solver
 *
 *  Solves the equation \f$ -\tilde{D}HG \phi = \rho \f$ for \f$\phi\f$ using the FFT forward
 *  (\f$\mathcal{F}\f$) and backward operators (\f$\mathcal{F}^{-1}\f$).
 *
 *  AMReX provides FFT operators from real to complex and from complex to real.
 *  In our case, both are needed.
 *
 *  Algorithm process:
 *  -# determine matrices \f$G\f$, \f$H\f$, \f$\tilde{D}\f$:
 *      - only need the first rows since they are all circulant
 *  -# calculate eigenvalues \f$\lambda\f$ of \f$-\tilde{D}HG\f$ for its diagonal form
 *      - the Hodge of the x-component is the tensor product \f$H^1_x \otimes H^0_y \otimes
 *        H^0_z\f$ of 1D stencils (see get_hodge_stencils), so it is not separable:
 *      - \f$\lambda_{total} = \lambda_x h^1_x h^0_y h^0_z + h^0_x \lambda_y h^1_y h^0_z +
 *        h^0_x h^0_y \lambda_z h^1_z\f$
 *      - \f$\lambda_{x_k} = (2-2\cos(2\pi k/N_x))\f$, \f$h_{x_k}\f$ the \f$k^{th}\f$ eigenvalue of
 *        the corresponding 1D Hodge stencil
 *  -# apply FFT :
 *      - \f$\phi = \mathcal{F}^{-1}(\mathcal{F}(\rho) / \lambda_{total})\f$
 *
 *  Due to the way the eigenvalues are calculated, it is necessary to perform the Fourier
 *  transform between cell-centered MultiFabs.
 *  rho and phi are both node-centered and as such, we need to copy values back and forth.
 *
 * @param[out] phi The solution vector
 * @param rho The right-hand side vector
 *
 */
class PoissonFFTSolver : public PoissonSolverMethod
{
private:
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_r2c;
    amrex::MultiFab m_rhoFft;
    amrex::MultiFab m_phiFft;

    ComputationalDomain m_compDom;
    // Per direction: eigenvalues of the 1D Laplacian times the Hodge stencil along the
    // derivative direction (including scaling), and of the Hodge stencil across it
    std::array<amrex::Gpu::DeviceVector<amrex::Real>, AMREX_SPACEDIM> m_eigenvaluesAlongGpu;
    std::array<amrex::Gpu::DeviceVector<amrex::Real>, AMREX_SPACEDIM> m_eigenvaluesAcrossGpu;

public:
    PoissonFFTSolver(ComputationalDomain const& compDom,
                     int const hodgeDegree,
                     Forms::HodgeScheme hodgeScheme = Forms::HodgeScheme::FDHodge);

    void solve(Forms::DeRhamField<Forms::Grid::primal, Forms::Space::node>& phi,
               Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell> const& rho) override;
};
} //namespace Gempic::FieldSolvers

/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_VLASOVMAXWELLIMPLICITFFT_H
#define GEMPIC_VLASOVMAXWELLIMPLICITFFT_H

#include <AMReX_FFT.H>
#include <AMReX_GpuComplex.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"

namespace Gempic::FieldSolvers
{

/** @brief VlasovMaxwell Semi-Implicit FFT solver
 *
 *  Solves the implicit Maxwell system using FFT for the time stepping.
 *  This solver handles the implicit coupling between electric and magnetic fields
 *  in the Maxwell equations using Fourier transforms for efficiency.
 *
 *  The solver implements System 3 from the VlasovMaxwell implicit scheme:
 *  - Updates E and B fields from \f$ E^n \f$, \f$ B^n \f$ to \f$ E^{n+1} \f$, \f$ B^{n+1} \f$
 *  - Uses FFT to solve the linear system efficiently
 *
 * Algorithm process:
 * 1. Setup RHS using current \f$ E^n \f$ and \f$ B^n \f$ fields
 * 2. Apply FFT to transform to frequency domain
 * 3. Solve the linear system in frequency domain using eigenvalues
 * 4. Apply inverse FFT to get updated fields
 *
 */
class VlasovMaxwellSemiImplicitFFTSolver
{
private:
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_r2c;
    ComputationalDomain m_compDom;
    std::shared_ptr<Forms::FDDeRhamComplex> m_deRham;

    // Eigenvalue vectors for derivative operators
    amrex::Gpu::DeviceVector<amrex::GpuComplex<amrex::Real>> m_eigenDxGpu, m_eigenDyGpu,
        m_eigenDzGpu;
    amrex::Gpu::DeviceVector<amrex::GpuComplex<amrex::Real>> m_eigenDTxGpu, m_eigenDTyGpu,
        m_eigenDTzGpu;

    amrex::GpuComplex<amrex::Real>* m_eigenDx; // Pointers to GPU-compatible adresses
    amrex::GpuComplex<amrex::Real>* m_eigenDy;
    amrex::GpuComplex<amrex::Real>* m_eigenDz;
    amrex::GpuComplex<amrex::Real>* m_eigenDTx;
    amrex::GpuComplex<amrex::Real>* m_eigenDTy;
    amrex::GpuComplex<amrex::Real>* m_eigenDTz;

    // Eigenvalue vectors for mass matrices
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenM1xGpu, m_eigenM1yGpu, m_eigenM1zGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenM2xGpu, m_eigenM2yGpu, m_eigenM2zGpu;

    amrex::Real* m_eigenM1x; // Pointers to GPU-compatible adresses
    amrex::Real* m_eigenM1y; // (or CPU adresses if no GPU present)
    amrex::Real* m_eigenM1z;
    amrex::Real* m_eigenM2x;
    amrex::Real* m_eigenM2y;
    amrex::Real* m_eigenM2z;

    int m_hodgeDegree;

    // Helper methods
    void setup_eigenvalues();
    void setup_mass_matrix_eigenvalues();
    void setup_derivative_eigenvalues();

public:
    VlasovMaxwellSemiImplicitFFTSolver(ComputationalDomain const& compDom,
                                       std::shared_ptr<Forms::FDDeRhamComplex> deRham,
                                       int const hodgeDegree);

    /** @brief Solve one implicit time step for Maxwell equations
     *
     * @param[in,out] E Electric field, updated from \f$ E^n \f$ to \f$ E^{n+1} \f$
     * @param[in,out] B Magnetic field, updated from \f$ B^n \f$ to \f$ B^{n+1} \f$
     * @param[in] dt Time step size
     */
    void solve_implicit_step(Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
                             Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
                             amrex::Real dt);

    /** @brief Setup RHS for the linear system
     *
     * @param[out] rhs Right-hand side vector
     * @param[in] E Electric field \f$ E^n \f$
     * @param[in] B Magnetic field \f$ B^n \f$
     * @param[in] dt Time step size
     */
    void setup_rhs(Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
                   Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& E,
                   Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
                   amrex::Real dt);

    /** @brief Update magnetic field
     *         \f$ B^{n+1} = B^n - \mathrm{curl}(E^n) dt/2 - \mathrm{curl}(E^{n+1}) dt/2 \f$
     *
     * @param[in] eOld Electric field \f$ E^n \f$
     * @param[in] eNew Electric field \f$ E^{n+1} \f$
     * @param[in,out] B Magnetic field to update
     * @param[in] dt Time step size
     */
    void update_magnetic_field(Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& eOld,
                               Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>& eNew,
                               Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>& B,
                               amrex::Real dt);
};

} // namespace Gempic::FieldSolvers

#endif // GEMPIC_VLASOVMAXWELLIMPLICITFFT_H
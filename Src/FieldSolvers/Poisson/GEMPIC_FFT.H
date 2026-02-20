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
 *      - \f$\lambda_{total} = \lambda_x + \lambda_y + \lambda_z\f$
 *      - \f$\lambda_{x_k} = (2-2\cos(2\pi k/N_x))\f$(\f$k^{th}\f$ eigenvalue of \f$H\f$)
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
class FFTSolver : public PoissonSolverMethod
{
private:
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_r2c;
    amrex::MultiFab m_rhoFft;
    amrex::MultiFab m_phiFft;

    ComputationalDomain m_compDom;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenvalues0xGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenvalues0yGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenvalues0zGpu;

    amrex::Real* m_eigenvalues0x; // Pointers to GPU-compatible adresses
    amrex::Real* m_eigenvalues0y; // (or CPU adresses if no GPU present)
    amrex::Real* m_eigenvalues0z;

public:
    FFTSolver(ComputationalDomain const& compDom,
              int const hodgeDegree,
              Forms::HodgeScheme hodgeScheme = Forms::HodgeScheme::FDHodge);

    void solve(Forms::DeRhamField<Grid::primal, Space::node>& phi,
               Forms::DeRhamField<Grid::dual, Space::cell> const& rho) override;
};
} //namespace Gempic::FieldSolvers

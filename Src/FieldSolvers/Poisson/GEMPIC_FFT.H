#include <AMReX_FFT.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"

namespace Gempic::FieldSolvers
{
/** @brief FFT Poisson solver
 *
 *  needs rhs dual grid cell rho and the solution we want in primal grid node
 *  Algorithm process:
 *  equation: GHG^{tilde} phi = rho
 *  1. input rho(cell value), calculate phi(node value)
 *  2. determine matrices G, H, G^{tilde}:
 *      ----dimension of equations (for G, G^{tilde}, H)
 *      ----order of Hodge matrix H (for H)
 *      ----only need the first rows since they are all circulant
 *  3.calculate eigenvalues of GHG^{tilde} for its diagonal form D
 *      ----eigenvalues_total = lamda_x + lamda_y + lamda_z
 *      ----lamda_x_k = [2-2*cos(2*M_PI*k/Nx)]*(kth eigenvalue of hodge_x)
 *  4.Apply FFT :
 *      ----phi = FFT_backward((FFT_forward(rho) / eigenvalues))
 *      ----amrex provides from real to complex and from complex to real.
 *          In our case, both are needed
 *
 *  Due to the way the eigenvalues are calculated, it is necessary to perform the Fourier
 *  transform between cell-centered MultiFabs.
 *  rho and phi are both node-centered and as such, we need to copy values back and forth.
 *
 * @param[out] phi : The solution vector
 * @param rho : The right-hand side vect
 *
 */
class FFTSolver : public PoissonSolverMethod
{
private:
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_r2c;

    ComputationalDomain m_compDom;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenvalues0xGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenvalues0yGpu;
    amrex::Gpu::DeviceVector<amrex::Real> m_eigenvalues0zGpu;

    amrex::Real* m_eigenvalues0x; // Pointers to GPU-compatible adresses
    amrex::Real* m_eigenvalues0y; // (or CPU adresses if no GPU present)
    amrex::Real* m_eigenvalues0z;

public:
    FFTSolver(const ComputationalDomain& compDom,
              const int hodgeDegree,
              Forms::HodgeScheme hodgeScheme = Forms::HodgeScheme::FDHodge);

    void solve(Forms::DeRhamField<Grid::primal, Space::node>& phi,
               Forms::DeRhamField<Grid::dual, Space::cell>& rho) override;
};
} //namespace Gempic::FieldSolvers

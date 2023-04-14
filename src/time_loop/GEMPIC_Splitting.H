#ifndef GEMPIC_SPLITTING_
#define GEMPIC_SPLITTING_

#include <AMReX.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_profiling.H>

using namespace Diagnostics_Output;

namespace Gempic {

namespace Time_Loop {

template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
class Splitting {
    
    public:
    void time_loop(std::shared_ptr<FDDeRhamComplex> deRham,
                   DeRhamField<Grid::dual, Space::cell>& rho,
                   DeRhamField<Grid::primal, Space::edge>& E,
                   DeRhamField<Grid::primal, Space::face>& B,
                   DeRhamField<Grid::dual, Space::face>& D,
                   DeRhamField<Grid::dual, Space::edge>& H,
                   DeRhamField<Grid::dual, Space::face>& J,
                   computational_domain infra,
                   const amrex::Real dt,
                   amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr,
                   const int nSteps,
                   int strang_order);

    virtual void treat_particles(std::shared_ptr<FDDeRhamComplex> deRham,
                                 DeRhamField<Grid::primal, Space::edge>& E,
                                 DeRhamField<Grid::primal, Space::face>& B,
                                 DeRhamField<Grid::dual, Space::face>& D,
                                 DeRhamField<Grid::dual, Space::face>& J,
                                 computational_domain infra,
                                 const amrex::Real dt,
                                 amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr) = 0;

    private:
    virtual void time_step(std::shared_ptr<FDDeRhamComplex> deRham,
                           DeRhamField<Grid::dual, Space::cell>& rho,
                           DeRhamField<Grid::primal, Space::edge>& E,
                           DeRhamField<Grid::primal, Space::face>& B,
                           DeRhamField<Grid::dual, Space::face>& D,
                           DeRhamField<Grid::dual, Space::edge>& H,
                           DeRhamField<Grid::dual, Space::face>& J,
                           DeRhamField<Grid::primal, Space::face>& auxPrimalF2,
                           DeRhamField<Grid::primal, Space::face>& auxPrimalF2_2,
                           DeRhamField<Grid::dual, Space::face>& auxDualF2,
                           DeRhamField<Grid::dual, Space::face>& auxDualF2_2,
                           computational_domain infra,
                           const amrex::Real dt,
                           amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr) = 0;

};

template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
class HamiltonianSplitting: public Splitting<vdim, numspec, degx, degy, degz, degmw,
         ndata, electromagnetic, profiling> {
        
    protected:
        /*!
        * Push the velocity with a given electric field
        */
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> push_v_efield(
            amrex::GpuArray<amrex::Real, vdim> vel, amrex::Real dt,
            amrex::Real chargemass,  // charge/mass
            amrex::GpuArray<amrex::Real, vdim> &Ep);
};

}  // namespace Time_Loop

}  // namespace Gempic

#include "GEMPIC_Splitting.cpp"

#endif
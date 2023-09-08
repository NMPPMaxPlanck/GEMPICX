#ifndef GEMPIC_SPLITTING_
#define GEMPIC_SPLITTING_

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>

using namespace CompDom;
using namespace Particles;
using namespace Profiling;
using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_PoissonSolver;

namespace Gempic {

namespace Time_Loop {

template<int vdim, int degx, int degy, int degz, int degmw>
class OperatorHamilton {
    public:

    // The template parameter is for unit testing purposes
    template<class SplineType = Spline::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_H_e_particle(
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos,
        amrex::Particle<0, 0>* AMREX_RESTRICT particles,
        computational_domain const &infra,
        amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &eA,
        SplineType &spline,
        amrex::GpuArray<amrex::Real, vdim> vel,
        amrex::ParticleReal* const AMREX_RESTRICT velx,
        amrex::ParticleReal* const AMREX_RESTRICT vely,
        amrex::ParticleReal* const AMREX_RESTRICT velz,
        amrex::Real const chargemass,
        amrex::Real const dt,
        const long pp
    ) const
    {
        // evaluate the electric field
        amrex::GpuArray<amrex::Real, vdim> efield =
            spline.template evalField<vdim, 1>(eA);

        // push v with the electric field
        amrex::GpuArray<amrex::Real, vdim> newPosE =
            push_v_efield<vdim>(vel, dt * 0.5, chargemass, efield);

        velx[pp] = newPosE[0];
        vely[pp] = newPosE[1];
        velz[pp] = newPosE[2];
    }

    template<int dim, class SplineType = Spline::SplineBase<degx, degy, degz>>
    AMREX_GPU_HOST_DEVICE inline void apply_H_p_i(
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> &pos,
                            amrex::GpuArray<amrex::Real, vdim> &vel,
                            computational_domain const &infra,
                            SplineType &spline,
                            amrex::GpuArray<amrex::Real, 2> bfields,
                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx, 
                            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> const &jA,
                            amrex::GpuArray<amrex::Array4<amrex::Real>, int(vdim / 2.5) * 2 + 1> const &bA,
                            amrex::Real const chargemass,
                            amrex::Real const weight, 
                            amrex::Real const dt
    ) const
    {
        constexpr int degrees[3] = {degx, degy, degz};

        amrex::Real x_new = pos[dim] + dt * 0.5 * vel[dim];

        spline.template update1DSplines<dim, degrees[dim]>(x_new, infra.plo[dim], infra.dxi[dim]);
        spline.template update1DPrimitive<dim, degrees[dim]>(x_new, infra.plo[dim], infra.dxi[dim]);

        accumulate_j_update_v_C2<SplineType,
                                    vdim, degrees[dim], degrees[(dim+1)%3],
                                    degrees[(dim+2)%3], dim>
                                    (spline, weight, dx, bA, jA, bfields);

        if (vdim > 1) vel[(dim+1)%3] = vel[(dim+1)%3] - chargemass * bfields[0];
        if (vdim > 2) vel[(dim+2)%3] = vel[(dim+2)%3] + chargemass * bfields[1];
        pos[dim] = x_new;
    }

    inline void apply_H_b(std::shared_ptr<FDDeRhamComplex> deRham,
                          const DeRhamField<Grid::primal, Space::face>& B,
                          DeRhamField<Grid::dual, Space::face>& D,
                          DeRhamField<Grid::dual, Space::edge>& H,
                          DeRhamField<Grid::dual, Space::face>& auxDualF2,
                          const amrex::Real dt
    )
    {
        // Operator H_B for dt/2
        // Solve Ampere-Maxwell
        deRham -> hodgeFD<degmw>(B, H);
        deRham -> curl(H, auxDualF2);
        auxDualF2 *= 0.5*dt;
        D += auxDualF2;
    }
    inline void apply_H_e_field(std::shared_ptr<FDDeRhamComplex> deRham,
                                DeRhamField<Grid::primal, Space::edge>& E,
                                DeRhamField<Grid::primal, Space::face>& B,
                                DeRhamField<Grid::dual, Space::face>& D,
                                DeRhamField<Grid::primal, Space::face>& auxPrimalF2,
                                const amrex::Real dt
    )
    {
        deRham -> hodgeFD<degmw>(D, E);
        // Operator H_E for dt/2
        deRham -> curl(E, auxPrimalF2);
        auxPrimalF2 *= 0.5*dt;
        B -= auxPrimalF2;
    }
};

template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
class Splitting {

    public:
    virtual void time_loop(std::shared_ptr<FDDeRhamComplex> deRham,
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
                   int strang_order,
                   OperatorHamilton<vdim, degx, degy, degz, degmw> operatorHamilton) = 0;

};

}  // namespace Time_Loop

}  // namespace Gempic

#endif

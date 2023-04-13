#ifndef GEMPIC_SPLITTING_
#define GEMPIC_SPLITTING_

#include <AMReX.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_diagnostics.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_profiling.H>

using namespace Diagnostics_Output;

namespace Gempic
{
namespace Time_Loop
{

template<int vdim, int numspec, int degx, int degy, int degz, int degmw,
         int ndata, bool electromagnetic, bool profiling>
class Splitting
{
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
                   int strang_order)
    {
        BL_PROFILE("time_loop_hs_zigzag_C2");

        //bool profiling_count = false;
        //timers profiling_timers(true);
        DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
        DeRhamField<Grid::primal, Space::face> auxPrimalF2_2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);
        DeRhamField<Grid::dual, Space::face> auxDualF2_2(deRham);

        /*
        for (int comp = 0; comp < 3; ++comp)
        {
            amrex::Print() << "infra_plo_mein: " << infra.plo[comp] << std::endl;
            amrex::Print() << "infra_dxi_mein: " << infra.dxi[comp] << std::endl;
        }
        */

        // if (profiling_timers.profiling) profiling_timers.counter_all -= MPI_Wtime();
        for (int t_step = 0; t_step < nSteps; t_step++)
        {
            BL_PROFILE("time_loop_hs_zigzag_C2::step");

            // Rescale the fields
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dr = {infra.geom.CellSize()[0], infra.geom.CellSize()[1], infra.geom.CellSize()[2]};
            //(rho.data).mult((dr[0] * dr[1] * dr[2]));

            (E.data[0]).mult((1/dr[0]));
            (E.data[1]).mult((1/dr[1]));
            (E.data[2]).mult((1/dr[2]));

            (D.data[0]).mult(1/(dr[1] * dr[2]));
            (D.data[1]).mult(1/(dr[0] * dr[2]));
            (D.data[2]).mult(1/(dr[0] * dr[1]));

            //(J.data[0]).mult(1/(dr[1] * dr[2]));
            //(J.data[1]).mult(1/(dr[0] * dr[2]));
            //(J.data[2]).mult(1/(dr[0] * dr[1]));

            (B.data[0]).mult(1/(dr[1] * dr[2]));
            (B.data[1]).mult(1/(dr[0] * dr[2]));
            (B.data[2]).mult(1/(dr[0] * dr[1]));

            (H.data[0]).mult(1/(dr[0]));
            (H.data[1]).mult(1/(dr[1]));
            (H.data[2]).mult(1/(dr[2]));


            //if (profiling_timers.profiling && t_step == 0) profiling_timers.counter_all -= MPI_Wtime();
            time_step(deRham, rho, E, B, D, H, J, auxPrimalF2, auxPrimalF2_2, auxDualF2, auxDualF2_2, infra, dt, partGr);

            amrex::Real D_rms = 0.0;
            for (amrex::MFIter mfi(D.data[0]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                amrex::Array4<amrex::Real> const &DMF = (D.data[0])[mfi].array();

                for (int i = lo.x; i <= hi.x; ++i) 
                    D_rms = DMF(i, 0, 0) * DMF(i, 0, 0);
            }
            D_rms = std::sqrt((1./16.)*D_rms);

            

            amrex::Print() << "finished time-step: " << t_step << " D_rms: " << D_rms << std::endl;
        }

        
    }
        
        /**
        * @brief Apply the Strang splitting to the Hamiltonian H_p.
        * 
        * Description:
        * ----------
        * Deconstruct the Hamiltonian H_p into
        * 
        * H_b * H_e * H_p1 * H_p2 * H_p3 * H_p3 * H_p2 * H_p1 * H_e * H_b
        * 
        * Parameters:
        * ----------
        * @tparam vdim: int, the number of dimensions of the velocity space v
        * @tparam numspec: int,
        * @tparam degx: int, polynomial degree of the kernel in x-dimension
        * @tparam degy: int, polynomial degree of the kernel in y-dimension
        * @tparam degz: int, polynomial degree of the kernel in z-dimension
        * @tparam degmw: int, the degree of the Maxwell-Yee approximation
        * @tparam ndata: int,
        * @tparam electromagnetic: bool, 
        * @tparam profiling: bool,
        * @param infra: computational_domain, object representing the 
        *               computational domain
        * @param mw_yee: maxwell_yee, object representing the Maxwell-Yee solver
        * @param coeff: amrex::Real, coefficient adjusting the time step size
        *               for different splitting methods
        * @param profiling_timers: timers, contains all timers for profiling the code
        * @param profiling_count: bool, indicate if it's the first time step or not
        * 
        * Returns:
        * ----------
        * Returns void
        * 
        * Raises:
        * ----------
        * Does not raise any exceptions 
        */

        virtual void treat_particles(std::shared_ptr<FDDeRhamComplex> deRham,
                                     DeRhamField<Grid::primal, Space::edge>& E,
                                     DeRhamField<Grid::primal, Space::face>& B,
                                     DeRhamField<Grid::dual, Space::face>& D,
                                     DeRhamField<Grid::dual, Space::face>& J,
                                     computational_domain infra,
                                     const amrex::Real dt,
                                     amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec>& partGr) = 0;

    private:
        /**
        * @brief Execute one time step on the Hamiltonian spliting.
        * 
        * Description:
        * ----------
        * Advance in time the individual components H_b, H_e, and H_p of the
        * Hamiltonian spliting
        * 
        * H = H_b + H_e + H_p.
        * 
        * Parameters:
        * ----------
        * @tparam vdim : int, the number of dimensions of the velocity space v
        * @tparam numspec: int,
        * @tparam degx: int, polynomial degree of the kernel in x-dimension
        * @tparam degy: int, polynomial degree of the kernel in y-dimension
        * @tparam degz: int, polynomial degree of the kernel in z-dimension
        * @tparam degmw: int, the degree of the Maxwell-Yee approximation
        * @tparam ndata: int, 
        * @tparam electromagnetic: bool, flag signaling if the model is electromagnetic
        * @tparam profiling: bool, indicates if profiling is supposed to be done
        * @param infra: computational_domain, object representing the 
        *               computational domain
        * @param mw_yee: maxwell_yee, object representing the Maxwell-Yee solver
        * @param coeff: amrex::Real, coefficient adjusting the time step size
        *               for different splitting methods
        * @param part_gr: amrex::GpuArray, array of particle_group objects
                        containing all particle information.
        * @param profiling_timers: timers, contains all timers for profiling the code
        * @param profiling_count: bool, indicate if it's the first time step or not
        * 
        * Returns:
        * ----------
        * Returns void
        * 
        * Raises:
        * ----------
        * Does not raise any exceptions 
        */
        
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
         ndata, electromagnetic, profiling>
{
    protected:
        /*!
        * Push the velocity with a given electric field
        */
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vdim> push_v_efield(
            amrex::GpuArray<amrex::Real, vdim> vel, amrex::Real dt,
            amrex::Real chargemass,  // charge/mass
            amrex::GpuArray<amrex::Real, vdim> &Ep)
        {
            amrex::GpuArray<amrex::Real, vdim> newPos;

            newPos[0] = vel[0] + dt * chargemass * (Ep[0]);
            if (vdim > 1)
            {
                newPos[1] = vel[1] + dt * chargemass * (Ep[1]);
            }
            if (vdim > 2)
            {
                newPos[2] = vel[2] + dt * chargemass * (Ep[2]);
            }
            return (newPos);
        }

};

}  // namespace Time_Loop
}  // namespace Gempic

#endif

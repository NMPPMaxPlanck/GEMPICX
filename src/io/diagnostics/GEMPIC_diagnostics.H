#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

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
#include <GEMPIC_checkpoint.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_profiling.H>

using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Utils;
using namespace Profiling;

namespace Gempic
{
namespace Diagnostics_Output
{
/**
 * @brief Compute the dot product of two MultiFabs with a mask
 *
 * @param mf1
 * @param mf2
 * @param mask
 * @return amrex::Real
 */
amrex::Real computeDotProd(amrex::MultiFab const &mf1, amrex::MultiFab const &mf2,
                           amrex::iMultiFab const &mask)
{
    auto const &data1 = mf1.const_arrays();
    auto const &data2 = mf2.const_arrays();
    auto const &mask_ma = mask.const_arrays();
    return amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real>{}, mf1,
        amrex::IntVect(0),  // zero ghost cells
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> amrex::Real {
            if (mask_ma[box_no](i, j, k))
            {
                amrex::Real prod = data1[box_no](i, j, k) * data2[box_no](i, j, k);
                return prod;
            }
            else
            {
                return 0.0;
            }
        });
}
/**
 * Class responsible for all diagnostics. The field norms, momentum and kinetic energy are stored
 * and written into files. Slice Data of fields can be generated.
 */

template <int vdim, int numspec, int degx, int degy, int degz, int degree>
class diagnostics
{
public:
    int nSteps;  // number of timesteps
    amrex::Vector<amrex::GpuArray<amrex::Real, int(vdim / 2.5) * 2 + 1>> Bfield_energy;
    amrex::Vector<amrex::GpuArray<amrex::Real, vdim>> Efield_energy, mom;
    amrex::Vector<amrex::Real> kin;
    amrex::Vector<amrex::Real> gauss_error;
    amrex::Real init_tot_energy;
    int save_fields;
    int save_particles;
    int save_checkpoint;
    int freq_save = 5;
    int lastSaved = -1;
    std::string sim_name;
    amrex::Real domain_volume;
    bool ctest;

    // The default constructor.
    /**
     * Default Constructor.
     */
    diagnostics() {}

    // Explicit const
    /**
     * Explicit constructor: object can be initialized by passing the parameters: number of time
     * steps @p n_t, the frequency with which to output positions @p fx and velocities @p fv, the
     * frequency with which to generate slices and the name of the simulation @p sim_name.
     */
    diagnostics(int n_t, int fx, int fv, int fs, std::string sim_n, amrex::Real vol,
                bool ctest = false)
    {
        initialize_diagnostics(n_t, fx, fv, fs, sim_n, vol, ctest);
    }

    void reinit(int n_t, int fx, int fv, int fs, std::string sim_n, amrex::Real vol,
                bool ctest = false)
    {
        initialize_diagnostics(n_t, fx, fv, fs, sim_n, vol, ctest);
    }

    void end_of_timestep(timers *profiling_timers, int t_step, computational_domain infra,
                         maxwell_yee<vdim> *mw_yee,
                         amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> &part_gr,
                         std::string simulation_name, bool ctest)
    {
        BL_PROFILE("diagnostics::end_of_timestep");

        if (profiling_timers->profiling) profiling_timers->counter_diagnostics -= MPI_Wtime();
        // compute reduced diagnostics at end of time step t_step, i.e. time
        ComputeReducedDiag(t_step + 1, infra, mw_yee, part_gr);

        // save reduced data every freq_save steps
        if ((t_step + 1) % freq_save == 0 || t_step + 1 == nSteps)
        {
            save_time_history(mw_yee->dt, simulation_name, t_step + 1);
        }
        // save Plot and Checkpoint-files
        if ((t_step + 1) % save_checkpoint == 0)
        {
            Gempic_WriteCheckpointFile<vdim, numspec>(mw_yee, part_gr, &infra, "Checkpoint",
                                                      sim_name, t_step + 1, 20);
        }
        // Save fields and particles
        if ((t_step + 1) % save_particles == 0)
        {
            bool save_part = true;
            Gempic_WritePlotFile<vdim, numspec>(part_gr, mw_yee, &infra, sim_name, t_step + 1,
                                                save_part);
        }
        // Save fields
        else if ((t_step + 1) % save_fields == 0)
        {
            bool save_part = false;
            Gempic_WritePlotFile<vdim, numspec>(part_gr, mw_yee, &infra, sim_name, t_step + 1,
                                                save_part);
        }

        if (profiling_timers->profiling) profiling_timers->counter_diagnostics += MPI_Wtime();
        amrex::Print() << "finished step " << t_step + 1 << std::endl;

        if (ctest) ctest_output(simulation_name, t_step);
    }

    /**
     * @brief Computes the reduced diagnostics that are integrated over space and
     * summed over particles at time step*dt
     *
     * @param step step number in time
     * @param infra compuational parameters
     * @param mw_yee
     * @param part_gr particle container
     */
    void ComputeReducedDiag(
        int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> &part_gr)
    {
        BL_PROFILE("diagnostics::ComputeReducedData");

        // make room for one more component in vector
        Bfield_energy.push_back({0., 0., 0.});
        Efield_energy.push_back({0., 0., 0.});
        mom.push_back({0., 0., 0.});
        kin.push_back(0.);
        gauss_error.push_back(0.);

        // E and B field
        mw_yee->template hodge_full<degree>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
        BL_PROFILE_VAR("diagnostics::electric energy", ehe);
        for (int cc = 0; cc < vdim; cc++)
        {
            amrex::Real val = amrex::MultiFab::Dot(*(mw_yee->E_Masks[cc]), *(mw_yee->E_Array[cc]),
                                                   0, *(mw_yee->HE_Array[cc]), 0, 1, 0);

            // amrex::Real val = computeDotProd(*(mw_yee->E_Array[cc]), *(mw_yee->HE_Array[cc]),
            //                                  *(mw_yee->E_Masks[cc]));

            val *= infra.dx[0] / infra.Length[0]
#if (GEMPIC_SPACEDIM > 1)
                   * infra.dx[1] / infra.Length[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
                   * infra.dx[2] / infra.Length[2]
#endif
                ;
            // amrex::ParallelDescriptor::ReduceRealSum(
            //     val, amrex::ParallelDescriptor::IOProcessorNumber());
            Efield_energy[step][cc] = mw_yee->epref_ep0 * val;
        }
        BL_PROFILE_VAR_STOP(ehe);

        mw_yee->template hodge_full<degree>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        BL_PROFILE_VAR("diagnostics::magnetic energy", bhb);
        for (int cc = 0; cc < int(vdim / 2.5) * 2 + 1; cc++)
        {
            amrex::Real val = amrex::MultiFab::Dot(*(mw_yee->B_Masks[cc]), *(mw_yee->B_Array[cc]),
                                                   0, *(mw_yee->HB_Array[cc]), 0, 1, 0);
            // amrex::Real val = computeDotProd(*(mw_yee->B_Array[cc]), *(mw_yee->HB_Array[cc]),
            //                                  *(mw_yee->B_Masks[cc]));
            val *= infra.dx[0] / infra.Length[0]
#if (GEMPIC_SPACEDIM > 1)
                   * infra.dx[1] / infra.Length[1]
#endif
#if (GEMPIC_SPACEDIM > 2)
                   * infra.dx[2] / infra.Length[2]
#endif
                ;
            // reduced sum over mpi ranks
            // amrex::ParallelDescriptor::ReduceRealSum(
            //     val, amrex::ParallelDescriptor::IOProcessorNumber());
            Bfield_energy[step][cc] = mw_yee->energy_b_scaling * val;
        }
        BL_PROFILE_VAR_STOP(bhb);

        // compute kinetic energy norm and momentum
        kin[step] = 0.0;
        for (int cmp = 0; cmp < vdim; cmp++)
        {
            mom[step][cmp] = 0.0;
        }
        BL_PROFILE_VAR("diagnostics::kinetic energy", kin);
        for (int spec = 0; spec < numspec; spec++)
        {  // TODO: multispecies
            auto kin_tmp = amrex::ReduceSum(
                *part_gr[spec],
                [=] AMREX_GPU_HOST_DEVICE(const amrex::Particle<vdim + 1, 0> &p) -> amrex::Real {
                    auto m = p.rdata(vdim);
                    auto vel = pow(std::abs(p.rdata(0)), 2) + pow(std::abs(p.rdata(1)), 2) +
                               pow(std::abs(p.rdata(2)), 2);
                    return (m * vel);
                });

            // reduced sum over mpi ranks
            amrex::ParallelDescriptor::ReduceRealSum(
                kin_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

            kin[step] += 0.5 * kin_tmp;
            BL_PROFILE_VAR_STOP(kin);

            BL_PROFILE_VAR("diagnostics::momentum", mom);
            for (int cmp = 0; cmp < vdim; cmp++)
            {
                auto mom_tmp =
                    amrex::ReduceSum(*part_gr[spec],
                                     [=] AMREX_GPU_HOST_DEVICE(
                                         const amrex::Particle<vdim + 1, 0> &p) -> amrex::Real {
                                         auto m = p.rdata(vdim);
                                         auto vel = p.rdata(cmp);
                                         return (m * vel);
                                     });

                // reduced sum over mpi ranks
                amrex::ParallelDescriptor::ReduceRealSum(
                    mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

                // save results for this species i_s into m_data
                mom[step][cmp] += mom_tmp;
            }
            BL_PROFILE_VAR_STOP(mom);
        }
        gauss_error[step] =
            mw_yee->template compute_gauss_error<numspec, degx, degy, degz>(infra, part_gr);

        // save initial total energy
        if (step == 0)
        {
            init_tot_energy =
                0.5 * (Efield_energy[0][0] + Efield_energy[0][1] + Efield_energy[0][2] +
                       Bfield_energy[0][0] + Bfield_energy[0][1] + Bfield_energy[0][2]) +
                kin[0] / domain_volume;
        }
    }

    /**
     * @brief Writes the time history data from the last freq_save steps so far (at time @p
     * dt * @p t_step) into an output file simulation.txt.
     *
     * @param dt time step
     * @param simulation_name
     * @param t_step time step number
     */
    void save_time_history(amrex::Real dt, std::string simulation_name, int t_step)
    {
        BL_PROFILE("diagnostics::save_time_history");

        int firstStep = lastSaved + 1;
        int lastSaved = t_step;

        if (amrex::ParallelDescriptor::IOProcessor())  // only IOProcessor prints
        {
            std::ofstream ofsfun(simulation_name + ".txt", std::ofstream::out | std::ofstream::app);
            if (t_step == 0)
            {
                firstStep = 0;
                lastSaved = 0;
            }
            for (int step = firstStep; step <= lastSaved; step++)
            {
                amrex::Real tot_energy = 0.5 * (Efield_energy[step][0] + Efield_energy[step][1] +
                                                Efield_energy[step][2] + Bfield_energy[step][0] +
                                                Bfield_energy[step][1] + Bfield_energy[step][2]) +
                                         kin[step] / domain_volume;
                amrex::Print(ofsfun).SetPrecision(20) << dt * (step) << " ";
                switch (vdim)
                {
                    case 1:
                        amrex::Print(ofsfun).SetPrecision(20) << Efield_energy[step][0];
                        break;
                    case 2:
                        amrex::Print(ofsfun).SetPrecision(20)
                            << Efield_energy[step][0] << " " << Efield_energy[step][1];
                        break;
                    case 3:
                        amrex::Print(ofsfun).SetPrecision(20)
                            << Efield_energy[step][0] << " " << Efield_energy[step][1] << " "
                            << Efield_energy[step][2];
                        break;
                }

                switch (int(vdim / 2.5) * 2 + 1)
                {
                    case 1:
                        amrex::Print(ofsfun).SetPrecision(20) << " " << Bfield_energy[step][0];
                        break;
                    case 3:
                        amrex::Print(ofsfun).SetPrecision(20)
                            << " " << Bfield_energy[step][0] << " " << Bfield_energy[step][1] << " "
                            << Bfield_energy[step][2];
                        break;
                }
                amrex::Print(ofsfun).SetPrecision(20) << " " << kin[step];
                switch (vdim)
                {
                    case 1:
                        amrex::Print(ofsfun).SetPrecision(20) << " " << mom[step][0];
                        break;
                    case 2:
                        amrex::Print(ofsfun).SetPrecision(20)
                            << " " << mom[step][0] << " " << mom[step][1];
                        break;
                    case 3:
                        amrex::Print(ofsfun).SetPrecision(20)
                            << " " << mom[step][0] << " " << mom[step][1] << " " << mom[step][2];
                        break;
                }
                amrex::Print(ofsfun).SetPrecision(20)
                    << " " << gauss_error[step] << " " << tot_energy << " "
                    << abs(tot_energy - init_tot_energy) << std::endl;
            }
            ofsfun.close();
        }
    }

    void ctest_output(std::string simulation_name, int t_step)
    {
        BL_PROFILE("diagnostics::ctest_output");

        std::string filename = simulation_name + ".output";
        switch (vdim)
        {
            case 1:
                amrex::PrintToFile(filename) << t_step << " " << Efield_energy[t_step][0];
                break;
            case 2:
                amrex::PrintToFile(filename)
                    << t_step << " " << Efield_energy[t_step][0] << " " << Efield_energy[t_step][1];
                break;
            case 3:
                amrex::PrintToFile(filename)
                    << t_step << " " << Efield_energy[t_step][0] << " " << Efield_energy[t_step][1]
                    << " " << Efield_energy[t_step][2];
                break;
        }
        switch (vdim)
        {
            case 1:  // no break, so that case 1 and 2 are handled the same way
            case 2:
                amrex::PrintToFile(filename) << " " << Bfield_energy[t_step][0];
                break;
            case 3:
                amrex::PrintToFile(filename)
                    << " " << Bfield_energy[t_step][0] << " " << Bfield_energy[t_step][1] << " "
                    << Bfield_energy[t_step][2];
                break;
        }
        amrex::PrintToFile(filename) << " " << kin[t_step] << " ";
        switch (vdim)
        {
            case 1:
                amrex::PrintToFile(filename) << mom[t_step][0];
                break;
            case 2:
                amrex::PrintToFile(filename) << mom[t_step][0] << " " << mom[t_step][1];
                break;
            case 3:
                amrex::PrintToFile(filename)
                    << mom[t_step][0] << " " << mom[t_step][1] << " " << mom[t_step][2];
                break;
        }
        amrex::PrintToFile(filename) << " " << gauss_error[t_step];
        amrex::PrintToFile(filename) << std::endl;
    }

#if (GEMPIC_SPACEDIM == 3)
    void cross_product(amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> &MultiInputE,
                       amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> &MultiInputB,
                       amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> &MultiOutput,
                       computational_domain infra)
    {
        BL_PROFILE("diagnostics::cross_product");

        amrex::MultiFab EyBz(convert(infra.grid, amrex::IntVect::TheCellVector()), infra.distriMap,
                             0, 0);
        amrex::MultiFab EzBy(convert(infra.grid, amrex::IntVect::TheCellVector()), infra.distriMap,
                             0, 0);
        amrex::MultiFab EzBx(convert(infra.grid, amrex::IntVect::TheCellVector()), infra.distriMap,
                             0, 0);
        amrex::MultiFab ExBz(convert(infra.grid, amrex::IntVect::TheCellVector()), infra.distriMap,
                             0, 0);
        amrex::MultiFab ExBy(convert(infra.grid, amrex::IntVect::TheCellVector()), infra.distriMap,
                             0, 0);
        amrex::MultiFab EyBx(convert(infra.grid, amrex::IntVect::TheCellVector()), infra.distriMap,
                             0, 0);

        std::cout << "before copy" << std::endl;
        // "E"
        EyBz.Copy(EyBz, *(MultiInputE[1]), 0, 0, 1, 0);
        EzBy.Copy(EzBy, *(MultiInputE[2]), 0, 0, 1, 0);
        EzBx.Copy(EzBx, *(MultiInputE[2]), 0, 0, 1, 0);
        ExBz.Copy(ExBz, *(MultiInputE[0]), 0, 0, 1, 0);
        ExBy.Copy(ExBy, *(MultiInputE[0]), 0, 0, 1, 0);
        EyBx.Copy(EyBx, *(MultiInputE[1]), 0, 0, 1, 0);

        std::cout << "before multiply" << std::endl;
        // "EB"
        EyBz.Multiply(EyBz, *(MultiInputB[2]), 0, 0, 1, 0);
        EzBy.Multiply(EzBy, *(MultiInputB[1]), 0, 0, 1, 0);
        EzBx.Multiply(EzBx, *(MultiInputB[0]), 0, 0, 1, 0);
        ExBz.Multiply(ExBz, *(MultiInputB[2]), 0, 0, 1, 0);
        ExBy.Multiply(ExBy, *(MultiInputB[1]), 0, 0, 1, 0);
        EyBx.Multiply(EyBx, *(MultiInputB[0]), 0, 0, 1, 0);

        std::cout << "before minus" << std::endl;
        // "EB - EB"
        EyBz.minus(EzBy, 0, 1, 0);
        EzBx.minus(ExBz, 0, 1, 0);
        ExBy.minus(EyBx, 0, 1, 0);

        std::cout << "before final copy" << std::endl;
        // Copy into out MF
        (*MultiOutput[0]).Copy(*MultiOutput[0], EyBz, 0, 0, 1, 0);
        (*MultiOutput[1]).Copy(*MultiOutput[1], EzBx, 0, 0, 1, 0);
        (*MultiOutput[2]).Copy(*MultiOutput[2], ExBy, 0, 0, 1, 0);
    }
#endif

private:
    void initialize_diagnostics(int n_t, int fx, int fv, int fs, std::string sim_n, amrex::Real vol,
                                bool ctest)
    {
        BL_PROFILE("diagnostics::initialize_diagnostics");

        nSteps = n_t;
        save_fields = fx;
        save_particles = fv;
        save_checkpoint = fs;
        sim_name = sim_n;
        domain_volume = vol;

        Bfield_energy.reserve(nSteps);
        Efield_energy.reserve(nSteps);
        mom.reserve(nSteps);
        kin.reserve(nSteps);
        gauss_error.reserve(nSteps);

        // Open time history file
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            std::ofstream ofsfun(sim_name + ".txt", std::ofstream::out);
            amrex::Print(ofsfun).SetPrecision(20)
                << "# time   E-energy-components   B-energy-components   "
                   "kinetic-energy   momentum-components gauss-error total-energy "
                   "energy-error"
                << std::endl;
            ofsfun.close();
        }

        if (ctest) amrex::PrintToFile(sim_name + ".output") << "\n";
    }
};

}  // namespace Diagnostics_Output
}  // namespace Gempic

#endif

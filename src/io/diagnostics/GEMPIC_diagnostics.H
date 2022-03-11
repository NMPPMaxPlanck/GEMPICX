#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

#include <AMReX.H>
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
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_profiling.H>
#include <GEMPIC_particle_positions.H>

using namespace Field_solvers;
using namespace CompDom;
using namespace Particles;
using namespace Utils;
using namespace Profiling;

namespace Gempic {
namespace Diagnostics_Output {

/**
 * Class responsible for all diagnostics. The field norms, momentum and kinetic energy are stored and written into files. Slice Data of fields can be generated.
 */

template<int vdim, int numspec, int degx, int degy, int degz, int degree>
class diagnostics
{
public:
    int n_time; // number of timesteps
    std::vector<amrex::GpuArray<amrex::Real,int(vdim/2.5)*2+1>> Bfield_energy;
    std::vector<amrex::GpuArray<amrex::Real,vdim>> Efield_energy, mom;
    std::vector<amrex::Real> kin;
    std::vector<amrex::Real> gauss_error;
    int freq_x;
    int freq_v;
    int freq_slice;
    std::string sim_name;
    amrex::Real domain_volume;

    // The default constructor.
    /**
     * Default Constructor.
     */
    diagnostics (){}

    // Explicit const
    /**
     * Explicit constructor: object can be initialized by passing the parameters: number of time steps @p n_t, the frequency with which to output positins @p fx and velocities @p fv,
     * the frequency with which to generate slices and the name of the simulation @p sim_name.
     */
    diagnostics (int n_t, int fx, int fv, int fs, std::string sim_n, amrex::Real vol){
        initialize_diagnostics(n_t, fx, fv, fs, sim_n, vol);
    }


    void reinit(int n_t, int fx, int fv, int fs, std::string sim_n, amrex::Real vol){
        initialize_diagnostics(n_t, fx, fv, fs, sim_n, vol);
    }


    void end_of_timestep(timers *profiling_timers, int t_step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                         particle_groups<vdim, numspec> *part_gr, std::string filename, bool ctest){
        if (profiling_timers->profiling)
          profiling_timers->counter_diagnostics -= MPI_Wtime();
        // save norms for diagnostics
        save_step_data(t_step+1, infra, mw_yee, part_gr); // false: slices are not saved

        // save temporary results every n steps
        if ((t_step+1)%5 == 0) {
            save_all_to_textfile(mw_yee->dt, filename, t_step+1);
        }
        if (profiling_timers->profiling)
          profiling_timers->counter_diagnostics += MPI_Wtime();
        amrex::Print() << "finished step " << t_step+1 << std::endl;

        if (ctest) ctest_output(filename, t_step);
    }

    // save position and velocities
    /**
     * This function saves all the relevant data for the step in field arrays and saves SlicePlots if needed according to the Slice Plot frequency.
     */
    void save_step_data(int step,
                        computational_domain infra,
                        maxwell_yee<vdim> *mw_yee,
                        particle_groups<vdim, numspec> *part_gr){

        Bfield_energy.push_back({0., 0., 0.});
        Efield_energy.push_back({0., 0., 0.});
        mom.push_back({0., 0., 0.});
        kin.push_back(0.);
        gauss_error.push_back(0.);

        // E and B field
        mw_yee->template hodge_full<degree>(infra, mw_yee->E_Array, mw_yee->HE_Array, true);
        for(int cc=0;cc<vdim;cc++){
            amrex::Real val = amrex::MultiFab::Dot(*(mw_yee->E_Masks[cc]), *(mw_yee->E_Array[cc]), 0, *(mw_yee->HE_Array[cc]), 0, 1, 0);
            val *= infra.dx[0]/infra.Length[0]
            #if(GEMPIC_SPACEDIM > 1)
                    *infra.dx[1]/infra.Length[1]
            #endif
            #if(GEMPIC_SPACEDIM > 2)
                    *infra.dx[2]/infra.Length[2]
            #endif
                    ;
            Efield_energy[step][cc] = mw_yee->epref_ep0*val;
        }

        mw_yee->template hodge_full<degree>(infra, mw_yee->B_Array, mw_yee->HB_Array, false);
        for(int cc=0;cc<int(vdim/2.5)*2+1;cc++){
            amrex::Real val = amrex::MultiFab::Dot(*(mw_yee->B_Masks[cc]), *(mw_yee->B_Array[cc]), 0, *(mw_yee->HB_Array[cc]), 0, 1, 0);
            val *= infra.dx[0]/infra.Length[0]
            #if(GEMPIC_SPACEDIM > 1)
                    *infra.dx[1]/infra.Length[1]
            #endif
            #if(GEMPIC_SPACEDIM > 2)
                    *infra.dx[2]/infra.Length[2]
            #endif
                    ;
            Bfield_energy[step][cc] = mw_yee->energy_b_scaling*val;
        }

        // compute kinetic energy norm and momentum
        kin[step] = 0.0;
        for (int cmp=0;cmp<vdim;cmp++) {
            mom[step][cmp] = 0.0;
        }
        for (int spec=0;spec<numspec;spec++){ // TODO: multispecies
            auto kin_tmp = amrex::ReduceSum( *(*part_gr).mypc[spec],
                                             [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real
            {
                auto m  = p.rdata(vdim);
                auto vel = pow(std::abs(p.rdata(0)),2)+pow(std::abs(p.rdata(1)),2)+pow(std::abs(p.rdata(2)),2);
                return (m*vel);
            });


            // reduced sum over mpi ranks
            amrex::ParallelDescriptor::ReduceRealSum
                    (kin_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

            kin[step] += 0.5*kin_tmp;
            for (int cmp=0;cmp<vdim;cmp++) {
                auto mom_tmp = amrex::ReduceSum( *(*part_gr).mypc[spec],
                                                 [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real
                {
                    auto m  = p.rdata(vdim);
                    auto vel = p.rdata(cmp);
                    return (m*vel);
                });


                // reduced sum over mpi ranks
                amrex::ParallelDescriptor::ReduceRealSum
                        (mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

                // save results for this species i_s into m_data
                mom[step][cmp] += mom_tmp;
            }
        }
        gauss_error[step] = (*mw_yee).template compute_gauss_error<numspec,degx,degy,degz>(infra, part_gr);

        // save Plot and Checkpoint-files
        //if ((step == 0) || ((step+1)%freq_slice == 0)) {
        if ((step+1)%freq_slice == 0) {
           Gempic_WriteCheckpointFile(mw_yee, part_gr, &infra, "Checkpoint/" + sim_name, step+1, 20);
           Gempic_WritePlotFile(part_gr, mw_yee, &infra, sim_name, step+1);
        }

    }

    /**
     * This function writes the data from all steps so far (at time @p dt) into an output file PIC_save.output or a file of alternative name if @p filename is passed. If only a
     * subset of the data should be saved, the argument @p partial, setting the last index to be saved, should be passed.
     */
    void save_all_to_textfile(amrex::Real dt, std::string filename = "vlasov_maxwell_save.output", int partial = 0){
        if (partial == 0){
            partial = n_time;
        }
        std::ofstream ofsfun(filename, std::ofstream::out);
        amrex::Real init_tot_energy = 0.5*(Efield_energy[0][0]+Efield_energy[0][1]+Efield_energy[0][2]+Bfield_energy[0][0]+Bfield_energy[0][1]+Bfield_energy[0][2])+kin[0]/domain_volume;
        amrex::Print(ofsfun).SetPrecision(20) << "# time   E-energy-components   B-energy-components   kinetic-energy   momentum-components gauss-error total-energy energy-error" << std::endl;
        for (int step=0;step<=partial;step++){
            amrex::Real tot_energy = 0.5*(Efield_energy[step][0]+Efield_energy[step][1]+Efield_energy[step][2]+Bfield_energy[step][0]+Bfield_energy[step][1]+Bfield_energy[step][2])+kin[step]/domain_volume;
            amrex::Print(ofsfun).SetPrecision(20) << dt*(step) << " ";
            switch(vdim) {
            case 1:
                amrex::Print(ofsfun).SetPrecision(20) << Efield_energy[step][0];
                break;
            case 2:
                amrex::Print(ofsfun).SetPrecision(20) << Efield_energy[step][0] << " " << Efield_energy[step][1];
                break;
            case 3:
                amrex::Print(ofsfun).SetPrecision(20) << Efield_energy[step][0] << " " << Efield_energy[step][1] << " " << Efield_energy[step][2];
                break;

            }

            switch (int(vdim/2.5)*2+1) {
            case 1:
                amrex::Print(ofsfun).SetPrecision(20) << " " << Bfield_energy[step][0];
                break;
            case 3:
                amrex::Print(ofsfun).SetPrecision(20) << " " << Bfield_energy[step][0] << " " << Bfield_energy[step][1] << " " << Bfield_energy[step][2];
                break;
            }
            amrex::Print(ofsfun).SetPrecision(20) << " " << kin[step];
            switch(vdim) {
            case 1:
                amrex::Print(ofsfun).SetPrecision(20) << " "  << mom[step][0];
                break;
            case 2:
                amrex::Print(ofsfun).SetPrecision(20) << " "  << mom[step][0] << " " << mom[step][1];
                break;
            case 3:
                amrex::Print(ofsfun).SetPrecision(20) << " "  << mom[step][0] << " " << mom[step][1] << " " << mom[step][2];
                break;

            }
            amrex::Print(ofsfun).SetPrecision(20) << " " << gauss_error[step]
                                    << " " << tot_energy << " " << abs(tot_energy-init_tot_energy) << std::endl;
        }
        ofsfun.close();
    }

    void ctest_output(std::string simulation_name, int t_step){
        switch (vdim) {
        case 1:
            amrex::AllPrintToFile(simulation_name) << t_step << " " << Efield_energy[t_step][0];
            break;
        case 2:
            amrex::AllPrintToFile(simulation_name) << t_step << " " << Efield_energy[t_step][0] << " " << Efield_energy[t_step][1];
            break;
        case 3:
            amrex::AllPrintToFile(simulation_name) << t_step << " " << Efield_energy[t_step][0] << " " << Efield_energy[t_step][1] << " " << Efield_energy[t_step][2];
            break;
        }
        switch (vdim) {
        case 1: // no break, so that case 1 and 2 are handled the same way
        case 2:
            amrex::AllPrintToFile(simulation_name) << " " << Bfield_energy[t_step][0];
            break;
        case 3:
            amrex::AllPrintToFile(simulation_name) << " " << Bfield_energy[t_step][0] << " " << Bfield_energy[t_step][1] << " " << Bfield_energy[t_step][2];
            break;
        }
        amrex::AllPrintToFile(simulation_name) << " " << kin[t_step] << " ";
        switch (vdim) {
        case 1:
            amrex::AllPrintToFile(simulation_name) << mom[t_step][0];
            break;
        case 2:
            amrex::AllPrintToFile(simulation_name) << mom[t_step][0] << " " << mom[t_step][1];
            break;
        case 3:
            amrex::AllPrintToFile(simulation_name) << mom[t_step][0] << " " << mom[t_step][1] << " " << mom[t_step][2];
            break;
        }
        amrex::AllPrintToFile(simulation_name) << " " << gauss_error[t_step];
        amrex::AllPrintToFile(simulation_name) << std::endl;
    }

#if (GEMPIC_SPACEDIM==3)
    void cross_product (amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> *MultiInputE,
                        amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> *MultiInputB,
                        amrex::GpuArray<std::unique_ptr<amrex::MultiFab>, 3> *MultiOutput,
                        computational_domain infra) {

        amrex::MultiFab EyBz(convert(infra.grid, amrex::IntVect::TheCellVector()),infra.distriMap,0,0);
        amrex::MultiFab EzBy(convert(infra.grid, amrex::IntVect::TheCellVector()),infra.distriMap,0,0);
        amrex::MultiFab EzBx(convert(infra.grid, amrex::IntVect::TheCellVector()),infra.distriMap,0,0);
        amrex::MultiFab ExBz(convert(infra.grid, amrex::IntVect::TheCellVector()),infra.distriMap,0,0);
        amrex::MultiFab ExBy(convert(infra.grid, amrex::IntVect::TheCellVector()),infra.distriMap,0,0);
        amrex::MultiFab EyBx(convert(infra.grid, amrex::IntVect::TheCellVector()),infra.distriMap,0,0);

        std::cout << "before copy" << std::endl;
        // "E"
        EyBz.Copy(EyBz, *((*MultiInputE)[1]), 0, 0, 1, 0);
        EzBy.Copy(EzBy, *((*MultiInputE)[2]), 0, 0, 1, 0);
        EzBx.Copy(EzBx, *((*MultiInputE)[2]), 0, 0, 1, 0);
        ExBz.Copy(ExBz, *((*MultiInputE)[0]), 0, 0, 1, 0);
        ExBy.Copy(ExBy, *((*MultiInputE)[0]), 0, 0, 1, 0);
        EyBx.Copy(EyBx, *((*MultiInputE)[1]), 0, 0, 1, 0);

        std::cout << "before multiply" << std::endl;
        // "EB"
        EyBz.Multiply(EyBz, *((*MultiInputB)[2]), 0, 0, 1, 0);
        EzBy.Multiply(EzBy, *((*MultiInputB)[1]), 0, 0, 1, 0);
        EzBx.Multiply(EzBx, *((*MultiInputB)[0]), 0, 0, 1, 0);
        ExBz.Multiply(ExBz, *((*MultiInputB)[2]), 0, 0, 1, 0);
        ExBy.Multiply(ExBy, *((*MultiInputB)[1]), 0, 0, 1, 0);
        EyBx.Multiply(EyBx, *((*MultiInputB)[0]), 0, 0, 1, 0);

        std::cout << "before minus" << std::endl;
        // "EB - EB"
        EyBz.minus(EzBy, 0, 1, 0);
        EzBx.minus(ExBz, 0, 1, 0);
        ExBy.minus(EyBx, 0, 1, 0);

        std::cout << "before final copy" << std::endl;
        // Copy into out MF
        (*((*MultiOutput)[0])).Copy(*((*MultiOutput)[0]), EyBz, 0, 0, 1, 0);
        (*((*MultiOutput)[1])).Copy(*((*MultiOutput)[1]), EzBx, 0, 0, 1, 0);
        (*((*MultiOutput)[2])).Copy(*((*MultiOutput)[2]), ExBy, 0, 0, 1, 0);
    }
#endif


private :

void initialize_diagnostics (int n_t, int fx, int fv, int fs, std::string sim_n, amrex::Real vol){
    n_time = n_t;
    freq_x = fx;
    freq_v = fv;
    freq_slice = fs;
    sim_name = sim_n;
    domain_volume = vol;

    Bfield_energy.reserve(n_time);
    Efield_energy.reserve(n_time);
    mom.reserve(n_time);
    kin.reserve(n_time);
    gauss_error.reserve(n_time);
}
};


}
}

#endif

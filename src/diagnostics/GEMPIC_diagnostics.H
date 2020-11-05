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
#include <GEMPIC_infrastructure.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>

using namespace Field_solvers;
using namespace Infra;
using namespace Particles;
using namespace Utils;

#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

namespace Gempic {
namespace Diagnostics_Output {

/**
 * Class responsible for all diagnostics. The field norms, momentum and kinetic energy are stored and written into files. Slice Data of fields can be generated.
 */

template<int vdim, int numspec, int degx, int degy, int degz>
class diagnostics
{
public:
    int n_time; // number of timesteps
    // 9 static so far!! need to fix this
    std::array<std::array<amrex::Real,int(floor(vdim/2.5)*2+1)>,50001> Bfield;
    std::array<std::array<amrex::Real,vdim>,50001> Efield, mom;
    std::array<amrex::Real,50001> kin;
    std::array<amrex::Real,50001> gauss_error;
    int freq_x;
    int freq_v;
    int freq_slice;
    std::string sim_name;

    // The default constructor.
    /**
     * Default Constructor.
     */
    diagnostics ();

    // Explicit const
    /**
     * Explicit constructor: object can be initialized by passing the parameters: number of time steps @p n_t, the frequency with which to output positins @p fx and velocities @p fv,
     * the frequency with which to generate slices and the name of the simulation @p sim_name.
     */
    diagnostics (int n_t, int fx, int fv, int fs, std::string sim_n){
        n_time = n_t;
        freq_x = fx;
        freq_v = fv;
        freq_slice = fs;
        sim_name = sim_n;
    }

    // save position and velocities
    /**
     * This function saves all the relevant data for the step in field arrays and saves SlicePlots if needed according to the Slice Plot frequency.
     */
    void save_step_data(int step,
                        infrastructure infra,
                        maxwell_yee<vdim> *mw_yee,
                        particle_groups<vdim, numspec> *part_gr){

        // E and B field
        for(int cc=0;cc<vdim;cc++){
            Efield[step][cc] = gempic_norm(&(*(*mw_yee).E_Array[cc]), infra, 2);
        }

        for(int cc=0;cc<int(floor(vdim/2.5)*2+1);cc++){
            Bfield[step][cc] = gempic_norm(&(*(*mw_yee).B_Array[cc]), infra, 2);
        }

        // compute kinetic energy norm and momentum
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

            kin[step] = 0.5*kin_tmp;
            for (int cmp=0;cmp<vdim;cmp++) {
                auto mom_tmp = amrex::ReduceSum( *(*part_gr).mypc[spec],
                                                 [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real
                {
                    auto m  = p.rdata(vdim);
                    auto vel = p.rdata(cmp);
                    return (m*std::abs(vel));
                });


                // reduced sum over mpi ranks
                amrex::ParallelDescriptor::ReduceRealSum
                        (mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

                // save results for this species i_s into m_data
                mom[step][cmp] = mom_tmp;
            }
            gauss_error[step] = (*mw_yee).template compute_gauss_error<numspec,degx,degy,degz>(infra, part_gr);
        }

        // save Plot and Checkpoint-files
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
        amrex::Print(ofsfun) << "# time   E-components   B-components   kinetic-energy   momentum-components gauss-error" << std::endl;
        for (int step=0;step<=partial;step++){
            amrex::Print(ofsfun) << dt*(step) << " ";
            switch(vdim) {
            case 1:
                amrex::Print(ofsfun) << Efield[step][0];
                break;
            case 2:
                amrex::Print(ofsfun) << Efield[step][0] << " " << Efield[step][1];
                break;
            case 3:
                amrex::Print(ofsfun) << Efield[step][0] << " " << Efield[step][1] << " " << Efield[step][2];
                break;

            }

            switch (int(floor(vdim/2.5)*2+1)) {
            case 1:
                amrex::Print(ofsfun) << " " << Bfield[step][0];
                break;
            case 3:
                amrex::Print(ofsfun) << " " << Bfield[step][0] << " " << Bfield[step][1] << " " << Bfield[step][2];
                break;
            }
            switch(vdim) {
            case 1:
                amrex::Print(ofsfun) << mom[step][0];
                break;
            case 2:
                amrex::Print(ofsfun) << mom[step][0] << " " << mom[step][1];
                break;
            case 3:
                amrex::Print(ofsfun) << mom[step][0] << " " << mom[step][1] << " " << mom[step][2];
                break;

            }
            amrex::Print(ofsfun) << " " << gauss_error[step]
                                    << std::endl;
        }
        ofsfun.close();
    }
};
}
}

#endif

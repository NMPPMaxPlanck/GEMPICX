#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hsall_fem.H>
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
#include <GEMPIC_time_loop_ls_rungekutta.H>
#include <GEMPIC_time_loop_particles.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>

using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Profiling;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;


AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z)
{
    amrex::Real val = 1.0 ;//+ 0.5 * std::cos(0.5 * x);
    return val;
}


template<int vdim, int numspec, int degx, int degy, int degz, int degvm, bool electromagnetic=true, bool output=true>
class vlasov_maxwell_simulation
{
public :
    gempic_parameters<vdim, numspec> params;
    computational_domain domain;
    maxwell_yee<vdim> mw_yee;
    particle_groups<vdim, numspec> part_gr;
    diagnostics<vdim, numspec,degx,degy,degz,degvm> diagn;
    bool readinfile = false;
    bool ctest = false;


     void initialize_vlasov_maxwell_from_file()
     {
         // ------------------------------------------------------------------------------
         // ------------PARAMETERS--------------------------------------------------------
         params.init_Nghost(degx, degy, degz);
         params.read_pp_params();
         params.set_computed_params();
     }

    void initialize_gempic_structures(amrex::GpuArray<int, 2> funcSelectRho, amrex::GpuArray<int, int(vdim/2.5)*2+1> funcSelectB)
    {
        domain.initialize_computational_domain(params.n_cell, params.max_grid_size, params.is_periodic, params.real_box);
        mw_yee.reinit(domain, params.dt, params.n_steps, params.Nghost);
        mw_yee.template init_rho_phi<degvm>(domain, funcSelectRho);
        // particles
        part_gr.reinit(params.charge, params.mass, domain);
        amrex::Real vol = (domain.geom.ProbHi(0)-domain.geom.ProbLo(0))*(domain.geom.ProbHi(1)-domain.geom.ProbLo(1))*(domain.geom.ProbHi(2)-domain.geom.ProbLo(2));
        diagn.reinit(mw_yee.nsteps, params.freq_x, params.freq_v, params.freq_slice, params.sim_name, vol);
        //------------------------------------------------------------------------------
        // initialize particles & loop preparation:
        init_particles_full_domain<vdim,numspec>(domain, part_gr, params.n_part_per_cell, params.VM, params.VD, params.VW, 0, wave_function);

        loop_preparation<vdim, numspec, degx, degy, degz, degvm, true>(params, domain, &mw_yee, &part_gr, &diagn, params.time_staggered, funcSelectB);

    }

    void initialize_gempic_structures_tinyexpr()
    {
        domain.initialize_computational_domain(params.n_cell, params.max_grid_size, params.is_periodic, params.real_box);
        mw_yee.reinit(domain, params.dt, params.n_steps, params.Nghost);

        amrex::GpuArray<std::string, 2> fields = {params.rho, params.phi};
        mw_yee.template init_rho_phi<degvm>(fields, params.k, domain);

        // particles
        part_gr.reinit(params.charge, params.mass, domain);

        amrex::Real vol = (domain.geom.ProbHi(0)-domain.geom.ProbLo(0))*(domain.geom.ProbHi(1)-domain.geom.ProbLo(1))*(domain.geom.ProbHi(2)-domain.geom.ProbLo(2));
        diagnostics<vdim, numspec,degx,degy,degz,degvm> diagn(mw_yee.nsteps, params.freq_x, params.freq_v, params.freq_slice, params.sim_name, vol);

        //------------------------------------------------------------------------------
        // initialize particles & loop preparation:
        amrex::GpuArray<std::string, int(vdim/2.5)*2+1> fields_B;
        fields_B[0] = params.Bx;
        if (int(vdim/2.5)*2+1 > 1) {
            fields_B[1] = params.By;
        }
        if (int(vdim/2.5)*2+1 > 1) {
            fields_B[2] = params.Bz;
        }
        if (params.propagator==100) {
            for (int spec=0; spec<numspec; spec++) {
                for(amrex::MFIter mfi=(*(part_gr).mypc[spec]).MakeMFIter(0); mfi.isValid(); ++mfi) {
                    if(mfi.index() == 0) {
                        amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(part_gr).mypc[spec]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                        (part_gr).add_particle({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(0.0, 0.0, 0.0)}, 1.0, particles);
                    }
                }
            }
            loop_preparation<vdim, numspec, degx, degy, degz, degvm, output>(params, domain, &mw_yee, &part_gr, &diagn, params.time_staggered, fields_B);
        } else {
        if (params.restart == 0) {
            if (readinfile) {
                for (int spec=0; spec<numspec; spec++) {
                    params.init_particles_from_file(&part_gr, spec, "particle_input.txt");
                }
            } else {
                for (int spec=0; spec<numspec; spec++) {
                    params.read_particle_spec(spec);
                    init_particles_full_domain<vdim,numspec>(domain, part_gr, params.n_part_per_cell, params.k, params.WF, params.VM, params.VD, params.VW, spec);
                }
            }

            loop_preparation<vdim, numspec, degx, degy, degz, degvm, output>(params, domain, &mw_yee, &part_gr, &diagn, params.time_staggered, fields_B);
        } else {
            Gempic_ReadCheckpointFile (&mw_yee, &part_gr, &domain, params.checkpoint_file, params.curr_step);
        }
        }
    }

    void run_time_loop()
    {
        auto start = std::chrono::high_resolution_clock::now();
        switch (params.propagator) {
            case 0:
                time_loop_boris_fd<vdim, numspec, degx, degy, degz, degvm, electromagnetic, false>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 1:
                time_loop_hs_fem<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 2:
                time_loop_hsall_fem<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 3:
                time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degvm, electromagnetic, false, output>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
        case 4:
                ls_rungekutta<3> rk_prop();
                rk_prop.time_loop(domain, &mw_yee, &art_gr);
            case 100:
              //time_loop_particles<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, &ofs);
                break;
            default:
                break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        amrex::Print()  << "execution lasted: " <<  duration.count() << " s" << std::endl;
        switch (params.propagator) {
          case 100:
            Gempic_WritePlotFile(&part_gr, &mw_yee, &domain, "Edipole", 10);
            break;
          default:
            break;
        }
    }

};


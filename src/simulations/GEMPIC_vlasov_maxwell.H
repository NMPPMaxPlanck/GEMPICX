#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
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
//#include <GEMPIC_time_loop_ls_rungekutta.H>
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
    maxwell_yee<vdim, int(vdim/2.5)*2+1> mw_yee;
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
         initialize_gempic_structures();
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
        init_particles_full_domain<vdim,numspec>(domain, part_gr, params.n_part_per_cell, params.meanVelocity[0], params.vThermal[0], params.vWeight[0], 0, wave_function);

        loop_preparation<vdim, numspec, degx, degy, degz, degvm, output>(params, domain, &mw_yee, &part_gr, &diagn, params.time_staggered, funcSelectB);
    }

    void initialize_gempic_structures()
    {
        domain.initialize_computational_domain(params.n_cell, params.max_grid_size, params.is_periodic, params.real_box);
        mw_yee.reinit(domain, params.dt, params.n_steps, params.Nghost);
        mw_yee.template init_rho_phi<degvm>(domain, params.rhoEval, params.phiEval);
        // particles
        part_gr.reinit(params.charge, params.mass, domain);
        amrex::Real vol = (domain.geom.ProbHi(0)-domain.geom.ProbLo(0))*(domain.geom.ProbHi(1)-domain.geom.ProbLo(1))*(domain.geom.ProbHi(2)-domain.geom.ProbLo(2));
        diagn.reinit(mw_yee.nsteps, params.freq_x, params.freq_v, params.freq_slice, params.sim_name, vol);
        //------------------------------------------------------------------------------
        // initialize particles & loop preparation:
        init_particles_full_domain<vdim,numspec>(domain, part_gr, params.n_part_per_cell, params.meanVelocity[0], params.vThermal[0], params.vWeight[0], 0, wave_function);

        loop_preparation<vdim, numspec, degx, degy, degz, degvm, output>(params, domain, &mw_yee, &part_gr, &diagn,
                                                                         params.time_staggered, params.BxEval, params.ByEval, params.BzEval);
    }

    void run_time_loop()
    {
        auto start = std::chrono::high_resolution_clock::now();
        switch (params.propagator) {
            case 0:
                time_loop_boris_fd<vdim, numspec, degx, degy, degz, degvm, electromagnetic, true>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 1:
                time_loop_hs_fem<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 2:
                time_loop_hsall_fem<vdim, numspec, degx, degy, degz, degvm, electromagnetic>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 3:
                time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degvm, electromagnetic, true, output>(domain, &mw_yee, &part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
           // case 4:
           //     ls_rungekutta<3> rk_prop();
           //     rk_prop.time_loop(domain, &mw_yee, &art_gr);
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


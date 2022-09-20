#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
#include <GEMPIC_time_loop_hsall_fem.H>
#include <GEMPIC_time_loop_ls_rungekutta.H>
#include <GEMPIC_time_loop_particles.H>

using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Profiling;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real wave_function(amrex::Real x, amrex::Real y,
                                                                amrex::Real z, amrex::Real t)
{
    amrex::Real val = 1.0;  //+ 0.5 * std::cos(0.5 * x);
    return val;
}

template <int vdim, int numspec, int degx, int degy, int degz, int degvm,
          bool electromagnetic = true, bool output = true, int ndata = 1>
class vlasov_maxwell_simulation
{
public:
    gempic_parameters<vdim, numspec> params;
    computational_domain domain;
    maxwell_yee<vdim, int(vdim / 2.5) * 2 + 1> mw_yee;
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> part_gr;
    diagnostics<vdim, numspec, degx, degy, degz, degvm> diagn;
    bool readinfile = false;
    bool ctest = false;

    void initialize_vlasov_maxwell_from_file()
    {
        BL_PROFILE("vlasov_maxwell_simulation::initialize_vlasov_maxwell_from_file");

        // ------------------------------------------------------------------------------
        // ------------PARAMETERS--------------------------------------------------------
        params.init_Nghost(degx, degy, degz);
        params.read_pp_params();
        initialize_gempic_structures();
    }

    void initialize_gempic_structures(amrex::GpuArray<int, 2> funcSelectRho,
                                      amrex::GpuArray<int, int(vdim / 2.5) * 2 + 1> funcSelectB)
    {
        BL_PROFILE("vlasov_maxwell_simulation::initialize_gempic_structures(args)");
        auto start = std::chrono::high_resolution_clock::now();

        domain.initialize_computational_domain(params.n_cell, params.max_grid_size,
                                               params.is_periodic, params.real_box);
        mw_yee.reinit(domain, params.dt, params.n_steps, params.Nghost);
        mw_yee.template init_rho_phi<degvm>(domain, funcSelectRho);
        // particles
        for (int s = 0; s < numspec; s++)
        {
            part_gr[s] = std::make_unique<particle_groups<vdim, ndata>>(params.charge[s],
                                                                        params.mass[s], domain);
        }

        amrex::Real vol = (domain.geom.ProbHi(0) - domain.geom.ProbLo(0)) *
                          (domain.geom.ProbHi(1) - domain.geom.ProbLo(1)) *
                          (domain.geom.ProbHi(2) - domain.geom.ProbLo(2));
        diagn.reinit(mw_yee.nsteps, params.save_fields, params.save_particles,
                     params.save_checkpoint, params.sim_name, vol, ctest);
        //------------------------------------------------------------------------------
        // initialize particles & loop preparation:
        init_particles_full_domain<vdim, numspec, ndata>(domain, part_gr, params.n_part_per_cell,
                                                         params.meanVelocity[0], params.vThermal[0],
                                                         params.vWeight[0], 0, wave_function);

        loop_preparation<vdim, numspec, degx, degy, degz, degvm, ndata, output>(
            params, domain, &mw_yee, part_gr, &diagn, params.time_staggered, funcSelectB);

        // Print AMReX grid information and warn about resource usage
        if (ParallelDescriptor::IOProcessor())
        {
            std::cout << "\nGrid Summary: \n";
            std::cout << "total number of boxes " << domain.grid.size() << "\n";
            std::cout << "Number of MPI ranks " << ParallelDescriptor::NProcs() << "\n\n";
            // Print Box List if there are less than 20 boxes
            if (domain.grid.size() < 20)
            {
                std::cout << "Box list " << domain.grid.boxList() << "\n";
            }
            if (ParallelDescriptor::NProcs() > domain.grid.size())
            {
                std::cout << "WARNING:\n";
                std::cout << "----------------------------------------------------------- \n";
                std::cout << "There is not enough work for all MPI ranks. \n";
                std::cout << "You should have more boxes than MPI ranks. \n";
                std::cout << "This can be adjusted with the max_grid_size input parameter. \n";
                std::cout << "----------------------------------------------------------- \n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        amrex::Print() << "initialize_gempic_structures execution lasted: " << duration.count()
                       << " s" << std::endl;
    }

    void initialize_gempic_structures()
    {
        BL_PROFILE("vlasov_maxwell_simulation::initialize_gempic_structures");
        auto start = std::chrono::high_resolution_clock::now();

        domain.initialize_computational_domain(params.n_cell, params.max_grid_size,
                                               params.is_periodic, params.real_box);
        mw_yee.reinit(domain, params.dt, params.n_steps, params.Nghost);
        mw_yee.template init_rho_phi<degvm>(domain, params.rhoEval, params.phiEval);
        // particles
        for (int s = 0; s < numspec; s++)
        {
            part_gr[s] = std::make_unique<particle_groups<vdim, ndata>>(params.charge[s],
                                                                        params.mass[s], domain);
        }
        amrex::Real vol = (domain.geom.ProbHi(0) - domain.geom.ProbLo(0)) *
                          (domain.geom.ProbHi(1) - domain.geom.ProbLo(1)) *
                          (domain.geom.ProbHi(2) - domain.geom.ProbLo(2));
        diagn.reinit(mw_yee.nsteps, params.save_fields, params.save_particles,
                     params.save_checkpoint, params.sim_name, vol, ctest);
//------------------------------------------------------------------------------
// initialize particles & loop preparation:
#if GEMPIC_GPU
        if (ctest)
        {
            init_particles_full_domain<vdim, numspec>(domain, part_gr, params.n_part_per_cell,
                                                      params.meanVelocity[0], params.vThermal[0],
                                                      params.vWeight[0], 0, params.densityEval[0]);
        }
        else
        {
            init_particles_full_domain_gpu<vdim, numspec, ndata>(
                domain, part_gr, params.n_part_per_cell, params.meanVelocity[0], params.vThermal[0],
                params.vWeight[0], 0, params.densityEval[0]);
        }
#else
        init_particles_full_domain<vdim, numspec>(domain, part_gr, params.n_part_per_cell,
                                                  params.meanVelocity[0], params.vThermal[0],
                                                  params.vWeight[0], 0, params.densityEval[0]);
#endif

        loop_preparation<vdim, numspec, degx, degy, degz, degvm, ndata, output>(
            params, domain, &mw_yee, part_gr, &diagn, params.time_staggered, params.BxEval,
            params.ByEval, params.BzEval);

        // Print AMReX grid information and warn about resource usage
        if (ParallelDescriptor::IOProcessor())
        {
            std::cout << "\nGrid Summary:\n";
            std::cout << "total number of boxes " << domain.grid.size() << "\n";
            std::cout << "Number of MPI ranks " << ParallelDescriptor::NProcs() << "\n\n";
            if (domain.grid.size() < 20)
            {
                std::cout << "Box list " << domain.grid.boxList() << "\n";
            }
            if (ParallelDescriptor::NProcs() > domain.grid.size())
            {
                std::cout << "WARNING:\n";
                std::cout << "----------------------------------------------------------- \n";
                std::cout << "There is not enough work for all MPI ranks. \n";
                std::cout << "You should have more boxes than MPI ranks. \n";
                std::cout << "This can be adjusted with the max_grid_size input parameter. \n";
                std::cout << "----------------------------------------------------------- \n";
            }
            amrex::Print() << "Number of ghost cells: " << params.Nghost << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        amrex::Print() << "initialize_gempic_structures execution lasted: " << duration.count()
                       << " s" << std::endl
                       << std::endl;
    }

    void run_time_loop()
    {
        BL_PROFILE("vlasov_maxwell_simulation::run_time_loop");

        amrex::Print() << "Propagator: " << params.propagator << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        switch (params.propagator)
        {
            case 0:
                time_loop_boris_fd<vdim, numspec, degx, degy, degz, degvm, ndata, electromagnetic,
                                   true>(domain, &mw_yee, part_gr, &diagn, ctest, params.sim_name,
                                         params.strang_order);
                break;
            case 1:
                time_loop_hs_fem<vdim, numspec, degx, degy, degz, degvm, ndata, electromagnetic>(
                    domain, &mw_yee, part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 2:
                time_loop_hsall_fem<vdim, numspec, degx, degy, degz, degvm, ndata, electromagnetic>(
                    domain, &mw_yee, part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 3:
                time_loop_hs_zigzag_C2<vdim, numspec, degx, degy, degz, degvm, ndata,
                                       electromagnetic, true, output>(
                    domain, &mw_yee, part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            case 4:
            {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    ndata == 1 + 2 * GEMPIC_SPACEDIM,
                    'Runge-Kutta integrator needs extra data in the particle array.');
                ls_rungekutta<3, vdim, numspec> rk_prop(domain, mw_yee);
                rk_prop.template time_loop<degx, degy, degz, degvm, ndata, electromagnetic>(
                    domain, mw_yee, part_gr, &diagn, ctest, params.sim_name, params.strang_order);
                break;
            }
            case 100:
                time_loop_particles<vdim, numspec, degx, degy, degz, degvm, ndata, electromagnetic>(
                    domain, &mw_yee, part_gr, &diagn, ctest, params.sim_name);
                break;
            default:
                break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(stop - start);
        amrex::Print() << "run_time_loop execution lasted: " << duration.count() << " s"
                       << std::endl;
        switch (params.propagator)
        {
            case 100:
                Gempic_WritePlotFile<vdim, numspec, ndata>(part_gr, &mw_yee, &domain, "Edipole",
                                                           10);
                break;
            default:
                break;
        }
    }
};

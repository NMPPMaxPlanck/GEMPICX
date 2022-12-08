/**
 * 
 * @file test_H_E_Velocity.cpp
 * @author marm@ipp.mpg.de
 * @brief Tests apply_H_e in GEMPIC_time_loop_hs_zigzag_C2.H
 * @version 0.1
 * @date 2022-10-20
 * @details None
 * @copyright Copyright (c) 2022
 * 
 */

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_time_loop_hs_zigzag_C2.H>
#include <GEMPIC_sampler.H>

using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;
using namespace Diagnostics_Output;
using namespace Time_Loop;

template<int vdim, int numspec, int degmw, int degx, int degy, int degz, int ndata, bool profiling>
void main_main()
{       
    amrex::Real const coeff = 1.0;
    bool profiling_count = false;
    timers profiling_timers(true);
    amrex::IntVect n_cell = {AMREX_D_DECL(16, 16, 16)};  // spatial discretization: number of cells in each direction, currently: 128x128x128
    int numstep = 1;        // number of timesteps
    amrex::Real dt = 0.01;  // size of timesteps

    amrex::IntVect is_periodic = {AMREX_D_DECL(1, 1, 1)};
    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("test_H_E_Velocity", n_cell, {1}, numstep, 100000, 100000, 100000,
                    is_periodic, {AMREX_D_DECL(4, 4, 4)}, dt, {-1.0}, {1.0}, 1.0);

    CompDom::computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr[spec] =
            std::make_unique<particle_groups<vdim>>(VlMa.charge[spec], VlMa.mass[spec], infra);
    }

    PrintToFile("test_H_E_Velocity.tmp", 0) << std::endl;

    Sampling::init_particles_full_domain<vdim, numspec>(infra, part_gr, VlMa.n_part_per_cell,
                                            VlMa.meanVelocity[0], VlMa.vThermal[0],
                                            VlMa.vWeight[0], 0, VlMa.densityEval[0]);

    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);
    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real chargemass = part_gr[spec]->getCharge() / part_gr[spec]->getMass();
        //PrintToFile("test_H_E_Velocity.tmp") << std::endl;
        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti) 
        {
            const long np = pti.numParticles();
    
            Time_Loop::apply_H_e<vdim, numspec, degx, degy, degz, degmw, ndata, profiling>(infra, &mw_yee,
                coeff, spec, chargemass, pti, part_gr, &profiling_timers, profiling_count);
        
            auto particle_attributes = &pti.GetStructOfArrays();

            amrex::ParticleReal* const AMREX_RESTRICT velx = 
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();

            for (int pp = 0; pp <= np; ++pp)
            {
                //PrintToFile("test_H_E_Velocity.tmp") << std::endl;
                AllPrintToFile("test_H_E_Velocity.tmp") << "velx: " << velx[pp] << std::endl;
                AllPrintToFile("test_H_E_Velocity.tmp") << "vely: " << vely[pp] << std::endl;
                AllPrintToFile("test_H_E_Velocity.tmp") << "velz: " << velz[pp] << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD, 
                      overwrite_amrex_parser_defaults);

    if (amrex::ParallelDescriptor::MyProc() == 0) remove("test_H_E_Velocity.tmp.0");
    
    const int vdim = 3, numspec = 1, degmw = 2, degx = 1, degy = 1, degz = 1, ndata = 1, profiling = false;
    main_main<vdim, numspec, degmw, degx, degy, degz, ndata, profiling>();

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_H_E_Velocity.tmp.0", "test_H_E_Velocity.output");

    amrex::Finalize();
}
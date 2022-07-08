#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>

using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;

template <int vdim, int numspec>
void main_main()
{
    // This test only checks if the initialization runs through, no values are checked

    //------------------------------------------------------------------------------
    // Initialize Infrastructure

    amrex::IntVect is_periodic = {AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect n_cell = {AMREX_D_DECL(64, 64, 64)};

    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(1, 1, 1);
    VlMa.set_params("initialize_ctest", n_cell, {1000}, 5, 10, 10, 10, is_periodic, {AMREX_D_DECL(32, 32, 32)},
                    0.01, {1.0}, {1.0}, 1);
    VlMa.set_computed_params();

    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);

    // need a multifab to be able to iterate later:
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    //------------------------------------------------------------------------------
    // Initialize Particle Groups
    particle_groups<vdim, numspec> part_gr(VlMa.charge, VlMa.mass, infra);

    // set particles for first cell (and copies in remaining cells)
    int Np_cell = 100;  // number of particles per cell
    int species = 0;    // all particles are same species for now
    amrex::GpuArray<double, GEMPIC_SPACEDIM> position;
    amrex::GpuArray<double, GEMPIC_SPACEDIM> shifted_position;
    amrex::GpuArray<double, vdim> velocity;
    Real weight;

    // normally distributed random number generator:
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normD(0, 1);

    amrex::GpuArray<double, GEMPIC_SPACEDIM> x;

    for (int pp = 0; pp < Np_cell; pp++)
    {
        // position in model cell [0,dx]x[0,dy]x[0,dz]:
        position[0] = ((Real)rand() / (RAND_MAX)) / infra.dx[0];
#if (GEMPIC_SPACEDIM > 1)
        position[1] = ((Real)rand() / (RAND_MAX)) / infra.dx[1];
#endif
#if (GEMPIC_SPACEDIM > 2)
        position[2] = ((Real)rand() / (RAND_MAX)) / infra.dx[2];
#endif

        velocity[0] = normD(gen);
        if (vdim > 1)
        {
            velocity[1] = normD(gen);
        }
        if (vdim > 2)
        {
            velocity[2] = normD(gen);
        }
        weight = 1.0;

        // MFI that adds particle from the modell cell to all cells
        for (MFIter mfi = (*(part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::IntVect lo = {bx.smallEnd()};
            amrex::IntVect hi = {bx.bigEnd()};

            auto &particles =
                (*part_gr.mypc[species])
                    .GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

#if (GEMPIC_SPACEDIM > 2)
            for (int k = lo[2]; k <= hi[2]; k++)
            {
                x[2] = infra.plo[2] + (double)k * infra.dx[2];
                shifted_position[2] = position[2] + x[2];
#endif
#if (GEMPIC_SPACEDIM > 1)
                for (int j = lo[1]; j <= hi[1]; j++)
                {
                    x[1] = infra.plo[1] + (double)j * infra.dx[1];
                    shifted_position[1] = position[1] + x[1];
#endif
                    for (int l = lo[0]; l <= hi[0]; l++)
                    {
                        x[0] = infra.plo[0] + (double)l * infra.dx[0];
                        shifted_position[0] = position[0] + x[0];
                        part_gr.add_particle(shifted_position, velocity, weight, particles);
                    }
#if (GEMPIC_SPACEDIM > 1)
                }
#endif
#if (GEMPIC_SPACEDIM > 2)
            }
#endif
        }
    }
    amrex::PrintToFile("test_initialize.tmp") << "" << std::endl;
    amrex::PrintToFile("test_initialize.tmp") << 1 << std::endl;
}

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(
        argc,
        argv,
        build_parm_parse,
        MPI_COMM_WORLD,
        overwrite_amrex_parser_defaults
    );

    if (ParallelDescriptor::MyProc() == 0) remove("test_initialize.tmp.0");

    main_main<3, 1>();

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_initialize.tmp.0", "test_initialize.output");

    amrex::Finalize();
}

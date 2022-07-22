#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_checkpoint.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_sampler.H>

using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Sampling;
using namespace Utils;

AMREX_GPU_HOST_DEVICE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    return 0.0;
}

AMREX_GPU_HOST_DEVICE amrex::Real zero(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    return 0.0;
}
template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // initialize parameters
    amrex::IntVect n_cell_vector = {AMREX_D_DECL(4, 4, 4)};
    amrex::GpuArray<int, numspec> n_part_per_cell = {1};
    int n_steps = 2;
    amrex::IntVect is_periodic_vector = {AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect max_grid_size = {AMREX_D_DECL(2, 2, 2)};
    amrex::Real dt = 0.1;
    amrex::GpuArray<amrex::Real, numspec> charge = {-1.0};
    amrex::GpuArray<amrex::Real, numspec> mass = {1.0};
    // std::array<amrex::Real,GEMPIC_SPACEDIM> k = {AMREX_D_DECL(1.25,1.25,1.25)};
    amrex::Real k = 1.25;
    amrex::Real tolerance_particles = 1.e-10;

    amrex::Vector<amrex::Vector<amrex::Real>> meanVelocity = {{0.0, 0.0, 0.0}};
    amrex::Vector<amrex::Vector<amrex::Real>> vThermal = {
        {0.0141421356237309493730949, 0.04898979485566356, 0.04898979485566356}};
    if (vdim == 2)
    {
        amrex::Vector<amrex::Vector<amrex::Real>> meanVelocity = {{0.0, 0.0}};
        amrex::Vector<amrex::Vector<amrex::Real>> vThermal = {
            {0.0141421356237309493730949, 0.04898979485566356}};
    }
    amrex::Vector<amrex::Real> vWeight = {1.0};

    // initialize amrex data structures from parameters
    amrex::IntVect n_cell(AMREX_D_DECL(n_cell_vector[0], n_cell_vector[1], n_cell_vector[2]));
    amrex::IntVect is_periodic(
        AMREX_D_DECL(is_periodic_vector[0], is_periodic_vector[1], is_periodic_vector[2]));

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    const int FX = 10, FV = 10, FS = 10;
    VlMa.set_params("checkpoint_ctest", n_cell_vector, n_part_per_cell, n_steps, FX, FV, FS,
                    is_periodic_vector, max_grid_size, dt, charge, mass, k, {"0"}, "0", "0", "0",
                    "0", "0", "0", "0", {1}, 0, tolerance_particles);
    VlMa.set_computed_params();

    // infrastructure
    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);
    mw_yee.template init_rho_phi<2>(zero, zero, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa.charge, VlMa.mass, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0;  // all particles are same species for now
    init_particles_full_domain<vdim, numspec>(infra, part_gr, VlMa.n_part_per_cell, meanVelocity,
                                              vThermal, vWeight, species, wave_function);

    //------------------------------------------------------------------------------
    // test:
    // Initialize fields
    for (size_t i = 0; i < vdim; i++)
    {
        mw_yee.J_Array[i]->setVal(0.0, 0);
        mw_yee.E_Array[i]->setVal(0.0, 0);
        mw_yee.B_Array[i]->setVal(0.0, 0);
    }

    mw_yee.J_Array[0]->setVal(1.0, 0);
    amrex::Real old_val = gempic_norm(*mw_yee.J_Array[0], infra, 0);
    Gempic_WriteCheckpointFile(&mw_yee, &part_gr, &infra, "test_checkpoint", 0, 20);

    mw_yee.J_Array[0]->setVal(2.0, 0);
    amrex::Real new_val = gempic_norm(*mw_yee.J_Array[0], infra, 0);

    Gempic_ReadCheckpointFile(&mw_yee, &part_gr, &infra, "test_checkpoint",
                              0);  // last 2 args: field, step
    amrex::Real read_val = gempic_norm(*mw_yee.J_Array[0], infra, 0);

    amrex::PrintToFile("test_checkpoint_additional.tmp") << "" << std::endl;
    amrex::PrintToFile("test_checkpoint_additional.tmp")
        << "Norm of MF that is written out: " << old_val << std::endl;
    amrex::PrintToFile("test_checkpoint_additional.tmp")
        << "Norm the MF has after changing it: " << new_val << std::endl;
    amrex::PrintToFile("test_checkpoint_additional.tmp")
        << "Norm of MF that is read in: " << read_val << std::endl;

    bool passed = (std::abs(old_val - read_val) < 1e-12) && (std::abs(old_val - new_val) > 1e-1);
    amrex::PrintToFile("test_checkpoint.tmp") << "" << std::endl;
    amrex::PrintToFile("test_checkpoint.tmp") << passed << std::endl;
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

    if (ParallelDescriptor::MyProc() == 0) remove("test_checkpoint.tmp.0");
    if (ParallelDescriptor::MyProc() == 0) remove("test_checkpoint_additional.tmp.0");

    const int vdim=3, numspec=1, degx=1, degy=1, degz=1;
    main_main<vdim, numspec, degx, degy, degz>();

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_checkpoint.tmp.0", "test_checkpoint.output");
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_checkpoint_additional.tmp.0", "test_checkpoint_additional.output");

    amrex::Finalize();
}

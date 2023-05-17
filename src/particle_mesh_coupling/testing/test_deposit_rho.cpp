#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_assertion.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_PlotFile.H>
#include <GEMPIC_sampler.H>

using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Utils;

// wave function
AMREX_GPU_HOST_DEVICE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z,
                                                amrex::Real t)
{
    amrex::Real val = 1.0;
    return val;
}

// wave function
AMREX_GPU_HOST_DEVICE amrex::Real funct_rho_phi(amrex::Real x, amrex::Real y, amrex::Real z,
                                                amrex::Real t)
{
    amrex::Real val = 1.0;
    return val;
}

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{
    //------------------------------------------------------------------------------
    // Initialize Function
    // std::array<amrex::Real,GEMPIC_SPACEDIM> k = {AMREX_D_DECL(0.5,0.5,0.5)};
    amrex::Real k = 0.5;

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    amrex::IntVect is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect n_cell{AMREX_D_DECL(8, 8, 8)};

    // Weibel parameters
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> meanVelocity{
        {{0.0, 0.0, 0.0}}};  // species, gaussian, vdim
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> vThermal{
        {{0.014142135623730949, 0.04898979485566356, 0.04898979485566356}}};
    ;
    if (vdim == 2)
    {
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> meanVelocity{
            {{0.0, 0.0}}};  // species, gaussian, vdim
        amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> vThermal{
            {{0.014142135623730949, 0.04898979485566356}}};
        ;
    }
    amrex::Vector<amrex::Vector<amrex::Real>> vWeight{{1.0}};

    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    const int NS = 0, FX = 2, FV = 2, FS = 2;
    VlMa.set_params("test_deposit_rho", n_cell, {1000}, NS, FX, FV, FS, is_periodic,
                    amrex::IntVect{AMREX_D_DECL(4, 4, 4)}, 0.02, {-1.0}, {1.0}, k, {"0"});

    VlMa.meanVelocity = meanVelocity;
    VlMa.vThermal = vThermal;
    VlMa.vWeight = vWeight;

    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);

    //------------------------------------------------------------------------------
    // Initialize fields and particles
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    int species = 0;  // all particles are same species for now

    // particles
    amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr;
    for (int spec = 0; spec < numspec; spec++)
    {
        part_gr[spec] =
            std::make_unique<particle_groups<vdim>>(VlMa.charge[spec], VlMa.mass[spec], infra);
    }
    //------------------------------------------------------------------------------
    // initialize particles:
    init_particles_cellwise<vdim, numspec>(infra, part_gr, VlMa.n_part_per_cell,
                                           VlMa.meanVelocity[species], VlMa.vThermal[species],
                                           VlMa.vWeight[species], species, wave_function);

    //------------------------------------------------------------------------------
    // initialize rho and phi, phi will solve the analytically exact solution, rho
    // will be overwritten in next paragraph
    mw_yee.template init_rho_phi<2>(funct_rho_phi, funct_rho_phi, infra);

    mw_yee.phi.mult(-1.0);

    //------------------------------------------------------------------------------
    // Deposit charges:
    mw_yee.rho.setVal(0.0);

    for (int spec = 0; spec < numspec; spec++)
    {
        amrex::Real charge = part_gr[spec]->getCharge();
        part_gr[spec]->Redistribute();  // assign particles to the tile they are in
        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();
            const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

            amrex::Array4<amrex::Real> const& rhoarr = (mw_yee.rho)[pti].array();

            // for (int pp = 0; pp < np; pp++)
            amrex::ParallelFor(np,
                               [=] AMREX_GPU_DEVICE(long pp)
                               {
                                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> pos;
                                   for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++)
                                   {
                                       pos[comp] = partData[pp].pos(comp);
                                   }
                                   splines_at_particles<degx, degy, degz> spline;
                                   spline.init_particles(pos, infra.plo, infra.dxi);
                                   gempic_deposit_rho_C3<degx, degy, degz>(
                                       spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                                       rhoarr);
                               });
        }
    }

#if (GEMPIC_SPACEDIM == 3)
    mw_yee.rho.SumBoundary(0, 1, {mw_yee.Nghost, mw_yee.Nghost, mw_yee.Nghost}, {0, 0, 0},
                           infra.geom.periodicity());
    mw_yee.rho.FillBoundary(infra.geom.periodicity());
#endif

    //------------------------------------------------------------------------------
    // Compute difference and store in phi:
    mw_yee.phi.minus(mw_yee.rho, 0, 1, 0);

    amrex::Real error = gempic_norm(mw_yee.phi, infra, 2) * gempic_norm(mw_yee.phi, infra, 2);
    bool passed = true;
    gempic_assert_err(passed, gempic_norm(mw_yee.rho, infra, 2), error);

    PrintToFile("test_deposit_rho.output") << "\n";
    PrintToFile("test_deposit_rho.output")
        << "Norm of error: "
        << gempic_norm(mw_yee.phi, infra, 2) * gempic_norm(mw_yee.phi, infra, 2) << std::endl;
}

int main(int argc, char* argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      overwrite_amrex_parser_defaults);

    const int numspec = 1, degx = 1, degy = 1, degz = 1;
#if (GEMPIC_SPACEDIM == 1)
    const int vdim1 = 1;
    main_main<vdim1, numspec, degx, degy, degz>();
    const int vdim2 = 2;
    main_main<vdim2, numspec, degx, degy, degz>();
#elif (GEMPIC_SPACEDIM == 2)
    const int vdim2 = 2;
    main_main<vdim2, numspec, degx, degy, degz>();
    const int vdim3 = 3;
    main_main<vdim3, numspec, degx, degy, degz>();
#elif (GEMPIC_SPACEDIM == 3)
    const int vdim = 3;
    main_main<vdim, numspec, degx, degy, degz>();
#endif
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_deposit_rho.output.0", "test_deposit_rho.output");

    amrex::Finalize();
}

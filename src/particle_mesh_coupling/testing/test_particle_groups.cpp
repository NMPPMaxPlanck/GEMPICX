#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_assertion.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_vlasov_maxwell.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Diagnostics_Output;
using namespace Particles;
using namespace Sampling;

// wave function
// we want particles to have the weight 1, in the sampler, the weight is scaled with dx*dy*dz/nppc, to make up for it here we
// multiply by 1/dx*1/dy*1/dz*nppc = (n_cell*k/(2*pi))^3*1
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z)
{
    amrex::Real val = (32*1.25/6.28318530718)*(32*1.25/6.28318530718)*(32*1.25/6.28318530718)*1.0;
    return val;
}


AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_Bz(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = 1e-3 * std::cos(1.25 * x);
    return val;
}


AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real zero(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    return 0.0;
}



template<int vdim, int numspec>
void main_main ()
{
    const int degmw = 2;
    //------------------------------------------------------------------------------
    //build objects:

    amrex::IntVect is_periodic = {AMREX_D_DECL(1,1,1)};
    amrex::IntVect n_cell = {AMREX_D_DECL(32,32,32)};
    amrex::IntVect max_grid_size = {2,2,2};

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
    if (vdim > 1) {
        VM[1].push_back(0.0);
        VD[1].push_back(1.0);
        VW[1].push_back(1.0);
    }
    if (vdim > 2) {
        VM[2].push_back(0.0);
        VD[2].push_back(1.0);
        VW[2].push_back(1.0);
    }

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(1, 1, 1);
    VlMa.set_params("part_gr_ctest", n_cell, {1}, 0, 2, 2, 2,
                    is_periodic, max_grid_size, 0.01, {1.0}, {1.0}, 1.25, " ");
    VlMa.set_computed_params();
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    // infrastructure
    computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic, VlMa.real_box);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    mw_yee.template initB<degmw>(zero, zero, funct_Bz, infra);
    mw_yee.template initE<degmw>(zero, zero, zero, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa.charge, VlMa.mass, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0; // all particles are same species for now
    init_particles_cellwise<vdim, numspec>(infra, part_gr, VlMa.n_part_per_cell, VlMa.VM, VlMa.VD, VlMa.VW, species, wave_function);
    (*(part_gr).mypc[0]).Redistribute();

    int spec = 0;

    // compute mass, momentum and kinetic energy
    auto mass = amrex::ReduceSum( *(part_gr).mypc[spec],
                                  [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real
    {
        auto m  = p.rdata(vdim);
        return (m);
    });
    amrex::ParallelDescriptor::ReduceRealSum
            (mass, amrex::ParallelDescriptor::IOProcessorNumber());

    // momentum
    amrex::GpuArray<amrex::Real,vdim> momentum;
    for (int cmp=0;cmp<vdim;cmp++) {
        auto mom_tmp = amrex::ReduceSum( *(part_gr).mypc[spec],
                                         [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real
        {
            auto m  = p.rdata(vdim);
            auto vel = p.rdata(cmp);
            return (m*vel);
        });

        amrex::ParallelDescriptor::ReduceRealSum
                (mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

        momentum[cmp] = mom_tmp;
    }

    // kinetic energy
    amrex::GpuArray<amrex::Real,vdim> kinetic_energy;
    for (int cmp=0;cmp<vdim;cmp++) {
        auto mom_tmp = amrex::ReduceSum( *(part_gr).mypc[spec],
                                         [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<vdim+1,0>& p) -> amrex::Real
        {
            auto m  = p.rdata(vdim);
            auto vel = p.rdata(cmp);
            return (m*vel*vel);
        });

        amrex::ParallelDescriptor::ReduceRealSum
                (mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

        kinetic_energy[cmp] = mom_tmp;
    }

    amrex::Real rel_mass = 32768;
    amrex::GpuArray<amrex::Real,3> rel_mom = {584.1995254, -12972.79719, -1805.405324};
    amrex::GpuArray<amrex::Real,3> rel_kin = {10.41531633, 5135.909025, 99.47169141};

    bool passed = true;
    gempic_assert(passed, rel_mass, mass);
    gempic_assert(passed, rel_mom[0], momentum[0]);
    gempic_assert(passed, rel_mom[1], momentum[1]);
    gempic_assert(passed, rel_mom[2], momentum[2]);
    gempic_assert(passed, rel_kin[0], kinetic_energy[0]);
    gempic_assert(passed, rel_kin[1], kinetic_energy[1]);
    gempic_assert(passed, rel_kin[2], kinetic_energy[2]);

    AllPrintToFile("test_particle_groups.tmp") << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << passed << std::endl;

    AllPrintToFile("test_particle_groups_additional.tmp") << std::endl;
    AllPrintToFile("test_particle_groups_additional.tmp") << "mass: " << mass << std::endl;
    AllPrintToFile("test_particle_groups_additional.tmp") << "momentum: " ;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_particle_groups_additional.tmp") << momentum[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_particle_groups_additional.tmp") << momentum[0] << " " << momentum[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_particle_groups_additional.tmp") << momentum[0] << " " << momentum[1] << " " << momentum[2] << std::endl;
        break;

    }
    AllPrintToFile("test_particle_groups_additional.tmp") << "kinetic energy: " ;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_particle_groups_additional.tmp") << kinetic_energy[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_particle_groups_additional.tmp") << kinetic_energy[0] << " " << kinetic_energy[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_particle_groups_additional.tmp") << kinetic_energy[0] << " " << kinetic_energy[1] << " " << kinetic_energy[2] << std::endl;
        break;

    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_particle_groups.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_particle_groups_additional.tmp.0");

    main_main<3, 1>();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_particle_groups.tmp.0", "test_particle_groups.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_particle_groups_additional.tmp.0", "test_particle_groups_additional.output");
    amrex::Finalize();
}


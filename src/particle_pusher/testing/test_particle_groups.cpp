#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

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

template<int vdim, int numspec>
void main_main ()
{
    const int degmw = 2;
    //------------------------------------------------------------------------------
    //build objects:

    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(32,32,32)};
    std::array<int,GEMPIC_SPACEDIM> max_grid_size = {2,2,2};

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

    // we want particles to have the weight 1, in the sampler, the weight is scaled with dx*dy*dz/nppc, to make up for it here we
    // multiply by 1/dx*1/dy*1/dz*nppc = (n_cell*k/(2*pi))^3*1
    std::string WF = "(32*1.25/6.28318530718)*(32*1.25/6.28318530718)*(32*1.25/6.28318530718)*1.0";
    double x, y, z;
    double k = 1.25;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvar", &k}};
    int varcount = 4;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(1, 1, 1);
    VlMa.set_params("part_gr_ctest", n_cell, {1}, 0, 2, 2, 2,
                    is_periodic, max_grid_size, 0.01, {1.0}, {1.0}, 1.25, WF);
    VlMa.set_computed_params();
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    // infrastructure
    infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    std::string Bx = "0.0";
    std::string By = "0.0";
    std::string Bz = "1e-3 * cos(kvar * x)";
    std::array<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_B[0] = Bx;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[1] = By;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[2] = Bz;
    }
    std::string Ex = "0.0";
    std::string Ey = "0.0";
    std::string Ez = "0.0";
    std::array<std::string, int(vdim/2.5)*2+1> fields_E;
    fields_E[0] = Ex;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_E[1] = Ey;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_E[2] = Ez;
    }
    mw_yee.template initB<degmw>(fields_B, VlMa.k, infra);
    mw_yee.template initE<degmw>(fields_E, VlMa.k, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0; // all particles are same species for now
    init_particles_cellwise<vdim, numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, species, WF_parse, &x, &y, &z);
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
    std::array<amrex::Real,vdim> momentum;
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
    std::array<amrex::Real,vdim> kinetic_energy;
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


    AllPrintToFile("test_particle_groups.tmp") << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "mass: " << mass << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: " ;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_particle_groups.tmp") << momentum[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_particle_groups.tmp") << momentum[0] << " " << momentum[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_particle_groups.tmp") << momentum[0] << " " << momentum[1] << " " << momentum[2] << std::endl;
        break;

    }
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: " ;
    switch (vdim) {
    case 1:
        AllPrintToFile("test_particle_groups.tmp") << kinetic_energy[0] << std::endl;
        break;
    case 2:
        AllPrintToFile("test_particle_groups.tmp") << kinetic_energy[0] << " " << kinetic_energy[1] << std::endl;
        break;
    case 3:
        AllPrintToFile("test_particle_groups.tmp") << kinetic_energy[0] << " " << kinetic_energy[1] << " " << kinetic_energy[2] << std::endl;
        break;

    }


}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    /* This ctest has a different output for each GEMPIC_SPACEDIM and vdim. Therefore, the expected_output file contains all outputs.
    For each dimension, apart from running the main_main for the dimension, the output for the other dimensions needs to be
    outputted, so that the comparison to the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1 vdim=1, GEMPIC_SPACEDIM=1 vdim=2, GEMPIC_SPACEDIM=2 vdim=2, GEMPIC_SPACEDIM=2 vdim=3, GEMPIC_SPACEDIM=3 vdim=3 */

#if (GEMPIC_SPACEDIM == 1)

    // Output for GEMPIC_SPACEDIM=1 vdim=1
    main_main<1, 1>();
    // Output for GEMPIC_SPACEDIM=1 vdim=2
    main_main<2, 1>();

    // Output for GEMPIC_SPACEDIM=2 vdim=2
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 6518.99" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -9092.65 642.819" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 12682.4 63.3867" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 6518.99" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -9092.65 642.819 5050.99" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 12682.4 63.3867 3913.57" << std::endl;

    // Output for GEMPIC_SPACEDIM=3 vdim=3
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 32768" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: 584.2 -12972.8 -1805.41" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 10.4153 5135.91 99.4717" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1 vdim=1
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 1296.91" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -392.235" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 118.627" << std::endl;

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 1296.91" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -392.235 -494.437" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 118.627 188.5" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=2
    main_main<2, 1>();
    // Output for GEMPIC_SPACEDIM=2 vdim=3
    main_main<3, 1>();

    // Output for GEMPIC_SPACEDIM=3 vdim=3
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 32768" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: 584.2 -12972.8 -1805.41" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 10.4153 5135.91 99.4717" << std::endl;

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1 vdim=1
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 1296.91" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -392.235" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 118.627" << std::endl;

    // Output for GEMPIC_SPACEDIM=1 vdim=2
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 1296.91" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -392.235 -494.437" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 118.627 188.5" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=2
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 6518.99" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -9092.65 642.819" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 12682.4 63.3867" << std::endl;

    // Output for GEMPIC_SPACEDIM=2 vdim=3
    AllPrintToFile("test_particle_groups.tmp") << std::endl;

    AllPrintToFile("test_particle_groups.tmp") << "mass: 6518.99" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "momentum: -9092.65 642.819 5050.99" << std::endl;
    AllPrintToFile("test_particle_groups.tmp") << "kinetic energy: 12682.4 63.3867 3913.57" << std::endl;


    // Output for GEMPIC_SPACEDIM=3 vdim=3
    main_main<3, 1>();
#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_particle_groups.tmp.0", "test_particle_groups.output");
    amrex::Finalize();
}


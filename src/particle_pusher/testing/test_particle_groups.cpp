#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_initializer.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Diagnostics_Output;
using namespace Init;
using namespace Particles;
using namespace Sampling;

void main_main ()
{
    //------------------------------------------------------------------------------
    //build objects:

    //initializer
    initializer init;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
    amrex::IntVect n_cell(AMREX_D_DECL(32,32,32));
    int max_grid_size = 2;

    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VM{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VD{};
    std::array<std::vector<amrex::Real>, GEMPIC_VDIM> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
#if (GEMPIC_VDIM > 1)
    VM[1].push_back(0.0);
    VD[1].push_back(1.0);
    VW[1].push_back(1.0);
#endif
#if (GEMPIC_VDIM > 2)
    VM[2].push_back(0.0);
    VD[2].push_back(1.0);
    VW[2].push_back(1.0);
#endif

    // we want particles to have the weight 1, in the sampler, the weight is scaled with dx*dy*dz/nppc, to make up for it here we
    // multiply by 1/dx*1/dy*1/dz*nppc = (n_cell*k/(2*pi))^3*1
    std::string WF = "(32*1.25/6.28318530718)*(32*1.25/6.28318530718)*(32*1.25/6.28318530718)*1.0";
    double x, y, z;
    double k = 1.25;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvar", &k}};
    int varcount = 4;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);


    init.initialize_from_parameters(n_cell,max_grid_size,is_periodic,1,0.01,0,{1.0},{1.0},1,1.25,VM,VD,VW,0);

    // infrastructure
    infrastructure infra(init);

    // maxwell_yee
    maxwell_yee mw_yee(init, infra, init.Nghost);
    std::string Bx = "0.0";
    std::string By = "0.0";
    std::string Bz = "1e-3 * cos(kvar * x)";
    std::string Ex = "0.0";
    std::string Ey = "0.0";
    std::string Ez = "0.0";
    te_expr *Bx_parse = te_compile(Bx.c_str(), read_vars, varcount, &err);
    te_expr *By_parse = te_compile(By.c_str(), read_vars, varcount, &err);
    te_expr *Bz_parse = te_compile(Bz.c_str(), read_vars, varcount, &err);
    te_expr *Ex_parse = te_compile(Ex.c_str(), read_vars, varcount, &err);
    te_expr *Ey_parse = te_compile(Ey.c_str(), read_vars, varcount, &err);
    te_expr *Ez_parse = te_compile(Ez.c_str(), read_vars, varcount, &err);
    mw_yee.initB(infra, Bx_parse, By_parse, Bz_parse, &x, &y, &z);
    mw_yee.initE(infra, Ex_parse, Ey_parse, Ez_parse, &x, &y, &z);

    // particles
    particle_groups part_gr(init, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0; // all particles are same species for now
    init_particles_cellwise(infra, part_gr, init, species, WF_parse, &x, &y, &z);
    (*(part_gr).mypc[0]).Redistribute();

    const int step = 0;
    int spec = 0;

    // compute mass, momentum and kinetic energy
    auto mass = amrex::ReduceSum( *(part_gr).mypc[spec],
                                  [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<4,0>& p) -> amrex::Real
    {
        auto m  = p.rdata(GEMPIC_VDIM);
        return (m);
    });
    amrex::ParallelDescriptor::ReduceRealSum
            (mass, amrex::ParallelDescriptor::IOProcessorNumber());

    // momentum
    std::array<amrex::Real,GEMPIC_VDIM> momentum;
    for (int cmp=0;cmp<GEMPIC_VDIM;cmp++) {
        auto mom_tmp = amrex::ReduceSum( *(part_gr).mypc[spec],
                                         [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<4,0>& p) -> amrex::Real
        {
            auto m  = p.rdata(GEMPIC_VDIM);
            auto vel = p.rdata(cmp);
            return (m*vel);
        });

        amrex::ParallelDescriptor::ReduceRealSum
                (mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

        momentum[cmp] = mom_tmp;
    }

    // kinetic energy
    std::array<amrex::Real,GEMPIC_VDIM> kinetic_energy;
    for (int cmp=0;cmp<GEMPIC_VDIM;cmp++) {
        auto mom_tmp = amrex::ReduceSum( *(part_gr).mypc[spec],
                                         [=] AMREX_GPU_HOST_DEVICE (const amrex::Particle<4,0>& p) -> amrex::Real
        {
            auto m  = p.rdata(GEMPIC_VDIM);
            auto vel = p.rdata(cmp);
            return (m*vel*vel);
        });

        amrex::ParallelDescriptor::ReduceRealSum
                (mom_tmp, amrex::ParallelDescriptor::IOProcessorNumber());

        kinetic_energy[cmp] = mom_tmp;
    }


    AllPrintToFile("test_output_pre_rename.output") << std::endl;
    AllPrintToFile("test_output_pre_rename.output") << "mass: " << mass << std::endl;
    AllPrintToFile("test_output_pre_rename.output") << "momentum: " << momentum[0]
                                                   #if (GEMPIC_VDIM > 1)
                                                            << " " << momentum[1]
                                                   #endif
                                                   #if (GEMPIC_VDIM > 2)
                                                            << " " << momentum[2]
                                                   #endif
                                                            << std::endl;
    AllPrintToFile("test_output_pre_rename.output") << "kinetic energy: " << kinetic_energy[0]
                                                   #if (GEMPIC_VDIM > 1)
                                                            << " " << kinetic_energy[1]
                                                   #endif
                                                   #if (GEMPIC_VDIM > 2)
                                                            << " " << kinetic_energy[2]
                                                   #endif
                                                            << std::endl;
    if (ParallelDescriptor::MyProc()==0) std::rename("test_output_pre_rename.output.0", "test_particle_groups.output");

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}


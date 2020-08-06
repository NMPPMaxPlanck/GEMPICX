#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Utils;

void main_main ()
{
    //------------------------------------------------------------------------------
    // Initialize Function
    double x, y, z;
    double k = 0.5;
    int err;
    std::string WF = "1.0 + 0.0 * cos(kvar * x)";
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvar", &k}};
    int varcount = 4;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    initializer init;
    amrex::IntVect is_periodic(AMREX_D_DECL(1,1,1));
    amrex::IntVect n_cell(AMREX_D_DECL(8,8,8));

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

    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(GEMPIC_DEG_X, GEMPIC_DEG_Y, GEMPIC_DEG_Z)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));

    init.initialize_from_parameters(n_cell,4,is_periodic,maxdeg,0.02,0,{-1.0},{1.0},1000,k,
    VM,VD,VW,0);
    //n_cell, max_grid_size, periodic, Nghost, dt, n_steps, charge, mass, n_part_per_cell, k, vel_mean, vel_dev, vel_weight, propagator

    infrastructure infra(init);

    //------------------------------------------------------------------------------
    // Initialize fields and particles
    maxwell_yee mw_yee(init, infra, init.Nghost);

    // particles
    particle_groups part_gr(init, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0; // all particles are same species for now
    init_particles_cellwise(infra, part_gr, init, species, WF_parse, &x, &y, &z);

    //------------------------------------------------------------------------------
    // initialize rho and phi, phi will solve the analytically exact solution, rho
    // will be overwritten in next paragraph
    mw_yee.init_rho_phi(infra, WF_parse, WF_parse, &x, &y, &z);
    mw_yee.phi.mult(-1.0);

    //------------------------------------------------------------------------------
    // Deposit charges:
    (mw_yee).rho.setVal(0.0, 0); // value and component
    (mw_yee).rho.FillBoundary(infra.geom.periodicity());

    for (int spec=0;spec<(part_gr).n_species;spec++) {
        (*(part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        for (amrex::ParIter<GEMPIC_VDIM+1,0,0,0> pti(*(part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
            amrex::Box tilebox;
            amrex::FArrayBox local_rho;

            tilebox = pti.tilebox();
            tilebox.grow((mw_yee).Nghost);
            const amrex::Box tb = amrex::convert(tilebox, amrex::IntVect::TheUnitVector());

            local_rho.resize(tb,1); // second arg: number of comps
            local_rho.setVal(0.0);

            auto& particles = pti.GetArrayOfStructs();
            const long np  = pti.numParticles();

            amrex::Array4<amrex::Real> const& rhoarr = local_rho.array();
            for (int pp=0;pp<np;pp++) {
                gempic_deposit_rho(particles[pp], (part_gr).charge[spec], rhoarr, infra.plo, infra.dxi);
              }
            ((mw_yee).rho)[pti].atomicAdd(local_rho,tb,tb,0,0,1);
          }
      }

    (mw_yee).rho.SumBoundary(infra.geom.periodicity());
    (mw_yee).rho.FillBoundary(infra.geom.periodicity());

    //------------------------------------------------------------------------------
    // Compute difference and store in phi:
    (mw_yee.phi).minus(mw_yee.rho, 0, 1, 0);

    std::ofstream ofs("test_deposit_rho.output", std::ofstream::out);
    amrex::Print(ofs) << std::endl;
    amrex::Print(ofs) << "Norm of error: " << gempic_norm(&(mw_yee.phi), infra, 2) << std::endl;
    ofs.close();
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}




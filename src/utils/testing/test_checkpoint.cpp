#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_checkpoint.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_sampler.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Sampling;
using namespace Utils;

template<int vdim, int numspec>
void main_main ()
{
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // compile parameters
    std::array<int, GEMPIC_SPACEDIM> degs = {AMREX_D_DECL(GEMPIC_DEG_X, GEMPIC_DEG_Y, GEMPIC_DEG_Z)};
    int maxdeg = *(std::max_element(degs.begin(), degs.end()));
    int Nghost = maxdeg;

    // initialize parameters
    std::array<int,GEMPIC_SPACEDIM> n_cell_vector = {AMREX_D_DECL(4,4,4)};
    int n_part_per_cell = 1;
    int n_steps = 2;
    std::array<int,GEMPIC_SPACEDIM> is_periodic_vector = {AMREX_D_DECL(1,1,1)};
    int max_grid_size = 2;
    amrex::Real dt = 0.1;
    std::array<amrex::Real, numspec> charge = {-1.0};
    std::array<amrex::Real, numspec> mass = {1.0};
    amrex::Real k = 1.25;
    std::string WF = "0.0";
    std::string phi = "0.0";
    std::string rho = "0.0";
    amrex::Real tolerance_particles = 1.e-10;

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VW[j].push_back(1.0);
    }
    VD[0].push_back(0.02/sqrt(2));
    VD[1].push_back(sqrt(12)*VD[0][0]);
    VD[2].push_back(VD[1][0]);

    // initialize amrex data structures from parameters
    amrex::IntVect n_cell(AMREX_D_DECL(n_cell_vector[0],n_cell_vector[1],n_cell_vector[2]));
    amrex::IntVect is_periodic(AMREX_D_DECL(is_periodic_vector[0],is_periodic_vector[1],is_periodic_vector[2]));

    // functions
    double x, y, z;
    int err;
    te_variable read_vars[] = {{"x", &x}, {"y", &y}, {"z", &z}, {"kvar", &k}};
    int varcount = 4;
    te_expr *WF_parse = te_compile(WF.c_str(), read_vars, varcount, &err);

    te_variable read_vars_poi[] = {{"x", &x}, {"y", &y}, {"z", &z}};
    varcount = 3;
    te_expr *rho_parse = te_compile(rho.c_str(), read_vars_poi, varcount, &err);
    te_expr *phi_parse = te_compile(phi.c_str(), read_vars_poi, varcount, &err);

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    //initializer
    initializer<vdim, numspec> init;
    init.initialize_from_parameters(n_cell,max_grid_size,is_periodic,Nghost,dt,n_steps,charge,mass,n_part_per_cell,k,
                                    VM,VD,VW,tolerance_particles);

    // infrastructure
    infrastructure infra;
    init.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(init, infra, init.Nghost);
    mw_yee.init_rho_phi(infra, phi_parse, rho_parse, &x, &y, &z);

    // particles
    particle_groups<vdim, numspec> part_gr(init, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0; // all particles are same species for now
    init_particles_full_domain(infra, part_gr, init, species, WF_parse, &x, &y, &z);

    //------------------------------------------------------------------------------
    // test:

    (*(mw_yee).J_Array[0]).setVal(1.0, 0);
    std::cout << gempic_norm(&(*(mw_yee).J_Array[0]), infra, 0) << std::endl;
    Gempic_WriteCheckpointFile (&mw_yee, &part_gr, &infra, "test_checkpoint", 0, 20);

    (*(mw_yee).J_Array[0]).setVal(2.0, 0);
    std::cout << gempic_norm(&(*(mw_yee).J_Array[0]), infra, 0) << std::endl;

    Gempic_ReadCheckpointFile (&mw_yee, &part_gr, &infra, "test_checkpoint", 0); // last 2 args: field, step
    std::cout << gempic_norm(&(*(mw_yee).J_Array[0]), infra, 0) << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main<3, 1>();

    amrex::Finalize();
}




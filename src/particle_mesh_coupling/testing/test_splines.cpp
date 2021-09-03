
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_splines.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;

void main_main ()
{

    //------------------------------------------------------------------------------
    // Init splines

    std::array<bool,GEMPIC_SPACEDIM> cent_eval = {AMREX_D_DECL(false, false, true)};
    std::array<bool,GEMPIC_SPACEDIM> int_eval = {AMREX_D_DECL(false, false, false)};
    splines spl(cent_eval, int_eval);

    int cell_index = 0;
    int dimension = 0;
    std::array<amrex::Real, GEMPIC_SPACEDIM> x = {AMREX_D_DECL(0.25, 0.25, 0.25)};
    std::cout << spl.compute_cell_index (cell_index, dimension, x[0], cent_eval[0]) << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}




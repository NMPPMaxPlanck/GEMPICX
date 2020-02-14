
#include <AMReX.H>
#include <AMReX_Print.H>
#include <particle_groups.H>
#include <maxwell_yee.H>
#include <initializer.H>
#include <splines.H>
#include <gempic_Config.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

using namespace std;
using namespace amrex;

double WF (std::array<double,GEMPIC_SPACEDIM> x, std::array<double,GEMPIC_VDIM> v,double k) {
    return(0.0);
}

void main_main ()
{

    //------------------------------------------------------------------------------
    // Init splines

    amrex::IntVect deg = amrex::IntVect{AMREX_D_DECL(1,2,1)};
    std::array<bool,GEMPIC_SPACEDIM> cent_eval = {false, false, true};
    std::array<bool,GEMPIC_SPACEDIM> int_eval = {false, false, false};
    splines spl(cent_eval, int_eval);

    int cell_index = 0;
    int dimension = 0;
    std::array<amrex::Real, GEMPIC_SPACEDIM> x = {AMREX_D_DECL(0.25, 0.25, 0.25)};
    std::cout << spl.compute_cell_index (cell_index, dimension, x[0], cent_eval[0]) << std::endl;

    std::array<int, GEMPIC_SPACEDIM> cell_vec = {AMREX_D_DECL(0,0,0)};
    //std::cout << spl.deposition_coefficient (x, cell_vec) << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}




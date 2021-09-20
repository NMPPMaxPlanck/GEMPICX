
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

    amrex::GpuArray<bool,GEMPIC_SPACEDIM> cent_eval = {AMREX_D_DECL(false, false, true)};
    amrex::GpuArray<bool,GEMPIC_SPACEDIM> int_eval = {AMREX_D_DECL(false, false, false)};
    splines spl(cent_eval, int_eval);

    int cell_index = 0;
    int dimension = 0;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> x = {AMREX_D_DECL(0.25, 0.25, 0.25)};
    int ind = spl.compute_cell_index (cell_index, dimension, x[0], cent_eval[0]);

    amrex::AllPrintToFile("test_splines_additional.tmp") << std::endl;
    amrex::AllPrintToFile("test_splines_additional.tmp") << ind << std::endl;

#if  !(GEMPIC_GPU)
    amrex::AllPrintToFile("test_splines.tmp") << std::endl;
#endif
    amrex::AllPrintToFile("test_splines.tmp") << (std::abs(ind-0)<1e-12) << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (ParallelDescriptor::MyProc()==0) remove("test_splines.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_splines_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_splines.tmp.0", "test_splines.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_splines_additional.tmp.0", "test_splines_additional.output");

    amrex::Finalize();
}




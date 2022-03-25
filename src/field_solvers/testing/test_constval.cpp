#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_initMF.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace FieldSolvers;

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{
    // initialize objects necessary for MultiFab
    int ncomp = 1;
    int ngrow = 0;
    int max_grid_size = 2;
    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(4, 4, 4));

    amrex::Box domain;
    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);

    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(max_grid_size);

    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // initialize MultiFabs
    MultiFab mf_in(grid, distriMap, ncomp, ngrow);
    MultiFab mf_out(grid, distriMap, ncomp, ngrow);

    // apply init function from header
    amrex::Real value = 2.0;
    initConst(mf_in, value);

    // set value with AMReX
    mf_out.setVal(2.0);

    // compare results
    mf_out.minus(mf_in, 0, 1, 0);
    amrex::Real norm_diff = mf_out.norm2();
    amrex::PrintToFile("test_constval.tmp") << std::endl;
    if (norm_diff < 1e-12)
    {
        amrex::PrintToFile("test_constval.tmp") << 1 << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_constval.tmp") << 0 << std::endl;
    }
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    const int vdim=3, numspec=1, degx=1, degy=1, degz=1;

    if (ParallelDescriptor::MyProc() == 0) remove("test_constval.tmp.0");
    main_main<vdim, numspec, degx, degy, degz>();
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_constval.tmp.0", "test_constval.output");

    amrex::Finalize();
}

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_amrex_init.H"
#include "GEMPIC_initMF.H"

using namespace amrex;
using namespace Gempic;

using namespace FieldSolvers;

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    // initialize objects necessary for MultiFab
    int ncomp = 1;
    int ngrow = 0;
    int maxGridSize = 2;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(4, 4, 4));

    amrex::Box domain;
    domain.setSmall(domLo);
    domain.setBig(domHi);

    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(maxGridSize);

    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // initialize MultiFabs
    MultiFab mfIn(grid, distriMap, ncomp, ngrow);
    MultiFab mfOut(grid, distriMap, ncomp, ngrow);

    // apply init function from header
    amrex::Real value = 2.0;
    init_const(mfIn, value);

    // set value with AMReX
    mfOut.setVal(2.0);

    // compare results
    mfOut.minus(mfIn, 0, 1, 0);
    amrex::Real normDiff = mfOut.norm2();
    amrex::PrintToFile("test_constval.tmp") << std::endl;
    if (normDiff < 1e-12)
    {
        amrex::PrintToFile("test_constval.tmp") << 1 << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_constval.tmp") << 0 << std::endl;
    }
}

int main (int argc, char *argv[])
{
    const bool buildParmParse = true;
    amrex::Initialize(argc, argv, buildParmParse, MPI_COMM_WORLD, overwrite_amrex_parser_defaults);
    const int vdim = 3, numspec = 1, degx = 1, degy = 1, degz = 1;

    if (ParallelDescriptor::MyProc() == 0) remove("test_constval.tmp.0");
    main_main<vdim, numspec, degx, degy, degz>();
    if (ParallelDescriptor::MyProc() == 0)
    {
        std::rename("test_constval.tmp.0", "test_constval.output");
    }

    amrex::Finalize();
}

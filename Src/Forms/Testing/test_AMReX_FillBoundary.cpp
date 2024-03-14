#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_Config.H"

using namespace amrex;

void main_main ()
{
    //-----------------------------------------------------------------------------
    // Initialize structures

    // Domain
    int isPeriodic[3] = {0, 1, 1};
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(3, 1, 1));
    amrex::Box domain;
    domain.setSmall(domLo);
    domain.setBig(domHi);
    amrex::RealBox realBox;
    realBox.setLo(amrex::RealVect{AMREX_D_DECL(0.0, 0.0, 0.0)});
    realBox.setHi(amrex::RealVect{AMREX_D_DECL(1.0, 1.0, 1.0)});

    // Grid
    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(2);

    // DistributionMapping
    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // Geometry
    amrex::Geometry geom;
    geom.define(domain, &realBox, amrex::CoordSys::cartesian, isPeriodic);

    // MultiFab
    amrex::IndexType indexA(amrex::IntVect{AMREX_D_DECL(1, 0, 0)});
    int nghost = 1;
    amrex::MultiFab testMf(convert(grid, indexA), distriMap, 1, nghost);
    testMf.setVal(0.0);

    //-----------------------------------------------------------------------------
    // Write values into MultiFab

    for (amrex::MFIter mfi(testMf); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        std::cout << "low: " << lo[xDir] << ", " << lo[yDir] << ", " << lo[zDir] << std::endl;
        std::cout << "low: " << hi[xDir] << ", " << hi[yDir] << ", " << hi[zDir] << std::endl;

        amrex::Array4<amrex::Real> const &vecMF = (testMf)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    { vecMF(i, j, k) = i * 100 + j * 10 + k; });
    }

    amrex::PrintToFile("test_AMReX_FillBoundary_additional.tmp") << std::endl;
    for (amrex::MFIter mfi(testMf); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_FillBoundary_additional.tmp") << testMf[mfi] << std::endl;
    }

    bool passed = true;
    passed = passed && (std::abs(testMf.norm1(0, nghost) - 4932) < 1e-6);
    testMf.FillBoundary(geom.periodicity());
    passed = passed && (std::abs(testMf.norm1(0, nghost) - 26304) < 1e-6);

    amrex::PrintToFile("test_AMReX_FillBoundary.tmp") << std::endl;
    amrex::PrintToFile("test_AMReX_FillBoundary.tmp") << passed << std::endl;

    amrex::PrintToFile("test_AMReX_FillBoundary_additional.tmp") << "FILLBOUNDARY" << std::endl;
    for (amrex::MFIter mfi(testMf); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_FillBoundary_additional.tmp") << testMf[mfi] << std::endl;
    }
}

int main (int argc, char *argv[])
{
    const bool buildParmParse = true;
    amrex::Initialize(argc, argv, buildParmParse, MPI_COMM_WORLD,
                      Gempic::overwrite_amrex_parser_defaults);

    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_FillBoundary.tmp.0");
    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_FillBoundary_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc() == 0)
    {
        std::rename("test_AMReX_FillBoundary.tmp.0", "test_AMReX_FillBoundary.output");
    }
    if (ParallelDescriptor::MyProc() == 0)
    {
        std::rename("test_AMReX_FillBoundary_additional.tmp.0",
                    "test_AMReX_FillBoundary_additional.output");
    }

    amrex::Finalize();
}

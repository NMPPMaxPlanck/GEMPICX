#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include "GEMPIC_AmrexInit.H"
#include "GEMPIC_Config.H"

using namespace amrex;

void main_main ()
{
    //-----------------------------------------------------------------------------
    // Initialize structures

    // Domain
    int isPeriodic[3] = {1, 1, 1};
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(3, 1, 1));
    int maxGridSize = 2;
    amrex::Box domain;
    domain.setSmall(domLo);
    domain.setBig(domHi);
    amrex::RealBox realBox;
    // real_box.setLo(amrex::RealVect{AMREX_D_DECL(0.0, 0.0, 0.0)});
    // real_box.setHi(amrex::RealVect{AMREX_D_DECL(6.0, 2.0, 2.0)});
    realBox.setLo(amrex::RealVect{AMREX_D_DECL(0.0, 0.0, 0.0)});
    realBox.setHi(amrex::RealVect{AMREX_D_DECL(2.0, 1.0, 1.0)});

    // Grid
    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(maxGridSize);

    // DistributionMapping
    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // Geometry
    amrex::Geometry geom;
    geom.define(domain, &realBox, amrex::CoordSys::cartesian, isPeriodic);

    // MultiFab
    amrex::IndexType indexA(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});  // nodal
    int nghost = 1;
    amrex::MultiFab testMf(convert(grid, indexA), distriMap, 1, nghost);
    testMf.setVal(0.0);

    // Ownermask
    amrex::iMultiFab mask(convert(grid, indexA), distriMap, 1, nghost);
    mask.setVal(0);

    //-----------------------------------------------------------------------------
    // Fill MultiFab

    for (amrex::MFIter mfi(testMf); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::Array4<amrex::Real> const &mfArr = (testMf)[mfi].array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // if-loop to exclude ownership for the point that is at the upper boundary
                        // for nodal directions
                        if ((i <= (indexA[xDir] == 0 ? hi[xDir] : (hi[xDir] - 1))) &&
                            (j <= (indexA[yDir] == 0 ? hi[yDir] : (hi[yDir] - 1))) &&
                            (k <= (indexA[zDir] == 0 ? hi[zDir] : (hi[zDir] - 1))))
                        {
                            mfArr(i, j, k) = 1000 * lo[xDir] + 100 * i + 10 * j + k;
                        }
                    });
    }

    for (amrex::MFIter mfi(testMf); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_OverrideSync_additional.tmp") << testMf[mfi] << std::endl;
    }

    //-----------------------------------------------------------------------------
    // Fill Owner Mask

    for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::Array4<int> const &maskArr = (mask)[mfi].array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // if-loop to exclude ownership for the point that is at the upper boundary
                        // for nodal directions
                        if ((i <= (indexA[xDir] == 0 ? hi[xDir] : (hi[xDir] - 1))) &&
                            (j <= (indexA[yDir] == 0 ? hi[yDir] : (hi[yDir] - 1))) &&
                            (k <= (indexA[zDir] == 0 ? hi[zDir] : (hi[zDir] - 1))))
                        {
                            maskArr(i, j, k) = 1;
                        }
                    });
    }

    bool passed = true;
    std::cout << testMf.norm1(0, nghost) << std::endl;
    passed = passed && (std::abs(testMf.norm1(0, nghost) - 18488) < 1e-12);
    testMf.OverrideSync(mask, geom.periodicity());
    std::cout << testMf.norm1(0, nghost) << std::endl;
    passed = passed && (std::abs(testMf.norm1(0, nghost) - 40938) < 1e-12);

    amrex::PrintToFile("test_AMReX_OverrideSync.tmp") << std::endl;
    amrex::PrintToFile("test_AMReX_OverrideSync.tmp") << passed << std::endl;

    amrex::PrintToFile("test_AMReX_OverrideSync_additional.tmp") << "OVERRIDESYNC" << std::endl;
    for (amrex::MFIter mfi(testMf); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_OverrideSync_additional.tmp") << testMf[mfi] << std::endl;
    }
}

int main (int argc, char *argv[])
{
    const bool buildParmParse = true;
    amrex::Initialize(argc, argv, buildParmParse, MPI_COMM_WORLD,
                      Gempic::overwrite_amrex_parser_defaults);

    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_OverrideSync.tmp.0");
    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_OverrideSync_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc() == 0)
    {
        std::rename("test_AMReX_OverrideSync.tmp.0", "test_AMReX_OverrideSync.output");
    }
    if (ParallelDescriptor::MyProc() == 0)
    {
        std::rename("test_AMReX_OverrideSync_additional.tmp.0",
                    "test_AMReX_OverrideSync_additional.output");
    }

    amrex::Finalize();
}

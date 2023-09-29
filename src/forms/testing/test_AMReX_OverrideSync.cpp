#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>
#include <GEMPIC_amrex_init.H>

using namespace amrex;

void main_main()
{
    //-----------------------------------------------------------------------------
    // Initialize structures

    // Domain
    int is_periodic[3] = {1, 1, 1};
    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(3, 1, 1));
    int max_grid_size = 2;
    amrex::Box domain;
    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);
    amrex::RealBox real_box;
    // real_box.setLo(amrex::RealVect{AMREX_D_DECL(0.0, 0.0, 0.0)});
    // real_box.setHi(amrex::RealVect{AMREX_D_DECL(6.0, 2.0, 2.0)});
    real_box.setLo(amrex::RealVect{AMREX_D_DECL(0.0, 0.0, 0.0)});
    real_box.setHi(amrex::RealVect{AMREX_D_DECL(2.0, 1.0, 1.0)});

    // Grid
    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(max_grid_size);

    // DistributionMapping
    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // Geometry
    amrex::Geometry geom;
    geom.define(domain, &real_box, amrex::CoordSys::cartesian, is_periodic);

    // MultiFab
    amrex::IndexType Index_A(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});  // nodal
    int Nghost = 1;
    amrex::MultiFab TestMF(convert(grid, Index_A), distriMap, 1, Nghost);
    TestMF.setVal(0.0);

    // Ownermask
    amrex::iMultiFab Mask(convert(grid, Index_A), distriMap, 1, Nghost);
    Mask.setVal(0);

    //-----------------------------------------------------------------------------
    // Fill MultiFab

    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::Array4<amrex::Real> const &mf_arr = (TestMF)[mfi].array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // if-loop to exclude ownership for the point that is at the upper boundary
                        // for nodal dimensions
                        if ((i <= (Index_A[0] == 0 ? hi[0] : (hi[0] - 1))) &&
                            (j <= (Index_A[1] == 0 ? hi[1] : (hi[1] - 1))) &&
                            (k <= (Index_A[2] == 0 ? hi[2] : (hi[2] - 1))))
                            mf_arr(i, j, k) = 1000 * lo[0] + 100 * i + 10 * j + k;
                    });
    }

    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_OverrideSync_additional.tmp") << TestMF[mfi] << std::endl;
    }

    //-----------------------------------------------------------------------------
    // Fill Owner Mask

    for (amrex::MFIter mfi(Mask); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::IntVect hi = {bx.bigEnd()};

        amrex::Array4<int> const &mask_arr = (Mask)[mfi].array();
        ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        // if-loop to exclude ownership for the point that is at the upper boundary
                        // for nodal dimensions
                        if ((i <= (Index_A[0] == 0 ? hi[0] : (hi[0] - 1))) &&
                            (j <= (Index_A[1] == 0 ? hi[1] : (hi[1] - 1))) &&
                            (k <= (Index_A[2] == 0 ? hi[2] : (hi[2] - 1))))
                            mask_arr(i, j, k) = 1;
                    });
    }

    bool passed = true;
    std::cout << TestMF.norm1(0, Nghost) << std::endl;
    passed = passed && (std::abs(TestMF.norm1(0, Nghost) - 18488) < 1e-12);
    TestMF.OverrideSync(Mask, geom.periodicity());
    std::cout << TestMF.norm1(0, Nghost) << std::endl;
    passed = passed && (std::abs(TestMF.norm1(0, Nghost) - 40938) < 1e-12);

    amrex::PrintToFile("test_AMReX_OverrideSync.tmp") << std::endl;
    amrex::PrintToFile("test_AMReX_OverrideSync.tmp") << passed << std::endl;

    amrex::PrintToFile("test_AMReX_OverrideSync_additional.tmp") << "OVERRIDESYNC" << std::endl;
    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_OverrideSync_additional.tmp") << TestMF[mfi] << std::endl;
    }
}

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      Gempic::overwrite_amrex_parser_defaults);

    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_OverrideSync.tmp.0");
    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_OverrideSync_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_AMReX_OverrideSync.tmp.0", "test_AMReX_OverrideSync.output");
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_AMReX_OverrideSync_additional.tmp.0",
                    "test_AMReX_OverrideSync_additional.output");

    amrex::Finalize();
}

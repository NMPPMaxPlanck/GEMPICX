#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

using namespace std;
using namespace amrex;


void main_main ()
{

    //-----------------------------------------------------------------------------
    // Initialize structures

    // Domain
    int is_periodic[3] =  {0,1,1};
    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(3, 1, 1));
    amrex::Box domain;
    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);
    amrex::RealBox real_box;
    real_box.setLo({0.0, 0.0, 0.0});
    real_box.setHi({1.0, 1.0, 1.0});

    // Grid
    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(2);

    // DistributionMapping
    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // Geometry
    amrex::Geometry geom;
    geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic);

    // MultiFab
    amrex::IndexType Index_A(amrex::IntVect{AMREX_D_DECL(1,0,0)});
    int Nghost = 1;
    amrex::MultiFab TestMF(convert(grid, Index_A),distriMap,1,Nghost);
    TestMF.setVal(0.0);
    TestMF.FillBoundary(geom.periodicity());

    //-----------------------------------------------------------------------------
    // Write values into MultiFab

    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        const amrex::Box& bx = mfi.validbox();

        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};

        std::cout << "low: " << lo[0] << ", " << lo[1] << ", " << lo[2] << std::endl;
        std::cout << "low: " << hi[0] << ", " << hi[1] << ", " << hi[2] << std::endl;

        amrex::Array4<amrex::Real> const& vecMF = (TestMF)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            vecMF(i,j,k) = (i)*100+(j)*10+(k);
        });
    }

    amrex::AllPrintToFile("test_AMReX_FillBoundary_additional.tmp") << std::endl;
    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_AMReX_FillBoundary_additional.tmp") << TestMF[mfi] << std::endl;
    }

    bool passed = true;
    passed = passed && (std::abs(TestMF.norm1(0,Nghost) - 4932) < 1e-6);
    TestMF.FillBoundary(geom.periodicity());
    passed = passed && (std::abs(TestMF.norm1(0,Nghost) - 26304) < 1e-6);

    amrex::AllPrintToFile("test_AMReX_FillBoundary.tmp") << std::endl;
    amrex::AllPrintToFile("test_AMReX_FillBoundary.tmp") << passed << std::endl;

    amrex::AllPrintToFile("test_AMReX_FillBoundary_additional.tmp") << "FILLBOUNDARY" << std::endl;
    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_AMReX_FillBoundary_additional.tmp") << TestMF[mfi] << std::endl;
    }

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (ParallelDescriptor::MyProc()==0) remove("test_AMReX_FillBoundary.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_AMReX_FillBoundary_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_FillBoundary.tmp.0", "test_AMReX_FillBoundary.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_FillBoundary_additional.tmp.0", "test_AMReX_FillBoundary_additional.output");

    amrex::Finalize();
}




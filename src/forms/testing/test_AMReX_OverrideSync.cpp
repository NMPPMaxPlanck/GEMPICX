#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>


using namespace std;
using namespace amrex;


void main_main ()
{

    //-----------------------------------------------------------------------------
    // Initialize structures

    // Domain
    int is_periodic[3] =  {1,1,1};
    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(3, 1, 1));
    int max_grid_size = 2;
    amrex::Box domain;
    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);
    amrex::RealBox real_box;
    //real_box.setLo({0.0, 0.0, 0.0});
    //real_box.setHi({6.0, 2.0, 2.0});
    real_box.setLo({0.0, 0.0, 0.0});
    real_box.setHi({2.0, 1.0, 1.0});

    // Grid
    amrex::BoxArray grid;
    grid.define(domain);
    grid.maxSize(max_grid_size);

    // DistributionMapping
    amrex::DistributionMapping distriMap;
    distriMap.define(grid);

    // Geometry
    amrex::Geometry geom;
    geom.define(domain,&real_box,amrex::CoordSys::cartesian,is_periodic);

    // MultiFab
    amrex::IndexType Index_A(amrex::IntVect{AMREX_D_DECL(1,1,0)}); // nodal
    int Nghost = 1;
    amrex::MultiFab TestMF(convert(grid, Index_A),distriMap,1,Nghost);
    TestMF.setVal(0.0,0);
    TestMF.FillBoundary(geom.periodicity());

    // Ownermask
    amrex::iMultiFab Mask(convert(grid, Index_A),distriMap,1,Nghost);
    Mask.setVal(0,0);
    Mask.FillBoundary(geom.periodicity());

    //-----------------------------------------------------------------------------
    // Fill MultiFab


    for ( amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};
        for(int k=lo[2]; k<=hi[2]; k++){
            for(int j=lo[1]; j<=hi[1]; j++){
                for(int l=lo[0]; l<=hi[0]; l++){
                    // the box for these values:
                    amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, Index_A);
                    TestMF[mfi].setVal(1000*lo[0]+100*l+10*j+k, cc, 0, 1);
                }

            }

        }
    }

    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_AMReX_OverrideSync_additional.tmp") << TestMF[mfi] << std::endl;
    }

    //-----------------------------------------------------------------------------
    // Fill Owner Mask


    for ( amrex::MFIter mfi(Mask); mfi.isValid(); ++mfi ) {
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};
        for(int k=lo[2]; k<=(Index_A[2]==0?hi[2]:(hi[2]-1)); k++){
            for(int j=lo[1]; j<=(Index_A[1]==0?hi[1]:(hi[1]-1)); j++){
                for(int l=lo[0]; l<=(Index_A[0]==0?hi[0]:(hi[0]-1)); l++){
                    // the box for these values:
                    amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, Index_A);
                    Mask[mfi].setVal(1, cc, 0, 1);
                }

            }

        }
    }

    bool passed = true;
    std::cout << TestMF.norm1(0,Nghost) << std::endl;
    passed = passed && (std::abs(TestMF.norm1(0,Nghost) - 43578) < 1e-12);
    TestMF.OverrideSync(Mask, geom.periodicity());
    std::cout << TestMF.norm1(0,Nghost) << std::endl;
    passed = passed && (std::abs(TestMF.norm1(0,Nghost) - 40938) < 1e-12);

    amrex::AllPrintToFile("test_AMReX_OverrideSync.tmp") << std::endl;
    amrex::AllPrintToFile("test_AMReX_OverrideSync.tmp") << passed << std::endl;

    amrex::AllPrintToFile("test_AMReX_OverrideSync_additional.tmp") << "OVERRIDESYNC" << std::endl;
    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_AMReX_OverrideSync_additional.tmp") << TestMF[mfi] << std::endl;
    }


}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (ParallelDescriptor::MyProc()==0) remove("test_AMReX_OverrideSync.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_AMReX_OverrideSync_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_OverrideSync.tmp.0", "test_AMReX_OverrideSync.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_OverrideSync_additional.tmp.0", "test_AMReX_OverrideSync_additional.output");

    amrex::Finalize();
}




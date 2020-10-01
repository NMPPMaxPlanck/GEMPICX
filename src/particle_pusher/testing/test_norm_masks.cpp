#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_initializer.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Diagnostics_Output;
using namespace Init;
using namespace Particles;
using namespace Sampling;

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
    real_box.setLo({0.0, 0.0, 0.0});
    real_box.setHi({1.0, 1.0, 1.0});

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
    amrex::IndexType Index_A(amrex::IntVect{AMREX_D_DECL(1,0,0)}); // nodal
    int Nghost = 1;
    amrex::MultiFab TestMF(convert(grid, Index_A),distriMap,1,Nghost);
    TestMF.setVal(0.0,0);
    TestMF.FillBoundary(geom.periodicity());

    //-----------------------------------------------------------------------------
    // Fill MultiFab


    for ( amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        const amrex::Box& bx = mfi.validbox();
        amrex::IntVect lo = {bx.smallEnd()};
        amrex::IntVect hi = {bx.bigEnd()};
        if (lo[0]==0) {
        amrex::Box cc0(amrex::IntVect{AMREX_D_DECL(0,0,0)}, amrex::IntVect{AMREX_D_DECL(0,0,0)}, Index_A);
        TestMF[mfi].setVal(1, cc0, 0, 1);
        amrex::Box cc1(amrex::IntVect{AMREX_D_DECL(1,0,0)}, amrex::IntVect{AMREX_D_DECL(1,0,0)}, Index_A);
        TestMF[mfi].setVal(4, cc1, 0, 1);
        amrex::Box cc2(amrex::IntVect{AMREX_D_DECL(2,0,0)}, amrex::IntVect{AMREX_D_DECL(2,0,0)}, Index_A);
        TestMF[mfi].setVal(6, cc2, 0, 1);
        amrex::Box cc3(amrex::IntVect{AMREX_D_DECL(0,1,0)}, amrex::IntVect{AMREX_D_DECL(0,1,0)}, Index_A);
        TestMF[mfi].setVal(10, cc3, 0, 1);
        amrex::Box cc4(amrex::IntVect{AMREX_D_DECL(1,1,0)}, amrex::IntVect{AMREX_D_DECL(1,1,0)}, Index_A);
        TestMF[mfi].setVal(40, cc4, 0, 1);
        amrex::Box cc5(amrex::IntVect{AMREX_D_DECL(2,1,0)}, amrex::IntVect{AMREX_D_DECL(2,1,0)}, Index_A);
        TestMF[mfi].setVal(60, cc5, 0, 1);
        amrex::Box cc6(amrex::IntVect{AMREX_D_DECL(0,0,1)}, amrex::IntVect{AMREX_D_DECL(0,0,1)}, Index_A);
        TestMF[mfi].setVal(100, cc6, 0, 1);
        amrex::Box cc7(amrex::IntVect{AMREX_D_DECL(1,0,1)}, amrex::IntVect{AMREX_D_DECL(1,0,1)}, Index_A);
        TestMF[mfi].setVal(400, cc7, 0, 1);
        amrex::Box cc8(amrex::IntVect{AMREX_D_DECL(2,0,1)}, amrex::IntVect{AMREX_D_DECL(2,0,1)}, Index_A);
        TestMF[mfi].setVal(600, cc8, 0, 1);
        amrex::Box cc9(amrex::IntVect{AMREX_D_DECL(0,1,1)}, amrex::IntVect{AMREX_D_DECL(0,1,1)}, Index_A);
        TestMF[mfi].setVal(1000, cc9, 0, 1);
        amrex::Box cc10(amrex::IntVect{AMREX_D_DECL(1,1,1)}, amrex::IntVect{AMREX_D_DECL(1,1,1)}, Index_A);
        TestMF[mfi].setVal(4000, cc10, 0, 1);
        amrex::Box cc11(amrex::IntVect{AMREX_D_DECL(2,1,1)}, amrex::IntVect{AMREX_D_DECL(2,1,1)}, Index_A);
        TestMF[mfi].setVal(6000, cc11, 0, 1);
} else {
        amrex::Box cc12(amrex::IntVect{AMREX_D_DECL(2,0,0)}, amrex::IntVect{AMREX_D_DECL(2,0,0)}, Index_A);
        TestMF[mfi].setVal(10000, cc12, 0, 1);
        amrex::Box cc13(amrex::IntVect{AMREX_D_DECL(3,0,0)}, amrex::IntVect{AMREX_D_DECL(3,0,0)}, Index_A);
        TestMF[mfi].setVal(40000, cc13, 0, 1);
        amrex::Box cc14(amrex::IntVect{AMREX_D_DECL(4,0,0)}, amrex::IntVect{AMREX_D_DECL(4,0,0)}, Index_A);
        TestMF[mfi].setVal(60000, cc14, 0, 1);
        amrex::Box cc15(amrex::IntVect{AMREX_D_DECL(2,1,0)}, amrex::IntVect{AMREX_D_DECL(2,1,0)}, Index_A);
        TestMF[mfi].setVal(100000, cc15, 0, 1);
        amrex::Box cc16(amrex::IntVect{AMREX_D_DECL(3,1,0)}, amrex::IntVect{AMREX_D_DECL(3,1,0)}, Index_A);
        TestMF[mfi].setVal(400000, cc16, 0, 1);
        amrex::Box cc17(amrex::IntVect{AMREX_D_DECL(4,1,0)}, amrex::IntVect{AMREX_D_DECL(4,1,0)}, Index_A);
        TestMF[mfi].setVal(600000, cc17, 0, 1);
        amrex::Box cc18(amrex::IntVect{AMREX_D_DECL(2,0,1)}, amrex::IntVect{AMREX_D_DECL(2,0,1)}, Index_A);
        TestMF[mfi].setVal(1000000, cc18, 0, 1);
        amrex::Box cc19(amrex::IntVect{AMREX_D_DECL(3,0,1)}, amrex::IntVect{AMREX_D_DECL(3,0,1)}, Index_A);
        TestMF[mfi].setVal(4000000, cc19, 0, 1);
        amrex::Box cc20(amrex::IntVect{AMREX_D_DECL(4,0,1)}, amrex::IntVect{AMREX_D_DECL(4,0,1)}, Index_A);
        TestMF[mfi].setVal(6000000, cc20, 0, 1);
        amrex::Box cc21(amrex::IntVect{AMREX_D_DECL(2,1,1)}, amrex::IntVect{AMREX_D_DECL(2,1,1)}, Index_A);
        TestMF[mfi].setVal(10000000, cc21, 0, 1);
        amrex::Box cc22(amrex::IntVect{AMREX_D_DECL(3,1,1)}, amrex::IntVect{AMREX_D_DECL(3,1,1)}, Index_A);
        TestMF[mfi].setVal(40000000, cc22, 0, 1);
        amrex::Box cc23(amrex::IntVect{AMREX_D_DECL(4,1,1)}, amrex::IntVect{AMREX_D_DECL(4,1,1)}, Index_A);
        TestMF[mfi].setVal(60000000, cc23, 0, 1);
}
        //for(int k=lo[2]; k<=hi[2]; k++){
            //for(int j=lo[1]; j<=hi[1]; j++){
                //for(int l=lo[0]; l<=hi[0]; l++){
                    // the box for these values:
                    //amrex::Box cc(amrex::IntVect{AMREX_D_DECL(l,j,k)}, amrex::IntVect{AMREX_D_DECL(l,j,k)}, Index_A);
                    //TestMF[mfi].setVal(1000*lo[0]+100*l+10*j+k, cc, 0, 1);
                    //TestMF[mfi].setVal(1000*lo[0]+100*l+10*j+k, cc, 0, 1);
                //}

            //}

        //}
    }

    std::cout << TestMF.norm1(0, geom.periodicity(), false) << std::endl;

    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_norm_masks") << TestMF[mfi] << std::endl;
    }


}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
}


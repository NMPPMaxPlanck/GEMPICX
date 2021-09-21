#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
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

        amrex::Array4<amrex::Real> const& mf_arr = TestMF[mfi].array();

        if (lo[0]==0) {
            mf_arr(0,0,0) = 1;
            mf_arr(1,0,0) = 4;
            mf_arr(2,0,0) = 6;
            mf_arr(0,1,0) = 10;
            mf_arr(1,1,0) = 40;
            mf_arr(2,1,0) = 60;
            mf_arr(0,0,1) = 100;
            mf_arr(1,0,1) = 400;
            mf_arr(2,0,1) = 600;
            mf_arr(0,1,1) = 1000;
            mf_arr(1,1,1) = 4000;
            mf_arr(2,1,1) = 6000;
        } else {
            mf_arr(2,0,0) = 10000;
            mf_arr(3,0,0) = 40000;
            mf_arr(4,0,0) = 60000;
            mf_arr(2,1,0) = 100000;
            mf_arr(3,1,0) = 400000;
            mf_arr(4,1,0) = 600000;
            mf_arr(2,0,1) = 1000000;
            mf_arr(3,0,1) = 4000000;
            mf_arr(4,0,1) = 6000000;
            mf_arr(2,1,1) = 10000000;
            mf_arr(3,1,1) = 40000000;
            mf_arr(4,1,1) = 60000000;
        }
    }

    amrex::AllPrintToFile("test_AMReX_NormMasks_additional.tmp") << "Norm of mask:" << TestMF.norm1(0, geom.periodicity(), false) << std::endl;
    amrex::AllPrintToFile("test_AMReX_NormMasks_additional.tmp") << std::endl;

    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_AMReX_NormMasks_additional.tmp") << TestMF[mfi] << std::endl;
    }
    amrex::AllPrintToFile("test_AMReX_NormMasks.tmp") << std::endl;
    amrex::AllPrintToFile("test_AMReX_NormMasks.tmp") << (std::abs(TestMF.norm1(0, geom.periodicity(), false)-83333332.5) < 1e-12) << std::endl;


}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    if (ParallelDescriptor::MyProc()==0) remove("test_AMReX_NormMasks.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_AMReX_NormMasks_additional.tmp.0");

    main_main();

    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_NormMasks.tmp.0", "test_AMReX_NormMasks.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_NormMasks_additional.tmp.0", "test_AMReX_NormMasks_additional.output");

    amrex::Finalize();
}


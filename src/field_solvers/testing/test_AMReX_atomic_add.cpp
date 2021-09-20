#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_assertion.H>

using namespace std;
using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;
using namespace Sampling;

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    amrex::IntVect n_cell = {AMREX_D_DECL(8,10,12)};
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.set_params("add_test", n_cell, {1}, 10, 12, 12, 12, {AMREX_D_DECL(1, 1, 1)}, {AMREX_D_DECL(8, 10, 12)});
    VlMa.set_computed_params();
    CompDom::computational_domain infra;
    VlMa.initialize_infrastructure(&infra);
    particle_groups<vdim, numspec> part_gr(VlMa, infra);
    const amrex::GpuArray<amrex::Real, 3>& dx = {0.6283, 0.5026, 0.4188};
    const amrex::GpuArray<amrex::Real, 3>& plo = {0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> x = {0.2, 0.2, 0.2};
    init_one_particle_cellwise<vdim>(dx, plo, &(*(part_gr).mypc[0]), x);

    //------------------------------------------------------------------------------
    amrex::MultiFab rho; // for Poisson
    const amrex::BoxArray& nba = amrex::convert(infra.grid,amrex::IntVect::TheNodeVector());
    int Nghost = 1;
    rho.define(nba, infra.distriMap, 1, Nghost);
    rho.setVal(0.0);

    amrex::Real testval = 2.3;
    for (amrex::ParIter<vdim+1,0,0,0> pti(*(part_gr).mypc[0], 0); pti.isValid(); ++pti) {

        amrex::Array4<amrex::Real> rhoarr = rho[pti].array();

        amrex::HostDevice::Atomic::Add(&(rhoarr)(5, 6, 7, 0), testval);

    }
    rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());

    amrex::Real readval[1];
    for ( amrex::MFIter mfi(rho); mfi.isValid(); ++mfi ) {
        amrex::AllPrintToFile("test_AMReX_atomic_add_additional.tmp") << rho[mfi] << std::endl;
        rho[mfi].getVal(readval,amrex::IntVect{5,6,7},0,1);
    }


    bool passed = true;
    gempic_assert(&passed, testval, *readval);
    amrex::AllPrintToFile("test_AMReX_atomic_add.tmp") << std::endl;
    amrex::AllPrintToFile("test_AMReX_atomic_add.tmp") << passed << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) std::remove("test_AMReX_atomic_add.tmp.0");
    if (ParallelDescriptor::MyProc()==0) std::remove("test_AMReX_atomic_add_additional.tmp.0");

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1>();
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1>();
    main_main<3, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_atomic_add.tmp.0", "test_AMReX_atomic_add.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_AMReX_atomic_add_additional.tmp.0", "test_AMReX_atomic_add_additional.output");
    amrex::Finalize();
}




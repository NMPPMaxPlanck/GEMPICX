#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_Particles.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_sampler.H>

using namespace amrex;
using namespace Particles;

template <int vdim, int degx, int degy, int degz>
void main_main()
{
    // This test shows how one can get a result from SumBoundary that has two slightly
    // different values for the same cell on different tiles

    //-----------------------------------------------------------------------------
    // Initialize structures

    // Domain
    int is_periodic[3] = {AMREX_D_DECL(1, 1, 1)};
    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(3, 1, 1));
    int max_grid_size = 2;
    amrex::Box domain;
    domain.setSmall(dom_lo);
    domain.setBig(dom_hi);
    amrex::RealBox real_box;
    real_box.setLo({AMREX_D_DECL(0.0, 0.0, 0.0)});
    real_box.setHi({AMREX_D_DECL(2.0, 1.0, 1.0)});

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
    amrex::IndexType Index_A(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});  // nodal | nodal | cell
    int Nghost = 1;
    amrex::MultiFab TestMF(convert(grid, Index_A), distriMap, 1, Nghost);
    TestMF.setVal(0.0);
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> plo;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> dx;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> dxi;
    dxi[GEMPIC_SPACEDIM] = 1.;
    for (int cc = 0; cc < 3; cc++)
    {
        plo[cc] = geom.ProbLo()[cc];
        dxi[cc] = (domain.bigEnd(cc) + 1) / real_box.hi(cc);
        dx[cc] = real_box.hi(cc) / (domain.bigEnd(cc) + 1);
        dxi[GEMPIC_SPACEDIM] *= dxi[cc];
    }
    // Particles
    amrex::Real charge = -1.0;
    particle_groups<vdim> pg(geom, distriMap, grid);

    Gempic::Sampling::init_one_particle_cellwise<vdim>(
        dx, plo, pg, {AMREX_D_DECL(2 * dx[0] / 5.0, 2 * dx[1] / 5.0, 0)});

    pg.Redistribute();
    //-----------------------------------------------------------------------------
    // Deposit charge
    // Deposit charges:
    for (amrex::ParIter<0,0,vdim + 1, 0> pti(pg, 0); pti.isValid(); ++pti) 
    {
        const auto& particles = pti.GetArrayOfStructs();
        const auto partData = particles().data();    
        const long np  = pti.numParticles();
        const auto& particle_attributes = pti.GetStructOfArrays();
        const auto weight = particle_attributes.GetRealData(vdim).data();

        amrex::Array4<amrex::Real> const& rhoarr = TestMF[pti].array();
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
        {
            splines_at_particles<degx,degy,degz> spline;
            amrex::GpuArray<amrex::Real,GEMPIC_SPACEDIM> pos;
            for (int comp = 0; comp < GEMPIC_SPACEDIM; comp++) 
            {
                pos[comp] = partData[pp].pos(comp);
            }
            spline.init_particles(pos , plo, dxi);
            Gempic::Particles::gempic_deposit_charge_indextype<amrex::Particle<vdim+1>,vdim,degx,degy,degz>(
                spline, charge * dxi[GEMPIC_SPACEDIM] * weight[pp], rhoarr, Index_A);
        });
    }
    amrex::PrintToFile("test_AMReX_SumBoundary_additional.tmp") << std::endl;
    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_SumBoundary_additional.tmp") << TestMF[mfi] << std::endl;
    }

    //-----------------------------------------------------------------------------
    // SumBoundary

    TestMF.SumBoundary(0, 1, {AMREX_D_DECL(Nghost, Nghost, Nghost)}, {AMREX_D_DECL(0, 0, 0)}, geom.periodicity());

    amrex::PrintToFile("test_AMReX_SumBoundary_additional.tmp") << "SUMBOUNDARY" << std::endl;
    for (amrex::MFIter mfi(TestMF); mfi.isValid(); ++mfi)
    {
        amrex::PrintToFile("test_AMReX_SumBoundary_additional.tmp").SetPrecision(20)
            << TestMF[mfi] << std::endl;
    }

    bool passed = (std::abs(TestMF.norm1(0, Nghost) - 7.8) < 1e-12);
    amrex::PrintToFile("test_AMReX_SumBoundary.tmp") << std::endl;
    amrex::PrintToFile("test_AMReX_SumBoundary.tmp") << passed << std::endl;
}

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(
        argc,
        argv,
        build_parm_parse,
        MPI_COMM_WORLD,
        overwrite_amrex_parser_defaults
    );
    const int vdim=3, degx=1, degy=1, degz=1;

    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_SumBoundary.tmp.0");
    if (ParallelDescriptor::MyProc() == 0) remove("test_AMReX_SumBoundary_additional.tmp.0");

    main_main<vdim,degx,degy,degz>();

    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_AMReX_SumBoundary.tmp.0", "test_AMReX_SumBoundary.output");
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_AMReX_SumBoundary_additional.tmp.0",
                    "test_AMReX_SumBoundary_additional.output");

    amrex::Finalize();
}

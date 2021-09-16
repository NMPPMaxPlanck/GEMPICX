#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <GEMPIC_assertion.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_vlasov_maxwell.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Utils;

// wave function
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real wave_function(amrex::Real x, amrex::Real y, amrex::Real z)
{
    amrex::Real val = 1.0;
    return val;
}

// wave function
AMREX_GPU_HOST_DEVICE AMREX_NO_INLINE amrex::Real funct_rho_phi(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t)
{
    amrex::Real val = 1.0;
    return val;
}

template<int vdim, int numspec, int degx, int degy, int degz>
void main_main ()
{
    //------------------------------------------------------------------------------
    // Initialize Function
    //std::array<amrex::Real,GEMPIC_SPACEDIM> k = {AMREX_D_DECL(0.5,0.5,0.5)};
    amrex::Real k = 0.5;

    //------------------------------------------------------------------------------
    // Initialize Infrastructure
    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(8,8,8)};

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    VM[0].push_back(0.0);
    VD[0].push_back(1.0);
    VW[0].push_back(1.0);
    if (vdim > 1) {
        VM[1].push_back(0.0);
        VD[1].push_back(1.0);
        VW[1].push_back(1.0);
    }
    if (vdim > 2) {
        VM[2].push_back(0.0);
        VD[2].push_back(1.0);
        VW[2].push_back(1.0);
    }

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("deposit_rho_ctest", n_cell, {1000}, 0, 2, 2, 2,
                    is_periodic, {4,4,4}, 0.02, {-1.0}, {1.0}, k, " ");
    VlMa.set_computed_params();
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    computational_domain infra;
    VlMa.initialize_infrastructure(&infra);

    //------------------------------------------------------------------------------
    // Initialize fields and particles
    maxwell_yee<vdim> mw_yee(VlMa, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0; // all particles are same species for now
    init_particles_cellwise<vdim, numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, species, wave_function);

    //------------------------------------------------------------------------------
    // initialize rho and phi, phi will solve the analytically exact solution, rho
    // will be overwritten in next paragraph
    mw_yee.template init_rho_phi<2>(funct_rho_phi, funct_rho_phi, VlMa.k_gpu, infra);

    mw_yee.phi.mult(-1.0);

    //------------------------------------------------------------------------------
    // Deposit charges:
    (mw_yee).rho.setVal(0.0, 0); // value and component
    (mw_yee).rho.FillBoundary(infra.geom.periodicity());

    for (int spec=0;spec<numspec;spec++) {
        (*(part_gr).mypc[spec]).Redistribute(); // assign particles to the tile they are in
        for (amrex::ParIter<vdim+1,0,0,0> pti(*(part_gr).mypc[spec], 0); pti.isValid(); ++pti) {
            amrex::Box validbox = pti.validbox();

            auto& particles = pti.GetArrayOfStructs();
            const long np  = pti.numParticles();

            amrex::Array4<amrex::Real> const& rhoarr = (mw_yee.rho)[pti].array();
            for (int pp=0;pp<np;pp++) {
                gempic_deposit_rho<amrex::Particle<vdim+1>,vdim, degx, degy, degz>(particles[pp], (part_gr).charge[spec], rhoarr, infra.plo, infra.dxi);
            }
        }
    }

    (mw_yee).rho.SumBoundary(0, 1, {(mw_yee).Nghost, (mw_yee).Nghost, (mw_yee).Nghost}, {0, 0, 0}, infra.geom.periodicity());
    (mw_yee).rho.FillBoundary(infra.geom.periodicity());

    //------------------------------------------------------------------------------
    // Compute difference and store in phi:
    (mw_yee.phi).minus(mw_yee.rho, 0, 1, 0);

    //std::ofstream ofs("test_deposit_rho.output", std::ofstream::out);
    AllPrintToFile("test_deposit_rho.tmp") << std::endl;
    amrex::Real error = gempic_norm(&(mw_yee.phi), infra, 2)*gempic_norm(&(mw_yee.phi), infra, 2);
    bool passed = true;
    gempic_assert_err(&passed, gempic_norm(&mw_yee.rho, infra, 2), error);
    AllPrintToFile("test_deposit_rho.tmp") << passed << std::endl;

    AllPrintToFile("test_deposit_rho_additional.tmp") << "Norm of error: " << gempic_norm(&(mw_yee.phi), infra, 2)*gempic_norm(&(mw_yee.phi), infra, 2) << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_deposit_rho.tmp.0");
    if (ParallelDescriptor::MyProc()==0) remove("test_deposit_rho_additional.tmp.0");


#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1, false>();
    main_main<2, 1, 1, 1, 1, false>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1, false>();
    main_main<3, 1, 1, 1, 1, false>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc()==0) std::rename("test_deposit_rho.tmp.0", "test_deposit_rho.output");
    if (ParallelDescriptor::MyProc()==0) std::rename("test_deposit_rho_additional.tmp.0", "test_deposit_rho_additional.output");


    amrex::Finalize();
}




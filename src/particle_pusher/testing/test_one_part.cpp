#include <tinyexpr.h>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Particles.H>

#include <GEMPIC_Config.H>
#include <GEMPIC_loop_preparation.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_vlasov_maxwell.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

template< int vdim, int numspec, int degx, int degy, int degz>
void main_main (bool ctest)
{
    // ------------------------------------------------------------------------------
    // ------------PARAMETERS--------------------------------------------------------

    // initialize parameters
    std::string sim_name = "One_Particle";
    std::array<int,GEMPIC_SPACEDIM> n_cell = {AMREX_D_DECL(4,4,4)};
    std::array<int, numspec> n_part_per_cell = {1};
    int n_steps = 1;
    int freq_x = 2;
    int freq_v = 2;
    int freq_slice = 1;
    std::array<int,GEMPIC_SPACEDIM> is_periodic = {AMREX_D_DECL(1,1,1)};
    int max_grid_size = 4;
    amrex::Real dt = 0.02;
    std::array<amrex::Real, numspec> charge = {-1.0};
    std::array<amrex::Real, numspec> mass = {1.0};
    //std::array<amrex::Real,GEMPIC_SPACEDIM> k = {AMREX_D_DECL(1.25,1.25,1.25)};
    amrex::Real k = 1.25;
    std::string WF = "1.0";
    std::string Bx = "0.0";
    std::string By = "0.0";
    std::string Bz = "1e-3 * cos(kvarx * x)";
    std::string phi = "4 * 0.5 * cos(0.5 * x)";
    std::string rho = "0.0";
    int propagator = 1;
    bool time_staggered = false;
    amrex::Real tolerance_particles = 1.e-10;

    std::array<std::vector<amrex::Real>, vdim> VM{};
    std::array<std::vector<amrex::Real>, vdim> VD{};
    std::array<std::vector<amrex::Real>, vdim> VW{};

    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VW[j].push_back(1.0);
    }
    VD[0].push_back(0.02/sqrt(2));
    VD[1].push_back(sqrt(12)*VD[0][0]);
    VD[2].push_back(VD[1][0]);

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params(sim_name, n_cell, n_part_per_cell, n_steps, freq_x, freq_v,
                    freq_slice, is_periodic, max_grid_size, dt, charge, mass, k,
                    WF, Bx, By, Bz, phi, 1, propagator, tolerance_particles);
    VlMa.set_computed_params();
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    // infrastructure
    infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    mw_yee.init_rho_phi(infra, VlMa);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    //------------------------------------------------------------------------------
    // initialize particles:
    int species = 0;
    for(amrex::MFIter mfi=(*(part_gr).mypc[species]).MakeMFIter(0); mfi.isValid(); ++mfi) {
        if(mfi.index() == 0) {
            using ParticleType = amrex::Particle<vdim+1, 0>; // Particle template
            amrex::ParticleTile<vdim+1, 0, 0, 0>& particles = (*(part_gr).mypc[species]).GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            std::array<amrex::Real,vdim> velocity;
            for (int comp = 0; comp < vdim; comp++) {
                velocity[comp] = 0.1;
            }
            (part_gr).add_particle({AMREX_D_DECL(2.512, 2.2, 2.3)}, velocity, 1.0, particles);
        }
    }

    //------------------------------------------------------------------------------
    // solve:
    diagnostics<vdim, numspec, degx, degy, degz> diagn(mw_yee.nsteps, freq_x, freq_v, freq_slice, sim_name);
    loop_preparation<vdim,numspec>(VlMa, infra, &mw_yee, &part_gr, &diagn, time_staggered);
    std::ofstream ofs("PIC.output", std::ofstream::out);
    amrex::Print(ofs) << endl;
    switch (propagator) {
    case 0:
        time_loop_boris_fd<vdim,numspec,degx,degy, degz>(infra, &mw_yee, &part_gr, &diagn, false, "test_one_part.tmp", &ofs);
        break;
    case 1:
        time_loop_hs_fem<vdim,numspec,degx,degy, degz>(infra, &mw_yee, &part_gr, &diagn, false, "test_one_part.tmp", &ofs);
        break;
    default:
        break;
    }

    AllPrintToFile("test_one_part.tmp") << std::endl;
    AllPrintToFile("test_one_part.tmp") << "Jx" << std::endl;
    for (amrex::MFIter mfi(*(mw_yee).J_Array[0]); mfi.isValid(); ++mfi ) {
        AllPrintToFile("test_one_part.tmp") << (*(mw_yee).J_Array[0])[mfi] << std::endl;
    }
    AllPrintToFile("test_one_part.tmp") << "Jy" << std::endl;
    for (amrex::MFIter mfi(*(mw_yee).J_Array[1]); mfi.isValid(); ++mfi ) {
        AllPrintToFile("test_one_part.tmp") << (*(mw_yee).J_Array[1])[mfi] << std::endl;
    }

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1>(argc==1);
    main_main<2, 1, 1, 1, 1>(argc==1);
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1>(argc==1);
    main_main<3, 1, 1, 1, 1>(argc==1);
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>(argc==1);
#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_one_part.tmp.0", "test_one_part.output");
    amrex::Finalize();
}




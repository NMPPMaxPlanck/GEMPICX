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
#include <GEMPIC_particle_positions.H>
#include <GEMPIC_sampler.H>
#include <GEMPIC_time_loop_boris_fd.H>
#include <GEMPIC_time_loop_hs_fem.H>
#include <GEMPIC_time_loop_hsall_fem.H>
#include <GEMPIC_vlasov_maxwell.H>
#include <GEMPIC_particle_groups.H>

using namespace std;
using namespace amrex;
using namespace Gempic;

using namespace Diagnostics_Output;
using namespace Field_solvers;
using namespace Particles;
using namespace Sampling;
using namespace Time_Loop;
using namespace Vlasov_Maxwell;

template<int vdim, int numspec, int degx, int degy, int degz, int degmw, int strang_order>
void main_main ()
{
    bool ctest = true;
    vlasov_maxwell<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params();
    VlMa.propagator = 2;
    VlMa.set_prop_related();
    if (int(vdim/2.5)*2+1 < 3) {
        VlMa.Bx = VlMa.Bz;
        VlMa.Bz = "0.0";
    }
    std::array<std::string, int(vdim/2.5)*2+1> fields_B;
    fields_B[0] = VlMa.Bx;
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[1] = VlMa.By;
    }
    if (int(vdim/2.5)*2+1 > 1) {
        fields_B[2] = VlMa.Bz;
    }
    if (GEMPIC_SPACEDIM==1 & vdim==1) {
        // For 1D1V change parameters to make a Landau damping
        VlMa.sim_name = "Landau";
        VlMa.n_part_per_cell = {10000};
        VlMa.k = {0.5,0.5,0.5};
        VlMa.WF = "1.0 + 0.5 * cos(kvarx * x)";
        VlMa.Bz = "0.0";
    }
    VlMa.n_steps = 10;
    VlMa.set_computed_params();

    std::array<std::vector<amrex::Real>, vdim> VM{}, VD{}, VW{};
    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VW[j].push_back(1.0);
    }
    VD[0].push_back(0.02/sqrt(2));
    VD[1].push_back(sqrt(12)*VD[0][0]);
    VD[2].push_back(VD[1][0]);
    VlMa.VM = VM;
    VlMa.VD = VD;
    VlMa.VW = VW;

    // ------------------------------------------------------------------------------
    // ------------INITIALIZE GEMPIC-STRUCTURES--------------------------------------

    // infrastructure
    infrastructure infra;
    VlMa.initialize_infrastructure(&infra);

    // maxwell_yee
    maxwell_yee<vdim> mw_yee(VlMa, infra);
    std::array<std::string, 2> fields = {VlMa.rho, VlMa.phi};
    mw_yee.template init_rho_phi<degmw>(fields, VlMa.k, infra);

    // particles
    particle_groups<vdim, numspec> part_gr(VlMa, infra);

    amrex::Real vol = (infra.geom.ProbHi(0)-infra.geom.ProbLo(0))*(infra.geom.ProbHi(1)-infra.geom.ProbLo(1))*(infra.geom.ProbHi(2)-infra.geom.ProbLo(2));
    diagnostics<vdim, numspec,degx,degy,degz,degmw> diagn(mw_yee.nsteps, VlMa.freq_x, VlMa.freq_v, VlMa.freq_slice, VlMa.sim_name, vol);

    //------------------------------------------------------------------------------
    // initialize particles & loop preparation:
    init_particles_full_domain<vdim,numspec>(infra, part_gr, VlMa, VlMa.VM, VlMa.VD, VlMa.VW, 0);
    loop_preparation<vdim, numspec, degx, degy, degz, degmw, true>(VlMa, infra, &mw_yee, &part_gr, &diagn, VlMa.time_staggered, fields_B);


    //------------------------------------------------------------------------------
    // timeloop
    time_loop_hsall_fem<vdim, numspec, degx, degy, degz, degmw, true>(infra, &mw_yee, &part_gr, &diagn, ctest, "test_vlasov_maxwell_hs_all", strang_order);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    if (ParallelDescriptor::MyProc()==0) remove("test_vlasov_maxwell_hs_all.tmp.0");

#if (GEMPIC_SPACEDIM == 1)

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "0 2.82775e-05 1.74142e-05 1.72514e-05 0 0 5e-07 0.315393 1.43412 4.90194 4.95606" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "1 2.8274e-05 1.74408e-05 1.72766e-05 2.69379e-12 1.35661e-11 4.99702e-07 0.31539 1.43414 4.90192 4.95602" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "2 2.82582e-05 1.74631e-05 1.7295e-05 4.2604e-11 2.139e-10 4.98979e-07 0.315388 1.4342 4.90191 4.95599" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "3 2.82305e-05 1.74822e-05 1.73069e-05 2.11588e-10 1.05717e-09 4.98324e-07 0.315388 1.43431 4.90189 4.95597" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "4 2.81909e-05 1.74959e-05 1.73127e-05 6.51505e-10 3.2308e-09 4.98481e-07 0.315389 1.43446 4.90188 4.95595" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "5 2.81396e-05 1.75035e-05 1.73101e-05 1.53975e-09 7.55231e-09 5.00336e-07 0.315391 1.43465 4.90187 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "6 2.80766e-05 1.75027e-05 1.72982e-05 3.07151e-09 1.48449e-08 5.04789e-07 0.315395 1.43488 4.90186 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "7 2.80016e-05 1.74913e-05 1.72747e-05 5.44034e-09 2.58122e-08 5.12622e-07 0.315401 1.43515 4.90186 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "8 2.79146e-05 1.74662e-05 1.72412e-05 8.81944e-09 4.09194e-08 5.24376e-07 0.315408 1.43546 4.90187 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "9 2.78161e-05 1.74281e-05 1.71962e-05 1.33441e-08 6.02953e-08 5.40262e-07 0.315417 1.43581 4.90189 4.95593" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "0 2.82775e-05 1.74142e-05 1.72514e-05 0 0 5e-07 0.315393 1.43412 4.90194 4.95606" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "1 2.8274e-05 1.74408e-05 1.72766e-05 2.69379e-12 1.35661e-11 4.99702e-07 0.31539 1.43414 4.90192 4.95602" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "2 2.82582e-05 1.74631e-05 1.7295e-05 4.2604e-11 2.139e-10 4.98979e-07 0.315388 1.4342 4.90191 4.95599" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "3 2.82305e-05 1.74822e-05 1.73069e-05 2.11588e-10 1.05717e-09 4.98324e-07 0.315388 1.43431 4.90189 4.95597" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "4 2.81909e-05 1.74959e-05 1.73127e-05 6.51505e-10 3.2308e-09 4.98481e-07 0.315389 1.43446 4.90188 4.95595" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "5 2.81396e-05 1.75035e-05 1.73101e-05 1.53975e-09 7.55231e-09 5.00336e-07 0.315391 1.43465 4.90187 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "6 2.80766e-05 1.75027e-05 1.72982e-05 3.07151e-09 1.48449e-08 5.04789e-07 0.315395 1.43488 4.90186 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "7 2.80016e-05 1.74913e-05 1.72747e-05 5.44034e-09 2.58122e-08 5.12622e-07 0.315401 1.43515 4.90186 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "8 2.79146e-05 1.74662e-05 1.72412e-05 8.81944e-09 4.09194e-08 5.24376e-07 0.315408 1.43546 4.90187 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs_all.tmp") << "9 2.78161e-05 1.74281e-05 1.71962e-05 1.33441e-08 6.02953e-08 5.40262e-07 0.315417 1.43581 4.90189 4.95593" << std::endl;

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=3
    main_main<3, 1, 1, 1, 1, 2, 2>();
    main_main<3, 1, 1, 1, 1, 2, 4>();
#endif

    if (ParallelDescriptor::MyProc()==0) std::rename("test_vlasov_maxwell_hs_all.tmp.0", "test_vlasov_maxwell_hs_all.output");
    amrex::Finalize();
}




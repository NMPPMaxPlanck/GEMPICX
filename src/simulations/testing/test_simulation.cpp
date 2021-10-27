

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <GEMPIC_Config.H>

#include <GEMPIC_vlasov_maxwell.H>

#define VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO 0
#define VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION 1

AMREX_GPU_HOST_DEVICE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real t, int funcSelect)
{
    switch(funcSelect){
    case VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION :
      return 1.0 ;//+ 0.5 * std::cos(0.5 * x);
    case VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO :
      return 0.0 ;
    }
    return 0.0;
}


AMREX_GPU_HOST_DEVICE amrex::Real zero(amrex::Real , amrex::Real , amrex::Real , amrex::Real )
{
    amrex::Real val = 0.0;
    return val;
}

template <int degx, int degy, int degz, int degmw, int vdim>
void vlasov_maxwell_run(std::string test_name, int propagator)
{
    std::string test_name_tmp = test_name + ".tmp.0";
    std::string test_name_end = test_name + ".output";

    if (amrex::ParallelDescriptor::MyProc()==0) remove(test_name_tmp.c_str());

    amrex::GpuArray<int, 2> funcSelectRho;
    funcSelectRho[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectRho[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    amrex::GpuArray<int, 3> funcSelectB;
    funcSelectB[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[2] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;

    // Output for GEMPIC_SPACEDIM=3
    //vlasov_maxwell_test<3, 1, 6, 5, 4, 4, 2, true>(3, test_name);
    vlasov_maxwell_simulation<3, 1, degx, degy, degz, degmw> sim;
    sim.params.init_Nghost(degx,degy,degz);
    sim.params.set_params(test_name, {12,8,8});
    sim.params.propagator = propagator;
    sim.params.set_prop_related();
    sim.params.n_steps = 5;
    sim.params.n_part_per_cell = {2000};
    sim.params.set_computed_params();
    sim.ctest = true;

    std::array<std::vector<amrex::Real>, vdim> VM{}, VD{}, VW{};
    for (int j=0; j<vdim; j++) {
        VM[j].push_back(0.0);
        VW[j].push_back(1.0);
    }
    VD[0].push_back(0.02/sqrt(2));
    VD[1].push_back(sqrt(12)*VD[0][0]);
    VD[2].push_back(VD[1][0]);
    sim.params.VM = VM;
    sim.params.VD = VD;
    sim.params.VW = VW;
    sim.initialize_gempic_structures(funcSelectRho, funcSelectB);
    sim.run_time_loop();
    if (amrex::ParallelDescriptor::MyProc()==0) std::rename(test_name_tmp.c_str(), test_name_end.c_str());
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    std::string test_name = "test_vlasov_maxwell_hs_zigzag_C2";

    vlasov_maxwell_run<6,5,4,4,3> (test_name, 3);

    amrex::Finalize();
}



#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_vlasov_maxwell.H>

#define VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO 0
#define VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION 1
#define VLASOV_MAXWELL_HS_ZIGZAG_C2_BZ 2

AMREX_GPU_HOST_DEVICE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z,
                                                      amrex::Real t, int funcSelect)
{
    switch (funcSelect)
    {
        case VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION:
            return 1.0;  //+ 0.5 * std::cos(0.5 * x);
        case VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO:
            return 0.0;
        case VLASOV_MAXWELL_HS_ZIGZAG_C2_BZ:
            return 1e-4 * cos(t * x);  // using t here as kx
    }
    return 0.0;
}

AMREX_GPU_HOST_DEVICE amrex::Real zero(amrex::Real, amrex::Real, amrex::Real, amrex::Real)
{
    amrex::Real val = 0.0;
    return val;
}

template <int degx, int degy, int degz, int degmw, int vdim, bool electromagnetic, bool output>
void vlasov_maxwell_run(std::string test_name, int propagator)
{
    std::string test_name_tmp = test_name + ".tmp.0";
    std::string test_name_end = test_name + ".output";

    if (amrex::ParallelDescriptor::MyProc() == 0) remove(test_name_tmp.c_str());

    amrex::GpuArray<int, 2> funcSelectRho;
    funcSelectRho[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectRho[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    amrex::GpuArray<int, 3> funcSelectB;
    funcSelectB[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[2] = VLASOV_MAXWELL_HS_ZIGZAG_C2_BZ;

    vlasov_maxwell_simulation<3, 1, degx, degy, degz, degmw, electromagnetic, output> sim;
    sim.params.init_Nghost(degx, degy, degz);

    const int nSteps = 5;
    amrex::IntVect nCell = {20, 20, 20};
    std::array<int, 1> nPartPerCell = {2000};
    const amrex::Real dt = 0.01;
    sim.params.set_params_Weibel(test_name, propagator, nSteps, nCell, nPartPerCell, dt);
    sim.ctest = true;
    sim.initialize_gempic_structures(funcSelectRho, funcSelectB);
    sim.run_time_loop();
    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename(test_name_tmp.c_str(), test_name_end.c_str());
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    std::string test_name = "test_vlasov_maxwell_hs_zigzag_C2";

    vlasov_maxwell_run<2, 2, 2, 2, 3, true, false>(test_name, 3);

    amrex::Finalize();
}

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_vlasov_maxwell.H>

#define VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO 0
#define VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION 1

AMREX_GPU_HOST_DEVICE amrex::Real function_to_project(amrex::Real x, amrex::Real y, amrex::Real z,
                                                      amrex::Real t, int funcSelect)
{
    switch (funcSelect)
    {
        case VLASOV_MAXWELL_HS_ZIGZAH_C2_WAVE_FUNCTION:
            return 1.0;  //+ 0.5 * std::cos(0.5 * x);
        case VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO:
            return 0.0;
    }
    return 0.0;
}

AMREX_GPU_HOST_DEVICE amrex::Real zero(amrex::Real, amrex::Real, amrex::Real, amrex::Real)
{
    amrex::Real val = 0.0;
    return val;
}

template <int degx, int degy, int degz, int degmw, int vdim, int ndata = 1>
void vlasov_maxwell_ctest(std::string test_name, int propagator)
{
    std::string test_name_tmp = test_name + ".output.0";
    std::string test_name_end = test_name + ".output";

    // if (amrex::ParallelDescriptor::MyProc()==0) remove(test_name_tmp.c_str());

    amrex::GpuArray<int, 2> funcSelectRho;
    funcSelectRho[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectRho[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    amrex::GpuArray<int, 3> funcSelectB;
    funcSelectB[0] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[1] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;
    funcSelectB[2] = VLASOV_MAXWELL_HS_ZIGZAH_C2_ZERO;

    const int numspec = 1;
    const bool electromagnetic = true;
    const bool output = true;
    vlasov_maxwell_simulation<vdim, numspec, degx, degy, degz, degmw, electromagnetic, output,
                              ndata>
        sim;
    sim.params.init_Nghost(degx, degy, degz);

    const int nSteps = 5;
    amrex::IntVect nCell{AMREX_D_DECL(12, 8, 8)};
    amrex::Array<int, numspec> nPartPerCell = {100};
    amrex::Real dt = 0.02;
    if (propagator == 4) dt = 0.005;
    sim.params.set_params_Weibel(test_name, propagator, nSteps, nCell, nPartPerCell, dt);

    amrex::Print() << sim.params.max_grid_size[0] << " grid size " << sim.params.n_cell[0]
                   << std::endl;

    sim.ctest = true;

    // sim.initialize_gempic_structures(funcSelectRho, funcSelectB);
    sim.initialize_gempic_structures();  // uses AMReX parser for initial functions
    sim.run_time_loop();
    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename(test_name_tmp.c_str(), test_name_end.c_str());
}

/*------------------------------------------------------------------------------
 Test 3D Maxwell Yee Solver (finite differences) on periodic grid

  We test the solution
  E(x,t) =  \begin{pmatrix} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\
                          -2\cos(x_1+x_2+x_3 - \sqrt(3) t) \\
                            \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
  B(x,t) = \begin{pmatrix} \sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \\
                            0 \\
                            -\sqrt{3} \cos(x_1+x_2+x_3 - \sqrt{3} t) \end{pmatrix}
------------------------------------------------------------------------------*/

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_maxwell_yee.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_positions.H>

using namespace std;
using namespace amrex;
using namespace Gempic;
using namespace Field_solvers;
using namespace Diagnostics_Output;

#define AMPERE_FARADAY_ZERO 0
#define AMPERE_FARADAY_E0_SIN 1
#define AMPERE_FARADAY_B1_SIN 2
#define AMPERE_FARADAY_OMEGA 3

AMREX_GPU_HOST_DEVICE amrex::Real function_to_project(amrex::Real, amrex::Real, amrex::Real z,
                                                      amrex::Real t, int funcSelect)
{
    amrex::Real omega = 1836.15267596 * 0.005;
    switch (funcSelect)
    {
        case AMPERE_FARADAY_E0_SIN:
            return 1.0 * std::sin(z - std::sqrt(omega) * t);
        case AMPERE_FARADAY_B1_SIN:
            return 1.0 / std::sqrt(omega) * std::sin(z - std::sqrt(omega) * t);
        case AMPERE_FARADAY_OMEGA:
            return omega;
        case AMPERE_FARADAY_ZERO:
            return 0.0;
    }
    return 0.0;
}

template <int vdim, int numspec, int degx, int degy, int degz>
void main_main()
{  //------------------------------------------------------------------------------
    // Analytical solutions - you can change them here

    amrex::GpuArray<std::string, vdim> fields_E;
    amrex::GpuArray<std::string, int(vdim / 2.5) * 2 + 1> fields_B;

    //------------------------------------------------------------------------------
    // Parameters that could be relevant for you:

    amrex::IntVect n_cell = {AMREX_D_DECL(
        16, 16,
        16)};  // spatial discretization: number of cells in each direction, currently: 128x128x128
    int numstep = 2;        // number of timesteps
    amrex::Real dt = 0.01;  // size of timesteps

    //------------------------------------------------------------------------------
    // Some setting up of data structures -- you can ignore this

    const int degree = 2;
    amrex::GpuArray<Real, vdim + int(vdim / 2.5) * 2 + 1> E_B_error;  // array for storing errors

    amrex::IntVect is_periodic = {AMREX_D_DECL(1, 1, 1)};
    gempic_parameters<vdim, numspec> VlMa;
    VlMa.init_Nghost(degx, degy, degz);
    VlMa.set_params("test_ampere_faraday", n_cell, {1}, numstep, 100000, 100000, 100000,
                    is_periodic, {AMREX_D_DECL(4, 4, 4)}, dt, {-1.0}, {1.0}, 1.0);

    CompDom::computational_domain infra;
    infra.initialize_computational_domain(VlMa.n_cell, VlMa.max_grid_size, VlMa.is_periodic,
                                          VlMa.real_box);

    maxwell_yee<vdim> mw_yee(infra, VlMa.dt, VlMa.n_steps, VlMa.Nghost);

    //------------------------------------------------------------------------------
    // Initialization of E and B: this is done via a projection-operator

    // mw_yee.template init_E_B<degree>(fields_E, fields_B, VlMa.k, infra);
    amrex::GpuArray<int, vdim> funcSelectE;
    funcSelectE[0] = AMPERE_FARADAY_E0_SIN;
    funcSelectE[1] = AMPERE_FARADAY_ZERO;
    funcSelectE[2] = AMPERE_FARADAY_ZERO;
    mw_yee.template initE<degree>(infra, funcSelectE);
    amrex::GpuArray<int, vdim> funcSelectB;
    funcSelectB[0] = AMPERE_FARADAY_ZERO;
    funcSelectB[1] = AMPERE_FARADAY_B1_SIN;
    funcSelectB[2] = AMPERE_FARADAY_ZERO;
    mw_yee.template initB<degree>(infra, funcSelectB);
    for (int comp = 0; comp < 3; comp++)
    {
        mw_yee.template projection<2>(0.0, infra, {false, false, false}, *mw_yee.E_Index[comp],
                                      *mw_yee.Alfven_Tensor[comp], AMPERE_FARADAY_OMEGA);
    }

    //------------------------------------------------------------------------------
    // This generates error output: comparing current E and B to the analytical solution
    // This output will be stored in a file test_ampere_faraday.output -- you can ignore the Code

    std::cout << "step: " << 0 << std::endl;
    E_B_error = mw_yee.template computeError<degree>(true, infra, funcSelectE, funcSelectB);
    PrintToFile("test_ampere_faraday.output") << endl;
    PrintToFile("test_ampere_faraday.output") << "Maxwell" << endl;
    PrintToFile("test_ampere_faraday.output") << "step " << 0 << endl;
    PrintToFile("test_ampere_faraday.output").SetPrecision(5)
        << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1]
        << " |Ez error: " << E_B_error[2] << std::endl;
    amrex::PrintToFile("test_ampere_faraday.output").SetPrecision(5)
        << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim + 1]
        << " |Bz error: " << E_B_error[vdim + 2] << std::endl;

    particle_groups<vdim, 1> part_gr(VlMa.charge, VlMa.mass, infra);
    //------------------------------------------------------------------------------
    // time loop
    // Gempic_WritePlotFile(&part_gr, &mw_yee, &infra, "Alfven_Test", 0);
    for (int n = 1; n <= mw_yee.nsteps; n++)
    {
        std::cout << "step: " << n << std::endl;

        //------------------------------------------------------------------------------
        // Ampere

        mw_yee.template hodge_full<degree>(infra, mw_yee.B_Array, mw_yee.HB_Array,
                                           false);  // we apply the hodge to B (you can ignore this,
                                                    // when degree=2, this is the identity)
        mw_yee.advance_E(infra, VlMa.dt, true, false, mw_yee.HB_Array, mw_yee.Alfven_Tensor,
                         mw_yee.E_Array);  // we apply curl to B and set E = E+dt*curl(B)
        // you can find the function advance_E in src/field_solvers/GEMPIC_maxwell_yee.H line 480
        // (you can ignore lines 513-548, they are for other cases)

        //------------------------------------------------------------------------------
        // Faraday
        mw_yee.template hodge_full<degree>(infra, mw_yee.E_Array, mw_yee.HE_Array,
                                           true);  // we apply the hodge to E (you can ignore this,
                                                   // when degree=2, this is the identity)
        mw_yee.advance_B(infra, VlMa.dt, mw_yee.HE_Array,
                         mw_yee.B_Array);  // we apply curl to E and set B = B+dt*curl(E)
        // you can find the function advance_B in src/field_solvers/GEMPIC_maxwell_yee.H line 574
        // (you can ignore lines 605-637, they are for other cases)

        //------------------------------------------------------------------------------
        // This generates error output once more: comparing current E and B to the analytical
        // solution -- you can ignore the code
        mw_yee.advance_time();
        E_B_error = mw_yee.template computeError<degree>(true, infra, funcSelectE, funcSelectB);
        PrintToFile("test_ampere_faraday.output") << "step " << n << endl;
        PrintToFile("test_ampere_faraday.output").SetPrecision(5)
            << "Ex error: " << E_B_error[0] << " |Ey error: " << E_B_error[1]
            << " |Ez error: " << E_B_error[2] << std::endl;
        amrex::PrintToFile("test_ampere_faraday.output").SetPrecision(5)
            << "Bx error: " << E_B_error[vdim] << " |By error: " << E_B_error[vdim + 1]
            << " |Bz error: " << E_B_error[vdim + 2] << std::endl;

        // Gempic_WritePlotFile(&part_gr, &mw_yee, &infra, "Alfven_Test", n);
    }
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

#if (GEMPIC_SPACEDIM == 1)
    main_main<1, 1, 1, 1, 1>();
    main_main<2, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 2)
    main_main<2, 1, 1, 1, 1>();
    main_main<3, 1, 1, 1, 1>();
#elif (GEMPIC_SPACEDIM == 3)
    main_main<3, 1, 1, 1, 1>();
#endif
    if (ParallelDescriptor::MyProc() == 0)
        std::rename("test_ampere_faraday.output.0", "test_ampere_faraday.output");
    amrex::Finalize();
}

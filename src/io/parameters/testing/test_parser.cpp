#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_assertion.H>
#include <GEMPIC_parameters.H>

template <unsigned int numspec>
void test_Weibel()
{
    gempic_parameters<numspec> WeibelParams;
    WeibelParams.set_params_Weibel();
    print_param(WeibelParams);
}

template <unsigned int numspec>
void test_read()
{
    gempic_parameters<numspec> params;
    params.read_pp_params();
    print_param(params);
}

template <unsigned int numspec>
void print_param(gempic_parameters<numspec> params)
{
    // Print data members in gempic_parameters
    amrex::PrintToFile("test_parser.output", 0) << params.sim_name << "\n";
    amrex::Print() << params.sim_name << "\n";
    amrex::PrintToFile("test_parser.output", 0) << params.real_box << "\n";
    amrex::Print() << params.real_box << "\n";
    amrex::PrintToFile("test_parser.output", 0) << params.is_periodic << "\n";
    amrex::Print() << params.is_periodic << "\n";
    amrex::PrintToFile("test_parser.output", 0) << params.n_cell << "\n";
    amrex::Print() << params.n_cell << "\n";
    for (int i = 0; i < GEMPIC_SPACEDIM; i++)
    {
        amrex::PrintToFile("test_parser.output", 0) << params.k[i] << " ";
        amrex::Print() << params.k[i] << " ";
    }
    amrex::PrintToFile("test_parser.output", 0) << "\n";
    amrex::Print() << "\n";
    amrex::PrintToFile("test_parser.output", 0) << params.n_steps << "\n";
    amrex::Print() << params.n_steps << "\n";
    amrex::PrintToFile("test_parser.output", 0) << params.dt << "\n";
    amrex::Print() << params.dt << "\n";
    amrex::PrintToFile("test_parser.output", 0) << params.propagator << "\n";
    amrex::Print() << params.propagator << "\n";

    auto vdim{params.vThermal[0][0].size()};
    // particle initialisation
    for (int i = 0; i < numspec; i++)
    {
        amrex::PrintToFile("test_parser.output", 0) << params.num_gaussians[i] << "\n";
        amrex::Print() << params.num_gaussians[i] << "\n";
        amrex::PrintToFile("test_parser.output", 0) << params.density[i] << "\n";
        amrex::Print() << params.density[i] << "\n";
        for (int j = 0; j < params.num_gaussians[i]; j++)
        {
            amrex::PrintToFile("test_parser.output", 0)
                << "species " << i << " gaussian " << j << "\n";
            amrex::Print() << "species " << i << " gaussian " << j << "\n";
            for (int k = 0; k < vdim; k++)
            {
                amrex::PrintToFile("test_parser.output", 0) << params.meanVelocity[i][j][k] << " ";
                amrex::Print() << params.meanVelocity[i][j][k] << " ";
            }
            amrex::PrintToFile("test_parser.output", 0) << "\n";
            amrex::Print() << "\n";
            for (int k = 0; k < vdim; k++)
            {
                amrex::PrintToFile("test_parser.output", 0) << params.vThermal[i][j][k] << " ";
                amrex::Print() << params.vThermal[i][j][k] << " ";
            }
            amrex::PrintToFile("test_parser.output", 0) << "\n";
            amrex::Print() << "\n";
            amrex::PrintToFile("test_parser.output", 0) << params.vWeight[i][j] << "\n";
            amrex::Print() << params.vWeight[i][j] << "\n";
        }
    }
    // test parser
    amrex::Real xlo = params.real_box.lo()[xDir];
    amrex::Real xhi = params.real_box.hi()[xDir];
    amrex::Real dx = (xhi - xlo) / params.n_cell[xDir];
    amrex::Real ylo = params.real_box.lo()[yDir];
    amrex::Real zlo = params.real_box.lo()[zDir];
    amrex::Real x, y, z, t = 1.0;
    amrex::Real maxerr = 0, maxloc = 0;
    for (int i = 0; i < params.n_cell[xDir]; i++)
    {
        x = xlo + i * dx;
        for (int j = 0; j < params.n_cell[yDir]; j++)
        {
            y = ylo + j * dx;
            for (int k = 0; k < params.n_cell[zDir]; k++)
            {
                z = zlo + k * dx;
                if (params.sim_name == "test_parser")
                {
                    maxloc = std::abs(params.densityEval[0](AMREX_D_DECL(x, y, z), t) -
                                      (1.0 + cos(params.k[xDir] * x) + sin(params.k[yDir] * y) +
                                       cos(2 * params.k[zDir] * z)));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.BxEval(AMREX_D_DECL(x, y, z), t) -
                                      sin(params.k[xDir] * x + params.k[yDir] * y + params.k[zDir] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.ByEval(AMREX_D_DECL(x, y, z), t) -
                                      cos(params.k[xDir] * x + params.k[yDir] * y + params.k[zDir] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.BzEval(AMREX_D_DECL(x, y, z), t) - 1e-3 * cos(params.k[xDir] * x));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.ExEval(AMREX_D_DECL(x, y, z), t) -
                                      sin(params.k[xDir] * x + params.k[yDir] * y + params.k[zDir] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.EyEval(AMREX_D_DECL(x, y, z), t) -
                                      cos(params.k[xDir] * x + params.k[yDir] * y + params.k[zDir] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.EzEval(AMREX_D_DECL(x, y, z), t) - 1e-3 * cos(params.k[xDir] * x));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.phiEval(AMREX_D_DECL(x, y, z), t) - 4 * 0.5 * cos(0.5 * x));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.rhoEval(AMREX_D_DECL(x, y, z), t) - (1 + 0.1 * cos(0.5 * x)));
                    maxerr = std::max(maxerr, maxloc);
                }
                else if (params.sim_name == "Weibel")
                {
                    amrex::Real t0 = 0.0;
                    maxloc = std::abs(params.densityEval[0](AMREX_D_DECL(x, y, z), t0) - 1.0);
                    maxerr = std::max(maxerr, maxloc);
                }
            }
        }
    }
    amrex::PrintToFile("test_parser.output", 0) << "parser error \n";
    amrex::PrintToFile("test_parser.output", 0) << maxerr << "\n";
}

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(argc, argv, build_parm_parse, MPI_COMM_WORLD,
                      overwrite_amrex_parser_defaults);

    {
        // Print empty line
        amrex::PrintToFile("test_parser.output", 0) << "\n";

        const unsigned int numSpecWeibel = 1;
        const unsigned int numSpecRead = 2;
        test_Weibel<numSpecWeibel>();
        amrex::Print() << "test_Weibel completed\n";
        test_read<numSpecRead>();
        amrex::Print() << "test_read completed\n";

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_parser.output.0", "test_parser.output");
        }
    }

    amrex::Finalize();
}

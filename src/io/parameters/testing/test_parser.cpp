#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_assertion.H>
#include <GEMPIC_parameters.H>

template <int vdim, int numspec>
void test_Weibel()
{
    gempic_parameters<vdim, numspec> WeibelParams;
    WeibelParams.set_params_Weibel();
    print_param(WeibelParams);
}

template <int vdim, int numspec>
void test_read()
{
    gempic_parameters<vdim, numspec> params;
    params.read_pp_params();
    print_param(params);
}

template <int vdim, int numspec>
void print_param(gempic_parameters<vdim, numspec> params)
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
    amrex::Real xlo = params.real_box.lo()[0];
    amrex::Real xhi = params.real_box.hi()[0];
    amrex::Real dx = (xhi - xlo) / params.n_cell[0];
    amrex::Real ylo = params.real_box.lo()[1];
    amrex::Real zlo = params.real_box.lo()[2];
    amrex::Real x, y, z, t = 1.0;
    amrex::Real maxerr = 0, maxloc = 0;
    for (int i = 0; i < params.n_cell[0]; i++)
    {
        x = xlo + i * dx;
        for (int j = 0; j < params.n_cell[1]; j++)
        {
            y = ylo + j * dx;
            for (int k = 0; k < params.n_cell[2]; k++)
            {
                z = zlo + k * dx;
                if (params.sim_name == "test_parser")
                {
                    maxloc = std::abs(params.densityEval[0](x, y, z, t) -
                                      (1.0 + cos(params.k[0] * x) + sin(params.k[1] * y) +
                                       cos(2 * params.k[2] * z)));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.BxEval(x, y, z, t) -
                                      sin(params.k[0] * x + params.k[1] * y + params.k[2] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.ByEval(x, y, z, t) -
                                      cos(params.k[0] * x + params.k[1] * y + params.k[2] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.BzEval(x, y, z, t) - 1e-3 * cos(params.k[0] * x));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.ExEval(x, y, z, t) -
                                      sin(params.k[0] * x + params.k[1] * y + params.k[2] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.EyEval(x, y, z, t) -
                                      cos(params.k[0] * x + params.k[1] * y + params.k[2] * z - t));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.EzEval(x, y, z, t) - 1e-3 * cos(params.k[0] * x));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.phiEval(x, y, z, t) - 4 * 0.5 * cos(0.5 * x));
                    maxerr = std::max(maxerr, maxloc);
                    maxloc = std::abs(params.rhoEval(x, y, z, t) - (1 + 0.1 * cos(0.5 * x)));
                    maxerr = std::max(maxerr, maxloc);
                }
                else if (params.sim_name == "Weibel")
                {
                    amrex::Real t0 = 0.0;
                    maxloc = std::abs(params.densityEval[0](x, y, z, t0) - 1.0);
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

        const int vdim = 3;
        const int numSpecWeibel = 1;
        const int numSpecRead = 2;
        test_Weibel<vdim, numSpecWeibel>();
        amrex::Print() << "test_Weibel completed\n";
        test_read<vdim, numSpecRead>();
        amrex::Print() << "test_read completed\n";

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_parser.output.0", "test_parser.output");
        }
    }

    amrex::Finalize();
}

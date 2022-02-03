//#include <AMReX.H>
#include <AMReX_Array.H>
//#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
//#include <AMReX_Print.H>

#include <GEMPIC_assertion.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_parameters.H>

template <int vdim, int numspec>
void test_Weibel()
{
    gempic_parameters<vdim, numspec> params;
    params.set_params_Weibel();
}

template <int vdim, int numspec>
void test_read()
{
    gempic_parameters<vdim, numspec> params;
    params.read_pp_params();

    // Print empty line
    amrex::AllPrintToFile("test_parser.tmp") << "\n";

    // Print data members in gempic_parameters
    amrex::AllPrintToFile("test_parser.tmp") << params.sim_name << "\n";
    amrex::AllPrintToFile("test_parser.tmp") << params.real_box << "\n";
    amrex::AllPrintToFile("test_parser.tmp") << params.is_periodic << "\n";
    amrex::AllPrintToFile("test_parser.tmp") << params.n_cell << "\n";
    for (int i = 0; i < GEMPIC_SPACEDIM; i++)
        amrex::AllPrintToFile("test_parser.tmp") << params.k[i] << "\n";
    amrex::AllPrintToFile("test_parser.tmp") << params.n_steps << "\n";
    amrex::AllPrintToFile("test_parser.tmp") << params.dt << "\n";
    amrex::AllPrintToFile("test_parser.tmp") << params.propagator << "\n";

    // particle initialisation
    for (int i = 0; i < numspec; i++)
    {
        for (int j = 0; j < params.num_gaussians[i]; j++)
        {
            amrex::AllPrintToFile("test_parser.tmp") << "species " << i << " gaussian " << j << "\n";
            for (int k = 0; k < vdim; k++)
            {
                amrex::AllPrintToFile("test_parser.tmp") << params.VM[i][j][k] << "\n";
                amrex::AllPrintToFile("test_parser.tmp") << params.VD[i][j][k] << "\n";
                amrex::AllPrintToFile("test_parser.tmp") << params.VW[i][j][k] << "\n";
            }
        }
    }
    amrex::AllPrintToFile("test_parser.tmp") << params.WF << "\n";
    amrex::Real xlo = params.real_box.lo()[0];
    amrex::Real xhi = params.real_box.hi()[0];
    amrex::Real dx = (xhi - xlo) / params.n_cell[0];
    amrex::Real ylo = params.real_box.lo()[1];
    amrex::Real yhi = params.real_box.hi()[1];
    amrex::Real dy = (xhi - xlo) / params.n_cell[1];
    amrex::Real zlo = params.real_box.lo()[2];
    amrex::Real zhi = params.real_box.hi()[2];
    amrex::Real dz = (xhi - xlo) / params.n_cell[2];
    amrex::Real x, y, z;
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
                maxloc = std::abs(params.WFeval(x, y, z) - (1.0 + cos(params.k[0] * x) + sin(params.k[1] * y) + cos(2 * params.k[2] * z)));
                maxerr = std::max(maxerr, maxloc);
            }
        }
    }
    amrex::AllPrintToFile("test_parser.tmp") << "parser error " << maxerr << "\n";
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    if (amrex::ParallelDescriptor::MyProc() == 0)
        remove("test_parser.tmp.0");
    {
        test_Weibel<3,1>();
        test_read<3, 2>();
    }
    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_parser.tmp.0", "test_parser.output");

    amrex::Finalize();
}
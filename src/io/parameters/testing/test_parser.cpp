#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>

#include <GEMPIC_assertion.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_parameters.H>

template <int vdim, int numspec>
void test_Weibel(std::ofstream& outputFile)
{
    gempic_parameters<vdim, numspec> params;
    params.set_params_Weibel();
}

template <int vdim, int numspec>
void test_read(std::ofstream& outputFile)
{
    gempic_parameters<vdim, numspec> params;
    params.read_pp_params();

    // Print empty line
    amrex::Print(outputFile) << "\n";

    // Print data members in gempic_parameters
    amrex::Print(outputFile) << params.sim_name << "\n";
    amrex::Print(outputFile) << params.real_box << "\n";
    amrex::Print(outputFile) << params.is_periodic << "\n";
    amrex::Print(outputFile) << params.n_cell << "\n";
    for (int i = 0; i < GEMPIC_SPACEDIM; i++)
        amrex::Print(outputFile) << params.k[i] << " ";
    amrex::Print(outputFile) << "\n";
    amrex::Print(outputFile) << params.n_steps << "\n";
    amrex::Print(outputFile) << params.dt << "\n";
    amrex::Print(outputFile) << params.propagator << "\n";

    // particle initialisation
    for (int i = 0; i < numspec; i++)
    {
        for (int j = 0; j < params.num_gaussians[i]; j++)
        {
            amrex::Print(outputFile) << "species " << i << " gaussian " << j << "\n";
            for (int k = 0; k < vdim; k++)
            {
                amrex::Print(outputFile) << params.VM[i][j][k] << " ";
            }
            amrex::Print(outputFile) << "\n";
            for (int k = 0; k < vdim; k++)
            {
                amrex::Print(outputFile) << params.VD[i][j][k] << " ";
            }
            amrex::Print(outputFile) << "\n";
            amrex::Print(outputFile) << params.VW[i][j] << "\n";
        }
    }
    amrex::Print(outputFile) << params.WF[0] << "\n";

    // test parser
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
                maxloc = std::abs(params.WFeval[0](x, y, z) - (1.0 + cos(params.k[0] * x) + sin(params.k[1] * y) + cos(2 * params.k[2] * z)));
                maxerr = std::max(maxerr, maxloc);
            }
        }
    }
    amrex::Print(outputFile) << "parser error \n";
    amrex::Print(outputFile) << maxerr << "\n";
}

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    std::ofstream outputFile("test_parser.output", std::ofstream::out);
    {
        test_Weibel<3, 1>(outputFile);
        test_read<3, 2>(outputFile);
    }
    outputFile.close();

    amrex::Finalize();
}
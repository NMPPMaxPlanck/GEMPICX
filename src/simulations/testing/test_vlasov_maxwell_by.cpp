#include <AMReX.H>
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_vlasov_maxwell_ctest.H>

int main(int argc, char *argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(
        argc,
        argv,
        build_parm_parse,
        MPI_COMM_WORLD,
        overwrite_amrex_parser_defaults
    );
    std::string test_name = "test_vlasov_maxwell_by";
    const int degx = 1;
    const int degy = 1;
    const int degz = 1;   // spline degrees
    const int degmw = 2;  // order Hodge
    const int vdim = 3;   // velocity dimension
    vlasov_maxwell_ctest<degx, degy, degz, degmw, vdim>(test_name, 0);

    amrex::Finalize();
}

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
    std::string test_name = "test_vlasov_maxwell_hs_zigzag_C2";
    const int degx = 6;
    const int degy = 5;
    const int degz = 4;   // spline degrees
    const int degmw = 4;  // order Hodge
    const int vdim = 3;   // velocity dimension
    
    vlasov_maxwell_ctest<degx, degy, degz, degmw, vdim>(test_name, 3);

    amrex::Finalize();
}

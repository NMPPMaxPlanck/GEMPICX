#include <GEMPIC_vlasov_maxwell_ctest.H>

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    std::string test_name = "test_vlasov_maxwell_hs_zigzag_C2";
    const int degx = 6;
    const int degy = 5;
    const int degz = 4;   // spline degrees
    const int degmw = 4;  // order Hodge
    const int vdim = 3;   // velocity dimension
    
    vlasov_maxwell_ctest<degx, degy, degz, degmw, vdim>(test_name, 3);

    amrex::Finalize();
}

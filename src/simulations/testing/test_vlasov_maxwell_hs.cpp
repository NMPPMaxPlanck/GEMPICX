#include <GEMPIC_vlasov_maxwell_ctest.H>

int main(int argc, char* argv[])
{

    amrex::Initialize(argc,argv);
    std::string test_name = "test_vlasov_maxwell_hs";
    const int degx = 1; const int degy = 1; const int degz = 1;  // spline degrees 
    const int degmw = 2; // order Hodge
    const int vdim = 3;  // velocity dimension
    vlasov_maxwell_ctest<degx, degy, degz, degmw, vdim> (test_name, 1);

    amrex::Finalize();

}

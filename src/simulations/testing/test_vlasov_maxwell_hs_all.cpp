#include <GEMPIC_vlasov_maxwell_ctest.H>

int main(int argc, char* argv[])
{

    amrex::Initialize(argc,argv);
    std::string test_name = "test_vlasov_maxwell_hs_all";

    vlasov_maxwell_ctest<1, 1, 1, 2, 3> (test_name, 2);

    amrex::Finalize();

}




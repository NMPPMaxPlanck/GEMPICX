#include <GEMPIC_vlasov_maxwell_ctest.H>

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    std::string test_name = "test_vlasov_maxwell_hs_zigzag_C2";

    vlasov_maxwell_ctest<6, 5, 4, 4, 3>(test_name, 3);

    amrex::Finalize();
}

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <GEMPIC_vlasov_maxwell_simulation.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    std::string test_name = "test_vlasov_maxwell_hs_zigzag_C2";
    std::string test_name_tmp = test_name + ".tmp.0";
    std::string test_name_end = test_name + ".output";

    if (amrex::ParallelDescriptor::MyProc()==0) remove(test_name_tmp.c_str());

    // Output for GEMPIC_SPACEDIM=3
    vlasov_maxwell_test<3, 1, 6, 5, 4, 4, 2, true>(3, test_name);

    if (amrex::ParallelDescriptor::MyProc()==0) std::rename(test_name_tmp.c_str(), test_name_end.c_str());
    amrex::Finalize();
}




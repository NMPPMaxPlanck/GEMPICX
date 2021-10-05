#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <GEMPIC_vlasov_maxwell_simulation.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    std::string test_name = "test_vlasov_maxwell_hs";
    std::string test_name_tmp = test_name + ".tmp.0";
    std::string test_name_end = test_name + ".output";
    if (amrex::ParallelDescriptor::MyProc()==0) remove(test_name_tmp.c_str());
    /* This ctest has a different output for each GEMPIC_SPACEDIM. Therefore, the expected_output file contains all outputs.
    For each dimension, apart from running the main_main for the dimension, the output for the other dimensions needs to be
    outputted, so that the comparison to the expected_output (which contains all dimensions) works The order of the outputs is:
    GEMPIC_SPACEDIM=1, GEMPIC_SPACEDIM=2, GEMPIC_SPACEDIM=3 */

#if (GEMPIC_SPACEDIM == 1)
    // Output for GEMPIC_SPACEDIM=1
    main_main<1, 1, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=2
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "0 3.69818e-05 2.9765e-05 5e-07 0.0321487 0.293507 0.967771" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "1 3.69566e-05 2.97362e-05 4.99698e-07 0.0321493 0.293512 0.967784" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "2 3.69131e-05 2.96885e-05 4.98936e-07 0.0321505 0.293523 0.9678" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "3 3.68493e-05 2.96266e-05 4.98131e-07 0.0321521 0.293546 0.967818" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "4 3.67656e-05 2.9548e-05 4.97897e-07 0.0321541 0.293579 0.96784" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "5 3.66624e-05 2.94545e-05 4.98951e-07 0.0321566 0.29362 0.967867" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "6 3.6538e-05 2.93378e-05 5.0199e-07 0.0321596 0.293671 0.967902" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "7 3.6393e-05 2.92042e-05 5.07584e-07 0.032163 0.293735 0.967943" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "8 3.62282e-05 2.90518e-05 5.16081e-07 0.032167 0.293816 0.967986" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "9 3.60436e-05 2.88822e-05 5.27541e-07 0.0321713 0.29391 0.968031" << std::endl;

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "0 2.82775e-05 1.74142e-05 1.72514e-05 0 0 5e-07 0.315393 1.43412 4.90194 4.95606" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "1 2.82736e-05 1.74398e-05 1.72751e-05 2.69293e-12 1.35775e-11 4.99391e-07 0.31539 1.43414 4.90193 4.95602" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "2 2.82574e-05 1.74621e-05 1.72926e-05 4.25678e-11 2.14137e-10 4.98359e-07 0.315389 1.43421 4.90191 4.95599" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "3 2.82294e-05 1.74806e-05 1.73044e-05 2.11396e-10 1.05865e-09 4.97397e-07 0.315388 1.43431 4.90189 4.95597" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "4 2.81897e-05 1.74944e-05 1.73097e-05 6.51175e-10 3.23578e-09 4.97251e-07 0.31539 1.43446 4.90188 4.95595" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "5 2.81381e-05 1.75013e-05 1.73066e-05 1.53963e-09 7.56317e-09 4.98807e-07 0.315392 1.43465 4.90187 4.95594" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "6 2.80749e-05 1.74998e-05 1.72936e-05 3.07237e-09 1.48651e-08 5.02967e-07 0.315396 1.43488 4.90187 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "7 2.79993e-05 1.74866e-05 1.72703e-05 5.44347e-09 2.58451e-08 5.10512e-07 0.315402 1.43515 4.90187 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "8 2.79121e-05 1.74598e-05 1.72373e-05 8.8258e-09 4.09633e-08 5.21991e-07 0.315409 1.43547 4.90188 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "9 2.78133e-05 1.74224e-05 1.7191e-05 1.33538e-08 6.0346e-08 5.3761e-07 0.315419 1.43582 4.9019 4.95594" << std::endl;

#elif (GEMPIC_SPACEDIM == 2)

    // Output for GEMPIC_SPACEDIM=1
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "0 0.497151 5e-07 6.26259 9.99417" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "1 0.497005 5e-07 6.26351 9.99487" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "2 0.496563 5e-07 6.26629 9.99712" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "3 0.495827 5e-07 6.27091 10.0009" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "4 0.494801 5e-07 6.27736 10.0062" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "5 0.493489 5e-07 6.28561 10.013" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "6 0.491887 5e-07 6.29567 10.0213" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "7 0.489994 5e-07 6.30758 10.0311" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "8 0.487813 5e-07 6.32129 10.0424" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "9 0.485357 5e-07 6.33672 10.055" << std::endl;

    // Output for GEMPIC_SPACEDIM=2
    main_main<2, 1, 1, 1, 1>();

    // Output for GEMPIC_SPACEDIM=3
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "0 2.82775e-05 1.74142e-05 1.72514e-05 0 0 5e-07 0.315393 1.43412 4.90194 4.95606" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "1 2.82736e-05 1.74398e-05 1.72751e-05 2.69293e-12 1.35775e-11 4.99391e-07 0.31539 1.43414 4.90193 4.95602" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "2 2.82574e-05 1.74621e-05 1.72926e-05 4.25678e-11 2.14137e-10 4.98359e-07 0.315389 1.43421 4.90191 4.95599" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "3 2.82294e-05 1.74806e-05 1.73044e-05 2.11396e-10 1.05865e-09 4.97397e-07 0.315388 1.43431 4.90189 4.95597" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "4 2.81897e-05 1.74944e-05 1.73097e-05 6.51175e-10 3.23578e-09 4.97251e-07 0.31539 1.43446 4.90188 4.95595" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "5 2.81381e-05 1.75013e-05 1.73066e-05 1.53963e-09 7.56317e-09 4.98807e-07 0.315392 1.43465 4.90187 4.95594" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "6 2.80749e-05 1.74998e-05 1.72936e-05 3.07237e-09 1.48651e-08 5.02967e-07 0.315396 1.43488 4.90187 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "7 2.79993e-05 1.74866e-05 1.72703e-05 5.44347e-09 2.58451e-08 5.10512e-07 0.315402 1.43515 4.90187 4.95592" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "8 2.79121e-05 1.74598e-05 1.72373e-05 8.8258e-09 4.09633e-08 5.21991e-07 0.315409 1.43547 4.90188 4.95593" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "9 2.78133e-05 1.74224e-05 1.7191e-05 1.33538e-08 6.0346e-08 5.3761e-07 0.315419 1.43582 4.9019 4.95594" << std::endl;

#elif (GEMPIC_SPACEDIM == 3)

    // Output for GEMPIC_SPACEDIM=1
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "0 0.497151 5e-07 6.26259 9.99417" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "1 0.497005 5e-07 6.26351 9.99487" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "2 0.496563 5e-07 6.26629 9.99712" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "3 0.495827 5e-07 6.27091 10.0009" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "4 0.494801 5e-07 6.27736 10.0062" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "5 0.493489 5e-07 6.28561 10.013" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "6 0.491887 5e-07 6.29567 10.0213" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "7 0.489994 5e-07 6.30758 10.0311" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "8 0.487813 5e-07 6.32129 10.0424" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "9 0.485357 5e-07 6.33672 10.055" << std::endl;

    // Output for GEMPIC_SPACEDIM=2
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << std::endl;

    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "0 3.69818e-05 2.9765e-05 5e-07 0.0321487 0.293507 0.967771" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "1 3.69566e-05 2.97362e-05 4.99698e-07 0.0321493 0.293512 0.967784" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "2 3.69131e-05 2.96885e-05 4.98936e-07 0.0321505 0.293523 0.9678" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "3 3.68493e-05 2.96266e-05 4.98131e-07 0.0321521 0.293546 0.967818" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "4 3.67656e-05 2.9548e-05 4.97897e-07 0.0321541 0.293579 0.96784" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "5 3.66624e-05 2.94545e-05 4.98951e-07 0.0321566 0.29362 0.967867" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "6 3.6538e-05 2.93378e-05 5.0199e-07 0.0321596 0.293671 0.967902" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "7 3.6393e-05 2.92042e-05 5.07584e-07 0.032163 0.293735 0.967943" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "8 3.62282e-05 2.90518e-05 5.16081e-07 0.032167 0.293816 0.967986" << std::endl;
    AllPrintToFile("test_vlasov_maxwell_hs.tmp") << "9 3.60436e-05 2.88822e-05 5.27541e-07 0.0321713 0.29391 0.968031" << std::endl;

    // Output for GEMPIC_SPACEDIM=3
    vlasov_maxwell_test<3, 1, 1, 1, 1, 2, 2, true>(1 ,test_name);
#endif

    if (amrex::ParallelDescriptor::MyProc()==0) std::rename(test_name_tmp.c_str(),test_name_end.c_str());

    amrex::Finalize();
}




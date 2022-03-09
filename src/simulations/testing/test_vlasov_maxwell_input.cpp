#include <AMReX.H>

#include <GEMPIC_Config.H>

#include <GEMPIC_vlasov_maxwell.H>


int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
#if (GEMPIC_SPACEDIM == 1)
    main_main<2, GEMPIC_NUMSPEC, 1, 1, 1, 2, 2, GEMPIC_ELECTROMAGNETIC, true>(argc==1);
#elif (GEMPIC_SPACEDIM == 2)
    main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, 2, 2, GEMPIC_ELECTROMAGNETIC, true>(argc==1);
#elif (GEMPIC_SPACEDIM == 3)
   // main_main<3, GEMPIC_NUMSPEC, 2, 2, 2, 2, 2, GEMPIC_ELECTROMAGNETIC, true>(argc==1);

    vlasov_maxwell_simulation<3,1,2,2,2,2> vlasovMaxwell;
    vlasovMaxwell.ctest = true;
    vlasovMaxwell.initialize_vlasov_maxwell_from_file();
    vlasovMaxwell.run_time_loop();

     if (amrex::ParallelDescriptor::MyProc()==0) 
        {
            std::rename("test_vlasov_maxwell_input.tmp.0","test_vlasov_maxwell_input.output");
        }

#endif
    }
    amrex::Finalize();
}




#include <AMReX.H>

#include <GEMPIC_amrex_init.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_vlasov_maxwell.H>

int main(int argc, char* argv[])
{
    const bool build_parm_parse = true;
    amrex::Initialize(
        argc,
        argv,
        build_parm_parse,
        MPI_COMM_WORLD,
        overwrite_amrex_parser_defaults
    );
    {
#if (GEMPIC_SPACEDIM == 1)
    //main_main<2, GEMPIC_NUMSPEC, 1, 1, 1, 2, 2, GEMPIC_ELECTROMAGNETIC, true>(argc==1);
#elif (GEMPIC_SPACEDIM == 2)
    //main_main<3, GEMPIC_NUMSPEC, 1, 1, 1, 2, 2, GEMPIC_ELECTROMAGNETIC, true>(argc==1);
#elif (GEMPIC_SPACEDIM == 3)
   // main_main<3, GEMPIC_NUMSPEC, 2, 2, 2, 2, 2, GEMPIC_ELECTROMAGNETIC, true>(argc==1);

    vlasov_maxwell_simulation<3,1,2,2,2,2> vlasovMaxwell;
    vlasovMaxwell.initialize_vlasov_maxwell_from_file();
    vlasovMaxwell.run_time_loop();

#endif
    }
    amrex::Finalize();
}
#include <AMReX.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_vlasov_maxwell.H>

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {

    const int vdim = 3, numspec = 1, degx = 2, degy = 2, degz = 2, degvm = 2;
    vlasov_maxwell_simulation<vdim, numspec, degx, degy, degz, degvm> vlasovMaxwell;
    
    vlasovMaxwell.ctest = true;
    vlasovMaxwell.initialize_vlasov_maxwell_from_file();
    vlasovMaxwell.run_time_loop();

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_vlasov_maxwell_input.output.0", "test_vlasov_maxwell_input.output");
        }
    }
    amrex::Finalize();
}

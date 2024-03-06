#ifndef AMREXTESTENV_H
#define AMREXTESTENV_H

#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

namespace GEMPIC_Tests
{

/** Global setup/teardown test enviroment configuration for unit tests
 * (stolen from AMR_wind)
 *
 *  This class is registered with GoogleTest infrastructure to perform global
 *  setup/teardown tasks. The base implementation calls the amrex::Initialize
 *  and amrex::Finalize calls.
 *
 *  During the AmrexTestEnv::SetUp call, it also finalizes the amrex::ParmParse
 *  global instance so that each test can utilize a clean "input file". The user
 *  can disable this feature by passing `utest.keep_parameters=1` at the command
 *  line.
 *
 */
class AmrexTestEnv : public ::testing::Environment
{
public:
    AmrexTestEnv(int& argc, char**& argv) : m_argc(argc), m_argv(argv) {}

    ~AmrexTestEnv() override = default;

    void SetUp () override
    {
        amrex::Initialize(m_argc, m_argv, true, MPI_COMM_WORLD,
                          [] ()
                          {
                              amrex::ParmParse pp("amrex");
                              if (!(pp.contains("v") || pp.contains("verbose")))
                              {
                                  pp.add("verbose", -1);
                                  pp.add("v", -1);
                              }

                              pp.add("throw_exception", 1);
                              pp.add("signal_handling", 0);

                              // Abort the simulation when we run out of GPU memory
                              // By default, AMReX starts swapping to CPU, which makes runs very
                              // slow. It's more useful to abort and tell the user to request more
                              // GPUs or scale down the simulation.
                              // https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters
                              bool abortOnOutOfGpuMemory = true;  // AMReX' default: false
                              pp.queryAdd("abort_on_out_of_gpu_memory", abortOnOutOfGpuMemory);
                              // Synchronizes CPU (host) memory with GPU (device) memory, which is
                              // useful for testing
                              pp.add("the_arena_is_managed", 1);
                          });

        // Save managed memory flag for future use
        {
            amrex::ParmParse pp("amrex");
            pp.query("the_arena_is_managed", m_hasManagedMemory);
        }

        // Call ParmParse::Finalize immediately to allow unit tests to start
        // with a clean "input file". However, allow user to override this
        // behavior through command line arguments.
        {
            amrex::ParmParse pp("utest");
            bool keepParameters = false;
            pp.query("keep_parameters", keepParameters);

            if (!keepParameters)
            {
                amrex::ParmParse::Finalize();
            }
        }
    }

    void TearDown () override { amrex::Finalize(); }

    bool has_managed_memory () const { return m_hasManagedMemory; }

protected:
    int& m_argc;
    char**& m_argv;

    bool m_hasManagedMemory{true};
};

}  // namespace GEMPIC_Tests

#endif /* AMREXTESTENV_H */
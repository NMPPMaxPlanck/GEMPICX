/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef AMREXTESTENV_H
#define AMREXTESTENV_H

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_Initialize.H"

namespace GempicTests
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

                              Gempic::Impl::overwrite_amrex_parser_defaults();
                              // Synchronizes CPU (host) memory with GPU (device) memory,
                              // which is required for amrex::FunctionParser to work
                              pp.add("the_arena_is_managed", 1);
                          });
        Kokkos::initialize(m_argc, m_argv);
        Gempic::Impl::set_amrex_external_stream();

        // Save managed memory flag for future use
        {
            amrex::ParmParse pp("amrex");
            pp.query("the_arena_is_managed", m_hasManagedMemory);
        }

        // Call ParmParse::Finalize immediately to allow unit tests to start
        // with a clean "input file". However, allow user to override this
        // behavior through command line arguments.
        {
            {
                amrex::ParmParse pp("utest");
                pp.query("keepParameters", m_keepParameters);
            }

            if (!m_keepParameters)
            {
                amrex::ParmParse::Finalize();
            }
        }
    }

    void TearDown () override
    {
        Gempic::Impl::reset_amrex_external_stream();
        Kokkos::finalize();
        amrex::Finalize();
    }

    bool has_managed_memory () const { return m_hasManagedMemory; }

protected:
    int& m_argc;
    char**& m_argv;

    bool m_hasManagedMemory{true};
    bool m_keepParameters{false};
};

class AmrexTestIsolation : public testing::EmptyTestEventListener
{
public:
    AmrexTestIsolation()
    {
        amrex::ParmParse pp("utest");
        pp.query("keepParameters", m_keepParameters);
    }

private:
    // Called after each test suite ends
    void OnTestSuiteEnd (testing::TestSuite const& /*testSuite*/) override
    {
        if (!m_keepParameters)
        {
            amrex::ParmParse::Finalize();
        }
    }

    // Called after each test ends
    void OnTestEnd (testing::TestInfo const& /*testInfo*/) override
    {
        if (!m_keepParameters)
        {
            amrex::ParmParse::Finalize();
        }
    }

protected:
    bool m_keepParameters{false};
};

} //namespace GempicTests

#endif /* AMREXTESTENV_H */

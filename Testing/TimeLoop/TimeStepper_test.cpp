/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>
#include <fstream>
#include <iostream>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include "GEMPIC_TestRealValue.H"
#include "GEMPIC_TimeStepper.H"

/** Example "dynamical system" object
 *
 * This steps the Harmonic Series through its values, as an example of a very simple
 * "dynamical system".
 * The purpose is to have a very basic example based on `TimeStepper` functionality.
 */

class HarmonicSeries : public Gempic::TimeStepper
{
private:
    // The dynamical system will be described by a collection of different meaningful data
    // structures, for this simple model just a real number
    RealValue m_sum;

public:
    HarmonicSeries() : Gempic::TimeStepper() { m_sum.set(0); }
    int initialize() override;
    int generate_initial_condition() override;
    int step() override;
    int finalize() override;
};

int HarmonicSeries::initialize ()
{
    // model dependent
    // read simname parameters

    // configure TimeStepper object with parameters provided
    this->set_timestep(0.01);
    this->set_iterations_to_do(100);
    this->set_checkpoints_to_keep(3);
    this->set_output_interval(20);
    this->set_diagnostics_interval(2);

    // model dependent
    // register model data for IO
    auto registryAddResult = this->get_registry().add_data_structure(
        &m_sum, std::string("sum") + std::to_string(amrex::ParallelDescriptor::MyProc()));

    // model independent
    // call generic initialization
    auto initResult = this->initialize_data();

    // model dependent
    // misc
    std::cout << "initial sum is " << m_sum.value() << std::endl;
    if ((registryAddResult != EXIT_SUCCESS) || (initResult != EXIT_SUCCESS))
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int HarmonicSeries::generate_initial_condition ()
{
    // generate appropriate initial condition
    m_sum.set(0.);
    return EXIT_SUCCESS;
}

int HarmonicSeries::step ()
{
    // apply physics equation, updating all data structures according to the appropriate numerical
    // method. for this example, we simply use the "exact solution" instead of an approximate
    // integration scheme.
    m_sum.set(m_sum.value() + 1. / (1 + this->discrete_time().current_step()));

    // test whether stopping mechanism works
    if (m_sum.value() > 20)
    {
        this->create_stop_request();
    }

    return EXIT_SUCCESS;
}

int HarmonicSeries::finalize ()
{
    // deallocate
    return EXIT_SUCCESS;
}

int main (int argc, char* argv[]) // generic
{
    amrex::Initialize(argc, argv);
    {
        /// create appropriate TimeStepper
        HarmonicSeries cc;

        /// execute TimeStepper
        cc.run();
    }
    amrex::Finalize();
    return EXIT_SUCCESS;
}

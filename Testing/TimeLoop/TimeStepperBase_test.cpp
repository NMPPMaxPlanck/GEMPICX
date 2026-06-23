/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>
#include <fstream>
#include <iostream>

#include "GEMPIC_TestRealValue.H"
#include "GEMPIC_TimeStepper.H"

/** Example "dynamical system" object
 *
 * This steps the Harmonic Series through its values, as an example of a very simple
 * "dynamical system".
 * The purpose is to have a very basic example based on `TimeStepper` functionality.
 */

template <class RNumber>
class HarmonicSeries : public TimeStepperBase
{
private:
    // The dynamical system will be described by a collection of different meaningful data
    // structures, for this simple model just a real number
    RNumber m_sum;

public:
    HarmonicSeries(std::string const simulationName) : TimeStepperBase(simulationName)
    {
        m_sum.set(0);
    }
    int initialize() override;
    int generate_initial_condition() override;
    int step() override;
    int finalize() override;

    double value () const { return m_sum.value(); }
};

template <class RNumber>
int HarmonicSeries<RNumber>::initialize ()
{
    // model dependent
    // read simname parameters

    // in principle we should configure TimeStepper object here with parameters provided
    // for this test we need to do it outside though.
    //this->set_iterations_to_do(100);
    //this->set_checkpoints_to_keep(3);
    //this->set_output_interval(20);
    //this->set_diagnostics_interval(2);

    // model dependent
    // register model data for IO
    auto registryAddResult = this->get_registry().add_data_structure(&m_sum, "sum");

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

template <class RNumber>
int HarmonicSeries<RNumber>::generate_initial_condition ()
{
    // generate appropriate initial condition
    m_sum.set(0.);
    return EXIT_SUCCESS;
}

template <class RNumber>
int HarmonicSeries<RNumber>::step ()
{
    // apply physics equation, updating all data structures according to the appropriate numerical
    // method. for this example, we simply use the "exact solution" instead of an approximate
    // integration scheme.
    m_sum.set(m_sum.value() + 1. / (1 + this->m_time.current_step()));

    // test whether stopping mechanism works
    if (m_sum.value() > 20)
    {
        this->create_stop_request();
    }

    return EXIT_SUCCESS;
}

template <class RNumber>
int HarmonicSeries<RNumber>::finalize ()
{
    // deallocate
    return EXIT_SUCCESS;
}

int main (int argc, char* argv[]) // generic
{
    /*****/
    /// read simulation name
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <simulation_name>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string const simulationName(argv[1]);
    /*****/

    /*****/
    /// check that code runs for a hundred iterations without errors
    HarmonicSeries<RealValue> cc0(simulationName + "0");
    cc0.set_iterations_to_do(100);
    cc0.set_checkpoints_to_keep(3);
    cc0.set_output_interval(20);
    cc0.set_diagnostics_interval(2);
    cc0.print_debug_info();
    cc0.run();
    /*****/

    /*****/
    // check that running two jobs of 10 iterations is the same as running
    // one job with 20 iterations
    HarmonicSeries<RealValueFreeForm> cc10(simulationName + "10");
    //HarmonicSeries<RealValueFreeForm> cc20(simulationName + "20");

    //cc10.set_iterations_to_do(10);
    //cc10.set_checkpoints_to_keep(2);
    //cc10.set_output_interval(10);
    //cc10.set_diagnostics_interval(2);

    //cc20.set_iterations_to_do(20);
    //cc20.set_checkpoints_to_keep(2);
    //cc20.set_output_interval(20);
    //cc20.set_diagnostics_interval(2);

    //cc10.print_debug_info();
    //cc10.run();
    //// "set_iterations_to_do" updates iter0 and iter1 for cc10
    //// (values needed in the actual iteration loop)
    //cc10.set_iterations_to_do(10);
    //cc10.print_debug_info();
    //cc10.run();
    //cc20.print_debug_info();
    //cc20.run();
    ///*****/

    //double const error = std::abs(cc10.value() - cc20.value());
    //// error is compared with 1e-6 because RealValue uses .txt files for checkpointing.
    //if ((error / cc10.value()) > 1e-6)
    //{
    //    std::cout << "TimesTepper test failed, error is " << error << ", relative error is "
    //              << error / cc10.value() << std::endl;
    //    return EXIT_FAILURE;
    //}
    //else
    //{
    //    return EXIT_SUCCESS;
    //}
    return EXIT_SUCCESS;
}

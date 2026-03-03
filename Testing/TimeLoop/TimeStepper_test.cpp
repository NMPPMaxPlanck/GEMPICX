/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>
#include <fstream>
#include <iostream>

#include "GEMPIC_TimeStepper.H"

/** Example of a "physical object" class
 *
 * In general holds some physically meaningful data structure, along with IO functionality.
 * For this simple example, a single real number
 */

class RealValue
{
private:
    double m_value;

public:
    void set (double const newValue) { m_value = newValue; }
    double value () const { return m_value; }

    int read(std::filesystem::path const fname);
    int write(std::filesystem::path const fname) const;
    int write_diagnostics(std::filesystem::path const fname) const;
    std::string extension () const { return std::string("txt"); }
};

int RealValue::read (std::filesystem::path const fname)
{
    std::ifstream iFile;
    iFile.open(fname, std::ios::in);
    iFile >> m_value;
    iFile.close();
    return EXIT_SUCCESS;
}

int RealValue::write (std::filesystem::path const fname) const
{
    std::ofstream oFile;
    oFile.open(fname, std::ios::out);
    oFile << m_value;
    oFile.close();
    return EXIT_SUCCESS;
}

int RealValue::write_diagnostics (std::filesystem::path const fname) const
{
    std::cout << "current sum is " << m_value << std::endl;
    std::ofstream oFile;
    oFile.open(fname, std::ios::out);
    oFile << m_value;
    oFile.close();
    return EXIT_SUCCESS;
}

/** Example "dynamical system" object
 *
 * This steps the Harmonic Series through its values, as an example of a very simple
 * "dynamical system".
 * The purpose is to have a very basic example based on `TimeStepper` functionality.
 */

class HarmonicSeries : public TimeStepperBase
{
private:
    // The dynamical system will be described by a collection of different meaningful data
    // structures, for this simple model just a real number
    RealValue m_sum;

public:
    HarmonicSeries(std::string const simulationName) : TimeStepperBase(simulationName)
    {
        m_sum.set(0);
    }
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
    this->set_iterations_to_do(100);
    this->set_checkpoints_to_keep(3);
    this->set_output_interval(20);
    this->set_diagnostics_interval(2);

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
    m_sum.set(m_sum.value() + 1. / (1 + this->iteration()));

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
    /*****/
    /// read simulation name
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <simulation_name>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string const simulationName(argv[1]);
    /*****/

    // FIXME: misc initialization (MPI, AMReX, etc)

    /// create appropriate TimeStepper object with provided simname
    HarmonicSeries cc(simulationName);

    /// execute TimeStepper
    cc.run();

    // FIXME: clean up (MPI, AMReX, etc)
    return EXIT_SUCCESS;
}

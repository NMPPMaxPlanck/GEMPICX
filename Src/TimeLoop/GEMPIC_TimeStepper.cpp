/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>
#include <fstream>
#include <iostream>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include "GEMPIC_PlaceholderIO.H"
#include "GEMPIC_TimeStepper.H"

TimeStepperBase::TimeStepperBase(std::string const simulationName) :
    m_simulationName(simulationName)
{
    m_iteration = 0;
    m_iter0 = 0;
    m_iter1 = 0;
    m_diagnosticsInterval = 1;
    read_current_iteration();
}

void TimeStepperBase::read_current_iteration ()
{
    std::ifstream iFile;
    std::filesystem::path simFolder(m_simulationName);
    iFile.open(simFolder / "iteration.txt", std::ios::in); // will be closed on method exit
    if (iFile.is_open())
    {
        iFile >> m_iteration;
    }
    else
    {
        m_iteration = 0;
        if (!std::filesystem::exists(simFolder))
        {
            std::filesystem::create_directory(simFolder);
        }
        this->write_current_iteration();
    }
}

void TimeStepperBase::write_current_iteration () const
{
    std::ofstream oFile;
    oFile.open(std::filesystem::path(m_simulationName) / "iteration.txt", std::ios::out);
    oFile << m_iteration;
    oFile.close();
}

int TimeStepperBase::initialize_data ()
{
    if (m_iteration > 0)
    {
        PlaceholderIO::handle_io_call(m_registry.read(this->checkpoint_path()), "read checkpoint");
    }
    else
    {
        auto result = this->generate_initial_condition();
        assert(result == EXIT_SUCCESS);
        PlaceholderIO::handle_io_call(m_registry.write(this->checkpoint_path()),
                                      "write checkpoint");
        this->write_current_iteration();
    }
    return EXIT_SUCCESS;
}

int TimeStepperBase::run ()
{
    read_current_iteration();
    auto const initResult = this->initialize(); // may include output of initial conditions
    if (initResult != EXIT_SUCCESS)
    {
        std::cerr << "TimeStepperBase::initialize() failed with result " << initResult << std::endl;
        // ideally we would deallocate all arrays successfully allocated within initialize,
        // but I don't think we can simply call `finalize` here.
        return EXIT_FAILURE;
    }
    bool stopCodeNow = false;
    // starting at iter0
    for (; (m_iteration <= m_iter1) && !stopCodeNow; m_iteration++)
    {
        auto const stepResult = this->step();
        if (stepResult != EXIT_SUCCESS)
        {
            std::cerr << "TimeStepperBase::step() failed with result " << stepResult << std::endl;
            std::cerr << "Attempting a clean stop of the code." << std::endl;
            // creating a stop request ensures that the user checks what went wrong
            // before continuing
            create_stop_request();
            // try to output statistics and checkpoint
            m_registry.write_diagnostics(this->diagnostics_path());
            PlaceholderIO::handle_io_call(m_registry.write(this->checkpoint_path()),
                                          "write checkpoint");
            this->write_current_iteration();
            // don't remove old checkpoint because we don't know what went wrong with "step()".
            break;
        }
        if ((m_iteration % m_diagnosticsInterval) == 0)
        {
            std::cout << "writing diagnostics for iteration " << m_iteration << std::endl;
            m_registry.write_diagnostics(this->diagnostics_path());
        }
        if ((m_iteration % m_outputInterval) == 0)
        {
            PlaceholderIO::handle_io_call(m_registry.write(this->checkpoint_path()),
                                          "write checkpoint");
            this->write_current_iteration();
            this->remove_old_checkpoint();
        }
        stopCodeNow = check_stop_request();
    }
    this->finalize();
    return EXIT_SUCCESS;
}

std::filesystem::path TimeStepperBase::get_path (int64_t const targetIteration,
                                                std::string const target) const
{
    return std::filesystem::path (std::filesystem::path(m_simulationName) / target /
                                 std::to_string(targetIteration));
}

int TimeStepperBase::remove_checkpoint (int64_t const targetIteration) const
{
    std::filesystem::path const checkpointToRemove = this->get_path(targetIteration, "checkpoint");
    if (std::filesystem::exists(checkpointToRemove))
    {
        std::filesystem::remove_all(checkpointToRemove);
    }
    return EXIT_SUCCESS;
}

std::filesystem::path TimeStepperBase::diagnostics_path () const
{
    auto const p = this->get_path(m_iteration, "diagnostics");
    if (!std::filesystem::exists(p))
    {
        std::filesystem::create_directories(p);
    }
    return p;
}

std::filesystem::path TimeStepperBase::checkpoint_path () const
{
    auto const p = this->get_path(m_iteration, "checkpoint");
    if (!std::filesystem::exists(p))
    {
        std::filesystem::create_directories(p);
    }
    return p;
}

void TimeStepperBase::remove_old_checkpoint () const
{
    auto result = this->remove_checkpoint(m_iteration - m_checkpointsToKeep * m_outputInterval);
    assert(result == EXIT_SUCCESS);
}

bool TimeStepperBase::check_stop_request () const
{
    std::filesystem::path const stoppingFile(std::filesystem::path(m_simulationName) /
                                             std::string("stop_request"));
    bool mustStopCode = std::filesystem::exists(stoppingFile);
    if (mustStopCode)
    {
        std::cerr << "Code stop requested at iteration " << m_iteration << std::endl;
        std::cout << "Code stop requested at iteration " << m_iteration << std::endl;
    }
    return mustStopCode;
}

void TimeStepperBase::create_stop_request () const
{
    std::filesystem::path const stoppingFile(std::filesystem::path(m_simulationName) /
                                             std::string("stop_request"));
    std::ofstream oFile;
    oFile.open(stoppingFile, std::ios::out);
    oFile << "TimeStepperBase: stop requested." << std::endl;
    oFile.close();
}

bool Gempic::TimeStepper::check_stop_request () const
{
    bool mustStopCode = false;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        mustStopCode = TimeStepperBase::check_stop_request();
    }
    amrex::ParallelDescriptor::ReduceBoolOr(mustStopCode);
    return mustStopCode;
}

void Gempic::TimeStepper::create_stop_request () const
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        TimeStepperBase::create_stop_request();
    }
}

void Gempic::TimeStepper::read_current_iteration ()
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        TimeStepperBase::read_current_iteration();
    }
    amrex::ParallelDescriptor::Bcast(&m_iteration, 1, amrex::ParallelDescriptor::IOProcessor());
}

void Gempic::TimeStepper::write_current_iteration () const
{
    int64_t ioProcessorIteration;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        ioProcessorIteration = m_iteration;
    }
    amrex::ParallelDescriptor::Bcast(&ioProcessorIteration, 1,
                                     amrex::ParallelDescriptor::IOProcessor());
    assert(m_iteration == ioProcessorIteration);
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        TimeStepperBase::write_current_iteration();
    }
}

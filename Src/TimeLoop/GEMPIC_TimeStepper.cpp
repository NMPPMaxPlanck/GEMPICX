/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Parameters.H"
#include "GEMPIC_PlaceholderIO.H"
#include "GEMPIC_TimeStepper.H"

std::string Gempic::TimeStepper::get_simulation_name_parameter () const
{
    std::string simulationName;
    Gempic::Io::Parameters params{};
    params.get_or_set("simName", simulationName);
    if (simulationName.length() == 0)
    {
        char* simname = new char[7];
        simname[6] = '\0';
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            std::random_device rd;  // a seed source for the random number engine
            std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
            std::string const charList(
                "01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
            std::uniform_int_distribution<> distrib(0, charList.length() - 1);
            for (auto i = 0; i < 6; i++)
            {
                simname[i] = charList[distrib(gen)];
            }
        }
        if (amrex::ParallelDescriptor::NProcs() > 1)
        {
            amrex::ParallelDescriptor::Bcast(simname, 6,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
        }
        simulationName = std::string(simname);
        assert(simulationName.length() == 6);
        delete[] simname;
        amrex::Warning("Warning: provided simName is empty, using \"" + simulationName +
                       "\" instead.");
    }
    return simulationName;
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
    size_t currentStep{m_time.current_step()};
    amrex::ParallelDescriptor::Bcast(&currentStep, 1,
                                     amrex::ParallelDescriptor::IOProcessorNumber());
}

void Gempic::TimeStepper::write_current_iteration () const
{
    int64_t ioProcessorIteration;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        ioProcessorIteration = m_time.current_step();
    }
    amrex::ParallelDescriptor::Bcast(&ioProcessorIteration, 1,
                                     amrex::ParallelDescriptor::IOProcessorNumber());
    assert(m_time.current_step() == ioProcessorIteration);
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        TimeStepperBase::write_current_iteration();
    }
}

int Gempic::TimeStepper::remove_checkpoint (int64_t const targetIteration) const
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        TimeStepperBase::remove_checkpoint(targetIteration);
    }
    return EXIT_SUCCESS;
}

/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef TIME_STEPPER_BASE_H_
#define TIME_STEPPER_BASE_H_

#include <cstdint>
#include <filesystem>
#include <string>

#include "GEMPIC_PlaceholderIO.H"

/** Basic "iterate solver" functionality.
 *
 * This abstract class offers a generic mechanism to iterate over a desired number of timesteps.
 * Solvers should inherit this class, and implement the four virtual methods `initialize`,
 * `generate_initial_condition`, `step` and `finalize` appropriately.
 *
 * Each object is initialized with a fixed simulation name, and it then uses the folder
 * `simulation_name` for storing all simulation data.
 * Within this folder, the file `iteration.txt` stores the iteration of the last successful
 * checkpoint (output).
 * If this file exists, the code reads the iteration from the file, otherwise it sets the current
 * iteration to 0.
 *
 * After initialization, the programmer should call the `set_iterations_to_do` method, to specify
 * how may iterations should be performed during the `run` method (which contains a simple loop).
 * During runtime, the user may create the file `simulation_name/stop_request` to trigger an early
 * exit from the main loop; this is considered normal behavior and the `run` method will still call
 * the `finalize` method and return `EXIT_SUCCESS`.
 */
class TimeStepperBase
{
protected:
    std::string const m_simulationName;
    Gempic::DiscreteTime m_time;
    int64_t m_checkpointsToKeep;
    int64_t m_outputInterval;
    int64_t m_diagnosticsInterval;
    PlaceholderIO::Registry m_registry;

    std::filesystem::path get_path(int64_t const targetIteration, std::string const target) const;
    virtual int remove_checkpoint(int64_t const targetIteration) const;

    TimeStepperBase(std::string const simulationName);

public:
    virtual ~TimeStepperBase() = default;

    Gempic::DiscreteTime const& discrete_time () const { return m_time; };

    std::string simulation_name () const { return m_simulationName; }

    void set_timestep (amrex::Real const dt)
    {
        m_time = Gempic::DiscreteTime{dt, m_time.current_step(), m_time.current_step()};
    }

    void set_iterations_to_do (int64_t const nIterToDo)
    {
        m_time = Gempic::DiscreteTime{m_time.dt(), m_time.current_step() + nIterToDo,
                                      m_time.current_step()};
    }

    void set_checkpoints_to_keep (int64_t const nCPToKeep) { m_checkpointsToKeep = nCPToKeep; }

    void set_output_interval (int64_t const outputInterval) { m_outputInterval = outputInterval; }

    void set_diagnostics_interval (int64_t const diagnosticsInterval)
    {
        m_diagnosticsInterval = diagnosticsInterval;
    }

    int print_debug_info () const
    {
        std::cout << "TimeStepperBase debug info" << std::endl;
        std::cout << "simname " << m_simulationName << std::endl;
        std::cout << "iteration " << m_time.current_step() << std::endl;
        std::cout << "iter0 " << m_time.initial_step() << std::endl;
        std::cout << "iter1 " << m_time.final_step() << std::endl;
        std::cout << "checkpoints to keep " << m_checkpointsToKeep << std::endl;
        std::cout << "output interval " << m_outputInterval << std::endl;
        std::cout << "diagnostics interval " << m_diagnosticsInterval << std::endl;
        return EXIT_SUCCESS;
    }

    int run();

    /** \brief Prepare model for time stepping.
     *
     * Children of `TimeStepperBase` shall use this method to prepare the calling object for
     * executing the `step` method.
     * Usually the method will consist of two parts:
     *  * allocate the various data structures;
     *  * call the `initialize_data` method (already defined by `TimeStepperBase`).
     *
     */
    virtual int initialize() = 0;

    /** \brief Populate data structures at iteration 0.
     *
     * This method populates relevant data
     * structures, using model-appropriate analytic expressions (that may contain
     * pseudo-random numbers).
     *
     * \warning: This method should not contain IO operations.
     *
     */
    virtual int generate_initial_condition() = 0;

    /** \brief Execute an in-place timestep.
     *
     * This method implements the algorithm that advances the model's state variables by one
     * timestep.
     *
     */
    virtual int step() = 0;

    /** \brief Deallocates data structures and "cleans up".
     *
     * This method shall deallocate those model data structures that permit it.
     * This method shall also perform any additional "cleanup" operations to complement the
     * relevant C++ destructors.
     *
     */
    virtual int finalize() = 0;

    /** \brief Populates data structures.
     *
     * This method populates the model data structures:
     *   * if the iteration is 0, it calls `generate_initial_condition`;
     *   * else it reads the most recent checkpoint.
     *
     */
    int initialize_data();

    /** \brief Reads the value of `m_iteration`.
     *
     * Reads the value of the current iteration from the file `simulation_name/iteration.txt`.
     */
    virtual void read_current_iteration();
    /** \brief Writes the value of `m_iteration`.
     *
     * Writes the value of the current iteration to the file `simulation_name/iteration.txt`.
     */
    virtual void write_current_iteration() const;
    virtual bool check_stop_request() const;
    virtual void create_stop_request() const;

    std::filesystem::path diagnostics_path() const;
    std::filesystem::path checkpoint_path() const;
    void remove_old_checkpoint() const;
    PlaceholderIO::Registry& get_registry () { return m_registry; }
};

#endif // TIME_STEPPER_BASE_H_

/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef TIME_STEPPER_H_
#define TIME_STEPPER_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include <AMReX_REAL.H>

#include "GEMPIC_DiscreteTime.H"
#include "GEMPIC_PlaceholderIO.H"
#include "GEMPIC_TimeStepperBase.H"

namespace Gempic
{
/** Basic "iterate solver" functionality, AMReX-compatible.
 *
 * Please see documentation for `TimeStepperBase`.
 * Here we simply add the required AMReX::ParallelDescriptor synchronization.
 */
class TimeStepper : public TimeStepperBase
{
protected:
    std::string get_simulation_name_parameter() const;
    int remove_checkpoint(int64_t const targetIteration) const override;

public:
    TimeStepper() : TimeStepperBase(get_simulation_name_parameter()) {}

    void read_current_iteration() override;
    void write_current_iteration() const override;
    bool check_stop_request() const override;
    void create_stop_request() const override;
};

} //namespace Gempic

#endif // TIME_STEPPER_H_

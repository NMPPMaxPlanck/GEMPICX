/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef DISCRETE_TIME_H_
#define DISCRETE_TIME_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include <AMReX_REAL.H>

namespace Gempic
{
// clang-format off
/**
 * @class DiscreteTime
 * @brief Representation of the temporal state of a simulation that advances
 *        in equidistant steps of size \f$\Delta t\f$.
 *
 * @details The time state is described by \f$ t \in [t_0,T]\f$ and \f$ t = n \Delta t\f$
 *          where
 *          - \f$t\f$ is given by @ref DiscreteTime::t.
 *          - \f$n\f$ is given by @ref DiscreteTime::current_step.
 *          - \f$\Delta t\f$ is given by @ref DiscreteTime::dt.
 *          - \f$t_0\f$ is given by @ref DiscreteTime::initial.
 *          - \f$T\f$ is given by @ref DiscreteTime::final.
 */
// clang-format on
class DiscreteTime
{
public:
    DiscreteTime(amrex::Real dt, size_t finalStep, size_t initialStep = 0);
    //! Final time
    amrex::Real final() const;
    //! Initial time
    amrex::Real initial() const;
    //! Step width
    amrex::Real const& dt() const;
    //! Current time
    amrex::Real t() const;
    //! Index identifier of the current step
    size_t const& current_step() const;
    //! Index identifier of the initial state
    size_t const& initial_step() const;
    //! Index identifier of the final state
    size_t const& final_step() const;

    //! Returns true if @ref DiscreteTime::current_step <= @ref DiscreteTime::final_step
    bool continue_simulation() const;
    //! Advance the current step by one (\f$n \leftarrow n + 1\f$)
    void step();

    bool operator==(DiscreteTime const&) const = default;

private:
    amrex::Real m_dt;
    size_t m_initialStep;
    size_t m_currentStep;
    size_t m_finalStep;
};

class H5GroupHandle;
void serialize(std::string const& label, DiscreteTime const&, H5GroupHandle const&);
void deserialize(std::string const& label, DiscreteTime&, H5GroupHandle const&);
} // namespace Gempic

#endif // DISCRETE_TIME_H_

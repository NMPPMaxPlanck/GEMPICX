/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef PROFILING_H_
#define PROFILING_H_

#include <AMReX.H>
#include <AMReX_REAL.H>

namespace Gempic
{
namespace Profiling
{
/**
 * Class responsible contains all timers for profiling the code
 */

class Timers
{
public:
    bool m_profiling;
    amrex::Real m_counterDeposition;
    amrex::Real m_counterRedistribute;
    amrex::Real m_counterJboundary;
    amrex::Real m_counterFields;
    amrex::Real m_counterDiagnostics;
    amrex::Real m_counterAll;

    // The default constructor.
    /**
     * Default Constructor.
     */
    Timers();

    // Explicit const
    /**
     * Explicit constructor.
     */
    Timers(bool prof)
    {
        m_profiling = prof;
        m_counterDeposition = 0.;
        m_counterRedistribute = 0.;
        m_counterJboundary = 0.;
        m_counterFields = 0.;
        m_counterDiagnostics = 0.;
        m_counterAll = 0.;
    }
};
} // namespace Profiling
} // namespace Gempic

#endif

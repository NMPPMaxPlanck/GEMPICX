#include <AMReX.H>
#include <AMReX_REAL.H>

#ifndef PROFILING_H_
#define PROFILING_H_

namespace Gempic {
namespace Profiling {

/**
 * Class responsible contains all timers for profiling the code
 */

class timers
{
public:
    bool profiling;
    amrex::Real counter_deposition;
    amrex::Real counter_redistribute;
    amrex::Real counter_jboundary;
    amrex::Real counter_fields;
    amrex::Real counter_diagnostics;
    amrex::Real counter_all;

    // The default constructor.
    /**
     * Default Constructor.
     */
    timers ();

    // Explicit const
    /**
     * Explicit constructor.
     */
    timers (bool prof){
        profiling = prof;
        counter_deposition = 0.;
        counter_redistribute = 0.;
        counter_jboundary = 0.;
        counter_fields = 0.;
        counter_diagnostics = 0.;
        counter_all = 0.;
    }

};
}
}

#endif

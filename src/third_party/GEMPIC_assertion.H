#include <AMReX.H>
#include <AMReX_REAL.H>

#ifndef GEMPIC_ASSERT_H_
#define GEMPIC_ASSERT_H_

#include <GEMPIC_Config.H>

namespace Gempic {

void gempic_assert(bool *passed, amrex::Real solution, amrex::Real value){

    // check absolute error
    *passed = *passed && (fabs(solution-value) < TOL);

    // check relative error
    *passed = *passed && (fabs(solution-value)/std::max(fabs(solution), std::numeric_limits<double>::epsilon()) < RELTOL);

}

void gempic_assert_err(bool *passed, amrex::Real solution, amrex::Real err){

    // check absolute error
    *passed = *passed && (fabs(err) < TOL);

    // check relative error
    *passed = *passed && (fabs(err)/std::max(fabs(solution), std::numeric_limits<double>::epsilon()) < RELTOL);

}
}

#endif

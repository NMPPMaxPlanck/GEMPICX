#ifndef GEMPIC_ASSERT_H_
#define GEMPIC_ASSERT_H_

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <GEMPIC_Config.H>

namespace Gempic {

void gempic_assert(bool *passed, amrex::Real solution, amrex::Real value){

    // check absolute error
    bool pass_abs = fabs(solution-value) < TOL;

    // check relative error
    bool pass_rel = fabs(solution-value)/std::max(fabs(solution), std::numeric_limits<double>::epsilon()) < RELTOL;

    *passed = *passed && (pass_abs || pass_rel);

}

void gempic_assert_err(bool *passed, amrex::Real solution, amrex::Real err){

    // check absolute error
    *passed = *passed && (fabs(err) < TOL);

    // check relative error
    *passed = *passed && (fabs(err)/std::max(fabs(solution), std::numeric_limits<double>::epsilon()) < RELTOL);

}
}

#endif

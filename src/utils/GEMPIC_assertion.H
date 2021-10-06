#ifndef GEMPIC_ASSERT_H_
#define GEMPIC_ASSERT_H_

#include <cmath>
#include <limits>

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <GEMPIC_Config.H>

namespace Gempic {

void gempic_assert(bool passed, amrex::Real solution, amrex::Real value){

    // check absolute error
    bool pass_abs = std::fabs(solution-value) < GEMPIC_CTEST_TOL;

    // check relative error
    bool pass_rel = std::fabs(solution-value)/std::max(std::fabs(solution), std::numeric_limits<double>::epsilon()) < GEMPIC_CTEST_RELTOL;

    passed = passed && (pass_abs || pass_rel);

}

void gempic_assert_err(bool &passed, amrex::Real solution, amrex::Real err){

    // check absolute error

    passed = passed && (std::fabs(err) < GEMPIC_CTEST_TOL);

    // check relative error
    passed = passed && (std::fabs(err)/std::max(std::fabs(solution), std::numeric_limits<double>::epsilon()) < GEMPIC_CTEST_RELTOL);

}
}

#endif

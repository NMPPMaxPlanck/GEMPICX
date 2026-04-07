/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_PARTICLE_UTILS_H_
#define GEMPIC_PARTICLE_UTILS_H_
#define _USE_MATH_DEFINES

#include <cmath>

#include <AMReX.H>

#include "GEMPIC_Config.H"

namespace Gempic::Utils
{

/**
 * Evaluate the Maxwellian distribution for a given velocity vector,
 * density, and uniform thermal velocity.
 *
 * @tparam vDim The number of velocity components.
 * @param vel The velocity vector.
 * @param density The density of the distribution.
 * @param vThermal The thermal velocity of the distribution.
 *
 * @return The value of the Maxwellian.
 */
template <std::size_t vDim>
AMREX_GPU_HOST_DEVICE amrex::Real eval_maxwellian (std::array<amrex::Real, vDim> const& vel,
                                                   amrex::Real density,
                                                   amrex::Real vThermal)
{
    amrex::Real exponent = 0;
    amrex::Real coeff = 1;
    constexpr amrex::Real sqrtTwoPi{2.5066282746310002};

    for (unsigned int d = 0; d < vDim; ++d)
    {
        exponent += vel[d] * vel[d];
        coeff *= vThermal * sqrtTwoPi;
    }
    exponent *= -0.5 / (vThermal * vThermal);

    return density * std::exp(exponent) / coeff;
}

/**
 * Evaluates the Maxwellian distribution for a given velocity vector, density, thermal velocity
 * vector, and mean velocity vector.
 *
 * @tparam vDim The number of velocity components.
 * @param vel The velocity vector.
 * @param density The density of the distribution.
 * @param vThermal The thermal velocity of the distribution.
 * @param vMean The mean velocity of the distribution.
 *
 * @return The value of the Maxwellian distribution.
 */
template <std::size_t vDim>
AMREX_GPU_HOST_DEVICE amrex::Real eval_maxwellian (std::array<amrex::Real, vDim> const& vel,
                                                   amrex::Real density,
                                                   std::array<amrex::Real, vDim> const& vThermal,
                                                   std::array<amrex::Real, vDim> const& vMean)
{
    amrex::Real exponent = 0;
    amrex::Real coeff = 1;
    constexpr amrex::Real sqrtTwoPi{2.5066282746310002};

    for (unsigned int d = 0; d < vDim; ++d)
    {
        exponent += (vel[d] - vMean[d]) * (vel[d] - vMean[d]) / (vThermal[d] * vThermal[d]);
        coeff *= vThermal[d] * sqrtTwoPi;
    }
    exponent *= -0.5;

    return density * std::exp(exponent) / coeff;
}

/**
 * Evaluates the square root of the Maxwellian distribution for a given velocity vector, density,
 * and uniform thermal velocity.
 *
 * @tparam vDim The number of velocity components.
 * @param vel The velocity vector.
 * @param density The density of the distribution.
 * @param vThermal The thermal velocity of the distribution.
 *
 * @return The square root of the Maxwellian distribution.
 */
template <std::size_t vDim>
AMREX_INLINE AMREX_GPU_HOST_DEVICE amrex::Real eval_sqrt_maxwellian (
    std::array<amrex::Real, vDim> const& vel, amrex::Real density, amrex::Real vThermal)
{
    return std::sqrt(eval_maxwellian(vel, density, vThermal));
}

} // namespace Gempic::Utils

#endif

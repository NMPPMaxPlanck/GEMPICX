/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef PARTICLE_DATA_INDICES_H_
#define PARTICLE_DATA_INDICES_H_

#include <limits>

#include <AMReX.H>
#include <AMReX_Particle.H>

namespace Gempic::Particle
{

/** \brief Collection of named array indices within SoA particle objects.
 *
 * This struct shall hold indices to use with the `rdata` method of `ParticleTileData` objects.
 *
 * The purpose of the struct is to provide useful names for the different arrays,
 * (e.g. `posx`, `velx` etc) within `amrex::ParallelFor` or `amrex::ParticleReduce` blocks. By
 * default all indices are MAX_INT, you should use the GEMPIC_ParticleSpecies::get_data_indices
 * method for initialization.
 * Please see `compute_diags` from `Io/Diagnostics/ReducedDiagnostics/GEMPIC_ParticleDiags.H` for
 * example usage.
 *
 */
struct ParticleDataIndices
{
    int m_iposx;
    int m_iposy;
    int m_iposz;
    int m_ivelx;
    int m_ively;
    int m_ivelz;
    int m_iweight;
    int m_is0;
    int m_isqrtf0;
    int m_is2X;
    int m_is2Y;
    int m_is2Z;
    int m_is2Vx;
    int m_is2Vy;
    int m_is2Vz;

    static constexpr int s_badIndex = std::numeric_limits<int>::max() - 1;

    ParticleDataIndices()
    {
        m_iposx = s_badIndex;
        m_iposy = s_badIndex;
        m_iposz = s_badIndex;
        m_ivelx = s_badIndex;
        m_ively = s_badIndex;
        m_ivelz = s_badIndex;
        m_iweight = s_badIndex;
        m_is0 = s_badIndex;
        m_isqrtf0 = s_badIndex;
        m_is2X = s_badIndex;
        m_is2Y = s_badIndex;
        m_is2Z = s_badIndex;
        m_is2Vx = s_badIndex;
        m_is2Vy = s_badIndex;
        m_is2Vz = s_badIndex;
    }
};
} // namespace Gempic::Particle

#endif // PARTICLE_DATA_INDICES_H_

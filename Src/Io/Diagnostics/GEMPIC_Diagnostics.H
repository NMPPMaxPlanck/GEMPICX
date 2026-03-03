/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPICX_DIAGNOSTICS_H_
#define GEMPICX_DIAGNOSTICS_H_

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FieldRegistry.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"

namespace Gempic::Io
{
/**
 * @brief: Convenience wrapper gathering diagnostics modules
 */
template <int xDeg, int yDeg, int zDeg, unsigned int vDim, unsigned int nData>
class Diagnostics
{
private:
    MultiDiagnostics<vDim, nData> m_fullDiagn;
    MultiReducedDiagnostics<vDim, xDeg, yDeg, zDeg, nData> m_redDiagn;

    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> m_particles;
    ComputationalDomain m_compDom;

public:
    template <Gempic::Forms::DeRhamComplexC DRC>
    Diagnostics(ComputationalDomain const& compDom,
                std::shared_ptr<DRC> const drc,
                FieldRegistry const& fieldRegistry,
                std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const&
                    particles) :
        m_fullDiagn{compDom, fieldRegistry, particles},
        m_redDiagn{drc, fieldRegistry},
        m_particles{particles},
        m_compDom{compDom}
    {
    }

    /**
     * Decides whether or not to compute diagnostics and write to output file(s), based on
     * whether stepNumber matches the output saveInterval(s) specified in the input file.
     * Can be overwritten with forceWrite = true.
     *
     * @param stepNumber Step number in simulation
     * @param time      Simulation time
     * @param forceWrite Forcibly write to output file, skipping the check
     */
    void compute_and_write_to_file (int stepNumber, amrex::Real time, bool forceWrite = false)
    {
        m_fullDiagn.filter_compute_pack_flush(stepNumber, time, forceWrite);
        m_redDiagn.compute_and_write_to_file(stepNumber, time, m_compDom, m_particles, forceWrite);
    }
};

/**
 * Convenience function to create diagnostics objects with partial tepmlate specialization
 */

template <int xDeg,
          int yDeg,
          int zDeg,
          unsigned int vDim,
          unsigned int nData,
          Gempic::Forms::DeRhamComplexC DRC>
Diagnostics<xDeg, yDeg, zDeg, vDim, nData> make_diagnostics (
    ComputationalDomain const& compDom,
    std::shared_ptr<DRC> const drc,
    FieldRegistry const& reg,
    std::vector<std::shared_ptr<Gempic::Particle::ParticleSpecies<vDim, nData>>> const& particles)
{
    return Diagnostics<xDeg, yDeg, zDeg, vDim, nData>(compDom, drc, reg, particles);
}
} //namespace Gempic::Io
#endif
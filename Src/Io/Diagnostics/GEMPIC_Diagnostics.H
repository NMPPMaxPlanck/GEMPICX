#ifndef GEMPICX_DIAGNOSTICS_H_
#define GEMPICX_DIAGNOSTICS_H_

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_MultiFullDiagnostics.H"
#include "GEMPIC_MultiReducedDiagnostics.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"

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

    std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>> m_particles;
    ComputationalDomain m_compDom;

public:
    Diagnostics(const ComputationalDomain& compDom,
                const std::shared_ptr<Forms::DeRhamComplex> drc,
                const std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>&
                    particles) :
        m_fullDiagn{compDom, drc->m_fieldsDiagnostics, drc->m_fieldsScaling, particles},
        m_redDiagn{drc},
        m_particles{particles},
        m_compDom{compDom}
    {
    }

    /**
     * Decides whether or not to compute diagnostics and write to output file(s), based on
     * whether stepNumber matches the output saveInterval(s) specified in the input file.
     * Can be overwritten with forceWrite = true.
     *
     * @param stepNumber: Step number in simulation
     * @param time:       Simulation time
     * @param forceWrite: Forcibly write to output file, skipping the check
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
template <int xDeg, int yDeg, int zDeg, unsigned int vDim, unsigned int nData>
Diagnostics<xDeg, yDeg, zDeg, vDim, nData> make_diagnostics (
    const ComputationalDomain& compDom,
    const std::shared_ptr<Forms::DeRhamComplex> drc,
    const std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vDim, nData>>>& particles)
{
    return Diagnostics<xDeg, yDeg, zDeg, vDim, nData>(compDom, drc, particles);
}
/*
    Io::MultiDiagnostics<vdim, ndata> fullDiagn(dt, infra, deRham->m_fieldsDiagnostics,
                                                deRham->m_fieldsScaling, ions);

    // Initialize reduced diagnostics and write initial time step
    Io::MultiReducedDiagnostics<vdim, degx, degy, degz, ndata> redDiagn(deRham);

    redDiagn.compute_and_write_to_file(tStep + 1, dt, infra, ions);
    fullDiagn.filter_compute_pack_flush(tStep + 1);
*/
}  //namespace Gempic::Io
#endif
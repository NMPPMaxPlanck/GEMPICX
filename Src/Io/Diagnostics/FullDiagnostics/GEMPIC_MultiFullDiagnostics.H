#ifndef GEMPICX_MULTIDIAGNOSTICS_H_
#define GEMPICX_MULTIDIAGNOSTICS_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include "GEMPIC_FullDiagnostics.H"

namespace Gempic::Io
{
/**
 * \brief This class contains a vector of all full diagnostics in the simulation.
 * Full diagnostics save all the field and/or particle data at chosen time steps
 */
template <unsigned int vDim, unsigned int nData>
class MultiDiagnostics
{
public:
    MultiDiagnostics(
        ComputationalDomain const& compDom,
        std::map<std::string, amrex::MultiFab const> const& fieldMap,
        std::map<std::string, amrex::Real const> const& fieldsScaling,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vDim, nData>>> const& particles)
    {
        BL_PROFILE("Gempic::Io::MultiDiagnostics()");
        read_parameters();
        /** Resize alldiags and initialize each element to a pointer to a
         * diagnostics. Calls the corresponding diagnostics constructor.
         */
        m_ndiags = static_cast<int>(m_diagsNames.size());
        m_alldiags.resize(m_ndiags);
        for (int i = 0; i < m_ndiags; i++)
        {
            m_alldiags[i] = std::make_unique<FullDiagnostics<vDim, nData>>(
                i, m_diagsNames[i], m_defaultSaveInterval, m_defaultSaveFolder);
        }

        init_data(compDom, fieldMap, fieldsScaling, particles);
    }
    /** \brief Read Input parameters. Called in constructor. */
    void read_parameters ()
    {
        BL_PROFILE("Gempic::Io::MultiDiagnostics::read_parameters()");
        Parameters params("FullDiagnostics");

        bool enableDiags{false};
        params.get_or_set("enable", enableDiags);
        if (enableDiags)
        {
            params.get("groupNames", m_diagsNames);
            params.get_or_set("saveInterval", m_defaultSaveInterval);
            params.get_or_set("saveFolder", m_defaultSaveFolder);
        }
    }
    /** \brief Loop over diags in alldiags and call their InitDiags */
    void init_data (
        ComputationalDomain const& compDom,
        std::map<std::string, amrex::MultiFab const> const& fieldMap,
        std::map<std::string, amrex::Real const> const& fieldsScaling,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vDim, nData>>> const& particles)
    {
        BL_PROFILE("Gempic::Io::MultiDiagnostics::init_data()");
        for (auto& diag : m_alldiags)
        {
            diag->init_data(compDom, fieldMap, fieldsScaling, particles);
        }
    }
    /** \brief Called at each iteration. Compute diags and flush. */
    void filter_compute_pack_flush (int step, amrex::Real simTime, bool forceFlush = false)
    {
        BL_PROFILE("Gempic::Io::MultiDiagnostics::filter_compute_pack_flush()");
        for (auto& diag : m_alldiags)
        {
            diag->filter_compute_pack_flush(step, simTime, forceFlush);
        }
    }
    /** \brief Called only at the last iteration. Loop over each diag and if m_dump_last_timestep
     *         is true, compute diags and flush with force_flush=true. */
    void filter_compute_pack_flush_last_timestep (int step, amrex::Real simTime)
    {
        BL_PROFILE("Gempic::Io::MultiDiagnostics::filter_compute_pack_flush_last_timestep()");
        for (auto& diag : m_alldiags)
        {
            if (diag->DoDumpLastTimestep())
            {
                constexpr bool forceFlush = true;
                diag->FilterComputePackFlush(step, simTime, forceFlush);
            }
        }
    }

    FullDiagnostics<vDim, nData>& get_diag (int idiag) { return *m_alldiags[idiag]; }
    int get_total_diags () { return m_ndiags; }
    int get_num_group_members (int idiag) { return m_alldiags[idiag]->get_num_group_members(); }

private:
    /** Vector of pointers to all diagnostics */
    amrex::Vector<std::unique_ptr<FullDiagnostics<vDim, nData>>> m_alldiags;
    int m_ndiags = 0; /**< number of different diagnostics */
    std::vector<std::string> m_diagsNames;
    int m_defaultSaveInterval{100}; // default save interval for all full diagnostics
    std::string m_defaultSaveFolder{"FullDiagnostics"};
};

} // namespace Gempic::Io

#endif // GEMPICX_MULTIDIAGNOSTICS_H_

/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGNOSTICS_MULTIREDUCEDDIAGNOSTICS_H_
#define REDUCEDDIAGNOSTICS_MULTIREDUCEDDIAGNOSTICS_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "GEMPIC_CurrentFieldEnergy.H"
#include "GEMPIC_ElecFieldEnergy.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GaussError.H"
#include "GEMPIC_MagFieldEnergy.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleDiags.H"
#include "GEMPIC_ReducedDiagnostics.H"

namespace Gempic::Io
{
/**
 * This class holds multiple instances of ReducedDiagnostics, and contains
 * general functions to initialize, compute, and write these diagnostics
 * to file.
 */
template <int vdim, int degx, int degy, int degz, int ndata = 1>
class MultiReducedDiagnostics
{
public:
    /// whether or not reduced diagnostics are activated
    bool m_plotRd = false;

    /// names of reduced diagnostics groups
    std::vector<std::string> m_rdNames;

    /// save frequency
    std::vector<int> m_saveIntervals = {1};

    /// save location
    std::vector<std::string> m_saveFolder = {"ReducedDiagnostics"};

    /// output extension (default)
    std::string m_extension = "txt";

    /// create fields needed for computations if they are missing
    std::vector<bool> m_computeMissingFields = {false};

    /// m_multi_rd stores a pointer to each reduced diagnostics
    std::vector<std::vector<std::unique_ptr<ReducedDiagnostics<vdim, degx, degy, degz, ndata>>>>
        m_multiRds;

    /// constructor
    MultiReducedDiagnostics(std::shared_ptr<Forms::DeRhamComplex> drc)
    {
        BL_PROFILE("Gempic::Io::MultiReduceDiagnostics()");
        read_params();
        if (!m_plotRd)
        {
            return;
        }
        init_data(drc);
    }
    // end constructor

    /** Loop over all ReducedDiagnostics and call their InitData
     */
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc)
    {
        BL_PROFILE("Gempic::Io::MultiReduceDiagnostics::init_data()");
        if (!m_plotRd)
        {
            return;
        }
        // loop over all reduced diags
        for (size_t iRd{0}; iRd < m_rdNames.size(); ++iRd)
        {
            for (size_t iType{0}; iType < m_multiRds[iRd].size(); ++iType)
            {
                m_multiRds[iRd][iType]->init_data(drc, m_computeMissingFields[iRd]);
            }
        }
    }

    /** Redistribute parallel data structures during load balance
     */
    void load_balance ()
    {
        BL_PROFILE("Gempic::Io::MultiReduceDiagnostics::load_balance()");
        // loop over all reduced diags
        for (size_t iRd{0}; iRd < m_rdNames.size(); ++iRd)
        {
            for (size_t iType{0}; iType < m_multiRds[iRd].size(); ++iType)
            {
                m_multiRds[iRd][iType]->LoadBalance();
            }
        }
    }

    /** Loop over all ReducedDiagnostics and call their ComputeDiags */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>> const& partGr)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::compute_diags()");
        if (!m_plotRd)
        {
            return;
        }

        // loop over all reduced diags
        for (size_t iRd{0}; iRd < m_rdNames.size(); ++iRd)
        {
            for (size_t iType{0}; iType < m_multiRds[iRd].size(); ++iType)
            {
                m_multiRds[iRd][iType]->compute_diags(infra, partGr);
            }
        }
    }

    /** Loop over all ReducedDiagnostics and call their write_to_file */
    void write_to_file (int step, amrex::Real simTime)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::write_to_file()");
        if (!m_plotRd)
        {
            return;
        }

        // loop over all reduced diags
        for (size_t iRd = 0; iRd < m_rdNames.size(); ++iRd)
        {
            if ((step) % m_saveIntervals[iRd] == 0)
            {
                for (size_t iType{0}; iType < m_multiRds[iRd].size(); ++iType)
                {
                    m_multiRds[iRd][iType]->write_to_file(step, simTime);
                }
            }
        }
    }

    /** Loop over all ReducedDiagnostics, update and call their write_to_file */
    void compute_and_write_to_file (
        int step,
        amrex::Real simTime,
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>> const& partGr,
        bool forceWrite = false)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::update_and_write_to_file()");
        if (!m_plotRd)
        {
            return;
        }
        // loop over all reduced diags
        for (size_t iRd = 0; iRd < m_rdNames.size(); ++iRd)
        {
            if (forceWrite || (step) % m_saveIntervals[iRd] == 0)
            {
                for (size_t iType{0}; iType < m_multiRds[iRd].size(); ++iType)
                {
                    m_multiRds[iRd][iType]->compute_diags(infra, partGr);
                    m_multiRds[iRd][iType]->write_to_file(step, simTime);
                }
            }
        }
    }

private:
    void read_params ()
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::read_params()");
        // read reduced diags names
        Parameters ppGempic("ReducedDiagnostics");
        m_plotRd = !ppGempic.get_or_set("groupNames", m_rdNames);
        ppGempic.get_or_set("enable", m_plotRd);

        // if names are not given (parameter was set, not read) or enable is off, reduced diags will
        // not be done
        if (!m_plotRd)
        {
            return;
        }

        m_multiRds.resize(m_rdNames.size());
        ppGempic.get_or_set("saveInterval", m_saveIntervals[0]);
        m_saveIntervals.resize(m_rdNames.size(), m_saveIntervals[0]);

        bool defaultCompute{m_computeMissingFields[0]};
        ppGempic.get_or_set("computeMissingFields", defaultCompute);
        m_computeMissingFields.resize(m_rdNames.size(), defaultCompute);

        std::string saveFolder{"ReducedDiagnostics"};
        ppGempic.get_or_set("saveFolder", saveFolder);

        std::string extension{"txt"};
        ppGempic.get_or_set("extension", extension);

        std::filesystem::path fileName{saveFolder};
        // user may or may not have provided a '/'
        fileName.has_parent_path() ? fileName : fileName += "/";
        for (size_t idx{0}; idx < m_rdNames.size(); ++idx)
        {
            Parameters ppRdName("ReducedDiagnostics." + m_rdNames[idx]);
            ppRdName.get_or_set("saveInterval", m_saveIntervals[idx]);
            bool groupDefaultCompute{m_computeMissingFields[idx]};
            ppRdName.get_or_set("computeMissingFields", groupDefaultCompute);
            m_computeMissingFields[idx] = groupDefaultCompute;

            fileName.replace_filename(m_rdNames[idx]);
            fileName.replace_extension(extension);

            // read reduced diags types
            std::vector<std::string> rdTypes;
            ppRdName.get("types", rdTypes);

            size_t const numTypes{rdTypes.size()};
            int indexNum{0};
            size_t typeNum{0};
            // loop over all reduced diags and fill m_multi_rd with requested reduced diags
            std::transform(rdTypes.begin(), rdTypes.end(), std::back_inserter(m_multiRds[idx]),
                           [&] (auto const& rdType)
                           {
                               bool isLastType{++typeNum == numTypes};
                               return get_diagnostics_type(rdType, m_rdNames[idx], indexNum,
                                                           isLastType, fileName.string());
                           });
        }
    }

    std::unique_ptr<ReducedDiagnostics<vdim, degx, degy, degz, ndata>> get_diagnostics_type (
        std::string const& diagType,
        std::string const& name,
        int& indexNum,
        bool isLastType,
        std::string const& fileName)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::get_diagnostics_type()");
        if (diagType == "ElecFieldEnergy")
        {
            using ElecEnergy = ElecFieldEnergy<vdim, degx, degy, degz, ndata>;
            return std::make_unique<ElecEnergy>(name, indexNum, isLastType, fileName);
        }
        else if (diagType == "MagFieldEnergy")
        {
            using MagEnergy = MagFieldEnergy<vdim, degx, degy, degz, ndata>;
            return std::make_unique<MagEnergy>(name, indexNum, isLastType, fileName);
        }
        else if (diagType == "CurrentFieldEnergy")
        {
            using CurrentEnergy = CurrentFieldEnergy<vdim, degx, degy, degz, ndata>;
            return std::make_unique<CurrentEnergy>(name, indexNum, isLastType, fileName);
        }
        else if (diagType == "GaussError")
        {
            using GaussErr = GaussError<vdim, degx, degy, degz, ndata>;
            return std::make_unique<GaussErr>(name, indexNum, isLastType, fileName);
        }
        else if (diagType == "Particle")
        {
            using Part = Particle<vdim, degx, degy, degz, ndata>;
            return std::make_unique<Part>(name, indexNum, isLastType, fileName);
        }
        else if (diagType == "ParticleLinVlasov")
        {
            using Part = ParticleLinVlasov<vdim, degx, degy, degz, ndata>;
            return std::make_unique<Part>(name, indexNum, isLastType, fileName);
        }
        else
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                false, diagType + " is not a valid type for reduced diagnostic in group " + name);
            // this is never reached
            return std::make_unique<ElecFieldEnergy<vdim, degx, degy, degz, ndata>>(
                name, indexNum, isLastType, fileName);
        }
    }
};

} // namespace Gempic::Io

#endif

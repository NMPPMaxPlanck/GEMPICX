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
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata = 1>
class MultiReducedDiagnostics
{
public:
    /// Bool: whether or not reduced diagnostics are activated
    int m_plotRd = 0;

    /// names of reduced diagnostics
    std::vector<std::string> m_rdNames;

    /// save freq
    std::vector<int> m_saveIntervals = {1};

    /// create fields needed for computations if they are missing
    bool m_computeMissingFields = false;

    /// m_multi_rd stores a pointer to each reduced diagnostics
    std::vector<std::vector<
        std::unique_ptr<ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>>>>
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
                m_multiRds[iRd][iType]->init_data(drc, m_computeMissingFields);
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
        ComputationalDomain& infra,
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        amrex::GpuArray<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>, numspec>&
            partGr)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::compute_diags()");
        if (!m_plotRd)
        {
            return;
        }
        // GEMPIC_PROFILE("MultiReducedDiagnostics::ComputeDiags()");

        // loop over all reduced diags
        for (size_t iRd{0}; iRd < m_rdNames.size(); ++iRd)
        {
            for (size_t iType{0}; iType < m_multiRds[iRd].size(); ++iType)
            {
                m_multiRds[iRd][iType]->compute_diags(infra, fieldsDiagnostics, partGr);
            }
        }
        // end loop over all reduced diags
    }

    /** Loop over all ReducedDiagnostics and call their WriteToFile */
    void write_to_file (int step, amrex::Real dt)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::write_to_file()");
        // Only the I/O rank does
        if (!amrex::ParallelDescriptor::IOProcessor() || !m_plotRd)
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
                    // call the write to file function
                    m_multiRds[iRd][iType]->write_to_file(dt, step);
                }
            }
        }
        // end loop over all reduced diags
    }

private:
    void read_params ()
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::read_params()");
        // read reduced diags names
        Parameters ppGempic("ReducedDiagnostics");
        ppGempic.get_or_set("computeMissingFields", m_computeMissingFields);
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

        for (size_t idx{0}; idx < m_rdNames.size(); ++idx)
        {
            Parameters ppRdName("ReducedDiagnostics." + m_rdNames[idx]);
            ppRdName.get_or_set("saveInterval", m_saveIntervals[idx]);

            // read reduced diags types
            std::vector<std::string> rdTypes;
            ppRdName.get("types", rdTypes);

            const size_t numTypes{rdTypes.size()};
            int indexNum{0};
            size_t typeNum{0};
            // loop over all reduced diags and fill m_multi_rd with requested reduced diags
            std::transform(rdTypes.begin(), rdTypes.end(), std::back_inserter(m_multiRds[idx]),
                           [&] (const auto& rdType)
                           {
                               bool isLastType{++typeNum == numTypes};
                               return get_diagnostics_type(rdType, m_rdNames[idx], indexNum,
                                                           isLastType);
                           });
        }
        // end loop over all reduced diags
    }

    std::unique_ptr<ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>>
    get_diagnostics_type (const std::string& diagType,
                          const std::string& name,
                          int& indexNum,
                          bool isLastType)
    {
        BL_PROFILE("Gempic::Io::MultiReducedDiagnostics::get_diagnostics_type()");
        if (diagType == "ElecFieldEnergy")
        {
            using ElecEnergy = ElecFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>;
            return std::make_unique<ElecEnergy>(name, indexNum, isLastType);
        }
        else if (diagType == "MagFieldEnergy")
        {
            using MagEnergy = MagFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>;
            return std::make_unique<MagEnergy>(name, indexNum, isLastType);
        }
        else if (diagType == "GaussError")
        {
            using GaussErr = GaussError<vdim, numspec, degx, degy, degz, degree, ndata>;
            return std::make_unique<GaussErr>(name, indexNum, isLastType);
        }
        else if (diagType == "Particle")
        {
            using Part = Particle<vdim, numspec, degx, degy, degz, degree, ndata>;
            return std::make_unique<Part>(name, indexNum, isLastType);
        }
        else
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                false, diagType + " is not a valid type for reduced diagnostic in group " + name);
            // this is never reached
            return std::make_unique<
                ElecFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>>(name, indexNum,
                                                                                 isLastType);
        }
    }
};

}  // namespace Gempic::Io

#endif

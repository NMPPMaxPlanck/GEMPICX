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

#include "ReducedDiagnostics.H"

#include "ElecFieldEnergy.H"
#include "MagFieldEnergy.H"
#include "Particle.H"
#include "GaussError.H"
#include "GEMPIC_parameters.H"
#include "GEMPIC_Fields.H"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
/**
 * This class holds multiple instances of ReducedDiagnostics, and contains
 * general functions to initialize, compute, and write these diagnostics
 * to file.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata=1>
class MultiReducedDiagnostics
{
public:

    /// Bool: whether or not reduced diagnostics are activated
    int m_plot_rd = 0;

    /// names of reduced diagnostics
    std::vector<std::string> m_rd_names;

    /// save freq
    std::vector<int> m_save_intervals = {1};

    /// create fields needed for computations if they are missing
    bool m_computeMissingFields = false;

    /// m_multi_rd stores a pointer to each reduced diagnostics
    std::vector<std::vector<std::unique_ptr<ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>>>> m_multi_rds;

    /// constructor
    MultiReducedDiagnostics (std::shared_ptr<GEMPIC_Fields::DeRhamComplex> drc)
    {
        readParams();
        if (!m_plot_rd) {return;}
        InitData(drc);
    }
    // end constructor

    /** Loop over all ReducedDiagnostics and call their InitData
     */
    void InitData (std::shared_ptr<GEMPIC_Fields::DeRhamComplex> drc)
    {
        if (!m_plot_rd) {return;}
        // loop over all reduced diags
        for (size_t i_rd{0}; i_rd < m_rd_names.size(); ++i_rd)
        {
            for (size_t i_type{0}; i_type < m_multi_rds[i_rd].size(); ++i_type)
            {
                m_multi_rds[i_rd][i_type] -> InitData(drc, m_computeMissingFields);
            }
        }
    }

    /** Redistribute parallel data structures during load balance
     */
    void LoadBalance ()
    {
        // loop over all reduced diags
        for (size_t i_rd{0}; i_rd < m_rd_names.size(); ++i_rd)
        {
            for (size_t i_type{0}; i_type < m_multi_rds[i_rd].size(); ++i_type)
            {
                m_multi_rds[i_rd][i_type] -> LoadBalance();
            }
        }
    }

    /** Loop over all ReducedDiagnostics and call their ComputeDiags */
    void ComputeDiags(Gempic::CompDom::computational_domain& infra, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> &part_gr)
    {
        if (!m_plot_rd) {return;}
        //GEMPIC_PROFILE("MultiReducedDiagnostics::ComputeDiags()");

        // loop over all reduced diags
        for (size_t i_rd{0}; i_rd < m_rd_names.size(); ++i_rd)
        {
            for (size_t i_type{0}; i_type < m_multi_rds[i_rd].size(); ++i_type)
            {
                m_multi_rds[i_rd][i_type] -> ComputeDiags(infra, fieldsDiagnostics, part_gr);
            }
        }
        // end loop over all reduced diags
    }

    /** Loop over all ReducedDiagnostics and call their WriteToFile */
    void WriteToFile (int step, amrex::Real dt)
    {
        // Only the I/O rank does
        if ( !amrex::ParallelDescriptor::IOProcessor() || !m_plot_rd) { return; }

        // loop over all reduced diags
        for (size_t i_rd = 0; i_rd < m_rd_names.size(); ++i_rd)
        {
            if((step) % m_save_intervals[i_rd] == 0)
            {
                for (size_t i_type{0}; i_type < m_multi_rds[i_rd].size(); ++i_type)
                {
                    // call the write to file function
                    m_multi_rds[i_rd][i_type]->WriteToFile(dt, step);
                }
            }
        }
        // end loop over all reduced diags
    }

    private:
    void readParams()
    {
        // read reduced diags names
        Parameters pp_gempic("reducedDiagnostics");
        pp_gempic.getOrSet("compute_missing_fields", m_computeMissingFields);
        m_plot_rd = !pp_gempic.getOrSet("group_names", m_rd_names);
        pp_gempic.getOrSet("enable", m_plot_rd);

        // if names are not given (parameter was set, not read) or enable is off, reduced diags will not be done
        if ( !m_plot_rd) { return; }

        using CS = const std::string& ;
        const auto reduced_diags_dictionary =
            std::map<std::string, std::function<std::unique_ptr<ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>>(CS)>>{
                {"ElecFieldEnergy",           [](CS s){return std::make_unique<ElecFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                {"MagFieldEnergy",           [](CS s){return std::make_unique<MagFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                {"GaussError",           [](CS s){return std::make_unique<GaussError<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                {"Particle",         [](CS s){return std::make_unique<Particle<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}}
            };

        m_multi_rds.resize(m_rd_names.size());
        pp_gempic.getOrSet("save_interval", m_save_intervals[0]);
        m_save_intervals.resize(m_rd_names.size(), m_save_intervals[0]);

        for (size_t idx{0}; idx < m_rd_names.size(); ++idx)
        {    
        Parameters pp_rd_name("reducedDiagnostics." + m_rd_names[idx]);
        pp_rd_name.getOrSet("save_interval", m_save_intervals[idx]);

        // read reduced diags types
        std::vector<std::string> rd_types;
        pp_rd_name.get("types", rd_types);
        
        // loop over all reduced diags and fill m_multi_rd with requested reduced diags
        std::transform(rd_types.begin(), rd_types.end(), std::back_inserter(m_multi_rds[idx]),
            [&](const auto& rd_type){
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    reduced_diags_dictionary.count(rd_type) != 0,
                    rd_type + " is not a valid type for reduced diagnostic in group " + m_rd_names[idx]
                );

                return reduced_diags_dictionary.at(rd_type)(m_rd_names[idx]);
            });
        }
        // end loop over all reduced diags
    }
};

#endif

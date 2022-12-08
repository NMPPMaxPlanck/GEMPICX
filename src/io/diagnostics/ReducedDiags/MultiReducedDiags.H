/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef REDUCEDDIAGS_MULTIREDUCEDDIAGS_H_
#define REDUCEDDIAGS_MULTIREDUCEDDIAGS_H_

#include "ReducedDiags.H"

#include "ElecFieldEnergy.H"
#include "MagFieldEnergy.H"
#include "Momentum.H"
#include "ParticleEnergy.H"
#include "GaussError.H"

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
class MultiReducedDiags
{
public:

    /// Bool: whether or not reduced diagnostics are activated
    int m_plot_rd = 1;

    /// names of reduced diagnostics
    std::vector<std::string> m_rd_names;

    /// save freq
    int save_reduced = 1;

    /// m_multi_rd stores a pointer to each reduced diagnostics
    std::vector<std::unique_ptr<ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>>> m_multi_rd;

    /// constructor
    MultiReducedDiags ()
    {
        // read reduced diags names
        amrex::ParmParse pp_gempic("gempic");
        m_plot_rd = pp_gempic.queryarr("reduced_diags_names", m_rd_names);
        pp_gempic.query("save_reduced", save_reduced);
        
        // if names are not given, reduced diags will not be done
        if ( m_plot_rd == 0 ) { return; }
        using CS = const std::string& ;
        const auto reduced_diags_dictionary =
            std::map<std::string, std::function<std::unique_ptr<ReducedDiags<vdim, numspec, degx, degy, degz, degree, ndata>>(CS)>>{
                {"ParticleEnergy",        [](CS s){return std::make_unique<ParticleEnergy<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                //{"ParticleMomentum",      [](CS s){return std::make_unique<ParticleMomentum>(s);}},
                {"ElecFieldEnergy",           [](CS s){return std::make_unique<ElecFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                {"MagFieldEnergy",           [](CS s){return std::make_unique<MagFieldEnergy<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                {"GaussError",           [](CS s){return std::make_unique<GaussError<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}},
                {"Momentum",         [](CS s){return std::make_unique<Momentum<vdim, numspec, degx, degy, degz, degree, ndata>>(s);}}//,
                /*{"FieldMaximum",          [](CS s){return std::make_unique<FieldMaximum>(s);}},
                {"FieldProbe",            [](CS s){return std::make_unique<FieldProbe>(s);}},
                {"FieldReduction",        [](CS s){return std::make_unique<FieldReduction>(s);}},
                {"RhoMaximum",            [](CS s){return std::make_unique<RhoMaximum>(s);}},
                {"BeamRelevant",          [](CS s){return std::make_unique<BeamRelevant>(s);}},
                {"LoadBalanceCosts",      [](CS s){return std::make_unique<LoadBalanceCosts>(s);}},
                {"LoadBalanceEfficiency", [](CS s){return std::make_unique<LoadBalanceEfficiency>(s);}},
                {"ParticleHistogram",     [](CS s){return std::make_unique<ParticleHistogram>(s);}},
                {"ParticleNumber",        [](CS s){return std::make_unique<ParticleNumber>(s);}},
                {"ParticleExtrema",       [](CS s){return std::make_unique<ParticleExtrema>(s);}}*/
            };
        // loop over all reduced diags and fill m_multi_rd with requested reduced diags
        std::transform(m_rd_names.begin(), m_rd_names.end(), std::back_inserter(m_multi_rd),
            [&](const auto& rd_name){
                amrex::ParmParse pp_rd_name(rd_name);

                // read reduced diags type
                std::string rd_type;
                pp_rd_name.get("type", rd_type);

                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    reduced_diags_dictionary.count(rd_type) != 0,
                    rd_type + " is not a valid type for reduced diagnostic " + rd_name
                );

                return reduced_diags_dictionary.at(rd_type)(rd_name);
            });
        // end loop over all reduced diags
    }
    // end constructor

    /** Loop over all ReducedDiags and call their InitData
     */
    void InitData ()
    {
        // loop over all reduced diags
        for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
        {
            m_multi_rd[i_rd] -> InitData();
        }
    }

    /** Redistribute parallel data structures during load balance
     */
    void LoadBalance ()
    {
        // loop over all reduced diags
        for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
        {
            m_multi_rd[i_rd] -> LoadBalance();
        }
    }

    /** Loop over all ReducedDiags and call their ComputeDiags
     *  @param[in] step current iteration time */
    void ComputeDiags (int step, computational_domain infra, maxwell_yee<vdim> *mw_yee,
                        amrex::GpuArray<std::unique_ptr<particle_groups<vdim, ndata>>, numspec> &part_gr)
    {
        //WARPX_PROFILE("MultiReducedDiags::ComputeDiags()");

        // loop over all reduced diags
        for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
        {
            m_multi_rd[i_rd] -> ComputeDiags(step, infra, mw_yee, part_gr);
        }
        // end loop over all reduced diags
    }

    /** Loop over all ReducedDiags and call their WriteToFile
     *  @param[in] step current iteration time */
    void WriteToFile (int step, maxwell_yee<vdim> *mw_yee)
    {
        // Only the I/O rank does
        if ( !amrex::ParallelDescriptor::IOProcessor() ) { return; }

        // loop over all reduced diags
        if((step+1) % save_reduced == 0)
        {
            for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
            {
                // call the write to file function
                m_multi_rd[i_rd]->WriteToFile(mw_yee->dt, step+1);
            }
        }
        // end loop over all reduced diags
    }
};

#endif

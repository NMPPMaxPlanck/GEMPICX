/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGS_CURRENTFIELDENERGY_H_
#define REDUCEDDIAGS_CURRENTFIELDENERGY_H_

#include <string>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_Profiling.H"
#include "GEMPIC_ReducedDiagnostics.H"

namespace Gempic::Io
{
/**
 *  This class mainly contains a function that
 *  computes the field energy.
 */
template <int vdim, int degx, int degy, int degz, int ndata>
class CurrentFieldEnergy : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    std::unique_ptr<Forms::DeRhamField<Grid::primal, Space::edge>> m_jField;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::face>> m_jFieldT;
    std::unique_ptr<Forms::DeRhamField<Grid::primal, Space::edge> const> m_constJField;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::face> const> m_constJFieldT;

    enum class ComputeMethod
    {
        missingNone,
        missingJField,
        missingJFieldT
    };

    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    CurrentFieldEnergy(std::string const& rdName,
                       int& indexNum,
                       bool isLastType,
                       std::string const& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName)
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy()");
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if (this->m_isNotRestart)
            {
                // open file
                std::ofstream ofs{this->m_path, std::ofstream::out | std::ofstream::app};
                // write header row
                if (this->m_isFirstType)
                {
                    ofs << "#";
                    ofs << "[" << ++indexNum << "]step()";
                    ofs << this->m_sep;
                    ofs << "[" << ++indexNum << "]time(s)";
                }
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]JFieldTx^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]JFieldTy^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]JFieldTz^2/2" + std::string("(J)");
                if (this->m_isLastType)
                {
                    ofs << std::endl;
                }
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * function to initialize data after amr levels are initialized.
     */
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc, bool cmf) final
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, cmf);
        m_computeMethod = choose_computation_method(this->m_deRham->m_fieldsDiagnostics);
    }

    /**
     * This function computes the current field energy component wise:
     */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>> const&) final
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_constJField, *m_constJFieldT, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingJFieldT:
                this->m_deRham->hodge(*m_jFieldT, *m_constJField);
                *m_jFieldT *= m_weightFunc;
                update_data(*m_constJField, *m_jFieldT, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingJField:
                this->m_deRham->hodge(*m_jField, *m_constJFieldT);
                *m_jField *= m_weightFunc;
                update_data(*m_jField, *m_constJFieldT, infra.m_geom.periodicity());
                return;
        }
    }
    // end void CurrentFieldEnergy::ComputeDiags

private:
    ComputeMethod m_computeMethod;
    amrex::Parser m_weightParser;
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> m_weightFunc;
    /**
     * Selects computation method based on missing fields and the input file switch
     * compute_missing_fields
     */
    ComputeMethod choose_computation_method (std::map<std::string, amrex::MultiFab const>& fieldMap)
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::choose_computation_method()");
        using namespace Forms;
        if (fieldMap.count("JFieldTx") && fieldMap.count("JFieldTy") &&
            fieldMap.count("JFieldTz") && fieldMap.count("JFieldx") && fieldMap.count("JFieldy") &&
            fieldMap.count("JFieldz"))
        {
            this->m_constJField = std::make_unique<DeRhamField<Grid::primal, Space::edge> const>(
                this->m_deRham, fieldMap["JFieldx"], fieldMap["JFieldy"], fieldMap["JFieldz"]);
            this->m_constJFieldT = std::make_unique<DeRhamField<Grid::dual, Space::face> const>(
                this->m_deRham, fieldMap["JFieldTx"], fieldMap["JFieldTy"], fieldMap["JFieldTz"]);
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldMap.count("JFieldx") && fieldMap.count("JFieldy") &&
                fieldMap.count("JFieldz") &&
                (!fieldMap.count("JFieldTx") || !fieldMap.count("JFieldTy") ||
                 !fieldMap.count("JFieldTz")))
            {
                auto [weightParser, weightFunc] = Utils::parse_function("DensityField");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int nComps{fieldMap["JFieldx"].nComp()};
                this->m_constJField =
                    std::make_unique<DeRhamField<Grid::primal, Space::edge> const>(
                        this->m_deRham, fieldMap["JFieldx"], fieldMap["JFieldy"],
                        fieldMap["JFieldz"]);

                this->m_jFieldT =
                    std::make_unique<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);

                return ComputeMethod::missingJFieldT;
            }
            else if (fieldMap.count("JFieldTx") && fieldMap.count("JFieldTy") &&
                     fieldMap.count("JFieldTz") &&
                     (!fieldMap.count("JFieldx") || !fieldMap.count("JFieldy") ||
                      !fieldMap.count("JFieldz")))
            {
                auto [weightParser, weightFunc] = Utils::parse_function("DensityFieldInv");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int nComps{fieldMap["JFieldTx"].nComp()};
                this->m_constJFieldT = std::make_unique<DeRhamField<Grid::dual, Space::face> const>(
                    this->m_deRham, fieldMap["JFieldTx"], fieldMap["JFieldTy"],
                    fieldMap["JFieldTz"]);

                this->m_jField = std::make_unique<DeRhamField<Grid::primal, Space::edge>>(
                    this->m_deRham, nComps);

                return ComputeMethod::missingJField;
            }
            else
            {
                GEMPIC_ERROR(
                    "Neither JField nor JFieldT was found: Either JFieldx, JFieldy and JFieldz or "
                    "JFieldTx, JFieldTy and JFieldTz are "
                    "needed to compute electric field "
                    "energy when computeMissingFields is set");
            }
        }
        else
        {
            GEMPIC_ERROR(
                "JField or JFieldT not found: JFieldx, JFieldy, JFieldz, JFieldTx, JFieldTy and "
                "JFieldTz are all needed to "
                "compute electric field energy when "
                "computeMissingFields is not set");
        }
        return ComputeMethod::missingNone; // never reached
    }

    /**
     * This function computes the current field energy component wise:
     */
    void update_data (Forms::DeRhamField<Grid::primal, Space::edge> const& fieldJ,
                      Forms::DeRhamField<Grid::dual, Space::face> const& fieldJT,
                      amrex::Periodicity const& periodicity)
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::update_data()");
        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        for (int dir = 0; dir < noutputs; dir++)
        {
            auto mask = amrex::OwnerMask(fieldJ.m_data[dir], periodicity);
            this->m_data[dir] =
                0.5 * amrex::MultiFab::Dot(*mask, fieldJ.m_data[dir], 0, fieldJT.m_data[dir], 0,
                                           fieldJ.m_data[dir].nComp(), 0, 0);
        }
    }
};

} // namespace Gempic::Io

#endif

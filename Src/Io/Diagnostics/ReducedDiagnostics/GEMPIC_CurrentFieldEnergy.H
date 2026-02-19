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
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_FieldRegistry.H"
#include "GEMPIC_FunctionParse.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Particle.H"
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
    std::unique_ptr<Forms::DeRhamComplex::Primal1Form> m_jField;
    std::unique_ptr<Forms::DeRhamComplex::Dual2Form> m_jFieldT;
    std::unique_ptr<Forms::DeRhamComplex::Primal1Form const> m_constJField;
    std::unique_ptr<Forms::DeRhamComplex::Dual2Form const> m_constJFieldT;

    enum class ComputeMethod
    {
        missingAll,
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
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc,
                    Io::FieldRegistry const& reg,
                    bool cmf) final
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, reg, cmf);
        m_computeMethod = choose_computation_method(reg);
    }

    /**
     * This function computes the current field energy component wise:
     */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>> const&) final
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_constJField, *m_constJFieldT, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingJFieldT:
                hodge(*m_jFieldT, *m_constJField);
                *m_jFieldT *= m_weightFunc;
                update_data(*m_constJField, *m_jFieldT, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingJField:
                hodge(*m_jField, *m_constJFieldT);
                *m_jField *= m_weightFunc;
                update_data(*m_jField, *m_constJFieldT, infra.m_geom.periodicity());
                return;
            default:
                amrex::Abort(
                    "Gempic::Io::CurrentFieldEnergy::compute_diags(): ComputeMethod missing all "
                    "required fields!");
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
    ComputeMethod choose_computation_method (FieldRegistry const& registry)
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::choose_computation_method()");
        using namespace Forms;
        using DRC = Forms::DeRhamComplex;
        auto const* jPtr = get_field_ptr_or_null<DRC::Primal1Form>(registry, "JFieldx");
        auto const* jtPtr = get_field_ptr_or_null<DRC::Dual2Form>(registry, "JFieldTx");

        if (jtPtr && jPtr)
        {
            this->m_constJField = std::make_unique<DRC::Primal1Form>(this->m_deRham, *jPtr);
            this->m_constJFieldT = std::make_unique<DRC::Dual2Form>(this->m_deRham, *jtPtr);
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (jPtr && !jtPtr)
            {
                auto [weightParser, weightFunc] = Utils::parse_function("DensityField");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int nComps = jPtr->get_ncomp(xDir);
                this->m_constJField = std::make_unique<DRC::Primal1Form>(this->m_deRham, *jPtr);
                this->m_jFieldT = std::make_unique<DRC::Dual2Form>(this->m_deRham, nComps);
                return ComputeMethod::missingJFieldT;
            }
            else if (!jPtr && jtPtr)
            {
                auto [weightParser, weightFunc] = Utils::parse_function("DensityFieldInv");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;
                int nComps = jtPtr->get_ncomp(xDir);
                this->m_constJFieldT = std::make_unique<DRC::Dual2Form>(this->m_deRham, *jtPtr);
                this->m_jField = std::make_unique<DRC::Primal1Form>(this->m_deRham, nComps);
                return ComputeMethod::missingJField;
            }
            else
            {
                GEMPIC_ERROR(
                    "Neither JField nor JFieldT was found: Either JFieldx, JFieldy and JFieldz or "
                    "JFieldTx, JFieldTy and JFieldTz are "
                    "needed to compute electric field "
                    "energy when computeMissingFields is set");
                return ComputeMethod::missingAll;
            }
        }
        GEMPIC_ERROR(
            "JField or JFieldT not found: JFieldx, JFieldy, JFieldz, JFieldTx, JFieldTy and "
            "JFieldTz are all needed to "
            "compute electric field energy when "
            "computeMissingFields is not set");
        return ComputeMethod::missingAll; // never reached
    }

    /**
     * This function computes the current field energy component wise:
     */
    void update_data (Forms::DeRhamComplex::Primal1Form const& fieldJ,
                      Forms::DeRhamComplex::Dual2Form const& fieldJT,
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

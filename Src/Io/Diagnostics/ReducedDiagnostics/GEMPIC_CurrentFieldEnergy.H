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
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class CurrentFieldEnergy : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<Forms::DeRhamField<Grid::primal, Space::edge>> m_jField;
    std::shared_ptr<Forms::DeRhamField<Grid::dual, Space::face>> m_jFieldT;

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
     */
    CurrentFieldEnergy(std::string rdName, int& indexNum, bool isLastType) :
        ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(
            rdName, indexNum, isLastType)
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy()");
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if (this->m_isNotRestart)
            {
                // open file
                std::ofstream ofs{this->m_path + this->m_rdName + "." + this->m_extension,
                                  std::ofstream::out | std::ofstream::app};
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
     * This function computes the current field energy component wise:
     */
    void compute_diags (ComputationalDomain& infra,
                        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
                        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>,
                                        numspec>& partGr) final
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::compute_diags()");
        static ComputeMethod computeMethod{choose_computation_method(fieldsDiagnostics)};

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs);

        std::unique_ptr<amrex::iMultiFab> maskX;
        std::unique_ptr<amrex::iMultiFab> maskY;
        std::unique_ptr<amrex::iMultiFab> maskZ;

        switch (computeMethod)
        {
            case ComputeMethod::missingNone:
                maskX = amrex::OwnerMask(*fieldsDiagnostics["JFieldx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["JFieldx"], 0,
                                               *fieldsDiagnostics["JFieldTx"], 0,
                                               fieldsDiagnostics["JFieldx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["JFieldy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["JFieldy"], 0,
                                               *fieldsDiagnostics["JFieldTy"], 0,
                                               fieldsDiagnostics["JFieldy"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["JFieldz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["JFieldz"], 0,
                                               *fieldsDiagnostics["JFieldTz"], 0,
                                               fieldsDiagnostics["JFieldz"]->nComp(), 0, 0);
                return;
            case ComputeMethod::missingJFieldT:
                this->m_deRham->hodge(*m_jField, *m_jFieldT);
                *m_jFieldT *= m_weightFunc;
                maskX = amrex::OwnerMask(*fieldsDiagnostics["JFieldx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["JFieldx"], 0,
                                               m_jFieldT->m_data[xDir], 0,
                                               fieldsDiagnostics["JFieldx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["JFieldy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["JFieldy"], 0,
                                               m_jFieldT->m_data[yDir], 0,
                                               fieldsDiagnostics["JFieldy"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["JFieldz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["JFieldz"], 0,
                                               m_jFieldT->m_data[zDir], 0,
                                               fieldsDiagnostics["JFieldz"]->nComp(), 0, 0);
                return;
            case ComputeMethod::missingJField:
                this->m_deRham->hodge(*m_jFieldT, *m_jField);
                *m_jField *= m_weightFunc;
                maskX =
                    amrex::OwnerMask(*fieldsDiagnostics["JFieldTx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["JFieldTx"], 0,
                                               m_jField->m_data[xDir], 0,
                                               fieldsDiagnostics["JFieldTx"]->nComp(), 0, 0);
                maskY =
                    amrex::OwnerMask(*fieldsDiagnostics["JFieldTy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["JFieldTy"], 0,
                                               m_jField->m_data[yDir], 0,
                                               fieldsDiagnostics["JFieldTy"]->nComp(), 0, 0);
                maskZ =
                    amrex::OwnerMask(*fieldsDiagnostics["JFieldTz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["JFieldTz"], 0,
                                               m_jField->m_data[zDir], 0,
                                               fieldsDiagnostics["JFieldTz"]->nComp(), 0, 0);
                return;
        }
    }
    // end void CurrentFieldEnergy::ComputeDiags

private:
    amrex::Parser m_weightParser;
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> m_weightFunc;
    /**
     * Selects computation method based on missing fields and the input file switch
     * compute_missing_fields
     */
    ComputeMethod choose_computation_method (
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        BL_PROFILE("Gempic::Io::CurrentFieldEnergy::choose_computation_method()");
        using namespace Forms;
        if (fieldsDiagnostics.count("JFieldTx") && fieldsDiagnostics.count("JFieldTy") &&
            fieldsDiagnostics.count("JFieldTz") && fieldsDiagnostics.count("JFieldx") &&
            fieldsDiagnostics.count("JFieldy") && fieldsDiagnostics.count("JFieldz"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldsDiagnostics.count("JFieldx") && fieldsDiagnostics.count("JFieldy") &&
                fieldsDiagnostics.count("JFieldz") &&
                (!fieldsDiagnostics.count("JFieldTx") || !fieldsDiagnostics.count("JFieldTy") ||
                 !fieldsDiagnostics.count("JFieldTz")))
            {
                auto [weightParser, weightFunc] = Utils::parse_function("DensityField");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int srcComp{0};
                int dstComp{0};
                int nComps{fieldsDiagnostics["JFieldx"]->nComp()};
                this->m_jField = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(
                    this->m_deRham, nComps);
                amrex::Copy(this->m_jField->m_data[xDir], *fieldsDiagnostics["JFieldx"], srcComp,
                            dstComp, nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_jField->m_data[yDir], *fieldsDiagnostics["JFieldy"], srcComp,
                            dstComp, nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_jField->m_data[zDir], *fieldsDiagnostics["JFieldz"], srcComp,
                            dstComp, nComps, this->m_deRham->get_n_ghost());

                this->m_jFieldT =
                    std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);

                return ComputeMethod::missingJFieldT;
            }
            else if (fieldsDiagnostics.count("JFieldTx") && fieldsDiagnostics.count("JFieldTy") &&
                     fieldsDiagnostics.count("JFieldTz") &&
                     (!fieldsDiagnostics.count("JFieldx") || !fieldsDiagnostics.count("JFieldy") ||
                      !fieldsDiagnostics.count("JFieldz")))
            {
                auto [weightParser, weightFunc] = Utils::parse_function("DensityFieldInv");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int srcComp{0};
                int dstComp{0};
                int nComps{fieldsDiagnostics["JFieldTx"]->nComp()};
                this->m_jFieldT =
                    std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);
                amrex::Copy(this->m_jFieldT->m_data[xDir], *fieldsDiagnostics["JFieldTx"], srcComp,
                            dstComp, nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_jFieldT->m_data[yDir], *fieldsDiagnostics["JFieldTy"], srcComp,
                            dstComp, nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_jFieldT->m_data[zDir], *fieldsDiagnostics["JFieldTz"], srcComp,
                            dstComp, nComps, this->m_deRham->get_n_ghost());

                this->m_jField = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(
                    this->m_deRham, nComps);

                return ComputeMethod::missingJField;
            }
            else
            {
                std::string str =
                    "Either JFieldx, JFieldy and JFieldz or JFieldTx, JFieldTy and JFieldTz are "
                    "needed to compute electric field "
                    "energy when computeMissingFields is set";
                amrex::Assert("Neither G nor J was found", "CurrentFieldEnergy.H", __LINE__,
                              str.c_str());
            }
        }
        else
        {
            std::string str =
                "JFieldx, JFieldy, JFieldz, JFieldTx, JFieldTy and JFieldTz are all needed to "
                "compute electric field energy when "
                "computeMissingFields is not set";
            amrex::Assert("JField or JFieldT not found", "CurrentFieldEnergy.H", __LINE__,
                          str.c_str());
        }
        return ComputeMethod::missingNone;  // never reached
    }
};

}  // namespace Gempic::Io

#endif

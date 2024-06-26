/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGS_COLDFIELDENERGY_H_
#define REDUCEDDIAGS_COLDFIELDENERGY_H_

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
class ColdFieldEnergy : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>> m_G;
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>> m_J;

    enum class ComputeMethod
    {
        missingNone,
        missingJ,
        missingG
    };

    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     */
    ColdFieldEnergy(std::string rdName, int& indexNum, bool isLastType) :
        ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(
            rdName, indexNum, isLastType)
    {
        BL_PROFILE("Gempic::Io::ColdFieldEnergy()");
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
                ofs << "[" << ++indexNum << "]Jx^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]Jy^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]Jz^2/2" + std::string("(J)");
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
     * This function computes the cold field energy component wise:
     */
    void compute_diags (ComputationalDomain& infra,
                        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
                        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>,
                                        numspec>& partGr) final
    {
        BL_PROFILE("Gempic::Io::ColdFieldEnergy::compute_diags()");
        static ComputeMethod computeMethod{choose_computation_method(fieldsDiagnostics)};

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs);

        std::unique_ptr<amrex::iMultiFab> maskX;
        std::unique_ptr<amrex::iMultiFab> maskY;
        std::unique_ptr<amrex::iMultiFab> maskZ;

        switch (computeMethod)
        {
            case ComputeMethod::missingNone:
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Gx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Gx"], 0,
                                               *fieldsDiagnostics["Jx"], 0,
                                               fieldsDiagnostics["Gx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Gy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Gy"], 0,
                                               *fieldsDiagnostics["Jy"], 0,
                                               fieldsDiagnostics["Gy"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Gz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Gz"], 0,
                                               *fieldsDiagnostics["Jz"], 0,
                                               fieldsDiagnostics["Gz"]->nComp(), 0, 0);
                return;
            case ComputeMethod::missingJ:
                this->m_deRham->hodge(*m_G, *m_J);
                *m_J *= m_weightFunc;
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Gx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Gx"], 0,
                                               m_J->m_data[xDir], 0,
                                               fieldsDiagnostics["Gx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Gy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Gy"], 0,
                                               m_J->m_data[yDir], 0,
                                               fieldsDiagnostics["Gy"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Gz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Gz"], 0,
                                               m_J->m_data[zDir], 0,
                                               fieldsDiagnostics["Gz"]->nComp(), 0, 0);
                return;
            case ComputeMethod::missingG:
                this->m_deRham->hodge(*m_J, *m_G);
                *m_G *= m_weightFunc;
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Jx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Jx"], 0,
                                               m_G->m_data[xDir], 0,
                                               fieldsDiagnostics["Jx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Jy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Jy"], 0,
                                               m_G->m_data[yDir], 0,
                                               fieldsDiagnostics["Jy"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Jz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Jz"], 0,
                                               m_G->m_data[zDir], 0,
                                               fieldsDiagnostics["Jz"]->nComp(), 0, 0);
                return;
        }
    }
    // end void ColdFieldEnergy::ComputeDiags

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
        BL_PROFILE("Gempic::Io::ColdFieldEnergy::choose_computation_method()");
        using namespace Forms;
        if (fieldsDiagnostics.count("Jx") && fieldsDiagnostics.count("Jy") &&
            fieldsDiagnostics.count("Jz") && fieldsDiagnostics.count("Gx") &&
            fieldsDiagnostics.count("Gy") && fieldsDiagnostics.count("Gz"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldsDiagnostics.count("Gx") && fieldsDiagnostics.count("Gy") &&
                fieldsDiagnostics.count("Gz") &&
                (!fieldsDiagnostics.count("Jx") || !fieldsDiagnostics.count("Jy") ||
                 !fieldsDiagnostics.count("Jz")))
            {
                auto [weightParser, weightFunc] = Utils::parse_function("ColdDensity");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int srcComp{0};
                int dstComp{0};
                int nComps{fieldsDiagnostics["Gx"]->nComp()};
                this->m_G = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham,
                                                                                     nComps);
                amrex::Copy(this->m_G->m_data[xDir], *fieldsDiagnostics["Gx"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_G->m_data[yDir], *fieldsDiagnostics["Gy"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_G->m_data[zDir], *fieldsDiagnostics["Gz"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());

                this->m_J =
                    std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);

                return ComputeMethod::missingJ;
            }
            else if (fieldsDiagnostics.count("Jx") && fieldsDiagnostics.count("Jy") &&
                     fieldsDiagnostics.count("Jz") &&
                     (!fieldsDiagnostics.count("Gx") || !fieldsDiagnostics.count("Gy") ||
                      !fieldsDiagnostics.count("Gz")))
            {
                auto [weightParser, weightFunc] = Utils::parse_function("ColdDensityInv");
                m_weightParser = weightParser;
                m_weightFunc = weightFunc;

                int srcComp{0};
                int dstComp{0};
                int nComps{fieldsDiagnostics["Jx"]->nComp()};
                this->m_J =
                    std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);
                amrex::Copy(this->m_J->m_data[xDir], *fieldsDiagnostics["Jx"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_J->m_data[yDir], *fieldsDiagnostics["Jy"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_J->m_data[zDir], *fieldsDiagnostics["Jz"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());

                this->m_G = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham,
                                                                                     nComps);

                return ComputeMethod::missingG;
            }
            else
            {
                std::string str =
                    "Either Gx, Gy and Gz or Jx, Jy and Jz are needed to compute electric field "
                    "energy when computeMissingFields is set";
                amrex::Assert("Neither G nor J was found", "ColdFieldEnergy.H", __LINE__,
                              str.c_str());
            }
        }
        else
        {
            std::string str =
                "Gx, Gy, Gz, Jx, Jy and Jz are all needed to compute electric field energy when "
                "computeMissingFields is not set";
            amrex::Assert("G or J not found", "ColdFieldEnergy.H", __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone;  // never reached
    }
};

}  // namespace Gempic::Io

#endif

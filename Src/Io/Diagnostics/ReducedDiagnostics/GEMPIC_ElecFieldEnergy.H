/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGS_ELECFIELDENERGY_H_
#define REDUCEDDIAGS_ELECFIELDENERGY_H_

#include <string>

#include "GEMPIC_ComputationalDomain.H"
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
class ElecFieldEnergy : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>> m_E;
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>> m_D;

    enum class ComputeMethod
    {
        missingNone,
        missingD,
        missingE
    };

    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     */
    ElecFieldEnergy(std::string rdName, int& indexNum, bool isLastType) :
        ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(
            rdName, indexNum, isLastType)
    {
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
                ofs << "[" << ++indexNum << "]Ex^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]Ey^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]Ez^2/2" + std::string("(J)");
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
     * This function computes the electric field energy (EF) component wise:
     * epref_ep0 is the vacuum permittivity
     */
    void compute_diags (ComputationalDomain& infra,
                        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
                        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>,
                                        numspec>& partGr) final
    {
        static ComputeMethod computeMethod{choose_computation_method(fieldsDiagnostics)};

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        std::unique_ptr<amrex::iMultiFab> maskX;
        std::unique_ptr<amrex::iMultiFab> maskY;
        std::unique_ptr<amrex::iMultiFab> maskZ;

        switch (computeMethod)
        {
            case ComputeMethod::missingNone:
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Ex"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Ex"], 0,
                                               *fieldsDiagnostics["Dx"], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Ey"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Ey"], 0,
                                               *fieldsDiagnostics["Dy"], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Ez"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Ez"], 0,
                                               *fieldsDiagnostics["Dz"], 0, 1, 0, 0);
                return;
            case ComputeMethod::missingD:
                this->m_deRham->hodge(*m_E, *m_D);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Ex"], infra.m_geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Ex"], 0,
                                                                m_D->m_data[xDir], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Ey"], infra.m_geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Ey"], 0,
                                                                m_D->m_data[yDir], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Ez"], infra.m_geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Ez"], 0,
                                                                m_D->m_data[zDir], 0, 1, 0, 0);
                return;
            case ComputeMethod::missingE:
                this->m_deRham->hodge(*m_D, *m_E);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Dx"], infra.m_geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Dx"], 0,
                                                                m_E->m_data[xDir], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Dy"], infra.m_geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Dy"], 0,
                                                                m_E->m_data[yDir], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Dz"], infra.m_geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Dz"], 0,
                                                                m_E->m_data[zDir], 0, 1, 0, 0);
                return;
        }
    }
    // end void ElecFieldEnergy::ComputeDiags

private:
    /**
     * Selects computation method based on missing fields and the input file switch
     * compute_missing_fields
     */
    ComputeMethod choose_computation_method (
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        using namespace Forms;
        if (fieldsDiagnostics.count("Dx") && fieldsDiagnostics.count("Dy") &&
            fieldsDiagnostics.count("Dz") && fieldsDiagnostics.count("Ex") &&
            fieldsDiagnostics.count("Ey") && fieldsDiagnostics.count("Ez"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldsDiagnostics.count("Ex") && fieldsDiagnostics.count("Ey") &&
                fieldsDiagnostics.count("Ez") &&
                (!fieldsDiagnostics.count("Dx") || !fieldsDiagnostics.count("Dy") ||
                 !fieldsDiagnostics.count("Dz")))
            {
                this->m_E =
                    std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham);
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                amrex::Copy(this->m_E->m_data[xDir], *fieldsDiagnostics["Ex"], srcComp, dstComp,
                            numComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_E->m_data[yDir], *fieldsDiagnostics["Ey"], srcComp, dstComp,
                            numComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_E->m_data[zDir], *fieldsDiagnostics["Ez"], srcComp, dstComp,
                            numComps, this->m_deRham->get_n_ghost());

                this->m_D = std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham);

                return ComputeMethod::missingD;
            }
            else if (fieldsDiagnostics.count("Dx") && fieldsDiagnostics.count("Dy") &&
                     fieldsDiagnostics.count("Dz") &&
                     (!fieldsDiagnostics.count("Ex") || !fieldsDiagnostics.count("Ey") ||
                      !fieldsDiagnostics.count("Ez")))
            {
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                this->m_D = std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham);
                amrex::Copy(this->m_D->m_data[xDir], *fieldsDiagnostics["Dx"], srcComp, dstComp,
                            numComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_D->m_data[yDir], *fieldsDiagnostics["Dy"], srcComp, dstComp,
                            numComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_D->m_data[zDir], *fieldsDiagnostics["Dz"], srcComp, dstComp,
                            numComps, this->m_deRham->get_n_ghost());

                this->m_E =
                    std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham);

                return ComputeMethod::missingE;
            }
            else
            {
                std::string str =
                    "Either Ex, Ey and Ez or Dx, Dy and Dz are needed to compute electric field "
                    "energy when computeMissingFields is set";
                amrex::Assert("Neither E nor D was found", "ElecFieldEnergy.H", __LINE__,
                              str.c_str());
            }
        }
        else
        {
            std::string str =
                "Ex, Ey, Ez, Dx, Dy and Dz are all needed to compute electric field energy when "
                "computeMissingFields is not set";
            amrex::Assert("E or D not found", "ElecFieldEnergy.H", __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone;  // never reached
    }
};

}  // namespace Gempic::Io

#endif

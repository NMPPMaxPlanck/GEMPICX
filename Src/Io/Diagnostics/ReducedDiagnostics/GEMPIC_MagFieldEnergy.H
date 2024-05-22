/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGS_MAGFIELDENERGY_H_
#define REDUCEDDIAGS_MAGFIELDENERGY_H_

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
class MagFieldEnergy : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::primal, Forms::Space::face>> m_B;
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::edge>> m_H;

    enum class ComputeMethod
    {
        missingNone,
        missingH,
        missingB
    };
    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     */
    MagFieldEnergy(std::string rdName, int& indexNum, bool isLastType) :
        ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(
            rdName, indexNum, isLastType)
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::MagFieldEnergy()");
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
                ofs << "[" << ++indexNum << "]Bx^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]By^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]Bz^2/2" + std::string("(J)");
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
     * This function computes the magnetic energy (EF) component wise
     */
    void compute_diags (ComputationalDomain& infra,
                        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
                        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>,
                                        numspec>& partGr) final
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::compute_diags()");
        static ComputeMethod computeMethod{choose_computation_method(fieldsDiagnostics)};

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        std::unique_ptr<amrex::iMultiFab> maskX;
        std::unique_ptr<amrex::iMultiFab> maskY;
        std::unique_ptr<amrex::iMultiFab> maskZ;

        switch (computeMethod)
        {
            case ComputeMethod::missingNone:
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Bx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Bx"], 0,
                                               *fieldsDiagnostics["Hx"], 0,
                                               fieldsDiagnostics["Bx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["By"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["By"], 0,
                                               *fieldsDiagnostics["Hy"], 0,
                                               fieldsDiagnostics["By"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Bz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Bz"], 0,
                                               *fieldsDiagnostics["Hz"], 0,
                                               fieldsDiagnostics["Bz"]->nComp(), 0, 0);
                return;
            case ComputeMethod::missingH:
                this->m_deRham->hodge(*m_B, *m_H);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Bx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Bx"], 0,
                                               m_H->m_data[xDir], 0,
                                               fieldsDiagnostics["Bx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["By"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["By"], 0,
                                               m_H->m_data[yDir], 0,
                                               fieldsDiagnostics["By"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Bz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Bz"], 0,
                                               m_H->m_data[zDir], 0,
                                               fieldsDiagnostics["Bz"]->nComp(), 0, 0);
                return;
            case ComputeMethod::missingB:
                this->m_deRham->hodge(*m_H, *m_B);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Hx"], infra.m_geom.periodicity());
                this->m_data[xDir] =
                    0.5 * amrex::MultiFab::Dot(*maskX, *fieldsDiagnostics["Hx"], 0,
                                               m_B->m_data[xDir], 0,
                                               fieldsDiagnostics["Hx"]->nComp(), 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Hy"], infra.m_geom.periodicity());
                this->m_data[yDir] =
                    0.5 * amrex::MultiFab::Dot(*maskY, *fieldsDiagnostics["Hy"], 0,
                                               m_B->m_data[yDir], 0,
                                               fieldsDiagnostics["Hy"]->nComp(), 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Hz"], infra.m_geom.periodicity());
                this->m_data[zDir] =
                    0.5 * amrex::MultiFab::Dot(*maskZ, *fieldsDiagnostics["Hz"], 0,
                                               m_B->m_data[zDir], 0,
                                               fieldsDiagnostics["Hz"]->nComp(), 0, 0);
                return;
        }
    }
    // end void MagFieldEnergy::ComputeDiags

private:
    /**
     * Selects computation method based on missing fields and the input file switch
     * computeMissingFields
     */
    ComputeMethod choose_computation_method (
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::choose_computation_method()");
        using namespace Forms;
        if (fieldsDiagnostics.count("Hx") && fieldsDiagnostics.count("Hy") &&
            fieldsDiagnostics.count("Hz") && fieldsDiagnostics.count("Bx") &&
            fieldsDiagnostics.count("By") && fieldsDiagnostics.count("Bz"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldsDiagnostics.count("Bx") && fieldsDiagnostics.count("By") &&
                fieldsDiagnostics.count("Bz") &&
                (!fieldsDiagnostics.count("Hx") || !fieldsDiagnostics.count("Hy") ||
                 !fieldsDiagnostics.count("Hz")))
            {
                int srcComp{0};
                int dstComp{0};
                int nComps{fieldsDiagnostics["Bx"]->nComp()};
                this->m_B = std::make_shared<DeRhamField<Grid::primal, Space::face>>(this->m_deRham,
                                                                                     nComps);
                amrex::Copy(this->m_B->m_data[xDir], *fieldsDiagnostics["Bx"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_B->m_data[yDir], *fieldsDiagnostics["By"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_B->m_data[zDir], *fieldsDiagnostics["Bz"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());

                this->m_H =
                    std::make_shared<DeRhamField<Grid::dual, Space::edge>>(this->m_deRham, nComps);

                return ComputeMethod::missingH;
            }
            else if (fieldsDiagnostics.count("Hx") && fieldsDiagnostics.count("Hy") &&
                     fieldsDiagnostics.count("Hz") &&
                     (!fieldsDiagnostics.count("Bx") || !fieldsDiagnostics.count("By") ||
                      !fieldsDiagnostics.count("Bz")))
            {
                int srcComp{0};
                int dstComp{0};
                int nComps{fieldsDiagnostics["Hx"]->nComp()};
                this->m_H =
                    std::make_shared<DeRhamField<Grid::dual, Space::edge>>(this->m_deRham, nComps);
                amrex::Copy(this->m_H->m_data[xDir], *fieldsDiagnostics["Hx"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_H->m_data[yDir], *fieldsDiagnostics["Hy"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());
                amrex::Copy(this->m_H->m_data[zDir], *fieldsDiagnostics["Hz"], srcComp, dstComp,
                            nComps, this->m_deRham->get_n_ghost());

                this->m_B = std::make_shared<DeRhamField<Grid::primal, Space::face>>(this->m_deRham,
                                                                                     nComps);

                return ComputeMethod::missingB;
            }
            else
            {
                std::string str =
                    "Either Bx, By and Bz or Hx, Hy and Hz are needed to compute magnetic energy "
                    "when computeMissingFields is set";
                amrex::Assert("Neither B nor H was found", "MagFieldEnergy.H", __LINE__,
                              str.c_str());
            }
        }
        else
        {
            std::string str =
                "Bx, By, Bz, Hx, Hy and Hz are all needed to compute magnetic energy when "
                "computeMissingFields is not set";
            amrex::Assert("B or H not found", "MagFieldEnergy.H", __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone;  // never reached
    }
};

}  // namespace Gempic::Io

#endif

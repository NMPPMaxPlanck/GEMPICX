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
template <int vdim, int degx, int degy, int degz, int ndata>
class MagFieldEnergy : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    std::unique_ptr<Forms::DeRhamField<Grid::primal, Space::face>> m_B;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::edge>> m_H;
    std::unique_ptr<const Forms::DeRhamField<Grid::primal, Space::face>> m_constB;
    std::unique_ptr<const Forms::DeRhamField<Grid::dual, Space::edge>> m_constH;

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
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    MagFieldEnergy(const std::string& rdName,
                   int& indexNum,
                   bool isLastType,
                   const std::string& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName)
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::MagFieldEnergy()");
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
     * function to initialize data after amr levels are initialized.
     */
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc, bool cmf) final
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, cmf);
        m_computeMethod = choose_computation_method(this->m_deRham->m_fieldsDiagnostics);
    }

    /**
     * This function computes the magnetic energy (EF) component wise
     */
    void compute_diags (
        const ComputationalDomain& infra,
        const std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>>&) final
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_constB, *m_constH, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingH:
                this->m_deRham->hodge(*m_constB, *m_H);
                update_data(*m_constB, *m_H, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingB:
                this->m_deRham->hodge(*m_constH, *m_B);
                update_data(*m_B, *m_constH, infra.m_geom.periodicity());
                return;
        }
    }
    // end void MagFieldEnergy::ComputeDiags

private:
    ComputeMethod m_computeMethod;

    /**
     * Selects computation method based on missing fields and the input file switch
     * computeMissingFields
     */
    ComputeMethod choose_computation_method (std::map<std::string, const amrex::MultiFab>& fieldMap)
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::choose_computation_method()");
        using namespace Forms;
        if (fieldMap.count("Hx") && fieldMap.count("Hy") && fieldMap.count("Hz") &&
            fieldMap.count("Bx") && fieldMap.count("By") && fieldMap.count("Bz"))
        {
            this->m_constB = std::make_unique<DeRhamField<Grid::primal, Space::face>>(
                this->m_deRham, fieldMap["Bx"], fieldMap["By"], fieldMap["Bz"]);
            this->m_constH = std::make_unique<DeRhamField<Grid::dual, Space::edge>>(
                this->m_deRham, fieldMap["Hx"], fieldMap["Hy"], fieldMap["Hz"]);
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldMap.count("Bx") && fieldMap.count("By") && fieldMap.count("Bz") &&
                (!fieldMap.count("Hx") || !fieldMap.count("Hy") || !fieldMap.count("Hz")))
            {
                int nComps{fieldMap["Bx"].nComp()};
                this->m_constB = std::make_unique<DeRhamField<Grid::primal, Space::face>>(
                    this->m_deRham, fieldMap["Bx"], fieldMap["By"], fieldMap["Bz"]);

                this->m_H =
                    std::make_unique<DeRhamField<Grid::dual, Space::edge>>(this->m_deRham, nComps);

                return ComputeMethod::missingH;
            }
            else if (fieldMap.count("Hx") && fieldMap.count("Hy") && fieldMap.count("Hz") &&
                     (!fieldMap.count("Bx") || !fieldMap.count("By") || !fieldMap.count("Bz")))
            {
                int nComps{fieldMap["Hx"].nComp()};
                this->m_constH = std::make_unique<DeRhamField<Grid::dual, Space::edge>>(
                    this->m_deRham, fieldMap["Hx"], fieldMap["Hy"], fieldMap["Hz"]);

                this->m_B = std::make_unique<DeRhamField<Grid::primal, Space::face>>(this->m_deRham,
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

    /**
     * This function computes the magnetic energy (EF) component wise
     */
    void update_data (const Forms::DeRhamField<Grid::primal, Space::face>& fieldB,
                      const Forms::DeRhamField<Grid::dual, Space::edge>& fieldH,
                      const amrex::Periodicity& periodicity)
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::update_data()");

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        for (int dir = 0; dir < noutputs; dir++)
        {
            auto mask = amrex::OwnerMask(fieldB.m_data[dir], periodicity);
            this->m_data[dir] =
                0.5 * amrex::MultiFab::Dot(*mask, fieldB.m_data[dir], 0, fieldH.m_data[dir], 0,
                                           fieldB.m_data[dir].nComp(), 0, 0);
        }
    }
};

}  // namespace Gempic::Io

#endif

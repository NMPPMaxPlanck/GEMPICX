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
template <int vdim, int degx, int degy, int degz, int ndata>
class ElecFieldEnergy : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    std::unique_ptr<Forms::DeRhamField<Grid::primal, Space::edge>> m_E;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::face>> m_D;
    std::unique_ptr<const Forms::DeRhamField<Grid::primal, Space::edge>> m_constE;
    std::unique_ptr<const Forms::DeRhamField<Grid::dual, Space::face>> m_constD;

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
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    ElecFieldEnergy(const std::string& rdName,
                    int& indexNum,
                    bool isLastType,
                    const std::string& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName)
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy()");
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
     * function to initialize data after amr levels are initialized.
     */
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc, bool cmf) final
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, cmf);
        m_computeMethod = choose_computation_method(this->m_deRham->m_fieldsDiagnostics);
    }

    /**
     * This function computes the electric field energy (EF) component wise:
     * epref_ep0 is the vacuum permittivity
     */
    void compute_diags (
        const ComputationalDomain& infra,
        const std::vector<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>>&) final
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_constE, *m_constD, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingD:
                this->m_deRham->hodge(*m_constE, *m_D);
                update_data(*m_constE, *m_D, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingE:
                this->m_deRham->hodge(*m_constD, *m_E);
                update_data(*m_E, *m_constD, infra.m_geom.periodicity());
                return;
        }
    }
    // end void ElecFieldEnergy::ComputeDiags

private:
    ComputeMethod m_computeMethod;

    /**
     * Selects computation method based on missing fields and the input file switch
     * compute_missing_fields
     */
    ComputeMethod choose_computation_method (std::map<std::string, const amrex::MultiFab>& fieldMap)
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::choose_computation_method()");
        using namespace Forms;
        if (fieldMap.count("Dx") && fieldMap.count("Dy") && fieldMap.count("Dz") &&
            fieldMap.count("Ex") && fieldMap.count("Ey") && fieldMap.count("Ez"))
        {
            this->m_constE = std::make_unique<const DeRhamField<Grid::primal, Space::edge>>(
                this->m_deRham, fieldMap["Ex"], fieldMap["Ey"], fieldMap["Ez"]);
            this->m_constD = std::make_unique<const DeRhamField<Grid::dual, Space::face>>(
                this->m_deRham, fieldMap["Dx"], fieldMap["Dy"], fieldMap["Dz"]);
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldMap.count("Ex") && fieldMap.count("Ey") && fieldMap.count("Ez") &&
                (!fieldMap.count("Dx") || !fieldMap.count("Dy") || !fieldMap.count("Dz")))
            {
                int nComps{fieldMap["Ex"].nComp()};
                this->m_constE = std::make_unique<const DeRhamField<Grid::primal, Space::edge>>(
                    this->m_deRham, fieldMap["Ex"], fieldMap["Ey"], fieldMap["Ez"]);

                this->m_D =
                    std::make_unique<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);

                return ComputeMethod::missingD;
            }
            else if (fieldMap.count("Dx") && fieldMap.count("Dy") && fieldMap.count("Dz") &&
                     (!fieldMap.count("Ex") || !fieldMap.count("Ey") || !fieldMap.count("Ez")))
            {
                int nComps{fieldMap["Dx"].nComp()};
                this->m_constD = std::make_unique<const DeRhamField<Grid::dual, Space::face>>(
                    this->m_deRham, fieldMap["Dx"], fieldMap["Dy"], fieldMap["Dz"]);

                this->m_E = std::make_unique<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham,
                                                                                     nComps);

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

    /**
     * This function computes the electric field energy (EF) component wise:
     * epref_ep0 is the vacuum permittivity
     */
    void update_data (const Forms::DeRhamField<Grid::primal, Space::edge>& fieldE,
                      const Forms::DeRhamField<Grid::dual, Space::face>& fieldD,
                      const amrex::Periodicity& periodicity)
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::update_data()");

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        for (int dir = 0; dir < noutputs; dir++)
        {
            auto mask = amrex::OwnerMask(fieldE.m_data[dir], periodicity);
            this->m_data[dir] =
                0.5 * amrex::MultiFab::Dot(*mask, fieldE.m_data[dir], 0, fieldD.m_data[dir], 0,
                                           fieldE.m_data[dir].nComp(), 0, 0);
        }
    }
};

}  // namespace Gempic::Io

#endif

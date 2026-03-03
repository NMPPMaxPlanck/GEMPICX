/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
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
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_FieldRegistry.H"
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
class ElecFieldEnergy : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    std::unique_ptr<Forms::DeRhamComplex::Primal1Form> m_E;
    std::unique_ptr<Forms::DeRhamComplex::Dual2Form> m_D;
    std::unique_ptr<Forms::DeRhamComplex::Primal1Form const> m_constE;
    std::unique_ptr<Forms::DeRhamComplex::Dual2Form const> m_constD;

    enum class ComputeMethod
    {
        missingAll,
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
    ElecFieldEnergy(std::string const& rdName,
                    int& indexNum,
                    bool isLastType,
                    std::string const& fileName) :
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
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc,
                    FieldRegistry const& reg,
                    bool cmf) final
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, reg, cmf);
        m_computeMethod = choose_computation_method(reg);
    }

    /**
     * This function computes the electric field energy (EF) component wise:
     * epref_ep0 is the vacuum permittivity
     */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>> const&) final
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_constE, *m_constD, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingD:
                hodge(*m_D, *m_constE);
                update_data(*m_constE, *m_D, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingE:
                hodge(*m_E, *m_constD);
                update_data(*m_E, *m_constD, infra.m_geom.periodicity());
                return;
            default:
                amrex::Abort(
                    "Gempic::Io::ElecFieldEnergy::compute_diags(): ComputeMethod missing all "
                    "required fields!");
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
    ComputeMethod choose_computation_method (FieldRegistry const& registry)
    {
        BL_PROFILE("Gempic::Io::ElecFieldEnergy::choose_computation_method()");
        using DRC = Forms::DeRhamComplex;
        auto const* dPtr = get_field_ptr_or_null<DRC::Dual2Form>(registry, "Dx");
        auto const* ePtr = get_field_ptr_or_null<DRC::Primal1Form>(registry, "Ex");
        if (dPtr && ePtr)
        {
            this->m_constD = std::make_unique<DRC::Dual2Form>(this->m_deRham, *dPtr);
            this->m_constE = std::make_unique<DRC::Primal1Form>(this->m_deRham, *ePtr);
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (!dPtr && ePtr)
            {
                int nComps = ePtr->get_ncomp(xDir);
                this->m_constE = std::make_unique<DRC::Primal1Form>(this->m_deRham, *ePtr);
                this->m_D = std::make_unique<DRC::Dual2Form>(this->m_deRham, nComps);
                return ComputeMethod::missingD;
            }
            else if (dPtr && !ePtr)
            {
                int nComps = dPtr->get_ncomp(xDir);
                this->m_constD = std::make_unique<DRC::Dual2Form>(this->m_deRham, *dPtr);
                this->m_E = std::make_unique<DRC::Primal1Form>(this->m_deRham, nComps);
                return ComputeMethod::missingE;
            }
            else
            {
                GEMPIC_ERROR(
                    "Neither E nor D was found: Either Ex, Ey and Ez or Dx, Dy and Dz are needed "
                    "to compute electric field "
                    "energy when computeMissingFields is set");
                return ComputeMethod::missingAll;
            }
        }

        GEMPIC_ERROR(
            "E or D not found: Ex, Ey, Ez, Dx, Dy and Dz are all needed to compute electric "
            "field energy when "
            "computeMissingFields is not set");
        return ComputeMethod::missingAll;
    }

    /**
     * This function computes the electric field energy (EF) component wise:
     * epref_ep0 is the vacuum permittivity
     */
    void update_data (Forms::DeRhamComplex::Primal1Form const& fieldE,
                      Forms::DeRhamComplex::Dual2Form const& fieldD,
                      amrex::Periodicity const& periodicity)
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

} // namespace Gempic::Io

#endif

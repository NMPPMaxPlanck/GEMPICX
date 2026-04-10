/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/

#ifndef REDUCEDDIAGS_MAGFIELDENERGY_H_
#define REDUCEDDIAGS_MAGFIELDENERGY_H_

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
class MagFieldEnergy : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    std::unique_ptr<Forms::DeRhamComplex::Primal2Form> m_B;
    std::unique_ptr<Forms::DeRhamComplex::Dual1Form> m_H;
    std::unique_ptr<Forms::DeRhamComplex::Primal2Form const> m_constB;
    std::unique_ptr<Forms::DeRhamComplex::Dual1Form const> m_constH;

    enum class ComputeMethod
    {
        missingAll,
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
    MagFieldEnergy(std::string const& rdName,
                   int& indexNum,
                   bool isLastType,
                   std::string const& fileName) :
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
    void init_data (std::shared_ptr<Forms::DeRhamComplex> drc,
                    FieldRegistry const& reg,
                    bool cmf) final
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, reg, cmf);
        m_computeMethod = choose_computation_method(reg);
    }

    /**
     * This function computes the magnetic energy (EF) component wise
     */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>> const&) final
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_constB, *m_constH, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingH:
                hodge(*m_H, *m_constB);
                update_data(*m_constB, *m_H, infra.m_geom.periodicity());
                return;
            case ComputeMethod::missingB:
                hodge(*m_B, *m_constH);
                update_data(*m_B, *m_constH, infra.m_geom.periodicity());
                return;
            default:
                amrex::Abort(
                    "Gempic::Io::MagFieldEnergy::compute_diags(): ComputeMethod missing all "
                    "required fields!");
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
    ComputeMethod choose_computation_method (FieldRegistry const& registry)
    {
        BL_PROFILE("Gempic::Io::MagFieldEnergy::choose_computation_method()");
        using DRC = Forms::DeRhamComplex;
        auto const* bPtr = get_field_ptr_or_null<DRC::Primal2Form>(registry, "Bx");
        auto const* hPtr = get_field_ptr_or_null<DRC::Dual1Form>(registry, "Hx");

        if (bPtr && hPtr)
        {
            this->m_constB = std::make_unique<DRC::Primal2Form>(this->m_deRham, *bPtr);
            this->m_constH = std::make_unique<DRC::Dual1Form>(this->m_deRham, *hPtr);
            return ComputeMethod::missingNone;
        }
        // Handle missing fields: construct from nComps if necessary
        else if (bPtr && !hPtr)
        {
            int nComps = bPtr->get_ncomp(xDir);
            this->m_constB = std::make_unique<DRC::Primal2Form>(this->m_deRham, *bPtr);
            this->m_H = std::make_unique<DRC::Dual1Form>(this->m_deRham, nComps);
            return ComputeMethod::missingH;
        }
        else if (hPtr && !bPtr)
        {
            int nComps = hPtr->get_ncomp(xDir);
            this->m_constH = std::make_unique<DRC::Dual1Form>(this->m_deRham, *hPtr);
            this->m_B = std::make_unique<DRC::Primal2Form>(this->m_deRham, nComps);
            return ComputeMethod::missingB;
        }

        GEMPIC_ERROR(
            "B or H not found: Bx, By, Bz, Hx, Hy and Hz are all needed to compute magnetic "
            "energy when "
            "computeMissingFields is not set");
        return ComputeMethod::missingAll; // never reached
    }

    /**
     * This function computes the magnetic energy (EF) component wise
     */
    void update_data (Forms::DeRhamComplex::Primal2Form const& fieldB,
                      Forms::DeRhamComplex::Dual1Form const& fieldH,
                      amrex::Periodicity const& periodicity)
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

} // namespace Gempic::Io

#endif

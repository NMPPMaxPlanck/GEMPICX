/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGS_GAUSSERROR_H_
#define REDUCEDDIAGS_GAUSSERROR_H_

#include <memory>
#include <string>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Profiling.H"
#include "GEMPIC_ReducedDiagnostics.H"

namespace Gempic::Io
{
/**
 *  This class mainly contains a function that
 *  computes the field energy.
 */
template <int vdim, int degx, int degy, int degz, int ndata>
class GaussError : public ReducedDiagnostics<vdim, degx, degy, degz, ndata>
{
public:
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::cell>> m_rho;
    std::unique_ptr<Forms::DeRhamField<Grid::primal, Space::edge> const> m_constE;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::face>> m_D;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::cell>> m_divD;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::face> const> m_constD;
    std::unique_ptr<Forms::DeRhamField<Grid::dual, Space::cell> const> m_constDivD;

    // Initialization value needs to be chosen so that it is unlilkely to match computed value
    amrex::Real m_rhoNorm{0.0};
    amrex::Real m_divDNorm{0.0};

    enum class ComputeMethod
    {
        missingNone,
        missingDivD,
        missingD
    };
    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     * @param[in] fileName default file name suggested by MultiReducedDiagnostics
     */
    GaussError(std::string const& rdName,
               int& indexNum,
               bool isLastType,
               std::string const& fileName) :
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>(rdName, indexNum, isLastType, fileName)
    {
        BL_PROFILE("Gempic::Io::GaussError::GaussError()");
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
                ofs << "[" << ++indexNum << "]GaussError";
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]divDNorm";
                ofs << this->m_sep;
                ofs << "[" << ++indexNum << "]rhoNorm";
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
        BL_PROFILE("Gempic::Io::GaussError::init_data()");
        ReducedDiagnostics<vdim, degx, degy, degz, ndata>::init_data(drc, cmf);
        m_computeMethod = choose_computation_method(this->m_deRham->m_fieldsDiagnostics);
    }

    /**
     * This function computes the Gauss Error
     */
    void compute_diags (
        ComputationalDomain const& infra,
        std::vector<std::shared_ptr<Particle::ParticleSpecies<vdim, ndata>>> const& particles) final
    {
        BL_PROFILE("Gempic::Io::GaussError::compute_diags()");

        switch (m_computeMethod)
        {
            case ComputeMethod::missingNone:
                update_data(*m_rho, *m_constDivD, infra);
                break;
            case ComputeMethod::missingD:
                hodge(*m_D, *m_constE);
                div(*m_divD, *m_D);
                update_data(*m_rho, *m_divD, infra);
                break;
            case ComputeMethod::missingDivD:
                div(*m_divD, *m_constD);
                update_data(*m_rho, *m_divD, infra);
                break;
        }
    }

private:
    ComputeMethod m_computeMethod;
    /**
     * Selects computation method based on missing fields and the input file switch
     * computeMissingFields
     */
    ComputeMethod choose_computation_method (std::map<std::string, amrex::MultiFab const>& fieldMap)
    {
        BL_PROFILE("Gempic::Io::GaussError::choose_computation_method()");
        using namespace Forms;
        if (fieldMap.count("rho") && fieldMap.count("divD"))
        {
            this->m_rho = std::make_unique<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham,
                                                                                 fieldMap["rho"]);
            this->m_constDivD = std::make_unique<DeRhamField<Grid::dual, Space::cell>>(
                this->m_deRham, fieldMap["divD"]);
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldMap.count("rho") && fieldMap.count("Dx") && fieldMap.count("Dy") &&
                fieldMap.count("Dz"))
            {
                int nComps{fieldMap["Dx"].nComp()};
                this->m_rho = std::make_unique<DeRhamField<Grid::dual, Space::cell>>(
                    this->m_deRham, fieldMap["rho"]);
                this->m_constD = std::make_unique<DeRhamField<Grid::dual, Space::face> const>(
                    this->m_deRham, fieldMap["Dx"], fieldMap["Dy"], fieldMap["Dz"]);

                this->m_divD =
                    std::make_unique<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham, nComps);
                return ComputeMethod::missingDivD;
            }
            else if (fieldMap.count("rho") && fieldMap.count("Ex") && fieldMap.count("Ey") &&
                     fieldMap.count("Ez"))
            {
                int nComps{fieldMap["Ex"].nComp()};
                this->m_rho = std::make_unique<DeRhamField<Grid::dual, Space::cell>>(
                    this->m_deRham, fieldMap["rho"]);
                this->m_constE = std::make_unique<DeRhamField<Grid::primal, Space::edge> const>(
                    this->m_deRham, fieldMap["Ex"], fieldMap["Ey"], fieldMap["Ez"]);

                this->m_D =
                    std::make_unique<DeRhamField<Grid::dual, Space::face>>(this->m_deRham, nComps);
                this->m_divD =
                    std::make_unique<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham, nComps);
                return ComputeMethod::missingD;
            }
            else
            {
                GEMPIC_ERROR(
                    "rho and (D or E or divD) not found: rho and either Dx, Dy and Dz or Ex, Ey "
                    "and Ez or divD are needed to compute "
                    "Gauss error when computeMissingFields is set");
            }
        }
        else
        {
            GEMPIC_ERROR(
                "rho or divD not found: rho and divD are both needed to compute Gauss error when "
                "computeMissingFields "
                "is not set");
        }
        return ComputeMethod::missingNone; // never reached
    }

    /**
     * This function computes the Gauss error by comparing rho computed from particles with
     * div D coming from Maxwell
     */
    void update_data (Forms::DeRhamField<Grid::dual, Space::cell> const& rho,
                      Forms::DeRhamField<Grid::dual, Space::cell> const& divD,
                      ComputationalDomain const& infra)
    {
        BL_PROFILE("Gempic::Io::GaussError::update_data()");
        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        amrex::Real rhoNorm = Gempic::Utils::gempic_norm(rho.m_data, infra, 2);
        amrex::Real divDNorm = Gempic::Utils::gempic_norm(divD.m_data, infra, 2);

        // Check that rho and divD have been updated
        if (std::abs(rhoNorm - m_rhoNorm) < 1e-12 || std::abs(divDNorm - m_divDNorm) < 1e-12)
        {
            amrex::Warning(
                std::format("Warning: rhoNorm: {0:e}\tdivDNorm: {1:e}\nm_rhoNorm: "
                            "{2:e}\tm_divDNorm: {3:e}\nGaussError: rho or divD may not "
                            "have been updated",
                            rhoNorm, divDNorm, m_rhoNorm, m_divDNorm));
        }
        else
        {
            m_rhoNorm = rhoNorm;
            m_divDNorm = divDNorm;
        }

        amrex::MultiFab difference(divD.m_data.boxArray(), divD.m_data.DistributionMap(),
                                   divD.m_data.nComp(), 0);
        int srcComp{0};
        int dstComp{0};
        amrex::Copy(difference, divD.m_data, srcComp, dstComp, divD.m_data.nComp(), 0);
        difference.minus(rho.m_data, srcComp, difference.nComp(), 0);
        amrex::Real diffNorm = Gempic::Utils::gempic_norm(difference, infra, 2);

        this->m_data[0] = diffNorm;
        this->m_data[1] = divDNorm;
        this->m_data[2] = rhoNorm;
    }
};

} // namespace Gempic::Io

#endif

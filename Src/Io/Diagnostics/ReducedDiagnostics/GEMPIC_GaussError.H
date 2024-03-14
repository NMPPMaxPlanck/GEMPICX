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
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Profiling.H"
#include "GEMPIC_ReducedDiagnostics.H"

namespace Gempic::Io
{
/**
 *  This class mainly contains a function that
 *  computes the field energy.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class GaussError : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::primal, Forms::Space::edge>> m_E;
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>> m_D;
    std::shared_ptr<Forms::DeRhamField<Forms::Grid::dual, Forms::Space::cell>> m_divD;

    enum class ComputeMethod
    {
        missingNone,
        missingDivD,
        missingD
    };
    /**
     * constructor
     * @param[in] rdName reduced diags names
     */
    GaussError(std::string rdName) :
        ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(rdName)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if (this->m_isNotRestart)
            {
                // open file
                std::ofstream ofs{this->m_path + this->m_rdName + "." + this->m_extension,
                                  std::ofstream::out};
                // write header row
                int c = 0;
                ofs << "#";
                ofs << "[" << c++ << "]step()";
                ofs << this->m_sep;
                ofs << "[" << c++ << "]time(s)";
                ofs << this->m_sep;
                ofs << "[" << c++ << "]";
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the Gauss Error
     */
    void compute_diags (ComputationalDomain& infra,
                        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
                        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>,
                                        numspec>& partGr) final
    {
        constexpr int noutputs = 1;
        this->m_data.resize(noutputs, 0.0);
        this->m_data[0] = compute_gauss_error(infra, fieldsDiagnostics, partGr);
    }

    amrex::Real compute_gauss_error (
        ComputationalDomain& infra,
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        amrex::GpuArray<std::shared_ptr<Gempic::Particle::ParticleGroups<vdim, ndata>>, numspec>&
            partGr)
    {
        BL_PROFILE("reduced_diag::compute_gauss_error");
        static ComputeMethod computeMethod{choose_computation_method(fieldsDiagnostics)};

        auto& rho = *fieldsDiagnostics["rho"];

        // Deposit charges in rho:
        rho.setVal(0.0);
        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real charge = partGr[spec]->get_charge();

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*partGr[spec], 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                const auto& particles = pti.GetArrayOfStructs()().data();
                const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        // Local arrays for particle position
                        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                        {
                            // positionParticle data structure needed for spline
                            positionParticle[d] = particles[pp].pos(d);
                        }

                        ParticleMeshCoupling::SplineBase<degx, degy, degz> spline(
                            positionParticle, infra.m_plo, infra.m_dxi);

                        ParticleMeshCoupling::gempic_deposit_rho(spline, charge * weight[pp],
                                                                 rhoarr);
                    });
            }
        }
        // get number of ghost cells
        rho.SumBoundary(0, 1, this->m_deRham->get_n_ghost(), amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                        infra.m_geom.periodicity());
        rho.AverageSync(infra.m_geom.periodicity());
        rho.FillBoundary(infra.m_geom.periodicity());

        amrex::Real rhoNorm = Gempic::Utils::gempic_norm(rho, infra, 2);

        amrex::MultiFab* divD;
        switch (computeMethod)
        {
            case ComputeMethod::missingNone:
                divD = fieldsDiagnostics["divD"];
                break;
            case ComputeMethod::missingD:
                this->m_deRham->hodge(*m_E, *m_D);
            case ComputeMethod::missingDivD:
                this->m_deRham->div(*m_D, *m_divD);
                divD = &(m_divD->m_data);
                break;
        }
        // Compare rho computed from particles with div D coming from Maxwell
        divD->minus(rho, 0, 1, 0);
        amrex::Real diffNorm = Gempic::Utils::gempic_norm(*divD, infra, 2);
        return (diffNorm / rhoNorm);
    }

private:
    /**
     * Selects computation method based on missing fields and the input file switch
     * computeMissingFields
     */
    ComputeMethod choose_computation_method (
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        using namespace Forms;
        if (fieldsDiagnostics.count("rho") && fieldsDiagnostics.count("divD"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields)
        {
            if (fieldsDiagnostics.count("rho") && fieldsDiagnostics.count("D"))
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

                this->m_divD =
                    std::make_shared<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham);
                return ComputeMethod::missingDivD;
            }
            else if (fieldsDiagnostics.count("rho") && fieldsDiagnostics.count("E"))
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
                this->m_divD =
                    std::make_shared<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham);
                return ComputeMethod::missingD;
            }
            else
            {
                std::string str =
                    "rho and either Dx, Dy and Dz or Ex, Ey and Ez or divD are needed to compute "
                    "Gauss error when computeMissingFields is set";
                amrex::Assert("rho and (D or E or divD) not found", "GaussError.H", __LINE__,
                              str.c_str());
            }
        }
        else
        {
            std::string str =
                "rho and divD are both needed to compute Gauss error when computeMissingFields "
                "is not set";
            amrex::Assert("rho or divD not found", "GaussError.H", __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone;  // never reached
    }
};

}  // namespace Gempic::Io

#endif

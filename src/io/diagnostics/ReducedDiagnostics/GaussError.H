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

#include "ReducedDiagnostics.H"
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_profiling.H>

#include <string>
#include <memory>

/**
 *  This class mainly contains a function that
 *  computes the field energy.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class GaussError: public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::primal, GEMPIC_Fields::Space::edge>> m_E;
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::dual, GEMPIC_Fields::Space::face>> m_D;
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::dual, GEMPIC_Fields::Space::cell>> m_divD;

    enum class ComputeMethod {missingNone, missingDivD, missingD};
    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    GaussError(std::string rd_name)
    : ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>(rd_name)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            if ( this->m_IsNotRestart )
            {
                // open file
                std::ofstream ofs{this->m_path + this->m_rd_name + "." + this->m_extension, std::ofstream::out};
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
    void ComputeDiags(Gempic::CompDom::computational_domain& infra, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        constexpr int noutputs = 1;
        this->m_data.resize(noutputs, 0.0);
        this->m_data[0] = compute_gauss_error(infra, fieldsDiagnostics, part_gr);
    }

    amrex::Real compute_gauss_error( Gempic::CompDom::computational_domain& infra, 
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> &part_gr)
    {
        BL_PROFILE("reduced_diag::compute_gauss_error");
        static ComputeMethod computeMethod{chooseComputationMethod(fieldsDiagnostics)};

        auto& rho = *fieldsDiagnostics["rho"];
        
        // Deposit charges in rho:
        rho.setVal(0.0);
        for (int spec = 0; spec < numspec; spec++)
        {
            amrex::Real charge = part_gr[spec]->getCharge();

            for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
            {
                const long np = pti.numParticles();
                const auto& particles = pti.GetArrayOfStructs()().data();
                const auto weight = pti.GetStructOfArrays().GetRealData(vdim).data();

                amrex::Array4<amrex::Real> const& rhoarr = rho[pti].array();

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
                {
                    // Local arrays for particle position
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionParticle;
                    for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                    {
                        // positionParticle data structure needed for spline
                        positionParticle[d] = particles[pp].pos(d);
                    }

                    Spline::SplineBase<degx, degy, degz> spline(positionParticle, infra.plo, infra.dxi);

                    Gempic::Particles::gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
                });
            }
        }
        // get number of ghost cells
        rho.SumBoundary(0, 1, this->m_deRham->getNGhost(), amrex::IntVect(AMREX_D_DECL(0,0,0)),  infra.geom.periodicity());
        rho.AverageSync(infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());

        amrex::Real rho_norm = Gempic::Utils::gempic_norm(rho, infra, 2);

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
                divD = &(m_divD->data);
                break;
        }
        // Compare rho computed from particles with div D coming from Maxwell
        divD->minus(rho, 0, 1, 0);
        amrex::Real diff_norm = Gempic::Utils::gempic_norm(*divD, infra, 2);
        return (diff_norm / rho_norm);  
    }

    private:
    /**
     * Selects computation method based on missing fields and the input file switch compute_missing_fields
     */
    ComputeMethod chooseComputationMethod (std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        using namespace GEMPIC_Fields;
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
                amrex::Copy(this->m_D->data[xDir], *fieldsDiagnostics["Dx"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_D->data[yDir], *fieldsDiagnostics["Dy"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_D->data[zDir], *fieldsDiagnostics["Dz"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());

                this->m_divD = std::make_shared<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham);
                return ComputeMethod::missingDivD;
            }
            else if (fieldsDiagnostics.count("rho") && fieldsDiagnostics.count("E"))
            {
                this->m_E = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham);
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                amrex::Copy(this->m_E->data[xDir], *fieldsDiagnostics["Ex"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_E->data[yDir], *fieldsDiagnostics["Ey"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_E->data[zDir], *fieldsDiagnostics["Ez"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());

                this->m_D = std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham);
                this->m_divD = std::make_shared<DeRhamField<Grid::dual, Space::cell>>(this->m_deRham);
                return ComputeMethod::missingD;

            }
            else
            {
                std::string str = "rho and either Dx, Dy and Dz or Ex, Ey and Ez or divD are needed to compute Gauss error when compute_missing_fields is set";
                amrex::Assert("rho and (D or E or divD) not found",  "GaussError.H" , __LINE__, str.c_str());
            }
        }
        else
        {
            std::string str = "rho and divD are both needed to compute Gauss error when compute_missing_fields is not set";
            amrex::Assert("rho or divD not found",  "GaussError.H" , __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone; //never reached
    }
};

#endif

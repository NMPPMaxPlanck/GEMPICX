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

#include "ReducedDiagnostics.H"
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_profiling.H>

#include <string>

/**
 *  This class mainly contains a function that
 *  computes the field energy.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class ElecFieldEnergy : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::primal, GEMPIC_Fields::Space::edge>> m_E;
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::dual, GEMPIC_Fields::Space::face>> m_D;

    enum class ComputeMethod {missingNone, missingD, missingE};

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ElecFieldEnergy(std::string rd_name)
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
                ofs << "[" << c++ << "]Ex^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]Ey^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]Ez^2/2" + std::string("(J)");
                ofs << std::endl;
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
    void ComputeDiags(Gempic::CompDom::computational_domain& infra, std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics, 
                        amrex::GpuArray<std::shared_ptr<Gempic::Particles::particle_groups<vdim, ndata>>, numspec> &part_gr) override final
    {   
        static ComputeMethod computeMethod{chooseComputationMethod(fieldsDiagnostics)};

        constexpr int noutputs = 3;
        this->m_data.resize(noutputs, 0.0);

        std::unique_ptr<amrex::iMultiFab> maskX;
        std::unique_ptr<amrex::iMultiFab> maskY;
        std::unique_ptr<amrex::iMultiFab> maskZ;

        switch (computeMethod)
        {
            case ComputeMethod::missingNone:
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Ex"], infra.geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX,*fieldsDiagnostics["Ex"], 0, *fieldsDiagnostics["Dx"], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Ey"], infra.geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY,*fieldsDiagnostics["Ey"], 0, *fieldsDiagnostics["Dy"], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Ez"], infra.geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ,*fieldsDiagnostics["Ez"], 0, *fieldsDiagnostics["Dz"], 0, 1, 0, 0);
                return;
            case ComputeMethod::missingD:
                this->m_deRham->hodge(*m_E, *m_D);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Ex"], infra.geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX,*fieldsDiagnostics["Ex"], 0, m_D->data[xDir], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Ey"], infra.geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY,*fieldsDiagnostics["Ey"], 0, m_D->data[yDir], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Ez"], infra.geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ,*fieldsDiagnostics["Ez"], 0, m_D->data[zDir], 0, 1, 0, 0);
                return;
            case ComputeMethod::missingE:
                this->m_deRham->hodge(*m_D, *m_E);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Dx"], infra.geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX,*fieldsDiagnostics["Dx"], 0, m_E->data[xDir], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Dy"], infra.geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY,*fieldsDiagnostics["Dy"], 0, m_E->data[yDir], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Dz"], infra.geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ,*fieldsDiagnostics["Dz"], 0, m_E->data[zDir], 0, 1, 0, 0);
                return;
        }
    }
    // end void ElecFieldEnergy::ComputeDiags

    private:
    /**
     * Selects computation method based on missing fields and the input file switch compute_missing_fields
     */
    ComputeMethod chooseComputationMethod (std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        using namespace GEMPIC_Fields;
        if (fieldsDiagnostics.count("Dx")
         && fieldsDiagnostics.count("Dy")
         && fieldsDiagnostics.count("Dz")
         && fieldsDiagnostics.count("Ex")
         && fieldsDiagnostics.count("Ey")
         && fieldsDiagnostics.count("Ez"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields) {
            if (fieldsDiagnostics.count("Ex")
             && fieldsDiagnostics.count("Ey")
             && fieldsDiagnostics.count("Ez")
             && (!fieldsDiagnostics.count("Dx")
              || !fieldsDiagnostics.count("Dy")
              || !fieldsDiagnostics.count("Dz")))
            {
                this->m_E = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham);
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                amrex::Copy(this->m_E->data[xDir], *fieldsDiagnostics["Ex"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_E->data[yDir], *fieldsDiagnostics["Ey"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_E->data[zDir], *fieldsDiagnostics["Ez"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());

                this->m_D = std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham);

                return ComputeMethod::missingD;
            }
            else if (fieldsDiagnostics.count("Dx")
                  && fieldsDiagnostics.count("Dy")
                  && fieldsDiagnostics.count("Dz")
                  && (!fieldsDiagnostics.count("Ex")
                   || !fieldsDiagnostics.count("Ey")
                   || !fieldsDiagnostics.count("Ez")))
            {
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                this->m_D = std::make_shared<DeRhamField<Grid::dual, Space::face>>(this->m_deRham);
                amrex::Copy(this->m_D->data[xDir], *fieldsDiagnostics["Dx"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_D->data[yDir], *fieldsDiagnostics["Dy"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_D->data[zDir], *fieldsDiagnostics["Dz"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());

                this->m_E = std::make_shared<DeRhamField<Grid::primal, Space::edge>>(this->m_deRham);

                return ComputeMethod::missingE;
            }
            else
            {
                std::string str = "Either Ex, Ey and Ez or Dx, Dy and Dz are needed to compute electric field energy when compute_missing_fields is set";
                amrex::Assert("Neither E nor D was found",  "ElecFieldEnergy.H" , __LINE__, str.c_str());
            }
        }
        else
        {
            std::string str = "Ex, Ey, Ez, Dx, Dy and Dz are all needed to compute electric field energy when compute_missing_fields is not set";
            amrex::Assert("E or D not found",  "ElecFieldEnergy.H" , __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone; //never reached
    }
};

#endif

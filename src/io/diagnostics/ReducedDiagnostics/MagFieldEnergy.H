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
class MagFieldEnergy : public ReducedDiagnostics<vdim, numspec, degx, degy, degz, degree, ndata>
{
public:
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::primal, GEMPIC_Fields::Space::face>> m_B;
    std::shared_ptr<GEMPIC_Fields::DeRhamField<GEMPIC_Fields::Grid::dual, GEMPIC_Fields::Space::edge>> m_H;

    enum class ComputeMethod {missingNone, missingH, missingB};
    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    MagFieldEnergy(std::string rd_name)
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
                ofs << "[" << c++ << "]Bx^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]By^2/2" + std::string("(J)");
                ofs << this->m_sep;
                ofs << "[" << c++ << "]Bz^2/2" + std::string("(J)");
                ofs << std::endl;
                // close file
                ofs.close();
            }
        }
    }
    // end constructor

    /**
     * This function computes the magnetic energy (EF) component wise
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
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Bx"], infra.geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX,*fieldsDiagnostics["Bx"], 0, *fieldsDiagnostics["Hx"], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["By"], infra.geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY,*fieldsDiagnostics["By"], 0, *fieldsDiagnostics["Hy"], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Bz"], infra.geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ,*fieldsDiagnostics["Bz"], 0, *fieldsDiagnostics["Hz"], 0, 1, 0, 0);
                return;
            case ComputeMethod::missingH:
                this->m_deRham->hodge(*m_B, *m_H);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Bx"], infra.geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX,*fieldsDiagnostics["Bx"], 0, m_H->data[xDir], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["By"], infra.geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY,*fieldsDiagnostics["By"], 0, m_H->data[yDir], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Bz"], infra.geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ,*fieldsDiagnostics["Bz"], 0, m_H->data[zDir], 0, 1, 0, 0);
                return;
            case ComputeMethod::missingB:
                this->m_deRham->hodge(*m_H, *m_B);
                maskX = amrex::OwnerMask(*fieldsDiagnostics["Hx"], infra.geom.periodicity());
                this->m_data[xDir] = 0.5 * amrex::MultiFab::Dot(*maskX,*fieldsDiagnostics["Hx"], 0, m_B->data[xDir], 0, 1, 0, 0);
                maskY = amrex::OwnerMask(*fieldsDiagnostics["Hy"], infra.geom.periodicity());
                this->m_data[yDir] = 0.5 * amrex::MultiFab::Dot(*maskY,*fieldsDiagnostics["Hy"], 0, m_B->data[yDir], 0, 1, 0, 0);
                maskZ = amrex::OwnerMask(*fieldsDiagnostics["Hz"], infra.geom.periodicity());
                this->m_data[zDir] = 0.5 * amrex::MultiFab::Dot(*maskZ,*fieldsDiagnostics["Hz"], 0, m_B->data[zDir], 0, 1, 0, 0);
                return;
        }
    }
    // end void MagFieldEnergy::ComputeDiags

    private:
    /**
     * Selects computation method based on missing fields and the input file switch compute_missing_fields
     */
    ComputeMethod chooseComputationMethod (std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics)
    {
        using namespace GEMPIC_Fields;
        if (fieldsDiagnostics.count("Hx")
         && fieldsDiagnostics.count("Hy")
         && fieldsDiagnostics.count("Hz")
         && fieldsDiagnostics.count("Bx")
         && fieldsDiagnostics.count("By")
         && fieldsDiagnostics.count("Bz"))
        {
            return ComputeMethod::missingNone;
        }
        else if (this->m_computeMissingFields) {
            if (fieldsDiagnostics.count("Bx")
             && fieldsDiagnostics.count("By")
             && fieldsDiagnostics.count("Bz")
             && (!fieldsDiagnostics.count("Hx")
              || !fieldsDiagnostics.count("Hy")
              || !fieldsDiagnostics.count("Hz")))
            {
                this->m_B = std::make_shared<DeRhamField<Grid::primal, Space::face>>(this->m_deRham);
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                amrex::Copy(this->m_B->data[xDir], *fieldsDiagnostics["Bx"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_B->data[yDir], *fieldsDiagnostics["By"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_B->data[zDir], *fieldsDiagnostics["Bz"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());

                this->m_H = std::make_shared<DeRhamField<Grid::dual, Space::edge>>(this->m_deRham);

                return ComputeMethod::missingH;
            }
            else if (fieldsDiagnostics.count("Hx")
                  && fieldsDiagnostics.count("Hy")
                  && fieldsDiagnostics.count("Hz")
                  && (!fieldsDiagnostics.count("Bx")
                   || !fieldsDiagnostics.count("By")
                   || !fieldsDiagnostics.count("Bz")))
            {
                this->m_H = std::make_shared<DeRhamField<Grid::dual, Space::edge>>(this->m_deRham);
                int srcComp{0};
                int dstComp{0};
                int numComps{1};
                amrex::Copy(this->m_H->data[xDir], *fieldsDiagnostics["Hx"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_H->data[yDir], *fieldsDiagnostics["Hy"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());
                amrex::Copy(this->m_H->data[zDir], *fieldsDiagnostics["Hz"], srcComp, dstComp, numComps, this->m_deRham->getNGhost());

                this->m_B = std::make_shared<DeRhamField<Grid::primal, Space::face>>(this->m_deRham);

                return ComputeMethod::missingB;
            }
            else
            {
                std::string str = "Either Bx, By and Bz or Hx, Hy and Hz are needed to compute magnetic energy when compute_missing_fields is set";
                amrex::Assert("Neither B nor H was found",  "MagFieldEnergy.H" , __LINE__, str.c_str());
            }
        }
        else
        {
            std::string str = "Bx, By, Bz, Hx, Hy and Hz are all needed to compute magnetic energy when compute_missing_fields is not set";
            amrex::Assert("B or H not found",  "MagFieldEnergy.H" , __LINE__, str.c_str());
        }
        return ComputeMethod::missingNone; //never reached
    }
};

#endif

/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Modified for GEMPICX
 */

#ifndef REDUCEDDIAGNOSTICS_REDUCEDDIAGNOSTICS_H_
#define REDUCEDDIAGNOSTICS_REDUCEDDIAGNOSTICS_H_

//#include "Utils/IntervalsParser.H"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Fields.H"

namespace Gempic::Io
{
/**
 *  Base class for reduced diagnostics. Each type of reduced diagnostics is
 *  implemented in a derived class, and must override the (pure virtual)
 *  function ComputeDiags.
 */
template <int vdim, int numspec, int degx, int degy, int degz, int degree, int ndata>
class ReducedDiagnostics
{
public:
    /// output path (default)
    std::string m_path = "ReducedDiagnostics/";

    /// output extension (default)
    std::string m_extension = "txt";

    /// diags name
    std::string m_rdName;

    /// is this instance the first type of a group?
    bool m_isFirstType = true;

    /// is this instance the last type of a group?
    bool m_isLastType = true;

    /// check if it is a restart run
    int m_isNotRestart = 1;

    /// separator in the output file
    std::string m_sep = " ";

    /// output data
    std::vector<amrex::Real> m_data;

    /// create fields needed for computations if they are missing
    bool m_computeMissingFields;

    /// deRham complex
    std::shared_ptr<Forms::DeRhamComplex> m_deRham;

    /**
     * constructor
     * @param[in] rdName reduced diags names
     * @param[in] indexNum the index of the print field in the header
     * @param[in] isLastType is this the last type in the group?
     */
    ReducedDiagnostics(std::string rdName, int& indexNum, bool isLastType) :
        m_rdName{rdName}, m_isLastType{isLastType}
    {
        BL_PROFILE("Gempic::Io::ReducedDiagnostics::ReducedDiagnostics()");
        m_isFirstType = (indexNum == 0);
        // BackwardCompatibility();

        Parameters ppRdName("ReducedDiagnostics." + m_rdName);

        ppRdName.get_or_set("path", m_path);

        if (m_isFirstType)
        {
            // ensure output directory exists
            std::filesystem::path dirName{m_path};
            dirName.remove_filename();
            if (!(std::filesystem::exists(dirName)))
            {
                std::filesystem::create_directories(dirName);
            }
        }
        else
        {
            // We only want output once per group (if at all)
            ppRdName.hide_class_output();
        }

        // read extension
        ppRdName.get_or_set("extension", m_extension);

        if (amrex::ParallelDescriptor::IOProcessor() && m_isFirstType)
        {
            // create folder
            // if (!UtilCreateDirectory(m_path, 0755))
            //{ CreateDirectoryFailed(m_path); }

            // replace / create output file
            if (m_isNotRestart)  // not a restart
            {
                std::ofstream ofs{m_path + m_rdName + "." + m_extension, std::ios::trunc};
                ofs.close();
            }
        }
        /*
                // read separator
                pp_rd_name.query("separator", m_sep);
        */
    }
    // end constructor

    /**
     * Virtual destructor for polymorphism
     */
    virtual ~ReducedDiagnostics() = default;

    /**
     * function to initialize data after amr
     * levels are initialized.
     */
    virtual void init_data (std::shared_ptr<Forms::DeRhamComplex> drc, bool cmf)
    {
        BL_PROFILE("Gempic::Io::ReducedDiagnostics::init_data()");
        m_deRham = drc;
        m_computeMissingFields = cmf;
    }

    /** Redistribute parallel data structures during load balance
     */
    virtual void load_balance () {}

    /**
     * function to compute diags
     */
    virtual void compute_diags(
        ComputationalDomain& infra,
        std::map<std::string, amrex::MultiFab*>& fieldsDiagnostics,
        amrex::GpuArray<std::shared_ptr<Particle::ParticleGroups<vdim, ndata>>, numspec>&
            partGr) = 0;

    /**
     * write to file function
     */
    virtual void write_to_file (amrex::Real dt, int step) const
    {
        BL_PROFILE("Gempic::Io::ReducedDiagnostics::write_to_file()");
        // open file
        std::ofstream ofs{m_path + m_rdName + "." + m_extension,
                          std::ofstream::out | std::ofstream::app};

        if (m_isFirstType)
        {
            // write step
            ofs << step;

            ofs << m_sep;

            // set precision
            ofs << std::fixed << std::setprecision(14) << std::scientific;

            // write time
            ofs << dt * (step);
        }
        else
        {
            // set precision
            ofs << std::fixed << std::setprecision(14) << std::scientific;
        }

        // loop over data size and write
        for (const auto& item : m_data) ofs << m_sep << item;

        // end loop over data size

        if (m_isLastType)
        {
            // end line
            ofs << std::endl;
        }

        // close file
        ofs.close();
    }
    // end ReducedDiagnostics::WriteToFile
};
}  // namespace Gempic::Io

#endif

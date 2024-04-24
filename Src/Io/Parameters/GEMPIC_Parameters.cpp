#include <type_traits>

#include <AMReX_ParallelDescriptor.H>

#include "GEMPIC_Parameters.H"

namespace Gempic::Io
{
namespace Impl
{
/* Overload variant printing
 * To understand the visit-variant connection go to
 * https://en.cppreference.com/w/cpp/utility/variant/visit
 */
std::ostream& operator<<(std::ostream& os, const parmParseType& val)
{
    BL_PROFILE("Gempic::Io::Impl::operator(os,parmParseType& val)");
    std::visit([&os] (auto&& arg) { os << arg; }, val);
    return os;
}

std::ostream& operator<<(std::ostream& os, const parmParseParameterType& val)
{
    BL_PROFILE("Gempic::Io::Impl::operator(os,parmParseParameterType& val)");
    std::visit(
        [&os] (auto&& arg)
        {
            using T = std::decay_t<decltype(arg)>;
            // make sure to quote strings
            if constexpr (std::is_same_v<T, std::string>)
            {
                os << '"' << arg << '"';
            }
            else
            {
                os << arg;
            }
        },
        val);
    return os;
}

std::ostream& operator<<(std::ostream& os, const parmParseVectorType& inputVector)
{
    BL_PROFILE("Gempic::Io::Impl::operator(os,inputVector)");
    for (const auto& elem : inputVector)
    {
        os << elem << " ";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const parmParseArrayType& inputArray)
{
    BL_PROFILE("Gempic::Io::Impl::operator(os,inputArray)");
    for (const auto& elem : inputArray)
    {
        os << elem << " ";
    }
    return os;
}
}  //namespace Impl

void Parameters::set_print_output (bool printOrNot)
{
    BL_PROFILE("Gempic::Io::set_print_output()");
    if (s_numParameterInstances == 0)
    {
        s_printOutput = printOrNot;
    }
    else
    {
        std::cerr << "Error: Parameters class already previously initialized!";
        exit(Error::ParametersAlreadyInitialized);
    }
}

Parameters::Parameters(const std::string& classPrefix, std::string printName) :
    m_classPrefix{classPrefix}
{
    BL_PROFILE("Gempic::Io::Parameters(class)");
    using namespace Impl;  // utility functions for this class
    if (s_numParameterInstances == 0)
    {
        std::cerr << "Error: Parameters class not previously initialized!";
        exit(Error::ParametersNotInitialized);
    }
    if (printName == "None")
    {
        printName = "class " + classPrefix;
    }
    m_className = printName;
    if (s_printOutput)
    {
        m_classOutput << "# Parameters for " << m_className << ":\n";
        m_isIOProcess = amrex::ParallelDescriptor::IOProcessor();
    }
    ++s_numParameterInstances;
}

Parameters::Parameters()
{
    BL_PROFILE("Gempic::Io::Parameters(main)");
    m_className = "Parameters class";
    m_classPrefix = "";

    if (s_numParameterInstances == 0)
    {
        m_isIOProcess = amrex::ParallelDescriptor::IOProcessor();
        if (s_printOutput && m_isIOProcess)
        {
            get("outputFile", s_outputFile);
            std::string simulationName{"unnamed simulation"};
            get_or_set("simName", simulationName);
            std::ofstream ofs{s_outputFile, std::ofstream::out};
            ofs << "# Output file for " << simulationName << ":\n";
            ofs.close();
        }
    }
    ++s_numParameterInstances;
}

Parameters::~Parameters()
{
    BL_PROFILE("Gempic::Io::~Parameters()");
    if (--s_numParameterInstances)
    {
        print_class_parameters();
    }
    else
    {
        print_shared_parameters();
        for ([[maybe_unused]] const auto& [variableName, variable] : s_sharedParams)
        {
            // Reset parameters in case we want to do a different simulation in a different scope
            // (e.g. for testing)
            *variable.get() = SharedParam{};
        }
        // N.B. This might be unexpected to users running several different parameter sets in the
        // same program
        s_printOutput = false;
    }
}

void Parameters::hide_class_output() { m_hideOutput = true; }

// Probably just have this in the destructor.
void Parameters::print_class_parameters ()
{
    BL_PROFILE("Gempic::Io::print_class_parameters()");
    if (s_printOutput && !m_hideOutput && m_classOutput.rdbuf()->in_avail() && m_isIOProcess)
    {
        // Space for next output section
        m_classOutput << "\n";
        // open file
        std::ofstream ofs{s_outputFile, std::ofstream::out | std::ofstream::app};
        ofs << m_classOutput.rdbuf();  // moves binary data from m_classOutput to file
        ofs.close();
    }
}

// Probably just have this in the destructor.
void Parameters::print_shared_parameters ()
{
    BL_PROFILE("Gempic::Io::print_shared_parameters()");
    using namespace Impl;  // utility functions for this class
    if (s_printOutput && m_isIOProcess)
    {
        std::ofstream ofs{s_outputFile, std::ofstream::out | std::ofstream::app};
        ofs << "# Shared parameters:\n";

        // Print all shared parameters.
        for (const auto& [variableName, variable] : s_sharedParams)
        {
            if (variable.get()->m_isSet)
            {
                ofs << variableName << " = " << variable.get()->m_ref << " # "
                    << variable.get()->m_setBy << '\n';
            }
            else
            {
                ofs << "#" << variableName << " " << variable.get()->m_setBy << '\n';
            }
        }
        ofs.close();
    }
}

bool Parameters::exists (const std::string& variableName)
{
    BL_PROFILE("Gempic::Io::exists()");
    // Check if variable is a shared parameter (contains is c++20).
    auto search{s_sharedParams.find(variableName)};
    if (search != s_sharedParams.end())
    {
        amrex::ParmParse pp;
        return (pp.contains(variableName.c_str()) || search->second.get()->m_isSet);
    }
    else
    {
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.contains(variableName.c_str());
    }
}

bool Parameters::is_in_input_file (const std::string& variableName)
{
    BL_PROFILE("Gempic::Io::is_in_input_file");
    // Check if variable is a shared parameter (contains is c++20).
    auto search{s_sharedParams.find(variableName)};
    if (search != s_sharedParams.end())
    {
        amrex::ParmParse pp;
        return pp.contains(variableName.c_str());
    }
    else
    {
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.contains(variableName.c_str());
    }
}

std::string Parameters::has_been_set_by (const std::string& variableName)
{
    BL_PROFILE("Gempic::Io::has_been_set_by");
    // Check if variable is a shared parameter (contains is c++20).
    auto search{s_sharedParams.find(variableName)};
    if (search != s_sharedParams.end())
    {
        return search->second.get()->m_setBy;
    }
    else
    {
        std::cerr << variableName << " is not a shared parameter!\n";
        return "";
    }
}

}  // namespace Gempic::Io

#include <type_traits>

#include <AMReX_ParallelDescriptor.H>

#include "GEMPIC_parameters.H"

namespace Gempic::Impl::Params {
/* Overload variant printing
 * To understand the visit-variant connection go to
 * https://en.cppreference.com/w/cpp/utility/variant/visit
 */
std::ostream& operator<< (std::ostream& os, const parmParseType& val)
{
    std::visit([&os](auto&& arg){os << arg; }, val);
    return os;
}

std::ostream& operator<< (std::ostream& os, const parmParseParameterType& val)
{
    std::visit([&os] (auto&& arg)
               {using T = std::decay_t<decltype(arg)>;
                // make sure to quote strings
                if constexpr (std::is_same_v<T, std::string>) os << '"' << arg << '"';
                else os << arg;
               }, val);
    return os;
}

std::ostream& operator<< (std::ostream& os, const parmParseVectorType& inputVector)
{
    for (auto &elem : inputVector)
    {
        os << elem << " ";
    }
    return os;
}

std::ostream& operator<< (std::ostream& os, const parmParseArrayType& inputArray)
{
    for (auto &elem : inputArray)
    {
        os << elem << " ";
    }
    return os;
}
} // namespace Gempic::Impl::Params

void Parameters::setPrintOutput(bool printOrNot)
{
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

Parameters::Parameters (const std::string& classPrefix, std::string printName) : m_classPrefix{classPrefix}
{
    using namespace Gempic::Impl::Params; // utility functions for this class
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

Parameters::Parameters ()
{
    m_className = "Parameters class";
    m_classPrefix = "";

    if (s_numParameterInstances == 0)
    {
        m_isIOProcess = amrex::ParallelDescriptor::IOProcessor();
        if (s_printOutput && m_isIOProcess)
        {
            get("output_file", s_outputFile);
            std::string simulationName{"unnamed simulation"};
            getOrSet("sim_name", simulationName);
            std::ofstream ofs{s_outputFile, std::ofstream::out};
            ofs << "# Output file for " << simulationName << ":\n";
            ofs.close();
        }
    }
    ++s_numParameterInstances;
}

Parameters::~Parameters ()
{
    if (--s_numParameterInstances)
    {
        printClassParameters();
    }
    else
    {
        printSharedParameters();
        for ([[maybe_unused]] auto &[variableName, variable] : s_sharedParams)
        {
            // Reset parameters in case we want to do a different simulation in a different scope
            // (e.g. for testing)
            *variable.get() = SharedParam{};
        }
        // N.B. This might be unexpected to users running several different parameter sets in the same program
        s_printOutput = false;
    }
}

// Probably just have this in the destructor.
void Parameters::printClassParameters ()
{
    if (s_printOutput && m_classOutput.rdbuf()->in_avail() && m_isIOProcess)
    {
        // Space for next output section
        m_classOutput << "\n";
        // open file
        std::ofstream ofs{s_outputFile, std::ofstream::out | std::ofstream::app};
        ofs << m_classOutput.rdbuf(); // moves binary data from m_classOutput to file
        ofs.close();
    }
}

// Probably just have this in the destructor.
void Parameters::printSharedParameters ()
{
    using namespace Gempic::Impl::Params; // utility functions for this class
    if (s_printOutput && m_isIOProcess)
    {
        std::ofstream ofs{s_outputFile, std::ofstream::out | std::ofstream::app};
        ofs << "# Shared parameters:\n";

        // Print all shared parameters.
        for (auto &[variableName, variable] : s_sharedParams)
        {
            if (variable.get()->isSet)
            {
                ofs << variableName << " = " << variable.get()->ref << " # " << variable.get()->setBy << '\n'; 
            }
            else
            {
                ofs << "#" << variableName << " " << variable.get()->setBy << '\n';
            }
        }
        ofs.close();
    }
}

bool Parameters::exists (const std::string& variableName)
{
    // Check if variable is a shared parameter (contains is c++20).
    auto search{s_sharedParams.find(variableName)}; 
    if (search != s_sharedParams.end())
    {
        amrex::ParmParse pp;
        return (pp.contains(variableName.c_str()) || search->second.get()->isSet);

    }
    else
    {
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.contains(variableName.c_str());
    }
}

bool Parameters::isInInputFile (const std::string& variableName)
{
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

std::string Parameters::hasBeenSetBy (const std::string& variableName)
{
    // Check if variable is a shared parameter (contains is c++20).
    auto search{s_sharedParams.find(variableName)}; 
    if (search != s_sharedParams.end())
    {
        return search->second.get()->setBy;
    }
    else
    {
        std::cerr << variableName << " is not a shared parameter!\n";
        return "";
    }
}
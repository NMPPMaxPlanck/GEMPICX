/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_PARAMETERS_H_
#define GEMPIC_PARAMETERS_H_
#include <map>
#include <memory>
#include <string>
#include <variant>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_Verbosity.H"

namespace Gempic::Io
{
// utility functions for this class
namespace Impl
{
// Exhaustive list of parameter types possible to get from ParmParse
using parmParseParameterType = std::
    variant<bool, int, long, long long, float, double, std::string, amrex::Box, amrex::IntVect>;
using parmParseVectorType = std::vector<parmParseParameterType>;
using parmParseArrayType = std::array<parmParseParameterType, AMREX_SPACEDIM>;
using parmParseType = std::variant<parmParseParameterType, parmParseVectorType, parmParseArrayType>;

inline constexpr std::array indexToName = {
    "bool",   "int",         "long",       "long long",      "float",
    "double", "std::string", "amrex::Box", "amrex::IntVect",
};

/**
 * @brief Overload parmParseType printing
 *
 */
std::ostream& operator<<(std::ostream& os, parmParseType const& val);

std::ostream& operator<<(std::ostream& os, parmParseParameterType const& val);
/**
 * @brief Overload parameter vector printing
 */
std::ostream& operator<<(std::ostream& os, parmParseVectorType const& inputVector);

/**
 * @brief Overload parameter array printing
 */
std::ostream& operator<<(std::ostream& os, parmParseArrayType const& inputArray);

std::string parm_parse_typename(parmParseType const& val);

/**
 * @brief Setter function from arbitrary value to shared parameter map
 *
 * @param sharedParam Shared parameter to be set
 * @param value Value of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, T const& value)
{
    BL_PROFILE("Gempic::Io::Impl::set_shared_param()");
    sharedParam = value;
}

/**
 * @brief Setter function from arbitrary vector to shared parameter map
 *
 * @param sharedParam Shared parameter to be set
 * @param value Values of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, std::vector<T> const& value)
{
    BL_PROFILE("Gempic::Io::Impl::set_shared_param(std::vector)");
    sharedParam = parmParseVectorType(value.begin(), value.end());
}

/**
 * @brief Setter function from arbitrary amrex::Vector to shared parameter map
 *
 * @param sharedParam Shared parameter to be set
 * @param value Values of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, amrex::Vector<T> const& value)
{
    BL_PROFILE("Gempic::Io::Impl::set_shared_param(amrex::Vector)");
    sharedParam = parmParseVectorType(value.begin(), value.end());
}

/**
 * @brief Setter function from arbitrary array to shared parameter map
 *
 * @param sharedParam Shared parameter to be set
 * @param value Values of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, std::array<T, AMREX_SPACEDIM> const& value)
{
    BL_PROFILE("Gempic::Io::Impl::set_shared_param(std::array)");
    parmParseArrayType tmp;
#ifdef _WIN32
    auto tmpIt{tmp.begin()};
#else
    auto* tmpIt{tmp.begin()};
#endif
    for (size_t i{0}; i < AMREX_SPACEDIM; ++i)
    {
        *tmpIt = value[i];
        ++tmpIt;
    }
    sharedParam = tmp;
}

/**
 * @brief Getter function from shared parameter map to arbitrary value
 *
 * @param variable variable to be set
 * @param sharedParam Shared parameter to load
 */
template <typename T>
void get_shared_param (T& variable, parmParseType const& sharedParam)
{
    BL_PROFILE("Gempic::Io::Impl::get_shared_param()");
    auto tmp = std::get<parmParseParameterType>(sharedParam);
    variable = std::get<T>(tmp);
}

/**
 * @brief Getter function from shared parameter map to arbitrary vector
 *
 * @param variable vector to be set
 * @param sharedParam Shared parameter to load
 */
template <typename T>
void get_shared_param (std::vector<T>& variable, parmParseType const& sharedParam)
{
    BL_PROFILE("Gempic::Io::Impl::get_shared_param(std::vector)");
    auto tmp = std::get<parmParseVectorType>(sharedParam);
    variable.resize(tmp.size());
    for (size_t i{0}; i < tmp.size(); i++)
    {
        variable[i] = std::get<T>(tmp[i]);
    }
}

/**
 * @brief Getter function from shared parameter map to arbitrary amrex::Vector
 *
 * @param variable amrex::Vector to be set
 * @param sharedParam Shared parameter to load
 */
template <typename T>
void get_shared_param (amrex::Vector<T>& variable, parmParseType const& sharedParam)
{
    BL_PROFILE("Gempic::Io::Impl::get_shared_param(amrex::Vector)");
    auto tmp = std::get<parmParseVectorType>(sharedParam);
    variable.resize(tmp.size());
    for (size_t i{0}; i < tmp.size(); i++)
    {
        variable[i] = std::get<T>(tmp[i]);
    }
}

/**
 * @brief Getter function from shared parameter map to arbitrary array
 *
 * @param variable array to be set
 * @param sharedParam Shared parameter to load
 */
template <typename T>
void get_shared_param (std::array<T, AMREX_SPACEDIM>& variable, parmParseType const& sharedParam)
{
    BL_PROFILE("Gempic::Io::Impl::get_shared_param(std::array)");
    auto tmp = std::get<parmParseArrayType>(sharedParam);
    for (size_t i{0}; i < AMREX_SPACEDIM; i++)
    {
        variable[i] = std::get<T>(tmp[i]);
    }
}

/**
 * @brief convert to parmParseType
 *
 * @details
 * Convenience function that converts to parmParseType in order to print correctly to output file
 *
 * @param variable Parameter to print
 */
template <typename T>
parmParseType convert_to_param (T const& variable)
{
    BL_PROFILE("Gempic::Io::Impl::convert_to_param()");
    parmParseType parameterVariable;
    set_shared_param(parameterVariable, variable);
    return parameterVariable;
}
} //namespace Impl

/**
 * @brief Objects of this class will mediate the information on and loading of parameters
 *
 * @details
 * There are two types of objects of this class: Main parameter objects and class parameter objects
 *
 * Main parameter objects are set at the outer scope and initialized with a value of true or false
 * to turn printing of parameters used on or off, respectively:
 * Parameters params{true}; // printing parameters
 * Subsequent calls to main parameter initializer cannot change printing behaviour, but will simply
 * act like un-prefixed class parameter objects
 *
 * Class parameter objects are initialized inside class initialization methods and called with a
 * class prefix:
 * Parameters params{"ClassAbc"}; // lets you load scoped parameters related to ClassAbc
 *
 */
class Parameters
{
private:
    /**
     * @brief Is this process an IO_Process?
     */
    bool m_isIOProcess;

    /**
     * @brief Dirty instance counter for construction and destruction logic purposes
     */
    inline static int s_numParameterInstances{0};

    /**
     * @brief Should we print the parameters used in our simulation? Default is false
     */
    inline static bool s_printOutput{false};

    /**
     * @brief Class prefix
     */
    std::string m_classPrefix;

    /**
     * @brief Optional class name
     */
    std::string m_className;

    /**
     * @brief Output stream for each class
     */
    std::stringstream m_classOutput;

    /**
     * @brief Flag to hide output for a specific instance when it would otherwise be printed
     */
    bool m_hideOutput{false};

    /**
     * @brief Structure of a shared parameter
     *
     * @param ref Direct reference to parameter value
     * @param isSet Has this parameter been set already?
     * @param setBy Which class or input file set this parameter?
     */
    struct SharedParam
    {
        Impl::parmParseType m_ref;
        bool m_isSet{false};
        std::string m_setBy{"not used"};
    };

    /**
     * @brief The file to print parameters to
     */
    inline static std::string s_outputFile;

    /**
     * @brief Construct the shared parameters map. Used once.
     *
     * @param sharedParamNames List of names of all shared parameters
     */
    static std::map<std::string, std::unique_ptr<SharedParam>> const init_shared_parameters (
        std::vector<std::string> const& sharedParamNames)
    {
        std::map<std::string, std::unique_ptr<SharedParam>> m;
        for (auto sharedParamName : sharedParamNames)
        {
            m.insert(std::make_pair(sharedParamName, std::make_unique<SharedParam>()));
        }
        return m;
    }

    /**
     * @brief Parameters needed by two or more classes. The names of all shared parameters go here
     *
     * @todo deprecate according to
     * <a href="https://gitlab.mpcdf.mpg.de/gempic/gempic/-/issues/157">#157</a>
     */
    inline static auto const s_sharedParams = init_shared_parameters({
        "outputFile",
        "simName",
        "nGhost", // NB: Cannot be set by user
        "nGhostExtra",
        "k",
        "sV",
        "sOmega",
    });

public:
    /**
     * @brief set the print behavior of parameters. Without calling this function, print behavior
     * is turned off by default.
     *
     * Calls to this function after the first parameters class has been initialized aborts
     * simulation
     *
     * @param printOrNot To print or not to print
     *
     */
    static void set_print_output(bool printOrNot = true);

    /**
     * @brief Main parameter initializer. Unprefixed input file scope.
     */
    explicit Parameters();

    /**
     * @brief Class parameter initializers
     * Class parameter objects are initialized inside class initialization methods and called with a
     * class prefix:
     * Parameters params{"ClassAbc"}; // lets you load scoped parameters related to ClassAbc
     */
    Parameters(std::string const& classPrefix, std::string printName = "None");

    ~Parameters();

    // Delete other default methods
    Parameters(Parameters const& rhs) = delete;            // copy constructor
    Parameters(Parameters&& rhs) = delete;                 // move constructor
    Parameters& operator=(Parameters const& rhs) = delete; // copy assignment
    Parameters& operator=(Parameters&& rhs) = delete;      // move assignment

    /**
     * @brief Get variable if it was previously set or exists in input file. Otherwise, set it.
     * Prefer get_or_set over set unless it is vital that the variable is set from here.
     *
     * This is technically doing double work. It shouldn't be a problem.
     *
     * @tparam T type of the variable. Automatically intuited
     *
     * @param variableName Name of the variable. Should match a name in the input file
     * @param variable variable to be loaded/saved
     *
     * @return bool: whether variable was set or not
     */
    template <typename T>
    bool get_or_set (std::string const& variableName, T& variable)
    {
        BL_PROFILE("Gempic::Io::Parameters::get_or_set()");
        // Check if variable is a shared parameter ('.contains' is c++20).
        auto search{s_sharedParams.find(variableName)};
        if (search != s_sharedParams.end())
        {
            // Get existing shared parameter
            if (search->second.get()->m_isSet)
            {
                get(variableName, variable);
                return false;
            }
            // Get parameter that exists in input file
            else if (query_shared(variableName, variable))
            {
                get(variableName, variable);
                return false;
            }
        }
        // Get parameter that exists in input file
        else if (query(variableName, variable))
        {
            get(variableName, variable);
            return false;
        }

        // Parameter hasn't previously been set and doesn't exist in input file.
        set(variableName, variable);
        return true;
    }

    /**
     * @brief Get variable if it was previously set or exists in input file. Else, abort simulation
     *
     * @tparam T type of the variable. Automatically intuited
     *
     * @param variableName Name of the variable. Should match a name in the input file
     * @param variable variable to be loaded
     */
    template <typename T>
    void get (std::string const& variableName, T& variable)
    {
        BL_PROFILE("Gempic::Io::Parameters::get()");
        using namespace Impl; // Necessary to access operator<<() nicely;
        // Check if variable is a shared parameter (contains is c++20).
        auto search{s_sharedParams.find(variableName)};
        if (search != s_sharedParams.end())
        {
            if (search->second.get()->m_isSet)
            {
                try
                {
                    get_shared_param(variable, search->second.get()->m_ref);
                }
                catch (std::bad_variant_access& e)
                {
                    GEMPIC_ERROR("Tried to get " + variableName + " of type " +
                                 parm_parse_typename(search->second.get()->m_ref) + " " +
                                 search->second.get()->m_setBy + ".\nFailed to copy into type " +
                                 typeid(variable).name() + " in " + m_className);
                }
            }
            else if (query_shared(variableName, variable))
            {
                set_shared_param(search->second.get()->m_ref,
                                 variable); // Trouble w/ uncopyable parameters?
                search->second.get()->m_isSet = true;
                search->second.get()->m_setBy = "read from file";
            }
            else
            {
                GEMPIC_ERROR("Parameter " + variableName + " not in input file!");
            }
            m_classOutput << variableName << " = " << convert_to_param(variable) << " # "
                          << search->second.get()->m_setBy << '\n';
        }
        // local parameter
        else
        {
            // special case for prefix-less parameter instances
            std::string prefixDot{m_classPrefix.empty() ? m_classPrefix : m_classPrefix + "."};
            if (!query(variableName, variable))
            {
                GEMPIC_ERROR("Parameter " + prefixDot + variableName + " not in input file!");
            }
            m_classOutput << prefixDot << variableName << " = " << convert_to_param(variable)
                          << "\n";
        }
        return;
    }

    /**
     * @brief Set variable if it was not previously set or loaded from input file. Else, abort
     * simulation. Prefer get_or_set unless it is vital that the variable is set from here.
     *
     * @tparam T type of the variable. Automatically intuited
     *
     * @param variableName Name of the variable
     * @param variable variable to be saved
     */
    template <typename T>
    void set (std::string const& variableName, T const& variable)
    {
        BL_PROFILE("Gempic::Io::Parameters::set()");
        using namespace Impl; // Necessary to access operator<<() nicely;
        // Check if variable is a shared parameter (contains is c++20).
        auto search{s_sharedParams.find(variableName)};
        if (search != s_sharedParams.end())
        {
            // variable is already set!
            if (search->second.get()->m_isSet)
            {
                GEMPIC_ERROR(m_className + " attempted to set parameter " + variableName +
                             " already " + search->second.get()->m_setBy);
            }
            else
            {
                set_shared_param(search->second.get()->m_ref, variable);
                search->second.get()->m_isSet = true;
                search->second.get()->m_setBy = "set by " + m_className;
            }
            m_classOutput << variableName << " = " << convert_to_param(variable) << " # set here\n";
        }
        else
        {
            if (is_in_input_file(variableName))
            {
                std::string prefixDot{m_classPrefix.empty() ? m_classPrefix : m_classPrefix + "."};
                GEMPIC_ERROR("Parameter " + prefixDot + variableName + " already set previously!");
            }
            set_local_param(variableName, variable);
            // Update output
            m_classOutput << m_classPrefix << "." << variableName << " = "
                          << convert_to_param(variable) << '\n';
        }
        if (Gempic::Utils::Verbosity::level() > 0)
        {
            std::stringstream ss;
            ss << "Warning: Variable " << variableName << " = " << convert_to_param(variable)
               << " set by default in " << m_className << "\n";
            amrex::Warning(ss.str().c_str());
        }
        return;
    }

    /**
     * @brief reset the parameters, including those read from input file
     *
     * Calls to this function while a parameters object is alive initialized aborts simulation
     *
     */
    static void reset();

    /**
     * @brief Hide class parameter output for this instance
     */
    void hide_class_output();

    /**
     * @brief Print class parameters. Automatically called by destructor.
     *
     * @details
     * Prints class related parameter information to s_outputFile if s_printOutput is set and
     * hide_class_output is not called.
     */
    void print_class_parameters();

    /**
     * @brief Print shared parameters. Automatically called by destructor.
     *
     * @details
     * Prints shared parameters used in the simulation to s_outputFile if s_printOutput is set.
     */
    void print_shared_parameters();

    /**
     * @brief Does variable exist, either set in shared parameters or in input file?
     *
     * @details Prefer get_or_set instead of implementing logic based on this return value
     */
    bool exists(std::string const& variableName);

    /**
     * @brief is variable in input file?
     *
     * @details This function should not be used except when debugging. Prefer to use get_or_set
     * instead of implementing logic based on this return value.
     */
    bool is_in_input_file(std::string const& variableName);

    /**
     * @brief Has shared parameter been set? Where?
     *
     * @details This function should not be used except when debugging. Prefer to use get_or_set
     * instead of implementing logic based on this return value.
     */
    std::string has_been_set_by(std::string const& variableName);

private:
    /**
     * @brief query local parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    template <typename T>
    bool query (std::string const& variableName, T& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query()");
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.query(variableName.c_str(), value);
    }

    /**
     * @brief query local vector parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     *
     * @todo amrex::ParmParse has no inherent size check. Alternatively, call queryarr with
     * (name, variable, startIndex, endIndex)?
     */
    template <typename T>
    bool query (std::string const& variableName, std::vector<T>& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query(std::vector)");
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query local amrex::Vector parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    template <typename T>
    bool query (std::string const& variableName, amrex::Vector<T>& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query(amrex::Vector)");
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query local amrex::IntVect parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    bool query (std::string const& variableName, amrex::IntVect& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query(amrex::IntVect)");
        amrex::Vector<int> tmp(AMREX_SPACEDIM);
        amrex::ParmParse ppPrefix(m_classPrefix);
        bool found = ppPrefix.queryarr(variableName.c_str(), tmp);
        if (found)
        {
            for (int i = 0; i < AMREX_SPACEDIM; ++i)
            {
                value[i] = tmp[i];
            }
        }
        return found;
    }

    /**
     * @brief query shared parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    template <typename T>
    bool query_shared (std::string const& variableName, T& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query_shared()");
        amrex::ParmParse pp;
        return pp.query(variableName.c_str(), value);
    }

    /**
     * @brief query shared vector parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    template <typename T>
    bool query_shared (std::string const& variableName, std::vector<T>& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query_shared(std::vector)");
        amrex::ParmParse pp;
        return pp.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query shared amrex::Vector parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    template <typename T>
    bool query_shared (std::string const& variableName, amrex::Vector<T>& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query_shared(amrex::Vector)");
        amrex::ParmParse pp;
        return pp.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query shared amrex::IntVect parameters
     *
     * @param variableName Name of the parameter to be queried
     * @param value Reference to write potentially loaded value to
     */
    bool query_shared (std::string const& variableName, amrex::IntVect& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::query_shared(amrex::IntVect)");
        amrex::Vector<int> tmp(AMREX_SPACEDIM);
        amrex::ParmParse pp;
        bool found = pp.queryarr(variableName.c_str(), tmp);
        if (found)
        {
            for (int i = 0; i < AMREX_SPACEDIM; ++i)
            {
                value[i] = tmp[i];
            }
        }
        return found;
    }

    /**
     * @brief set local (i.e. non-shared) parameters
     *
     * @param variableName Name of the parameter to be added
     * @param value Value of the parameter to be added
     */
    template <typename T>
    void set_local_param (std::string const& variableName, T const& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::set_local_param()");
        amrex::ParmParse ppPrefix(m_classPrefix);
        ppPrefix.add(variableName.c_str(), value);
    }

    /**
     * @brief set local (i.e. non-shared) vector parameters
     *
     * @param variableName Name of the parameter to be added
     * @param value Value of the parameter to be added
     */
    template <typename T>
    void set_local_param (std::string const& variableName, std::vector<T> const& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::set_local_param(std::vector)");
        amrex::ParmParse ppPrefix(m_classPrefix);
        ppPrefix.addarr(variableName.c_str(), value);
    }

    /**
     * @brief set local (i.e. non-shared) array parameters
     *
     * @param variableName Name of the parameter to be added
     * @param value Value of the parameter to be added
     */
    template <typename T, size_t s>
    void set_local_param (std::string const& variableName, std::array<T, s> const& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::set_local_param(std::array)");
        amrex::ParmParse ppPrefix(m_classPrefix);
        std::vector<T> v;
        v.resize(s);
        for (std::size_t i = 0; i < s; ++i)
        {
            v[i] = value[i];
        }
        ppPrefix.addarr(variableName.c_str(), v);
    }

    /**
     * @brief set local (i.e. non-shared) amrex::Vector parameters
     *
     * @param variableName Name of the parameter to be added
     * @param value Value of the parameter to be added
     */
    template <typename T>
    void set_local_param (std::string const& variableName, amrex::Vector<T> const& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::set_local_param(amrex::Vector)");
        amrex::ParmParse ppPrefix(m_classPrefix);
        ppPrefix.addarr(variableName.c_str(), value);
    }

    /**
     * @brief set local (i.e. non-shared) amrex::IntVect parameters
     *
     * @param variableName Name of the parameter to be added
     * @param value Value of the parameter to be added
     */
    template <typename T>
    void set_local_param (std::string const& variableName, amrex::IntVect const& value)
    {
        BL_PROFILE("Gempic::Io::Parameters::set_local_param(amrex::IntVect)");
        amrex::Vector<int> tmp(AMREX_SPACEDIM);
        for (int i = 0; i < AMREX_SPACEDIM; ++i)
        {
            tmp[i] = value[i];
        }
        amrex::ParmParse ppPrefix(m_classPrefix);
        ppPrefix.addarr(variableName.c_str(), tmp);
    }

    /**
     * @brief print shared parameter
     *
     * @details
     * Convenience function to differentiate between shared parameters that have been set and those
     * that have not
     *
     * @param ofs Output file stream
     * @param variableName Name of the parameter to be printed
     * @param variable Reference to the value of the parameter
     */
    template <typename T>
    void print_shared_parameter (std::ofstream& ofs, std::string const& variableName, T& variable)
    {
        BL_PROFILE("Gempic::Io::Parameters::print_shared_parameter()");
        using namespace Impl; // Necessary to access operator<<() nicely;
        if (s_sharedParams.at(variableName).get()->m_isSet)
        {
            ofs << variableName << " " << convertToParam(variable) << " # "
                << s_sharedParams.at(variableName).get()->m_setBy << '\n';
        }
        else
        {
            ofs << variableName << " # " << s_sharedParams.at(variableName).get()->m_setBy << '\n';
        }
    }
};

} // namespace Gempic::Io

#endif

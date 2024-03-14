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
// Exhaustive list of parameter types possible to get from ParmParse (except for IntVect, because
// that's broken)
using parmParseParameterType =
    std::variant<bool, int, long, long long, float, double, std::string, amrex::Box>;
using parmParseVectorType = std::vector<parmParseParameterType>;
using parmParseArrayType = std::array<parmParseParameterType, GEMPIC_SPACEDIM>;
using parmParseType = std::variant<parmParseParameterType, parmParseVectorType, parmParseArrayType>;

/**
 * @brief Overload parmParseType printing
 *
 */
std::ostream& operator<<(std::ostream& os, const parmParseType& val);

std::ostream& operator<<(std::ostream& os, const parmParseParameterType& val);
/**
 * @brief Overload parameter vector printing
 */
std::ostream& operator<<(std::ostream& os, const parmParseVectorType& inputVector);

/**
 * @brief Overload parameter array printing
 */
std::ostream& operator<<(std::ostream& os, const parmParseArrayType& inputArray);

/**
 * @brief Setter function from arbitrary value to shared parameter map
 *
 * @param sharedParam : Shared parameter to be set
 * @param value : Value of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, const T& value)
{
    sharedParam = value;
}

/**
 * @brief Setter function from arbitrary vector to shared parameter map
 *
 * @param sharedParam : Shared parameter to be set
 * @param value : Values of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, const std::vector<T>& value)
{
    sharedParam = parmParseVectorType(value.begin(), value.end());
}

/**
 * @brief Setter function from arbitrary amrex::Vector to shared parameter map
 *
 * @param sharedParam : Shared parameter to be set
 * @param value : Values of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, const amrex::Vector<T>& value)
{
    sharedParam = parmParseVectorType(value.begin(), value.end());
}

/**
 * @brief Setter function from arbitrary array to shared parameter map
 *
 * @param sharedParam : Shared parameter to be set
 * @param value : Values of the parameter
 */
template <typename T>
void set_shared_param (parmParseType& sharedParam, const std::array<T, GEMPIC_SPACEDIM>& value)
{
    parmParseArrayType tmp;
    auto* tmpIt{tmp.begin()};
    for (size_t i{0}; i < GEMPIC_SPACEDIM; ++i)
    {
        *tmpIt = value[i];
        ++tmpIt;
    }
    sharedParam = tmp;
}

/**
 * @brief Getter function from shared parameter map to arbitrary value
 *
 * @param variable : variable to be set
 * @param sharedParam : Shared parameter to load
 */
template <typename T>
void get_shared_param (T& variable, const parmParseType& sharedParam)
{
    auto tmp = std::get<parmParseParameterType>(sharedParam);
    variable = std::get<T>(tmp);
}

/**
 * @brief Getter function from shared parameter map to arbitrary vector
 *
 * @param variable : vector to be set
 * @param sharedParam : Shared parameter to load
 */
template <typename T>
void get_shared_param (std::vector<T>& variable, const parmParseType& sharedParam)
{
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
 * @param variable : amrex::Vector to be set
 * @param sharedParam : Shared parameter to load
 */
template <typename T>
void get_shared_param (amrex::Vector<T>& variable, const parmParseType& sharedParam)
{
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
 * @param variable : array to be set
 * @param sharedParam : Shared parameter to load
 */
template <typename T>
void get_shared_param (std::array<T, GEMPIC_SPACEDIM>& variable, const parmParseType& sharedParam)
{
    auto tmp = std::get<parmParseArrayType>(sharedParam);
    for (size_t i{0}; i < GEMPIC_SPACEDIM; i++)
    {
        variable[i] = std::get<T>(tmp[i]);
    }
}

/**
 * @brief convert to parmParseType
 *
 * Description:
 * Convenience function that converts to parmParseType in order to print correctly to output file
 *
 * @param variable : Parameter to print
 */
template <typename T>
parmParseType convert_to_param (T variable)
{
    parmParseType parameterVariable;
    set_shared_param(parameterVariable, variable);
    return parameterVariable;
}
}  //namespace Impl

/**
 * @brief Objects of this class will mediate the information on and loading of parameters
 *
 * Description:
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
     * @brief Structure of a shared parameter
     *
     * @param ref : Direct reference to parameter value
     * @param isSet : Has this parameter been set already?
     * @param setBy : Which class or input file set this parameter?
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
     * @param sharedParamNames : List of names of all shared parameters
     */
    static const std::map<std::string, std::unique_ptr<SharedParam>> init_shared_parameters (
        const std::vector<std::string>& sharedParamNames)
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
     */
    inline static const auto s_sharedParams = init_shared_parameters({
        "outputFile",
        "simName",
        "nGhost",  // NB: Cannot be set by user
        "nGhostExtra",
        "nCellVector",
        "maxGridSizeVector",
        "isPeriodicVector",
        "k",
        "domainLo",
        "domainHi",
    });

public:
    /**
     * @brief set the print behavior of parameters. Without calling this function, print behavior
     * is turned off by default.
     *
     * Calls to this function after the first parameters class has been initialized aborts
     * simulation
     *
     * @param printOrNot : To print or not to print
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
     * Parameters params{"ClassAbc"}; // lets you load scoped parameters related to ClassABC
     */
    Parameters(const std::string& classPrefix, std::string printName = "None");

    ~Parameters();

    // Delete other default methods
    Parameters(Parameters const& rhs) = delete;             // copy constructor
    Parameters(Parameters&& rhs) = delete;                  // move constructor
    Parameters& operator=(const Parameters& rhs) = delete;  // copy assignment
    Parameters& operator=(Parameters&& rhs) = delete;       // move assignment

    /**
     * @brief Get variable if it was previously set or exists in input file. Otherwise, set it.
     * Prefer getOrSet over set unless it is vital that the variable is set from here.
     *
     * This is technically doing double work. It shouldn't be a problem.
     *
     * @tparam T : type of the variable. Automatically intuited
     *
     * @param variableName : Name of the variable. Should match a name in the input file
     * @param variable : variable to be loaded/saved
     *
     * @return variableWasSet : whether variable was set or not
     */
    template <typename T>
    bool get_or_set (const std::string& variableName, T& variable)
    {
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
     * @tparam T : type of the variable. Automatically intuited
     *
     * @param variableName : Name of the variable. Should match a name in the input file
     * @param variable : variable to be loaded
     */
    template <typename T>
    void get (const std::string& variableName, T& variable)
    {
        using namespace Impl;  // utility functions for this class
        // Check if variable is a shared parameter (contains is c++20).
        auto search{s_sharedParams.find(variableName)};
        if (search != s_sharedParams.end())
        {
            if (search->second.get()->m_isSet)
            {
                get_shared_param(variable, search->second.get()->m_ref);
            }
            else if (query_shared(variableName, variable))
            {
                set_shared_param(search->second.get()->m_ref,
                                 variable);  // Trouble w/ uncopyable parameters?
                search->second.get()->m_isSet = true;
                search->second.get()->m_setBy = "read from file";
            }
            else
            {
                std::cerr << "Error: Parameter " << variableName << " not in input file!";
                exit(Error::InvalidInput);
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
                std::cerr << "Error: Parameter " << prefixDot << variableName
                          << " not in input file!";
                exit(Error::InvalidInput);
            }
            m_classOutput << prefixDot << variableName << " = " << convert_to_param(variable)
                          << "\n";
        }
        return;
    }

    /**
     * @brief Set variable if it was not previously set or loaded from input file. Else, abort
     * simulation. Prefer getOrSet unless it is vital that the variable is set from here.
     *
     * @tparam T : type of the variable. Automatically intuited
     *
     * @param variableName : Name of the variable. Should match a name in the input file
     * @param variable : variable to be saved
     */
    template <typename T>
    void set (const std::string& variableName, T& variable)
    {
        using namespace Impl;  // utility functions for this class
        // Check if variable is a shared parameter (contains is c++20).
        auto search{s_sharedParams.find(variableName)};
        if (search != s_sharedParams.end())
        {
            // variable is already set!
            if (search->second.get()->m_isSet)
            {
                std::cerr << "Error: " << m_className << " attempted to set parameter "
                          << variableName << " already " << search->second.get()->m_setBy;
                exit(Error::AttemptedParameterChange);
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
            // Update output
            m_classOutput << m_classPrefix << "." << variableName << " = "
                          << convert_to_param(variable) << '\n';
        }
        if (Gempic::Utils::Verbosity::level() > 0)
        {
            std::cerr << "Variable" << variableName << " = " << convert_to_param(variable)
                      << " set by default in " << m_className << "\n";
        }
        return;
    }

    /**
     * @brief Print class parameters. Automatically called by destructor.
     *
     * Description:
     * Prints class related parameter information to s_outputFile if s_printOutput is set.
     */
    void print_class_parameters();

    /**
     * @brief Print shared parameters. Automatically called by destructor.
     *
     * Description:
     * Prints shared parameters used in the simulation to s_outputFile if s_printOutput is set.
     */
    void print_shared_parameters();

    /**
     * @brief Does variable exist, either set in shared parameters or in input file?
     *
     * Description: Prefer to use getOrSet instead of implementing logic based on this return value
     */
    bool exists(const std::string& variableName);

    /**
     * @brief is variable in input file?
     *
     * Description: This function should not be used except when debugging. Prefer to use getOrSet
     * instead of implementing logic based on this return value.
     */
    bool is_in_input_file(const std::string& variableName);

    /**
     * @brief Has shared parameter been set? Where?
     *
     * Description: This function should not be used except when debugging. Prefer to use getOrSet
     * instead of implementing logic based on this return value.
     */
    std::string has_been_set_by(const std::string& variableName);

private:
    /**
     * @brief query local parameters
     *
     * @param variableName : Name of the parameter to be queried
     * @param value : Reference to write potentially loaded value to
     */
    template <typename T>
    bool query (std::string variableName, T& value)
    {
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.query(variableName.c_str(), value);
    }

    /**
     * @brief query local vector parameters
     *
     * @param variableName : Name of the parameter to be queried
     * @param value : Reference to write potentially loaded value to
     *
     * @todo amrex::ParmParse has no inherent size check. Alternatively, call queryarr with
     * (name, variable, startIndex, endIndex)?
     */
    template <typename T>
    bool query (std::string variableName, std::vector<T>& value)
    {
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query local amrex::Vector parameters
     *
     * @param variableName : Name of the parameter to be queried
     * @param value : Reference to write potentially loaded value to
     */
    template <typename T>
    bool query (std::string variableName, amrex::Vector<T>& value)
    {
        amrex::ParmParse ppPrefix(m_classPrefix);
        return ppPrefix.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query shared parameters
     *
     * @param variableName : Name of the parameter to be queried
     * @param value : Reference to write potentially loaded value to
     */
    template <typename T>
    bool query_shared (std::string variableName, T& value)
    {
        amrex::ParmParse pp;
        return pp.query(variableName.c_str(), value);
    }

    /**
     * @brief query shared vector parameters
     *
     * @param variableName : Name of the parameter to be queried
     * @param value : Reference to write potentially loaded value to
     */
    template <typename T>
    bool query_shared (std::string variableName, std::vector<T>& value)
    {
        amrex::ParmParse pp;
        return pp.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief query shared amrex::Vector parameters
     *
     * @param variableName : Name of the parameter to be queried
     * @param value : Reference to write potentially loaded value to
     */
    template <typename T>
    bool query_shared (std::string variableName, amrex::Vector<T>& value)
    {
        amrex::ParmParse pp;
        return pp.queryarr(variableName.c_str(), value);
    }

    /**
     * @brief print shared parameter
     *
     * Description:
     * Convenience function to differentiate between shared parameters that have been set and those
     * that have not
     *
     * @param ofs : Output file stream
     * @param variableName : Name of the parameter to be printed
     * @param variable : Reference to the value of the parameter
     */
    template <typename T>
    void print_shared_parameter (std::ofstream& ofs, std::string variableName, T& variable)
    {
        using namespace Impl;  // utility functions for this class
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

}  // namespace Gempic::Io

#endif

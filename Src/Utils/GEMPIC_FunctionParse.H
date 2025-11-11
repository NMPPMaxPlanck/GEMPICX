#ifndef GEMPIC_FUNCTION_PARSE_H_
#define GEMPIC_FUNCTION_PARSE_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_Parser.H>

#include "GEMPIC_Parameters.H"

namespace Gempic::Utils
{
/**
 * @brief Parse functions based on string array. Please keep parseArray alive.
 *
 * @details
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @tparam functionNumber Number of functions to parse
 *
 * @param variableStrings List of strings to parse
 * @param params Optional parameter class object for parsing scoped functions
 *
 * @return [parseArray, functionArray]
 */
template <size_t functionNumber>
auto parse_functions (amrex::Array<std::string, functionNumber> const& variableStrings,
                      Io::Parameters& params)
{
    BL_PROFILE("Gempic::Utils::parse_functions(variableStrings,params)");
    amrex::Vector<amrex::Real> k;
    bool kExists{params.exists("k")};
    if (kExists)
    {
        params.get("k", k);
    }

    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, functionNumber> evalArray;
    amrex::Array<amrex::Parser, functionNumber> parseArray;

    size_t index{0};
    for (auto& variableString : variableStrings)
    {
        std::string parseString;
        params.get(variableString, parseString);
        parseArray[index] = amrex::Parser(parseString);
        if (kExists)
        {
            AMREX_D_TERM(parseArray[index].setConstant("kvarx", k[xDir]);
                         , parseArray[index].setConstant("kvary", k[yDir]);
                         , parseArray[index].setConstant("kvarz", k[zDir]);)
        }
        parseArray[index].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        evalArray[index] = parseArray[index].template compile<AMREX_SPACEDIM + 1>();
        ++index;
    }
    return std::make_tuple(parseArray, evalArray);
}

/**
 * @brief Parse functions based on string array. Please keep parseArray alive.
 *
 * @details
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @tparam functionNumber Number of functions to parse
 *
 * @param variableStrings List of strings to parse
 *
 * @return [parseArray, functionArray]
 */
template <size_t functionNumber>
auto parse_functions (amrex::Array<std::string, functionNumber> const& variableStrings)
{
    BL_PROFILE("Gempic::Utils::parse_functions(variableStrings)");
    Io::Parameters params{"Function", "function parse_functions"};
    return parse_functions(variableStrings, params);
}

/**
 * @brief Compiles an array of AMReX parsers into an array of compiled executors.
 *
 * @details
 * Convenience function to compile functions using AMReX's Parser module from uncompiled parse
 * objects (parseArray).
 *
 * @tparam functionNumber The number of functions in the parser array.
 * @param parseArray An array of AMReX parsers to compile.
 *
 * @return An array of compiled executors for the given parsers.
 */
template <size_t functionNumber>
auto compile_functions (amrex::Array<amrex::Parser, functionNumber> parseArray)
{
    BL_PROFILE("Gempic::Utils::compile_functions(parseArray)");

    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, functionNumber> evalArray;

    for (size_t index{0}; index < functionNumber; ++index)
    {
        evalArray[index] = parseArray[index].template compile<AMREX_SPACEDIM + 1>();
    }
    return evalArray;
}

/**
 * @brief Parse function based on string. Please keep parse object (return arg 1) alive.
 *
 * @details
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @param variableString String to parse
 * @param params Optional parameter class object for parsing a scoped function
 *
 * @return [parser, function]
 */
inline auto parse_function (std::string const& variableString, Io::Parameters& params)
{
    BL_PROFILE("Gempic::Utils::parse_function(variableString,params)");
    auto [parser, eval] = parse_functions<1>({variableString}, params);
    return std::make_tuple(parser[0], eval[0]);
}

/**
 * @brief Parse function based on string. Please keep parse object (return arg 1) alive.
 *
 * @details
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @param variableString String to parse
 *
 * @return [parser, function]
 */
inline auto parse_function (std::string const& variableString)
{
    BL_PROFILE("Gempic::Utils::parse_function(variableString)");
    Io::Parameters params{"Function", "function parseFunction"};
    return parse_function(variableString, params);
}

/**
 * @brief Compiles an AMReX parser into a compiled executor.
 *
 * @details
 * Convenience function to compile a single function using AMReX's Parser module from uncompiled
 * parse object.
 *
 * @param parser A single AMReX parser to compile.
 *
 * @return Compiled executor for the given parser.
 */
inline auto compile_function (amrex::Parser parser)
{
    BL_PROFILE("Gempic::Utils::compile_function(parser)");
    auto eval = parser.template compile<AMREX_SPACEDIM + 1>();
    return eval;
}

} // namespace Gempic::Utils
#endif
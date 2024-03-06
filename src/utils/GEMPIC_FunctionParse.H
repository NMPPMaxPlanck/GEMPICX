#ifndef GEMPIC_FUNCTION_PARSE_H_
#define GEMPIC_FUNCTION_PARSE_H_

#include <AMReX_Parser.H>

#include "GEMPIC_Parameters.H"

namespace Gempic::Utils
{
/**
 * @brief Parse functions based on string array. Please keep parseArray alive.
 *
 * Description:
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @tparam functionNumber : Number of functions to parse
 *
 * @param variableStrings : List of strings to parse
 * @param params : Optional parameter class object for parsing scoped functions
 *
 * @return [parseArray, functionArray]
 */
template <size_t functionNumber>
auto parse_functions (const amrex::Array<std::string, functionNumber>& variableStrings,
                      Io::Parameters& params)
{
    amrex::Vector<amrex::Real> k;
    bool kExists{params.exists("k")};
    if (kExists)
    {
        params.get("k", k);
    }

    amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, functionNumber> evalArray;
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
        evalArray[index] = parseArray[index].template compile<GEMPIC_SPACEDIM + 1>();
        ++index;
    }
    return std::make_tuple(parseArray, evalArray);
}

/**
 * @brief Parse functions based on string array. Please keep parseArray alive.
 *
 * Description:
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @tparam functionNumber : Number of functions to parse
 *
 * @param variableStrings : List of strings to parse
 *
 * @return [parseArray, functionArray]
 */
template <size_t functionNumber>
auto parse_functions (const amrex::Array<std::string, functionNumber>& variableStrings)
{
    Io::Parameters params{"function", "function parseFunctions"};
    return parse_functions(variableStrings, params);
}

/**
 * @brief Parse function based on string. Please keep parse object (return arg 1) alive.
 *
 * Description:
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @param variableString : String to parse
 * @param params : Optional parameter class object for parsing a scoped function
 *
 * @return [parser, function]
 */
inline auto parse_function (const std::string& variableString, Io::Parameters& params)
{
    auto [parser, eval] = parse_functions<1>({variableString}, params);
    return std::make_tuple(parser[0], eval[0]);
}

/**
 * @brief Parse function based on string. Please keep parse object (return arg 1) alive.
 *
 * Description:
 * Convenience function to load and compile functions using AMReX's Parser module.
 * The uncompiled parse object (parseArray) MUST be kept alive as long as the compiled object needs
 * to be used.
 *
 * @param variableString : String to parse
 *
 * @return [parser, function]
 */
inline auto parse_function (const std::string& variableString)
{
    Io::Parameters params{"function", "function parseFunction"};
    return parse_function(variableString, params);
}
}  // namespace Gempic::Utils
#endif
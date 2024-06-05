#ifndef GEMPIC_VERBOSITY_H
#define GEMPIC_VERBOSITY_H

namespace Gempic::Utils
{
/**
 * @brief Get and set verbosity level for Gempic
 *
 * @param level : Non-negative integer sets the level of verbosity
 */
struct Verbosity
{
    /**
     * @brief Set verbosity level for Gempic. May only be called once before the verbosity is
     * queried.
     *
     * @param level : Non-negative integer sets the level of verbosity
     */
    explicit Verbosity(int level);

    /**
     * @brief Set verbosity level for Gempic. May only be called once before the verbosity is
     * queried.
     *
     * @param level : Non-negative integer sets the level of verbosity
     */
    static void set_level(int level);

    /**
     * @brief Get verbosity level for Gempic
     */
    static int level();

private:
    inline static int s_verbosityLevel{0};
    inline static bool s_settable{true};
};
}  // namespace Gempic::Utils
#endif
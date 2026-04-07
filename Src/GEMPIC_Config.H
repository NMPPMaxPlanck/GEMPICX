/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_
#include <math.h>
#include <numbers>
#include <variant>

#ifndef M_PI /// M_PI is gnu
#define M_PI std::numbers::pi_v<amrex::Real>
#endif

#if defined(_DEBUG) || defined(DEBUG)
#define GEMPIC_FORCE_INLINE inline
#else
#define GEMPIC_FORCE_INLINE AMREX_FORCE_INLINE
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_MULT(a,b,c) macro multiplies the elements of a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#define GEMPIC_D_MULT(a, b, c) (a)
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_MULT(a, b, c) ((a) * (b))
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_MULT(a, b, c) ((a) * (b) * (c))
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_ADD(a,b,c) macro adds the elements of a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#define GEMPIC_D_ADD(a, b, c) (a)
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_ADD(a, b, c) ((a) + (b))
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_ADD(a, b, c) ((a) + (b) + (c))
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_PAD(a,b,c) macro replaces the last arguments by zeros
 * in a comma-separated list of the 3 arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#define GEMPIC_D_PAD(a, b, c) a, 0, 0
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_PAD(a, b, c) a, b, 0
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_PAD(a, b, c) a, b, c
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_PAD_ONE(a,b,c) macro replaces the last arguments by one
 * in a comma-separated list of the 3 arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#define GEMPIC_D_PAD_ONE(a, b, c) a, 1, 1
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_PAD_ONE(a, b, c) a, b, 1
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_PAD_ONE(a, b, c) a, b, c
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_EXCL(a,b,c) macro evaluates the elements of a
 * comma-separated list of the third, second or all three of the arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 * Chiefly used to define constants that may be lacking in 1D or 2D.
 */
// Tell clang-tidy that we know what we are doing with this macro:
// NOLINTBEGIN(bugprone-macro-parentheses)
#define GEMPIC_D_EXCL(a, b, c) a b c
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_EXCL(a, b, c) b c
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_EXCL(a, b, c) c
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_LOOP_BEGIN(a,b,c) macro initializes nested for loops using a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#define GEMPIC_D_LOOP_BEGIN(a, b, c) \
    a                                \
    {
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_LOOP_BEGIN(a, b, c) \
    b                                \
    {                                \
        a                            \
        {
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_LOOP_BEGIN(a, b, c) \
    c                                \
    {                                \
        b                            \
        {                            \
            a                        \
            {
#endif
// NOLINTEND(bugprone-macro-parentheses)

/**
 * The GEMPIC_D_LOOP_END macro ends a nested for loop begun by GEMPIC_D_LOOP_BEGIN
 */
#if AMREX_SPACEDIM == 1
#define GEMPIC_D_LOOP_END }
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_LOOP_END \
    }                     \
    }
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_LOOP_END \
    }                     \
    }                     \
    }
#endif

/**
 * The GEMPIC_D_AND(a,b,c) macro compare conditions of a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of AMREX_SPACEDIM.
 */
#if AMREX_SPACEDIM == 1
#define GEMPIC_D_AND(a, b, c) a
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_AND(a, b, c) ((a) && (b))
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_AND(a, b, c) ((a) && (b) && (c))
#endif

/**
 * Tiny profiler macro for host device functions. Profiling uses a host function which cannot be
 * called from a device context.
 */
#ifdef GEMPIC_PROFILE_KERNELS
#define BL_PROFILE_HOST_DEVICE(fname) AMREX_IF_ON_HOST((BL_PROFILE(fname)))
#else
#define BL_PROFILE_HOST_DEVICE(fname)
#endif

/**
 * The GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE and GEMPIC_ERROR macros allow for dynamic messages at run
 * time
 */
#define GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(EX, MSG) \
    (EX) ? ((void)0) : amrex::Assert(#EX, __FILE__, __LINE__, std::string(MSG).c_str())

#define GEMPIC_ERROR(MSG) amrex::Assert(std::string(MSG).c_str(), __FILE__, __LINE__)

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_PAD_ONEX(a), like GEMPIC_D_PAD_ONE
 * but working only on the x component
 */
#define GEMPIC_D_PAD_ONEX(a) a
#define GEMPIC_D_PAD_ONEY(a) 1
#define GEMPIC_D_PAD_ONEZ(a) 1
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_PAD_ONEX(a) a
#define GEMPIC_D_PAD_ONEY(a) a
#define GEMPIC_D_PAD_ONEZ(a) 1
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_PAD_ONEX(a) a
#define GEMPIC_D_PAD_ONEY(a) a
#define GEMPIC_D_PAD_ONEZ(a) a
#endif

#if AMREX_SPACEDIM == 1
/**
 * The GEMPIC_D_PAD_X(a), like GEMPIC_D_PAD
 * but working only on the x component
 */
#define GEMPIC_D_PAD_X(a) a
#define GEMPIC_D_PAD_Y(a) 0
#define GEMPIC_D_PAD_Z(a) 0
#elif AMREX_SPACEDIM == 2
#define GEMPIC_D_PAD_X(a) a
#define GEMPIC_D_PAD_Y(a) a
#define GEMPIC_D_PAD_Z(a) 0
#elif AMREX_SPACEDIM == 3
#define GEMPIC_D_PAD_X(a) a
#define GEMPIC_D_PAD_Y(a) a
#define GEMPIC_D_PAD_Z(a) a
#endif

///
///Use to avoid "unused parameter" warnings.
///
///If a function must have a parameter in order to respect an API, but it does not in fact use it,
///DON'T call this macro; comment out the parameter in the function signature, e.g. replace
///``int i,`` with ``int /*i*/,``.
///
///If the parameter is only used in some dimensions/constexpr decisions, then in the others,
///DO call this macro with the parameter as an argument in order to avoid unused variable compiler
///warnings.
///

#define UNUSED(pp) ((void)(pp))

/**
 * A simple "print debug message" function.
 *
 * \note
 * If anyone needs to use this for a multi-process run:
 *
 *      * It's very useful to concatenate everything into a single
 *      string before feeding it to "std::cerr", and also to use "std::endl"
 *      to ensure that the buffer is flushed as soon as possible.
 *      Otherwise messages of different processes are likely to be merged.
 *      * It is useful to prepand the MPI rank to the message.
 */
#ifdef GEMPIC_DEBUG_MESSAGES
#define GEMPIC_DEBUG(MSG) std::cerr << std::string(MSG) << std::endl;
#else
#define GEMPIC_DEBUG(MSG)
#endif
// Useful for static_assert, because without a layer of indirection a false condition triggers
// even in code that's never supposed to be reached
template <typename> // only accepts types
constexpr bool gempic_always_false ()
{
    return false;
}
template <auto> // only accepts values
constexpr bool gempic_always_false ()
{
    return false;
}

//! Directions available in GEMPIC
enum Direction : unsigned int
{
    xDir = 0,
    yDir = 1,
    zDir = 2,
    noDir
};

// deRham enum
enum class Sequence
{
    primal,
    dual
};

enum class Form
{
    zero,
    one,
    two,
    three
};

template <typename T>
using canonical_t = std::remove_cvref_t<T>;

// Checks if T is in the variant
template <typename T, typename Variant>
struct is_variant_member;

/**
 * @brief check if type T is a variant alternative of the variants
 * @tparam T a type
 * @tparam ...Ts parameter pack listing all variant alternatives
 * @note    - iterate over the variant alternatives automatically, without manually listing each of
 * them.
 *          - disjunction_v is a helper for disjunction<...>::value
 *          - disjunction: once a true trait is found, the rest are not evaluated
 */
template <typename T, typename... Ts>
struct is_variant_member<T, std::variant<Ts...>> : std::disjunction<std::is_same<T, Ts>...>
{
};

template <std::size_t... i, typename Func>
constexpr void ct_for_index_sequence (std::index_sequence<i...>, Func&& f)
{
    ((f(std::integral_constant<std::size_t, i>{})), ...);
}

#endif

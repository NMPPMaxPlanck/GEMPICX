#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_
#include <math.h>

#ifndef M_PI /// M_PI is gnu @todo: switch to c++20 and std::numbers::pi_v<amrex::Real>
#define M_PI 3.14159265358979323846
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

enum Direction : unsigned int
{
    xDir = 0,
    yDir = 1,
    zDir = 2,
    noDir = 3,
};

enum class Grid
{
    primal,
    dual
};

enum class Space
{
    node,
    edge,
    face,
    cell
};

enum class Field
{
    PrimalZeroForm,
    PrimalOneForm,
    PrimalTwoForm,
    PrimalThreeForm,

    DualZeroForm,
    DualOneForm,
    DualTwoForm,
    DualThreeForm,
};

template <Field field>
constexpr Field complement_form ()
{
    switch (field)
    {
        case Field::PrimalZeroForm:
            return Field::DualThreeForm;
        case Field::PrimalOneForm:
            return Field::DualTwoForm;
        case Field::PrimalTwoForm:
            return Field::DualOneForm;
        case Field::PrimalThreeForm:
            return Field::DualZeroForm;
        case Field::DualZeroForm:
            return Field::DualThreeForm;
        case Field::DualOneForm:
            return Field::DualTwoForm;
        case Field::DualTwoForm:
            return Field::DualOneForm;
        case Field::DualThreeForm:
            return Field::DualZeroForm;
    }
}

#endif

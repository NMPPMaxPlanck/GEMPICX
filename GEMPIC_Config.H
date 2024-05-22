#ifndef GEMPIC_CONFIG_H_
#define GEMPIC_CONFIG_H_

#define GEMPIC_SPACEDIM AMREX_SPACEDIM

#define GEMPIC_NUMSPEC 1
#define GEMPIC_ELECTROMAGNETIC true

#define MAXGAUSS 2  // maximum number of velocity gaussians

#define GEMPIC_CTEST_TOL 1e-4
#define GEMPIC_CTEST_RELTOL 1e-4

#if GEMPIC_SPACEDIM == 1
/**
 * The GEMPIC_D_MULT(a,b,c) macro multiplies the elements of a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of GEMPIC_SPACEDIM.
 */
#define GEMPIC_D_MULT(a, b, c) (a)
#elif GEMPIC_SPACEDIM == 2
#define GEMPIC_D_MULT(a, b, c) ((a) * (b))
#elif GEMPIC_SPACEDIM == 3
#define GEMPIC_D_MULT(a, b, c) ((a) * (b) * (c))
#endif

#if GEMPIC_SPACEDIM == 1
/**
 * The GEMPIC_D_ADD(a,b,c) macro adds the elements of a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of GEMPIC_SPACEDIM.
 */
#define GEMPIC_D_ADD(a, b, c) (a)
#elif GEMPIC_SPACEDIM == 2
#define GEMPIC_D_ADD(a, b, c) ((a) + (b))
#elif GEMPIC_SPACEDIM == 3
#define GEMPIC_D_ADD(a, b, c) ((a) + (b) + (c))
#endif

#if GEMPIC_SPACEDIM == 1
/**
 * The GEMPIC_D_PAD(a,b,c) macro replaces the last arguments by zeros
 * in a comma-separated list of the 3 arguments of the call,
 * depending on the value of GEMPIC_SPACEDIM.
 */
#define GEMPIC_D_PAD(a, b, c) a, 0, 0
#elif GEMPIC_SPACEDIM == 2
#define GEMPIC_D_PAD(a, b, c) a, b, 0
#elif GEMPIC_SPACEDIM == 3
#define GEMPIC_D_PAD(a, b, c) a, b, c
#endif

#if GEMPIC_SPACEDIM == 1
/**
 * The GEMPIC_D_EXCL(a,b,c) macro evaluates the elements of a
 * comma-separated list of the third, second or all three of the arguments of the call,
 * depending on the value of GEMPIC_SPACEDIM.
 * Chiefly used to define constants that may be lacking in 1D or 2D.
 */
// Tell clang-tidy that we know what we are doing with this macro:
// NOLINTBEGIN(bugprone-macro-parentheses)
#define GEMPIC_D_EXCL(a, b, c) a b c
#elif GEMPIC_SPACEDIM == 2
#define GEMPIC_D_EXCL(a, b, c) b c
#elif GEMPIC_SPACEDIM == 3
#define GEMPIC_D_EXCL(a, b, c) c
#endif

#if GEMPIC_SPACEDIM == 1
/**
 * The GEMPIC_D_LOOP_BEGIN(a,b,c) macro initializes nested for loops using a
 * comma-separated list of 1, 2, or all 3 of the arguments of the call,
 * depending on the value of GEMPIC_SPACEDIM.
 */
#define GEMPIC_D_LOOP_BEGIN(a, b, c) \
    a                                \
    {
#elif GEMPIC_SPACEDIM == 2
#define GEMPIC_D_LOOP_BEGIN(a, b, c) \
    b                                \
    {                                \
        a                            \
        {
#elif GEMPIC_SPACEDIM == 3
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
#if GEMPIC_SPACEDIM == 1
#define GEMPIC_D_LOOP_END }
#elif GEMPIC_SPACEDIM == 2
#define GEMPIC_D_LOOP_END \
    }                     \
    }
#elif GEMPIC_SPACEDIM == 3
#define GEMPIC_D_LOOP_END \
    }                     \
    }                     \
    }
#endif

enum Direction : unsigned int
{
    xDir = 0,
    yDir = 1,
    zDir = 2,
    noDir = 3,
};

enum Error : int
{
    InvalidInput = 1,
    AttemptedParameterChange = 2,
    ParametersNotInitialized = 3,
    ParametersAlreadyInitialized = 4,
    VerbosityAlreadySet = 5,
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

#ifndef GEMPIC_DERHAMCOMPLEX_H_
#define GEMPIC_DERHAMCOMPLEX_H_

#include "GEMPIC_Config.H"

// Primary template deleted
template <typename T>
constexpr bool is_primal() = delete;
template <typename T>
constexpr bool is_dual() = delete;
template <typename T>
constexpr bool is_zero_form() = delete;
template <typename T>
constexpr bool is_three_form() = delete;
template <typename T>
constexpr bool is_zero_or_three_form() = delete;
template <typename T>
constexpr bool is_one_form() = delete;
template <typename T>
constexpr bool is_two_form() = delete;
template <typename T>
constexpr bool is_one_or_two_form() = delete;
template <typename T>
constexpr Sequence to_sequence() = delete;
template <typename T>
constexpr Form to_form() = delete;

// Grid to Sequence
template <Sequence sequence>
constexpr Sequence to_sequence ()
{
    return sequence;
}

// Space to Form
template <Form form>
constexpr Form to_form ()
{
    return form;
}

// template specialization
template <Sequence c>
constexpr bool is_primal ()
{
    return (c == Sequence::primal);
}

template <Sequence c>
constexpr bool is_dual ()
{
    return !is_primal<c>();
}

// template specialization
template <Form f>
constexpr bool is_zero_form ()
{
    return (f == Form::zero);
}
template <Form f>
constexpr bool is_three_form ()
{
    return (f == Form::three);
}
template <Form f>
constexpr bool is_zero_or_three_form ()
{
    return (is_zero_form<f>() || is_three_form<f>());
}

template <Form f>
constexpr bool is_one_form ()
{
    return (f == Form::one);
}

template <Form f>
constexpr bool is_two_form ()
{
    return (f == Form::two);
}

template <Form f>
constexpr bool is_one_or_two_form ()
{
    return (is_one_form<f>() || is_two_form<f>());
}

// (sequence,form) <=> (grid,space)

// multifab enum
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

// Grid to Sequence
template <Grid grid>
constexpr Sequence to_sequence ()
{
    return (grid == Grid::primal ? Sequence::primal : Sequence::dual);
}

// Space to Form
template <Space space>
constexpr Form to_form ()
{
    return (space == Space::node   ? Form::zero
            : space == Space::edge ? Form::one
            : space == Space::face ? Form::two
                                   : Form::three);
}

template <Space space>
constexpr bool is_zero_form ()
{
    return is_zero_form<to_form<space>()>();
}
template <Space space>
constexpr bool is_three_form ()
{
    return is_three_form<to_form<space>()>();
}
template <Space space>
constexpr bool is_zero_or_three_form ()
{
    return is_zero_or_three_form<to_form<space>()>();
}

template <Space space>
constexpr bool is_one_form ()
{
    return is_one_form<to_form<space>()>();
}
template <Space space>
constexpr bool is_two_form ()
{
    return is_two_form<to_form<space>()>();
}
template <Space space>
constexpr bool is_one_or_two_form ()
{
    return is_one_or_two_form<to_form<space>()>();
}

// Space-based concepts
template <Space space>
concept OneFormSpaceC = is_one_form<space>();
template <Space space>
concept TwoFormSpaceC = is_two_form<space>();
template <Space space>
concept OneOrTwoFormSpaceC = is_one_or_two_form<space>();

template <Space space>
concept ZeroFormSpaceC = is_zero_form<space>();
template <Space space>
concept ThreeFormSpaceC = is_three_form<space>();
template <Space space>
concept ZeroOrThreeFormSpaceC = is_zero_or_three_form<space>();

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

namespace Gempic::Forms
{

enum class HodgeScheme
{
    FDHodge,
    GalerkinHodge,
    GDECHodge,     // Galerkin Differences. No dual grid is used.
    GDECLumpHodge, // with Mass lumping
};

// concept for deRham complexes
template <typename T>
concept DeRhamComplexC = requires(T drc) {
    //{ T::s_hodgeScheme } -> std::convertible_to<HodgeScheme>; // static member, no instance
    { drc.m_hodgeDegree() } -> std::convertible_to<int>; // non static member
    typename T::Primal0Form;
    typename T::Primal1Form;
    typename T::Primal2Form;
    typename T::Primal3Form;
    typename T::Dual0Form;
    typename T::Dual1Form;
    typename T::Dual2Form;
    typename T::Dual3Form;
};

// // map of pointers to the field multifabs for the diagnostics
// static std::map<std::string, amrex::MultiFab*> fieldsDiagnostics;
// // map of scaling parameters to go from degrees of freedom to point values for diagnostics
// static std::map<std::string, amrex::Real> fieldsScaling;

// Helpful for enable_if magic. @todo: If we switch to C++20, we should use concepts, as shown in
// https://gitlab.mpcdf.mpg.de/gempic/gempic/-/merge_requests/127#note_225990
//template <Space space>
//inline constexpr bool isZeroOrThreeForm{false};
//
//template <>
//inline constexpr bool is_zero_or_three_form<Space::node>{true};
//
//template <>
//inline constexpr bool is_zero_or_three_form<Space::cell>{true};
//
//template <Space space>
//inline constexpr bool isOneOrTwoForm{!is_zero_or_three_form<space>};

template <typename SrcFormType>
struct DualOf;

// Handle const/ref wrappers
template <typename SrcFormType>
    requires(!std::same_as<SrcFormType, canonical_t<SrcFormType>>)
struct DualOf<SrcFormType> : DualOf<canonical_t<SrcFormType>>
{
};

template <typename SrcFormType>
struct DualOf
{
    static_assert(gempic_always_false<SrcFormType>(),
                  "DualOf specialization missing for this form type");
};

template <typename T>
using dual_t = typename DualOf<T>::type;

// c++20 concept for one two forms
template <typename T>
concept DeRhamFormC = requires {
    { canonical_t<T>::s_sequence } -> std::convertible_to<Sequence>;
    { canonical_t<T>::s_form } -> std::convertible_to<Form>;
};

// c++20 concept for zero three forms
template <typename T>
concept PrimalFormC = DeRhamFormC<canonical_t<T>> && is_primal<canonical_t<T>::s_sequence>();
// c++20 concept for zero three forms
template <typename T>
concept DualFormC = DeRhamFormC<canonical_t<T>> && is_dual<canonical_t<T>::s_sequence>();
// c++20 concept for zero three forms
template <typename T>
concept ZeroFormC = DeRhamFormC<canonical_t<T>> && is_zero_form<canonical_t<T>::s_form>();
template <typename T>
concept ThreeFormC = DeRhamFormC<canonical_t<T>> && is_three_form<canonical_t<T>::s_form>();
template <typename T>
concept ZeroThreeFormC =
    DeRhamFormC<canonical_t<T>> && is_zero_or_three_form<canonical_t<T>::s_form>();

template <typename T>
concept PrimalZeroFormC = DeRhamFormC<canonical_t<T>> && is_zero_form<canonical_t<T>::s_form>() &&
                          is_primal<canonical_t<T>::s_sequence>();
template <typename T>
concept PrimalThreeFormC = DeRhamFormC<canonical_t<T>> && is_three_form<canonical_t<T>::s_form>() &&
                           is_primal<canonical_t<T>::s_sequence>();
template <typename T>
concept DualZeroFormC = DeRhamFormC<canonical_t<T>> && is_zero_form<canonical_t<T>::s_form>() &&
                        is_dual<canonical_t<T>::s_sequence>();
template <typename T>
concept DualThreeFormC = DeRhamFormC<canonical_t<T>> && is_three_form<canonical_t<T>::s_form>() &&
                         is_dual<canonical_t<T>::s_sequence>();
// c++20 concept for zero three forms
template <typename T>
concept OneFormC = DeRhamFormC<canonical_t<T>> && is_one_form<canonical_t<T>::s_form>();
template <typename T>
concept TwoFormC = DeRhamFormC<canonical_t<T>> && is_two_form<canonical_t<T>::s_form>();
template <typename T>
concept OneTwoFormC = DeRhamFormC<canonical_t<T>> && is_one_or_two_form<canonical_t<T>::s_form>();

template <typename T>
concept PrimalOneFormC = DeRhamFormC<canonical_t<T>> && is_one_form<canonical_t<T>::s_form>() &&
                         is_primal<canonical_t<T>::s_sequence>();
template <typename T>
concept PrimalTwoFormC = DeRhamFormC<canonical_t<T>> && is_two_form<canonical_t<T>::s_form>() &&
                         is_primal<canonical_t<T>::s_sequence>();
template <typename T>
concept DualOneFormC = DeRhamFormC<canonical_t<T>> && is_one_form<canonical_t<T>::s_form>() &&
                       is_dual<canonical_t<T>::s_sequence>();
template <typename T>
concept DualTwoFormC = DeRhamFormC<canonical_t<T>> && is_two_form<canonical_t<T>::s_form>() &&
                       is_dual<canonical_t<T>::s_sequence>();
template <Sequence sequence, Form form>
struct DeRhamFormBase
{
public:
    static constexpr Form s_form = form;
    static constexpr Sequence s_sequence = sequence;
};

template <typename FormA, typename FormB>
concept MutualDual = std::same_as<canonical_t<dual_t<FormA>>, canonical_t<FormB>> &&
                     std::same_as<canonical_t<dual_t<FormB>>, canonical_t<FormA>> &&
                     ((ZeroThreeFormC<FormA> && ZeroThreeFormC<FormB>) ||
                      (OneTwoFormC<FormA> && OneTwoFormC<FormB>));

template <typename FormA, typename FormB>
concept SameSequence = std::same_as<decltype(canonical_t<FormA>::s_sequence),
                                    decltype(canonical_t<FormB>::s_sequence)> &&
                       (canonical_t<FormA>::s_sequence == canonical_t<FormB>::s_sequence);
} //namespace Gempic::Forms

#endif

#ifndef GEMPIC_DERHAMCOMPLEX_H_
#define GEMPIC_DERHAMCOMPLEX_H_

#include <type_traits>
#include <variant>

#include "GEMPIC_Config.H"

namespace Gempic::Forms
{

template <typename DRC>
concept HasHodgeScheme = requires {
    { static_cast<int>(DRC::s_hodgeScheme) } -> std::convertible_to<int>;
};

// concept for deRham complexes
template <typename T>
concept DeRhamComplexC = requires(T drc) {
    //{ T::s_hodgeDegree } -> std::convertible_to<int>; // static member, no instance
    { drc.get_hodge_degree() } -> std::convertible_to<int>; // non static member
    typename T::Primal0Form;
    typename T::Primal1Form;
    typename T::Primal2Form;
    typename T::Primal3Form;
    typename T::Dual0Form;
    typename T::Dual1Form;
    typename T::Dual2Form;
    typename T::Dual3Form;
};

template <DeRhamComplexC Drc>
using FieldVariantPtr = std::variant<typename Drc::Primal0Form,
                                     typename Drc::Primal1Form,
                                     typename Drc::Primal2Form,
                                     typename Drc::Primal3Form,
                                     typename Drc::Dual0Form,
                                     typename Drc::Dual1Form,
                                     typename Drc::Dual2Form,
                                     typename Drc::Dual3Form>;

// Concept for forms in a DeRham complex
template <typename T, typename DRC>
concept FormOfComplexC = DeRhamComplexC<DRC> && is_variant_member<T, FieldVariantPtr<DRC>>::value;

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

// Primary template (undefined for unsupported types)
template <typename Field>
struct field_to_drc;

template <typename Field>
using field_to_drc_t = typename field_to_drc<canonical_t<Field>>::type;

enum class HodgeScheme
{
    FDHodge,
    GalerkinHodge,
    GDECHodge,     // Galerkin Differences. No dual grid is used.
    GDECLumpHodge, // with Mass lumping
};

template <HodgeScheme hodgeScheme>
constexpr auto valid_degrees() = delete;

template <>
constexpr auto valid_degrees<HodgeScheme::FDHodge>()
{
    constexpr std::array<int, 3> validDegreesFdHodge{2, 4, 6};
    return validDegreesFdHodge;
}

template <>
constexpr auto valid_degrees<HodgeScheme::GalerkinHodge>()
{
    constexpr std::array<int, 3> validDegreesGalerkinHodge{2, 4, 6};
    return validDegreesGalerkinHodge;
}

template <>
constexpr auto valid_degrees<HodgeScheme::GDECHodge>()
{
    // for GDEC we have theoretical arbitrary order of accuracy
    constexpr std::array<int, 5> validDegreesGdecHodge{2, 4, 6, 8, 10};
    return validDegreesGdecHodge;
}

template <>
constexpr auto valid_degrees<HodgeScheme::GDECLumpHodge>()
{
    constexpr std::array<int, 3> validDegreesGdecLumpHodge{2, 4, 6};
    return validDegreesGdecLumpHodge;
}

template <typename SrcFormType, DeRhamComplexC DRC>
    requires FormOfComplexC<SrcFormType, DRC>
struct DualOf;

// Handle const/ref wrappers
template <typename SrcFormType, DeRhamComplexC DRC>
    requires(FormOfComplexC<SrcFormType, DRC> &&
             !std::same_as<SrcFormType, canonical_t<SrcFormType>>)
struct DualOf<SrcFormType, DRC> : DualOf<canonical_t<SrcFormType>, DRC>
{
};

template <typename SrcFormType, DeRhamComplexC DRC>
    requires FormOfComplexC<SrcFormType, DRC>
struct DualOf
{
    static_assert(gempic_always_false<SrcFormType>(),
                  "DualOf specialization missing for this form SrcFormType type");
    static_assert(gempic_always_false<DRC>(), "DualOf specialization missing for this DRC");
};

template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Primal0Form, DRC>
{
    using type = typename DRC::Dual3Form;
};
template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Dual3Form, DRC>
{
    using type = typename DRC::Primal0Form;
};
template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Primal2Form, DRC>
{
    using type = typename DRC::Dual1Form;
};
template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Dual1Form, DRC>
{
    using type = typename DRC::Primal2Form;
};

template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Primal1Form, DRC>
{
    using type = typename DRC::Dual2Form;
};
template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Dual2Form, DRC>
{
    using type = typename DRC::Primal1Form;
};
template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Primal3Form, DRC>
{
    using type = typename DRC::Dual0Form;
};
template <DeRhamComplexC DRC>
struct DualOf<typename DRC::Dual0Form, DRC>
{
    using type = typename DRC::Primal3Form;
};

template <typename T, DeRhamComplexC DRC>
using dual_t = typename DualOf<canonical_t<T>, DRC>::type;

// c++20 concept forms
template <typename T>
concept DeRhamFormC = requires {
    { canonical_t<T>::s_sequence } -> std::convertible_to<Sequence>;
    { canonical_t<T>::s_form } -> std::convertible_to<Form>;
};

template <typename T>
concept PrimalFormC = DeRhamFormC<T> && is_primal<canonical_t<T>::s_sequence>();
template <typename T>
concept DualFormC = DeRhamFormC<T> && is_dual<canonical_t<T>::s_sequence>();

template <typename T>
concept ZeroFormC = DeRhamFormC<T> && is_zero_form<canonical_t<T>::s_form>();
template <typename T>
concept ThreeFormC = DeRhamFormC<T> && is_three_form<canonical_t<T>::s_form>();
template <typename T>
concept ZeroThreeFormC = ZeroFormC<T> || ThreeFormC<T>;

template <typename T>
concept PrimalZeroFormC = ZeroFormC<T> && PrimalFormC<T>;
template <typename T>
concept PrimalThreeFormC = ThreeFormC<T> && PrimalFormC<T>;
template <typename T>
concept DualZeroFormC = ZeroFormC<T> && DualFormC<T>;
template <typename T>
concept DualThreeFormC = ThreeFormC<T> && DualFormC<T>;

template <typename T>
concept OneFormC = DeRhamFormC<T> && is_one_form<canonical_t<T>::s_form>();
template <typename T>
concept TwoFormC = DeRhamFormC<T> && is_two_form<canonical_t<T>::s_form>();
template <typename T>
concept OneTwoFormC = OneFormC<T> || TwoFormC<T>;

template <typename T>
concept PrimalOneFormC = OneFormC<T> && PrimalFormC<T>;
template <typename T>
concept PrimalTwoFormC = TwoFormC<T> && PrimalFormC<T>;
template <typename T>
concept DualOneFormC = OneFormC<T> && DualFormC<T>;
template <typename T>
concept DualTwoFormC = TwoFormC<T> && DualFormC<T>;
template <Sequence sequence, Form form>
struct DeRhamFormBase
{
public:
    static constexpr Form s_form = form;
    static constexpr Sequence s_sequence = sequence;
};

template <typename FormA, typename FormB>
concept MutualDual =
    requires { typename field_to_drc_t<FormA>; } && DeRhamComplexC<field_to_drc_t<FormA>> &&
    std::same_as<dual_t<FormA, field_to_drc_t<FormA>>, canonical_t<FormB>> &&
    std::same_as<dual_t<FormB, field_to_drc_t<FormA>>, canonical_t<FormA>>;

template <typename FormA, typename FormB>
concept SameSequence = std::same_as<decltype(canonical_t<FormA>::s_sequence),
                                    decltype(canonical_t<FormB>::s_sequence)> &&
                       (canonical_t<FormA>::s_sequence == canonical_t<FormB>::s_sequence);
} //namespace Gempic::Forms

#endif

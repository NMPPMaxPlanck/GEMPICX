#ifndef GEMPIC_CUSTOMDIAGNOSTICSTRATEGIES_H_
#define GEMPIC_CUSTOMDIAGNOSTICSTRATEGIES_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_DiagnosticStrategies.H"
#include "GEMPIC_FieldRegistry.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"

// -----------------------------------------------------------------------------
// Registration lifetime contract
//
// All custom output processors (class-based) and custom operators (callable-based)
// must be registered *before the first diagnostics evaluation*.
//
// In practice, registration should happen during program initialization,
// typically before the time-stepping loop and before any diagnostics objects
// are constructed or executed.
//
// The internal registries are host-only, process-local, and not synchronized.
// Registering processors after diagnostics have been created or used results
// in undefined behavior and is therefore not supported.
// -----------------------------------------------------------------------------

namespace Gempic::Io::Impl
{
// Factory type: builds a ComputeDiagOutputProcessor from AnyFieldPtr and crseRatio
using ConstructorType = std::function<std::unique_ptr<ComputeDiagOutputProcessor>(
    Io::AnyFieldPtr const& dataSrc, amrex::IntVect const crseRatio)>;

// Helper to wrap a derived processor class T into a factory
template <typename T>
std::unique_ptr<ComputeDiagOutputProcessor> dynamic_convert (Io::AnyFieldPtr const& dataSrc,
                                                             amrex::IntVect const crseRatio)
{
    return std::make_unique<T>(dataSrc, crseRatio);
}
// -------------------- Helpers for CustomOperatorOutputProcessor --------------------
// Type Erasure helper class a la Iglberger ("C++ Software Design", 2022 O'Reilly Media, Inc.)
// OwningModel is separate because CUDA complained otherwise.
// Type-erasure for device-callables
template <typename T>
struct OwningModel
{
    OwningModel(T object) : m_object{std::move(object)} {}

    void operator()(amrex::Box const& bx,
                    amrex::Array4<amrex::Real> dst,
                    amrex::Array4<amrex::Real const> src,
                    int nSrcComp,
                    double scaling,
                    double ishift,
                    double jshift,
                    double kshift)
    {
        ParallelFor(bx, [=, object = this->m_object] AMREX_GPU_DEVICE(int i, int j, int k)
                    { object(dst, src, nSrcComp, i, j, k, scaling, ishift, jshift, kshift); });
    }

    T m_object;
};

// Type Erasure class a la Iglberger ("C++ Software Design", 2022 O'Reilly Media, Inc.)
// (section: Manual Implementation of Function Dispatch)
// Allows passing arbitrary lambdas.
// Changing lambda signature requires changing:
//   - m_calc signature
//   - OwningModel<T>::operator(), CustomOperatorType::operator() signature
//   - CalcOperation signature
// https://docs.nvidia.com/cuda/archive/10.1/cuda-c-programming-guide/index.html#extended-lambda-restrictions
class CustomOperatorType
{
public:
    template <typename T>
    CustomOperatorType(T object) :
        m_pimpl(new OwningModel<T>(std::move(object)),
                [] (void* p) { delete static_cast<OwningModel<T>*>(p); }),
        m_calc(
            [] (void* p,
                amrex::Box const& bx,
                amrex::Array4<amrex::Real> dst,
                amrex::Array4<amrex::Real const> src,
                int nSrcComp,
                double scaling,
                double ishift,
                double jshift,
                double kshift)
            {
                auto* model = static_cast<OwningModel<T>*>(p);
                (*model)(bx, dst, src, nSrcComp, scaling, ishift, jshift, kshift);
            }),
        m_clone(
            [] (void* p) -> void*
            {
                auto* model = static_cast<OwningModel<T>*>(p);
                return new OwningModel<T>(*model);
            })
    {
    }

    CustomOperatorType(CustomOperatorType const& other) :
        m_pimpl(other.m_clone(other.m_pimpl.get()), other.m_pimpl.get_deleter()),
        m_calc(other.m_calc),
        m_clone(other.m_clone)
    {
    }

    CustomOperatorType& operator=(CustomOperatorType const& other)
    {
        using std::swap;
        CustomOperatorType tmp(other);
        swap(m_pimpl, tmp.m_pimpl);
        swap(m_calc, tmp.m_calc);
        swap(m_clone, tmp.m_clone);
        return *this;
    }

    ~CustomOperatorType() = default;
    CustomOperatorType(CustomOperatorType&&) = default;
    CustomOperatorType& operator=(CustomOperatorType&&) = default;

    void operator()(amrex::Box const& bx,
                    amrex::Array4<amrex::Real> dst,
                    amrex::Array4<amrex::Real const> src,
                    int nSrcComp,
                    double scaling,
                    double ishift,
                    double jshift,
                    double kshift) const
    {
        m_calc(m_pimpl.get(), bx, dst, src, nSrcComp, scaling, ishift, jshift, kshift);
    }

private:
    using DestroyOperation = void(void*);
    using CalcOperation = void(void*,
                               amrex::Box const& bx,
                               amrex::Array4<amrex::Real> dst,
                               amrex::Array4<amrex::Real const> const src,
                               int nSrcComp,
                               double scaling,
                               double ishift,
                               double jshift,
                               double kshift);
    using CloneOperation = void*(void*);

    std::unique_ptr<void, DestroyOperation*> m_pimpl;
    CalcOperation* m_calc{nullptr};
    CloneOperation* m_clone{nullptr};
};

} // namespace Gempic::Io::Impl

//STRATEGY: free registry function + public static API on the class
//Define host-only registry accessors as free functions
namespace Gempic::Io::Detail
{

// Host-only registries (function-local statics)
inline std::map<std::string, Impl::ConstructorType>& class_registry();

inline std::map<std::string, Impl::CustomOperatorType>& operator_registry();

} //namespace Gempic::Io::Detail

namespace Gempic::Io
{

// Base class with class-factory registry (friend-free)
/**
 * \brief OutputProcessor for custom OutputProcessor classes. Inherit from this if you want to
 * create your own outputProcessor from scratch. Otherwise, if you only need to customize the
 * operator, use CustomOperatorOutputProcessor.
 *
 * Users can add their own outputProcessors before the first instantiation of FullDiagnostics.
 * inside your custom class you must override the operator() method,
 *    void operator()(amrex::MultiFab& mfDst, int dcomp) const override {...}
 * or
 *    void operator()(amrex::MultiFab& mfDst, int dcomp) const final {...}
 * and outside (somewhere in the main program) you must add your class via the static function
 *    CustomOutputProcessor::add_output_processor<YourOutputProcessorClass>(yourIdString);
 * and then select the class in the input file using yourIdString,
 *    FullDiagnostics.{groupName}.outputProcessor = "custom"
 *    FullDiagnostics.{groupName}.customId = "yourIdString"
 */
class CustomOutputProcessor : public ComputeDiagOutputProcessor
{
public:
    /** Constructor.
     *
     * \param[in] dataSrc source AnyFieldPtr register (type AnyFieldPtr).
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
                  to the output diagnostic MultiFab, mfDst.
     */
    CustomOutputProcessor(Io::AnyFieldPtr const& dataSrc, amrex::IntVect const crseRatio);

    /** \brief copy m_dataSrc to mfDst.
     *
     * \param[out] mfDst output MultiFab where the result is written
     */
    void operator()(amrex::MultiFab& mfDst, int /*dcomp*/) const override;

    amrex::BoxArray get_box_array(amrex::IndexType& t) override;

    //public static register_class/find_class
    static void register_class(std::string key, Impl::ConstructorType ctor);

    static Impl::ConstructorType const* find_class(std::string_view const& key);

protected:
    /** Non-owning handle to a registered diagnostics field (type-erased, possibly multi-component)
     */
    Io::AnyFieldPtr const* m_dataSrc;
};

// Derived class with kernel-callable registry (friend-free)
/**
 * \brief OutputProcessor for customizing only the diagnostics operator
 *
 * Users can add their own operators before the first instantiation of FullDiagnostics to be called
 * on the GPU to calculate the output fields from the input fields specified by name in the input
 * file.
 * The custom operator can either be a device lambda,
 *     [=] AMREX_GPU_DEVICE(amrex::Array4<amrex::Real> dst,
 *                          const amrex::Array4<const amrex::Real> src,
 *                          int nSrcComp,
 *                          int i, int j, int k,
 *                          double scaling,
 *                          double ishift, double jshift, double kshift) {...}
 * or a device function,
 *     AMREX_GPU_DEVICE void functionName(amrex::Array4<amrex::Real> dst,
 *                                        const amrex::Array4<const amrex::Real> src,
 *                                        int nSrcComp,
 *                                        int i, int j, int k,
 *                                        double scaling,
 *                                        double ishift, double jshift, double kshift) {...}
 *
 * and is added with an id string via Gempic::Io::CustomOutputProcessor::add_output_processor().
 * The id string is then used to select for this operator in the input file,
 *     FullDiagnostics.{groupName}.outputProcessor = "custom"
 *     FullDiagnostics.{groupName}.customId = "whateverYourIdWas"
 *
 *
 * If more control than a custom operator is needed, use the CustomOutputProcessor class.
 */
class CustomOperatorOutputProcessor final : public CustomOutputProcessor
{
public:
    using CustomOperatorType = Impl::CustomOperatorType;
    /** Constructor.
     *
     * \param[in] dataSrc source AnyFieldPtr register (type AnyFieldPtr).
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
                  to the output diagnostic MultiFab, mfDst.
     * \param[in] f the constructor of the custom outputProcessor chosen from the input file.
     */
    CustomOperatorOutputProcessor(Io::AnyFieldPtr const& dataSrc,
                                  amrex::IntVect const crseRatio,
                                  CustomOperatorType const& f);

    void operator()(amrex::MultiFab& mfDst, int /*dcomp*/) const final;

    //public static register_operator/find_operator
    static void register_operator(std::string key, CustomOperatorType op);
    static CustomOperatorType const* find_operator(std::string_view const& key);

private:
    CustomOperatorType m_f;
    amrex::Real m_ishift{0}, m_jshift{0}, m_kshift{0};
};

// Free registration for class-based custom processors
template <typename T>
inline void add_output_processor (std::string const& key)
{
    static_assert(std::is_base_of_v<CustomOutputProcessor, T>,
                  "T must derive from CustomOutputProcessor");
    CustomOutputProcessor::register_class(key, Impl::dynamic_convert<T>);
}

// Free registration for callable-based operators
inline void add_output_processor (std::string const& key, Impl::CustomOperatorType const& f)
{
    CustomOperatorOutputProcessor::register_operator(key, f);
}

template <class F>
    requires(!std::same_as<std::decay_t<F>, Impl::CustomOperatorType>)
inline void add_output_processor (std::string const& key, F&& f)
{
    add_output_processor(key, Impl::CustomOperatorType(std::forward<F>(f)));
}

} // namespace Gempic::Io

namespace Gempic::Io::Impl
{
// Factory lookup (consult both registries): try class factories first, then operator kernels
ConstructorType match_custom_id(std::string const& key);

} // namespace Gempic::Io::Impl

namespace Gempic::Io
{
/**
 * A higher resolution writer: high-order interpolation of a Field (AnyFieldPtr) on a finer grid
 * (MultiFab).
 */
class HighResSubcellOutputProcessor : public CustomOutputProcessor
{
public:
    void operator()(amrex::MultiFab& mfDst, int dcomp) const final;

    amrex::BoxArray get_box_array(amrex::IndexType& t) final;
    /**
     * @brief
     * @tparam AnyFieldPtr allow OutputProcessor for different field types
     * @param dataSrc
     * @param crseRatio
     */
    HighResSubcellOutputProcessor(Io::AnyFieldPtr const& dataSrc, amrex::IntVect const crseRatio);

    /**
     * @brief evaluate field to destination multifab grid points
     */
    void interpolate_any_field_to_multifab(amrex::MultiFab& mfDst, int dcomp) const;
};

} //namespace Gempic::Io

#endif // GEMPIC_CUSTOMDIAGNOSTICSTRATEGIES_H_

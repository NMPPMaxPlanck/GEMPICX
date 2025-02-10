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

namespace Gempic::Io::Impl
{
// ------------------------------- Helpers for CustomOutputProcessor -------------------------------
using ConstructorType = std::function<std::unique_ptr<ComputeDiagOutputProcessor>(
    const amrex::MultiFab& mfSrc, const amrex::Real scaling, const amrex::IntVect crseRatio)>;

template <typename T>
std::unique_ptr<ComputeDiagOutputProcessor> dynamic_convert (const amrex::MultiFab& mfSrc,
                                                             const amrex::Real scaling,
                                                             const amrex::IntVect crseRatio)
{
    return std::make_unique<T>(mfSrc, scaling, crseRatio);
}

ConstructorType match_custom_id(const std::string& key);

// -------------------- Helpers for CustomOperatorOutputProcessor --------------------
// Type Erasure helper class a la Iglberger ("C++ Software Design", 2022 O'Reilly Media, Inc.)
// OwningModel is separate because CUDA complained otherwise.
template <typename T>
struct OwningModel
{
    OwningModel(T object) : m_object{std::move(object)} {}

    void operator()(const amrex::Box& bx,
                    amrex::Array4<amrex::Real> dst,
                    const amrex::Array4<const amrex::Real> src,
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
                [] (void* operatorBytes)
                {
                    auto* const model = static_cast<OwningModel<T>*>(operatorBytes);
                    delete model;
                }),
        m_calc(
            [] (void* operatorBytes,
                const amrex::Box& bx,
                amrex::Array4<amrex::Real> dst,
                const amrex::Array4<const amrex::Real> src,
                int nSrcComp,
                double scaling,
                double ishift,
                double jshift,
                double kshift)
            {
                auto* const model = static_cast<OwningModel<T>*>(operatorBytes);
                model->operator()(bx, dst, src, nSrcComp, scaling, ishift, jshift, kshift);
            }),
        m_clone(
            [] (void* operatorBytes) -> void*
            {
                auto* const model = static_cast<OwningModel<T>*>(operatorBytes);
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
        CustomOperatorType copy(other);
        swap(m_pimpl, copy.m_pimpl);
        swap(m_calc, copy.m_calc);
        swap(m_clone, copy.m_clone);
        return *this;
    }

    ~CustomOperatorType() = default;
    CustomOperatorType(CustomOperatorType&&) = default;
    CustomOperatorType& operator=(CustomOperatorType&&) = default;
    void operator()(const amrex::Box& bx,
                    amrex::Array4<amrex::Real> dst,
                    const amrex::Array4<const amrex::Real> src,
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
                               const amrex::Box& bx,
                               amrex::Array4<amrex::Real> dst,
                               const amrex::Array4<const amrex::Real> src,
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
} //namespace Gempic::Io::Impl

namespace Gempic::Io
{
template <typename T>
void add_output_processor(const std::string& str);

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
    using ConstructorType = Impl::ConstructorType;
    /** Constructor.
     *
     * \param[in] mfSrc source MultiFab.
     * \param[in] scaling scaling factor.
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
                  to the output diagnostic MultiFab, mfDst.
     */
    CustomOutputProcessor(const amrex::MultiFab& mfSrc,
                          const amrex::Real scaling,
                          const amrex::IntVect crseRatio);

    /** \brief copy m_mfSrc to mfDst.
     *
     * \param[out] mfDst output MultiFab where the result is written
     */
    void operator()(amrex::MultiFab& mfDst, int /*dcomp*/) const override;

protected:
    /** Alias of source MultiFab (can be multi-component) */
    const amrex::MultiFab m_mfSrc;
    // Scaling factor
    const amrex::Real m_scaling{1.0};

private:
    /** map of all user-provided outputProcessor classes */
    inline static std::map<std::string, ConstructorType> s_outputProcessorClassMap{};

    template <typename T>
    friend void add_output_processor(const std::string&);

    friend ConstructorType Impl::match_custom_id(const std::string&);
};

// Add custom class to the list along with a key so it can be selected in the input file
template <typename T>
void add_output_processor (const std::string& key)
{
    // MUST provide a derived class
    static_assert(std::is_base_of_v<CustomOutputProcessor, T>);
    if (CustomOutputProcessor::s_outputProcessorClassMap.find(key) !=
        CustomOutputProcessor::s_outputProcessorClassMap.end())
    {
        std::string msg{"Already defined custom diagnostic Id: '" + key + "'."};
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }
    CustomOutputProcessor::s_outputProcessorClassMap.emplace(key, Impl::dynamic_convert<T>);
}

// Add custom lamba to the list along with a key so it can be selected in the input file
void add_output_processor(const std::string& key, const Impl::CustomOperatorType& f);

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
     * \param[in] mfSrc source MultiFab.
     * \param[in] scaling scaling factor.
     * \param[in] crseRatio for interpolating field values from the simulation MultiFab, srcMf,
                  to the output diagnostic MultiFab, mfDst.
     * \param[in] f the constructor of the custom outputProcessor chosen from the input file.
     */
    CustomOperatorOutputProcessor(const amrex::MultiFab& mfSrc,
                                  const amrex::Real scaling,
                                  const amrex::IntVect crseRatio,
                                  const CustomOperatorType& f);

    /** \brief copy m_mfSrc to mfDst.
     *
     * \param[out] mfDst output MultiFab where the result is written
     */
    void operator()(amrex::MultiFab& mfDst, int /*dcomp*/) const final;

private:
    /** User-provided operator specific to this instance */
    CustomOperatorType m_f;
    amrex::Real m_ishift{0}, m_jshift{0}, m_kshift{0};
    /** map of all user-provided outputProcessors */
    inline static std::map<std::string, CustomOperatorType> s_outputProcessorMap{};

    friend void add_output_processor(const std::string& key, const CustomOperatorType& f);
    friend Impl::ConstructorType Impl::match_custom_id(const std::string&);
};
} //namespace Gempic::Io

#endif // GEMPIC_CUSTOMDIAGNOSTICSTRATEGIES_H_

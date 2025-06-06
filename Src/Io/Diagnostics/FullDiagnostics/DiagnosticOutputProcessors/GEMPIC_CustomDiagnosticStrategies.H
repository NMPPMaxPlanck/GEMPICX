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
#include "GEMPIC_Interpolation.H"

namespace Gempic::Io::Impl
{
// ------------------------------- Helpers for CustomOutputProcessor -------------------------------
using ConstructorType = std::function<std::unique_ptr<ComputeDiagOutputProcessor>(
    amrex::MultiFab const& mfSrc, amrex::Real const scaling, amrex::IntVect const crseRatio)>;

template <typename T>
std::unique_ptr<ComputeDiagOutputProcessor> dynamic_convert (amrex::MultiFab const& mfSrc,
                                                             amrex::Real const scaling,
                                                             amrex::IntVect const crseRatio)
{
    return std::make_unique<T>(mfSrc, scaling, crseRatio);
}

ConstructorType match_custom_id(std::string const& key);

// -------------------- Helpers for CustomOperatorOutputProcessor --------------------
// Type Erasure helper class a la Iglberger ("C++ Software Design", 2022 O'Reilly Media, Inc.)
// OwningModel is separate because CUDA complained otherwise.
template <typename T>
struct OwningModel
{
    OwningModel(T object) : m_object{std::move(object)} {}

    void operator()(amrex::Box const& bx,
                    amrex::Array4<amrex::Real> dst,
                    amrex::Array4<amrex::Real const> const src,
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
                amrex::Box const& bx,
                amrex::Array4<amrex::Real> dst,
                amrex::Array4<amrex::Real const> const src,
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
    void operator()(amrex::Box const& bx,
                    amrex::Array4<amrex::Real> dst,
                    amrex::Array4<amrex::Real const> const src,
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
} //namespace Gempic::Io::Impl

namespace Gempic::Io
{
template <typename T>
void add_output_processor(std::string const& str);

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
    CustomOutputProcessor(amrex::MultiFab const& mfSrc,
                          amrex::Real const scaling,
                          amrex::IntVect const crseRatio);

    /** \brief copy m_mfSrc to mfDst.
     *
     * \param[out] mfDst output MultiFab where the result is written
     */
    void operator()(amrex::MultiFab& mfDst, int /*dcomp*/) const override;

    amrex::BoxArray get_box_array(amrex::IndexType& t) override;

protected:
    /** Alias of source MultiFab (can be multi-component) */
    amrex::MultiFab const m_mfSrc;
    // Scaling factor
    amrex::Real const m_scaling{1.0};

private:
    /** map of all user-provided outputProcessor classes */
    inline static std::map<std::string, ConstructorType> s_outputProcessorClassMap{};

    template <typename T>
    friend void add_output_processor(std::string const&);

    friend ConstructorType Impl::match_custom_id(std::string const&);
};

// Add custom class to the list along with a key so it can be selected in the input file
template <typename T>
void add_output_processor (std::string const& key)
{
    // MUST provide a derived class
    static_assert(std::is_base_of_v<CustomOutputProcessor, T>);
    if (CustomOutputProcessor::s_outputProcessorClassMap.find(key) !=
        CustomOutputProcessor::s_outputProcessorClassMap.end())
    {
        GEMPIC_ERROR("Already defined custom diagnostic Id: '" + key + "'.");
    }
    CustomOutputProcessor::s_outputProcessorClassMap.emplace(key, Impl::dynamic_convert<T>);
}

// Add custom lamba to the list along with a key so it can be selected in the input file
void add_output_processor(std::string const& key, Impl::CustomOperatorType const& f);

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
    CustomOperatorOutputProcessor(amrex::MultiFab const& mfSrc,
                                  amrex::Real const scaling,
                                  amrex::IntVect const crseRatio,
                                  CustomOperatorType const& f);

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

    friend void add_output_processor(std::string const& key, CustomOperatorType const& f);
    friend Impl::ConstructorType Impl::match_custom_id(std::string const&);
};

/**
 * A higher resolution writer: high-order interpolation on a finer grid.
 */
class HighResSubcellOutputProcessor : public CustomOutputProcessor
{
public:
    void operator()(amrex::MultiFab& mfDst, int dcomp) const final;

    amrex::BoxArray get_box_array(amrex::IndexType& t) final;

    HighResSubcellOutputProcessor(amrex::MultiFab const& mfSrc,
                                  amrex::Real const scaling,
                                  amrex::IntVect const crseRatio);

    // V0, V3 interpolation
    /**
     * interpolation function from the source to a finer multifab.
     *
     * @param mfDst[out] : destination cell-centered finer multifab (output data)
     *
     * @param mfDst[in] : destination cell-centered finer multifab iterator (output data)
     *
     * @param mfiDst0[in] : destination cell-centered non-refined multifab iterator
     *
     * @param mfiSrc[in] : source non-refined multifab iterator
     */
    template <int degree, Grid grid, Space space>
    void io_interpolation(amrex::MultiFab& mfDst, amrex::MFIter& mfiSrc) const;
    // V1, V2 interpolation
    /**
     * interpolation function from the source to a finer multifab.
     *
     * @param mfDst[out] : destination cell-centered finer multifab (output data)
     *
     * @param mfDst[in] : destination cell-centered finer multifab iterator (output data)
     *
     * @param mfiDst0[in] : destination cell-centered non-refined multifab iterator
     *
     * @param mfiSrc[in] : source non-refined multifab iterator
     */
    template <int degree, Grid grid, Space space, Direction dir>
    void io_interpolation(amrex::MultiFab& mfDst, amrex::MFIter& mfiSrc) const;

private:
    static constexpr int s_polyrecDegree{4};
    std::array<int, 3> m_sNSubCell3d{1, 1, 1};

    std::array<int, 3> m_shiftNode{1, 1, 1};
    std::array<int, 3> m_shiftCell{0, 0, 0};
    std::array<int, 3> m_shiftXFace{1, 0, 0};
    std::array<int, 3> m_shiftYFace{0, 1, 0};
    std::array<int, 3> m_shiftZFace{0, 0, 1};
    std::array<int, 3> m_shiftXEdge{0, 1, 1};
    std::array<int, 3> m_shiftYEdge{1, 0, 1};
    std::array<int, 3> m_shiftZEdge{1, 1, 0};
    std::array<int, 3> m_shiftV{0, 0, 0};
};

} //namespace Gempic::Io

#endif // GEMPIC_CUSTOMDIAGNOSTICSTRATEGIES_H_

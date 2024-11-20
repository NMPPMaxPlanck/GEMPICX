#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_CustomDiagnosticStrategies.H"
#include "GEMPIC_DiagnosticStrategies.H"

namespace Gempic::Io
{
CustomOutputProcessor::CustomOutputProcessor(const amrex::MultiFab& mfSrc,
                                             const amrex::Real scaling,
                                             const amrex::IntVect crseRatio) :
    ComputeDiagOutputProcessor{mfSrc.nComp(), crseRatio},
    m_mfSrc{amrex::MultiFab(mfSrc, amrex::make_alias, 0, mfSrc.nComp())},
    m_scaling{scaling}
{
}

void CustomOutputProcessor::operator ()(amrex::MultiFab& mfDst, int /*dcomp*/) const
{
    amrex::Assert("This operator should be overridden!", __FILE__, __LINE__);
}

Impl::ConstructorType Impl::match_custom_id (const std::string& key)
{
    if (CustomOutputProcessor::s_outputProcessorClassMap.find(key) !=
        CustomOutputProcessor::s_outputProcessorClassMap.end())
    {
        return (CustomOutputProcessor::s_outputProcessorClassMap.at(key));
    }
    else if (CustomOperatorOutputProcessor::s_outputProcessorMap.find(key) !=
             CustomOperatorOutputProcessor::s_outputProcessorMap.end())
    {
        return [=] (const amrex::MultiFab& mfSrc, const amrex::Real scaling,
                    const amrex::IntVect crseRatio) -> std::unique_ptr<ComputeDiagOutputProcessor>
        {
            return std::make_unique<CustomOperatorOutputProcessor> (
                mfSrc, scaling, crseRatio,
                CustomOperatorOutputProcessor::s_outputProcessorMap.at(key));
        };
    }
    else
    {
        std::string msg{"Unknown custom diagnostic Id: '" + key + "'."};
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
        return Impl::dynamic_convert<CustomOutputProcessor>;
    }
}

void add_output_processor (const std::string& key, const Impl::CustomOperatorType& f)
{
    if (CustomOperatorOutputProcessor::s_outputProcessorMap.find(key) !=
        CustomOperatorOutputProcessor::s_outputProcessorMap.end())
    {
        std::string msg{"Already defined custom diagnostic Id: '" + key + "'."};
        amrex::Assert(msg.c_str(), __FILE__, __LINE__);
    }
    else
    {
        CustomOperatorOutputProcessor::s_outputProcessorMap.emplace(key, f);
    }
}

CustomOperatorOutputProcessor::CustomOperatorOutputProcessor(const amrex::MultiFab& mfSrc,
                                                             const amrex::Real scaling,
                                                             const amrex::IntVect crseRatio,
                                                             const CustomOperatorType& f) :
    CustomOutputProcessor{mfSrc, scaling, crseRatio}, m_f{f}
{
    BL_PROFILE("CustomOperatorOutputProcessor::CustomOperatorOutputProcessor()");

    const amrex::IntVect indexTypeSrc = mfSrc.boxArray().ixType().toIntVect();
    AMREX_D_TERM(m_ishift = indexTypeSrc[xDir];, m_jshift = indexTypeSrc[yDir];
                 , m_kshift = indexTypeSrc[zDir];)
}

void CustomOperatorOutputProcessor::operator ()(amrex::MultiFab& mfDst, int /*dcomp*/) const
{
    BL_PROFILE("CustomOperatorOutputProcessor::operator()");
    int nComps = m_mfSrc.nComp();

    for (amrex::MFIter mfi(mfDst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4 dst = mfDst[mfi].array();
        amrex::Array4 const src = m_mfSrc[mfi].array();

        m_f(bx, dst, src, nComps, m_scaling, m_ishift, m_jshift, m_kshift);
    }
}
}  //namespace Gempic::Io

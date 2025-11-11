#ifndef GEMPIC_FDDERHAMCOMPLEX_H_
#define GEMPIC_FDDERHAMCOMPLEX_H_

#include <AMReX_SPACE.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_DeRhamComplex.H"
#include "GEMPIC_FieldMethods.H"

namespace Gempic::Forms
{

template <>
struct DualOf<DeRhamComplex::Primal0Form>
{
    using type = DeRhamComplex::Dual3Form;
};
template <>
struct DualOf<DeRhamComplex::Dual3Form>
{
    using type = DeRhamComplex::Primal0Form;
};
template <>
struct DualOf<DeRhamComplex::Primal2Form>
{
    using type = DeRhamComplex::Dual1Form;
};
template <>
struct DualOf<DeRhamComplex::Dual1Form>
{
    using type = DeRhamComplex::Primal2Form;
};

template <>
struct DualOf<DeRhamComplex::Primal1Form>
{
    using type = DeRhamComplex::Dual2Form;
};
template <>
struct DualOf<DeRhamComplex::Dual2Form>
{
    using type = DeRhamComplex::Primal1Form;
};
template <>
struct DualOf<DeRhamComplex::Primal3Form>
{
    using type = DeRhamComplex::Dual0Form;
};
template <>
struct DualOf<DeRhamComplex::Dual0Form>
{
    using type = DeRhamComplex::Primal3Form;
};

/// FDDeRhamComplex
class FDDeRhamComplex : public DeRhamComplex
{
private:
    amrex::RealVect m_dr;
    int m_nGhost;
    HodgeScheme m_hodgeScheme;

    // Temporary MultiFabs used in the Hodges
    amrex::MultiFab m_tempPrimalZeroForm;
    amrex::MultiFab m_tempDualZeroForm;
    amrex::Array<amrex::MultiFab, 3> m_tempPrimalOneForm;
    amrex::Array<amrex::MultiFab, 3> m_tempDualOneForm;
    amrex::Array<amrex::MultiFab, 3> m_tempPrimalTwoForm;
    amrex::Array<amrex::MultiFab, 3> m_tempDualTwoForm;
    amrex::MultiFab m_tempPrimalThreeForm;
    amrex::MultiFab m_tempDualThreeForm;

    // define scaling parameters for the Hodges
    amrex::Real m_sV;
    amrex::Real m_sOmega;

public:
    /// FDDeRhamComplex constructor
    /// @param infra
    /// @param hodgeDegree
    /// @param maxSplineDegree
    /// @param hodgeScheme
    FDDeRhamComplex(ComputationalDomain const& infra,
                    int const hodgeDegree,
                    int const maxSplineDegree,
                    HodgeScheme const hodgeScheme = HodgeScheme::FDHodge,
                    int nComp = 1);

    HodgeScheme get_hodge_scheme () const override { return m_hodgeScheme; }
    amrex::MultiFab& get_temp_primal_zero_form () override { return m_tempPrimalZeroForm; }
    amrex::MultiFab& get_temp_dual_zero_form () override { return m_tempDualZeroForm; }
    amrex::Array<amrex::MultiFab, 3>& get_temp_primal_one_form () override
    {
        return m_tempPrimalOneForm;
    }
    amrex::Array<amrex::MultiFab, 3>& get_temp_dual_one_form () override
    {
        return m_tempDualOneForm;
    }
    amrex::Array<amrex::MultiFab, 3>& get_temp_primal_two_form () override
    {
        return m_tempPrimalTwoForm;
    }
    amrex::Array<amrex::MultiFab, 3>& get_temp_dual_two_form () override
    {
        return m_tempDualTwoForm;
    }
    amrex::MultiFab& get_temp_primal_three_form () override { return m_tempPrimalThreeForm; }
    amrex::MultiFab& get_temp_dual_three_form () override { return m_tempDualThreeForm; }

    ~FDDeRhamComplex() override;

    // Getter function to access m_sV
    inline amrex::Real get_s_v () const { return m_sV; }
    inline amrex::Real get_s_omega () const { return m_sOmega; }

    /**
     * @brief Returns the scaling factor for the Hodges \f$I_0\f$
     */
    amrex::Real scaling_dto_e();
    amrex::Real scaling_eto_d();
    amrex::Real scaling_bto_h();

private:
};

} // namespace Gempic::Forms

#endif

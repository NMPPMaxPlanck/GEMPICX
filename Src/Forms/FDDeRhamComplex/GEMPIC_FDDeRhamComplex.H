#ifndef GEMPIC_FDDERHAMCOMPLEX_H_
#define GEMPIC_FDDERHAMCOMPLEX_H_

#include <AMReX_SPACE.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"

namespace Gempic::Forms
{
enum class HodgeScheme
{
    FDHodge,
    GalerkinHodge,
    GDECHodge,     // Galerkin Differences. No dual grid is used.
    GDECLumpHodge, // with Mass lumping
};

/**
 * @brief Fills the arrays of Hodge stencils.
 *
 * @details Computes stencils for periodic boundary conditions for either
 * finite difference of Galerkin Hodge
 *
 * @tparam degree int, degree of basis functions used
 * @tparam scheme int, 0 for FDHodge, 1 for Galerkin Hodge
 * @return std::tuple containing the stencils in the following order:
 *                      stencilNodeToCell, stencilCellToNode
 */
// NOLINTBEGIN(readability-magic-numbers)
template <int degree, HodgeScheme scheme>
inline auto get_hodge_stencils ()
{
    BL_PROFILE("Gempic::Forms::get_hodge_stencils()");
    if constexpr (scheme == HodgeScheme::FDHodge)
    {
        constexpr unsigned int stencilLength{degree - 1};
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        switch (degree)
        {
            case 2:
                stencilNodeToCell[0] = 1.0;
                stencilCellToNode[0] = 1.0;
                break;
            case 4:
                stencilNodeToCell[0] = 1. / 24.;
                stencilNodeToCell[1] = 22. / 24.;
                stencilNodeToCell[2] = 1. / 24.;
                stencilCellToNode[0] = -1. / 24.;
                stencilCellToNode[1] = 13. / 12.;
                stencilCellToNode[2] = -1. / 24.;
                break;
            case 6:
                stencilNodeToCell[0] = -17. / 5760.;
                stencilNodeToCell[1] = 154. / 2880.;
                stencilNodeToCell[2] = 863. / 960.;
                stencilNodeToCell[3] = 154. / 2880.;
                stencilNodeToCell[4] = -17. / 5760.;
                stencilCellToNode[0] = 3. / 640.;
                stencilCellToNode[1] = -29. / 480.;
                stencilCellToNode[2] = 1067. / 960.;
                stencilCellToNode[3] = -29. / 480.;
                stencilCellToNode[4] = 3. / 640.;
                break;
            default:
                AMREX_ASSERT("Degree not implemented");
                break;
        }

        return std::make_tuple(stencilNodeToCell, stencilCellToNode);
    }
    if constexpr (scheme == HodgeScheme::GalerkinHodge)
    {
        constexpr unsigned int stencilLength{2 * degree - 1};
        amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
        amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

        switch (degree)
        {
            case 2:
                stencilCellToNode[0] = 1. / 8.;
                stencilCellToNode[1] = 3. / 4.;
                stencilCellToNode[2] = 1. / 8.;
                stencilNodeToCell = stencilCellToNode;
                break;
            case 4:
                stencilCellToNode[0] = 121. / 46080.;
                stencilCellToNode[1] = -959. / 23040.;
                stencilCellToNode[2] = 4663. / 46080.;
                stencilCellToNode[3] = 10087. / 11520.;
                stencilCellToNode[4] = 4663. / 46080.;
                stencilCellToNode[5] = -959. / 23040.;
                stencilCellToNode[6] = 121. / 46080.;
                stencilNodeToCell = stencilCellToNode;
                break;
            case 6:
                stencilCellToNode[0] = 365053. / 3715891200.;
                stencilCellToNode[1] = -2981533. / 1857945600.;
                stencilCellToNode[2] = 55805753. / 3715891200.;
                stencilCellToNode[3] = -27050951. / 464486400.;
                stencilCellToNode[4] = 147416293. / 1857945600.;
                stencilCellToNode[5] = 864656441. / 928972800.;
                stencilCellToNode[6] = 147416293. / 1857945600.;
                stencilCellToNode[7] = -27050951. / 464486400.;
                stencilCellToNode[8] = 55805753. / 3715891200.;
                stencilCellToNode[9] = -2981533. / 1857945600.;
                stencilCellToNode[10] = 365053. / 3715891200.;
                stencilNodeToCell = stencilCellToNode;
                break;
            default:
                AMREX_ASSERT("Degree not implemented");
                break;
        }

        return std::make_tuple(stencilNodeToCell, stencilCellToNode);
    }
    if constexpr (scheme == HodgeScheme::GDECHodge)
    {
        constexpr unsigned int order{
            degree}; // the so called degree is actually the order of accuracy (=degree+1) =>
        constexpr unsigned int stencilLength{2 * order - 1};
        amrex::GpuArray<amrex::Real, stencilLength>
            stencilHodgeV0; // V0 or V1 refer to the 1D complex V0 --dx--> V1
        amrex::GpuArray<amrex::Real, stencilLength>
            stencilHodgeV1; // V0 or V1 refer to the 1D complex V0 --dx--> V1

        switch (order)
        {
            case 2:
                stencilHodgeV0[0] = 1. / 6.;
                stencilHodgeV0[1] = 2. / 3.;
                stencilHodgeV0[2] = 1. / 6.;
                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = 1.;
                stencilHodgeV1[2] = 0.;
                break;
            case 4:
                stencilHodgeV0[0] = 31. / 15120.;
                stencilHodgeV0[1] = -3. / 70.;
                stencilHodgeV0[2] = 257. / 1680.;
                stencilHodgeV0[3] = 733. / 945.;
                stencilHodgeV0[4] = 257. / 1680.;
                stencilHodgeV0[5] = -3. / 70.;
                stencilHodgeV0[6] = 31. / 15120.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -7. / 360.;
                stencilHodgeV1[2] = -1. / 180.;
                stencilHodgeV1[3] = 21. / 20.;
                stencilHodgeV1[4] = -1. / 180.;
                stencilHodgeV1[5] = -7. / 360.;
                stencilHodgeV1[6] = 0.;
                break;
            case 6:
                stencilHodgeV0[0] = 313. / 4435200.;
                stencilHodgeV0[1] = -3937. / 3326400.;
                stencilHodgeV0[2] = 59053. / 4435200.;
                stencilHodgeV0[3] = -12421. / 207900.;
                stencilHodgeV0[4] = 906919. / 6652800.;
                stencilHodgeV0[5] = 455963. / 554400.;
                stencilHodgeV0[6] = 906919. / 6652800.;
                stencilHodgeV0[7] = -12421. / 207900.;
                stencilHodgeV0[8] = 59053. / 4435200.;
                stencilHodgeV0[9] = -3937. / 3326400.;
                stencilHodgeV0[10] = 313. / 4435200.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -89. / 129600.;
                stencilHodgeV1[2] = 4547. / 453600.;
                stencilHodgeV1[3] = -4003. / 113400.;
                stencilHodgeV1[4] = -9691. / 453600.;
                stencilHodgeV1[5] = 99307. / 90720.;
                stencilHodgeV1[6] = -9691. / 453600.;
                stencilHodgeV1[7] = -4003. / 113400.;
                stencilHodgeV1[8] = 4547. / 453600.;
                stencilHodgeV1[9] = -89. / 129600.;
                stencilHodgeV1[10] = 0.;
                break;
            default:
                AMREX_ASSERT("order not implemented");
                break;
        }

        return std::make_tuple(stencilHodgeV0, stencilHodgeV1);
    }
    if constexpr (scheme == HodgeScheme::GDECLumpHodge)
    {
        constexpr unsigned int order{
            degree}; // the so called degree is actually the order of accuracy (=degree+1) =>
        constexpr unsigned int stencilLength{2 * order - 1};
        amrex::GpuArray<amrex::Real, stencilLength>
            stencilHodgeV0; // V0 or V1 refer to the 1D complex V0 --dx--> V1
        amrex::GpuArray<amrex::Real, stencilLength>
            stencilHodgeV1; // V0 or V1 refer to the 1D complex V0 --dx--> V1

        switch (order)
        {
            case 2:
                stencilHodgeV0[0] = 0.;
                stencilHodgeV0[1] = 1.;
                stencilHodgeV0[2] = 0.;
                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = 1.;
                stencilHodgeV1[2] = 0.;
                break;
            case 4:
                stencilHodgeV0[0] = 0.;
                stencilHodgeV0[1] = 0.;
                stencilHodgeV0[2] = 0.;
                stencilHodgeV0[3] = 1.;
                stencilHodgeV0[4] = 0.;
                stencilHodgeV0[5] = 0.;
                stencilHodgeV0[6] = 0.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -7. / 360.;
                stencilHodgeV1[2] = -1. / 180.;
                stencilHodgeV1[3] = 21. / 20.;
                stencilHodgeV1[4] = -1. / 180.;
                stencilHodgeV1[5] = -7. / 360.;
                stencilHodgeV1[6] = 0.;
                break;
            case 6:
                stencilHodgeV0[0] = 0.;
                stencilHodgeV0[1] = 0.;
                stencilHodgeV0[2] = 0.;
                stencilHodgeV0[3] = 0.;
                stencilHodgeV0[4] = 0.;
                stencilHodgeV0[5] = 1.;
                stencilHodgeV0[6] = 0.;
                stencilHodgeV0[7] = 0.;
                stencilHodgeV0[8] = 0.;
                stencilHodgeV0[9] = 0.;
                stencilHodgeV0[10] = 0.;

                stencilHodgeV1[0] = 0.;
                stencilHodgeV1[1] = -89. / 129600.;
                stencilHodgeV1[2] = 4547. / 453600.;
                stencilHodgeV1[3] = -4003. / 113400.;
                stencilHodgeV1[4] = -9691. / 453600.;
                stencilHodgeV1[5] = 99307. / 90720.;
                stencilHodgeV1[6] = -9691. / 453600.;
                stencilHodgeV1[7] = -4003. / 113400.;
                stencilHodgeV1[8] = 4547. / 453600.;
                stencilHodgeV1[9] = -89. / 129600.;
                stencilHodgeV1[10] = 0.;
                break;
            default:
                AMREX_ASSERT("order not implemented");
                break;
        }

        return std::make_tuple(stencilHodgeV0, stencilHodgeV1);
    }
}
// NOLINTEND(readability-magic-numbers)

/**
 * @brief Computes a matrix-vector product along 1 spatial direction.
 *
 * @details
 * Computes M*u for all edges in a given direction, where M is given
 * by a stencil and u are the values of the input along one edge.
 *
 * @tparam stencilLength int, length of the stencil array defining the matrix
 * @tparam direction Direction, direction along which matrix vector product is calculated
 * @param geom Geometry, needed for FillBoundary
 * @param stencil GpuArray, defining the matrix
 * @param multiInput MultiFab, containing the values of the vector being multiplied
 * @param multiOutput MultiFab, containing the result
 */

template <Direction direction, int hodgeDegree, unsigned int stencilLength>
inline void apply_1d_hodge (amrex::Geometry geom,
                            amrex::GpuArray<amrex::Real, stencilLength> stencil,
                            amrex::MultiFab const& multiInput,
                            amrex::MultiFab& multiOutput)
{
    BL_PROFILE("Gempic::Forms::apply_1d_hodge()");
    int nComps{multiInput.nComp()};
    for (amrex::MFIter mfi(multiOutput); mfi.isValid(); ++mfi)
    {
        amrex::Box bx = mfi.validbox();

        // Grow box to include sufficient number of ghost cells for stencil application in following
        // non-periodic dimensions. In the periodic case, this is handled by FillBoundary.
#if AMREX_SPACEDIM >= 2
        constexpr int nGrow = hodgeDegree / 2 - 1;
        if constexpr (direction == xDir)
        {
            if (!geom.isPeriodic(yDir))
            {
                bx.grow(yDir, nGrow);
            }
        }
#endif
#if AMREX_SPACEDIM == 3
        if constexpr (direction == xDir || direction == yDir)
        {
            if (!geom.isPeriodic(zDir))
            {
                bx.grow(zDir, nGrow);
            }
        }
#endif

        amrex::Array4<amrex::Real const> const& vecIn = multiInput[mfi].const_array();
        amrex::Array4<amrex::Real> const& vecOut = multiOutput[mfi].array();

        ParallelFor(bx, nComps,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        amrex::Real val = 0.0;

                        for (int d = 0; d < stencilLength; d++)
                        {
                            int shift = d - (stencilLength + 1) / 2 + 1;
                            val += stencil[d] * vecIn(i + (direction == xDir ? shift : 0),
                                                      j + (direction == yDir ? shift : 0),
                                                      k + (direction == zDir ? shift : 0), n);
                        }
                        vecOut(i, j, k, n) = val;
                    });
    }
    multiOutput.FillBoundary(geom.periodicity());
}

/// FDDeRhamComplex
class FDDeRhamComplex : public DeRhamComplex
{
private:
    static constexpr std::array<int, 3> s_validDegreesFDHodge{2, 4, 6};
    static constexpr std::array<int, 3> s_validDegreesGalerkinHodge{2, 4, 6};
    static constexpr std::array<int, 3> s_validDegreesGDECHodge{2, 4, 6};
    static constexpr std::array<int, 3> s_validDegreesGDECLumpHodge{2, 4, 6};

    template <HodgeScheme hodgeScheme>
    static constexpr auto valid_degrees ()
    {
        if constexpr (hodgeScheme == HodgeScheme::FDHodge)
        {
            return s_validDegreesFDHodge;
        }
        else if constexpr (hodgeScheme == HodgeScheme::GalerkinHodge)
        {
            return s_validDegreesGalerkinHodge;
        }
        else if constexpr (hodgeScheme == HodgeScheme::GDECHodge)
        {
            return s_validDegreesGDECHodge;
        }
        else if constexpr (hodgeScheme == HodgeScheme::GDECLumpHodge)
        {
            return s_validDegreesGDECLumpHodge;
        }
        else
        {
            return s_validDegreesNoScheme;
        }
    }

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
                    HodgeScheme const hodgeScheme = HodgeScheme::FDHodge);

    ~FDDeRhamComplex() override;

    // Getter function to access m_sV
    inline amrex::Real get_s_v () const { return m_sV; }
    inline amrex::Real get_s_omega () const { return m_sOmega; }

    // 0-/3-forms
    void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                    amrex::Real t,
                    DeRhamField<Grid::primal, Space::node>& field) override;
    void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                    amrex::Real t,
                    DeRhamField<Grid::primal, Space::node>& field) override;

    void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                    amrex::Real t,
                    DeRhamField<Grid::primal, Space::cell>& field,
                    int gaussNodes = 6) override;
    void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                    amrex::Real t,
                    DeRhamField<Grid::primal, Space::cell>& field,
                    int gaussNodes = 6) override;

    void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                    amrex::Real t,
                    DeRhamField<Grid::dual, Space::node>& field) override;
    void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                    amrex::Real t,
                    DeRhamField<Grid::dual, Space::node>& field) override;

    void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                    amrex::Real t,
                    DeRhamField<Grid::dual, Space::cell>& field,
                    int gaussNodes = 6) override;
    void projection(amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>> const& funcs,
                    amrex::Real t,
                    DeRhamField<Grid::dual, Space::cell>& field,
                    int gaussNodes = 6) override;

    // 1-/2-forms
    void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                    amrex::Real t,
                    DeRhamField<Grid::dual, Space::edge>& field,
                    int gaussNodes = 6) override;
    void projection(
        amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
        amrex::Real t,
        DeRhamField<Grid::dual, Space::edge>& field,
        int gaussNodes = 6) override;

    void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                    amrex::Real t,
                    DeRhamField<Grid::primal, Space::face>& field,
                    int gaussNodes = 6) override;
    void projection(
        amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
        amrex::Real t,
        DeRhamField<Grid::primal, Space::face>& field,
        int gaussNodes = 6) override;

    void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                    amrex::Real t,
                    DeRhamField<Grid::primal, Space::edge>& field,
                    int gaussNodes = 6) override;
    void projection(
        amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
        amrex::Real t,
        DeRhamField<Grid::primal, Space::edge>& field,
        int gaussNodes = 6) override;

    void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                    amrex::Real t,
                    DeRhamField<Grid::dual, Space::face>& field,
                    int gaussNodes = 6) override;
    void projection(
        amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>> const& funcs,
        amrex::Real t,
        DeRhamField<Grid::dual, Space::face>& field,
        int gaussNodes = 6) override;

    void hodge(DeRhamField<Grid::primal, Space::node>& zeroForm,
               DeRhamField<Grid::dual, Space::cell> const& threeForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::dual, Space::node>& zeroForm,
               DeRhamField<Grid::primal, Space::cell> const& threeForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::primal, Space::edge>& oneForm,
               DeRhamField<Grid::dual, Space::face> const& twoForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::dual, Space::edge>& oneForm,
               DeRhamField<Grid::primal, Space::face> const& twoForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::primal, Space::face>& twoForm,
               DeRhamField<Grid::dual, Space::edge> const& oneForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::dual, Space::face>& twoForm,
               DeRhamField<Grid::primal, Space::edge> const& oneForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::primal, Space::cell>& threeForm,
               DeRhamField<Grid::dual, Space::node> const& zeroForm,
               amrex::Real weight = 1.) override;
    void hodge(DeRhamField<Grid::dual, Space::cell>& threeForm,
               DeRhamField<Grid::primal, Space::node> const& zeroForm,
               amrex::Real weight = 1.) override;

    /**
     * @brief Computes the Interpolation operator \f$I_0\f$
     *
     * @details
     * Using the geometric degrees of freedeom:
     * Interpolates nodes of the primal grid using Lagrange polynomials of degree (dim - 1)
     *
     * @param field DeRhamField<dual, Space::cell>, 3-form \f$u^3\f$ holding the cell integrals
     * @param evalShift amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>, point to be interpolated
     * @param pointValues MultiFab in which to store the interpolated values
     */
    void eval_form(DeRhamField<Grid::primal, Space::cell> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues) override;
    void eval_form(DeRhamField<Grid::dual, Space::cell> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues) override;
    void eval_form(DeRhamField<Grid::primal, Space::face> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues,
                   Direction dir) override;
    void eval_form(DeRhamField<Grid::dual, Space::face> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues,
                   Direction dir) override;
    void eval_form(DeRhamField<Grid::primal, Space::edge> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues,
                   Direction dir) override;
    void eval_form(DeRhamField<Grid::dual, Space::edge> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues,
                   Direction dir) override;
    void eval_form(DeRhamField<Grid::primal, Space::node> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues) override;
    void eval_form(DeRhamField<Grid::dual, Space::node> const& field,
                   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
                   amrex::MultiFab& pointValues) override;

    /**
     * @brief Returns the scaling factor for the Hodges \f$I_0\f$
     */
    amrex::Real scaling_dto_e();
    amrex::Real scaling_eto_d();
    amrex::Real scaling_bto_h();

private:
    /**
     * @brief Selects a precompiled Hodge operator scheme from a runtime scheme parameter
     *
     * @param f1 const source field
     * @param f2 destination field
     * @param weight optional weight for the Hodge
     */
    void hodge_scheme_selector(anyFieldConstRef f1, anyFieldRef f2, amrex::Real weight = 1.);

    /**
     * @brief Selects a precompiled Hodge operator from a runtime Hodge degree.
     *
     * Recursively calls itself until the runtime Hodge degree == validDegrees[hodgeDegreeIdx],
     * then calls hodge<hodgeDegree, hodgeScheme>.
     *
     * @tparam hodgeScheme    Hodge scheme as found by hodge_scheme_selector
     * @tparam hodgeDegreeIdx Not the Hodge degree! The index of the validDegrees array
     * corresponding to the Hodge scheme.
     *
     * @param hodgeSrc const source field
     * @param hodgeDst destination field
     * @param weight optional weight for the hodge
     */
    template <HodgeScheme hodgeScheme, int hodgeDegreeIdx = 0>
    void hodge_degree_selector(anyFieldConstRef hodgeSrc,
                               anyFieldRef hodgeDst,
                               amrex::Real weight = 1.);

    /**
     * @brief Selects a precompiled direction for eval_form from a runtime direction parameter
     *
     * @param field const source field
     * @param evalShift vector of shifts from lower corner of primal cell (between 0 and 1)
     * @param pointValues Multifab in which to store the interpolated values
     * @param dir Direction in which to evaluate the interpolation(s)
     */
    inline void eval_form_direction_selector (
        anyFieldConstRef field,
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
        amrex::MultiFab& pointValues,
        Direction dir)
    {
        switch (dir)
        {
            case xDir:
                this->eval_form_degree_selector<xDir>(field, evalShift, pointValues);
                break;
            case yDir:
                this->eval_form_degree_selector<yDir>(field, evalShift, pointValues);
                break;
            case zDir:
                this->eval_form_degree_selector<zDir>(field, evalShift, pointValues);
                break;
            default:
                AMREX_ASSERT("Unrecognized direction passed to eval_form");
                break;
        }
    }

    /**
     * @brief Selects a precompiled eval_form operator from a runtime Hodge degree.
     *
     * Recursively calls itself until the runtime Hodge degree == validDegrees[hodgeDegreeIdx],
     * then calls
     *   eval_form_component<degree, grid, space> for 0-/3-forms, or
     *   eval_form_component<degree, grid, space, dir> for 1-/2-forms
     *
     * @tparam dir            Optional direction as found by eval_form_direction_selector
     * @tparam hodgeDegreeIdx Not the Hodge degree! The index of the validDegrees array
     * corresponding to the Hodge scheme. NOTE: Only Finite Difference Hodge schemes work for now.
     *
     * @param field const source field
     * @param evalShift vector of shifts from lower corner of primal cell (between 0 and 1)
     * @param pointValues Multifab in which to store the interpolated values
     */
    template <Direction dir = xDir, int hodgeDegreeIdx = 0>
    void eval_form_degree_selector(
        anyFieldConstRef field,
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
        amrex::MultiFab& pointValues);

    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::dual, Space::cell> const& threeForm,
                        DeRhamField<Grid::primal, Space::node>& zeroForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::primal, Space::cell> const& threeForm,
                        DeRhamField<Grid::dual, Space::node>& zeroForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::dual, Space::face> const& twoForm,
                        DeRhamField<Grid::primal, Space::edge>& oneForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::primal, Space::face> const& twoForm,
                        DeRhamField<Grid::dual, Space::edge>& oneForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::dual, Space::edge> const& oneForm,
                        DeRhamField<Grid::primal, Space::face>& twoForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::primal, Space::edge> const& oneForm,
                        DeRhamField<Grid::dual, Space::face>& twoForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::dual, Space::node> const& zeroForm,
                        DeRhamField<Grid::primal, Space::cell>& threeForm,
                        amrex::Real weight = 1.);
    template <int hodgeDegree, HodgeScheme hodgeScheme>
    void hodge_operator(DeRhamField<Grid::primal, Space::node> const& zeroForm,
                        DeRhamField<Grid::dual, Space::cell>& threeForm,
                        amrex::Real weight = 1.);
};

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_3\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_3\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param threeForm DeRhamField<Grid::dual, Space::cell>, 3-form \f$\tilde{u}^3\f$
 * @param zeroForm DeRhamField<Grid::primal, Space::node>, resulting 0-form \f$u^0\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::dual, Space::cell> const& threeForm,
                                     DeRhamField<Grid::primal, Space::node>& zeroForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(dual,cell,primal,node,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
        geom, stencilCellToNode, threeForm.m_data,
        AMREX_D_PICK(zeroForm.m_data, this->m_tempPrimalZeroForm, zeroForm.m_data)));
                 , (apply_1d_hodge<yDir, hodgeDegree>(
                       geom, stencilCellToNode,
                       AMREX_D_PICK((void), this->m_tempPrimalZeroForm, zeroForm.m_data),
                       AMREX_D_PICK((void), zeroForm.m_data, this->m_tempPrimalZeroForm)));
                 , (apply_1d_hodge<zDir, hodgeDegree>(
                       geom, stencilCellToNode, this->m_tempPrimalZeroForm, zeroForm.m_data)););

    zeroForm.m_data.mult(weight / GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]));

    zeroForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_3\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_3\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param threeForm DeRhamField<Grid::primal, Space::cell>, 3-form \f$u^3\f$
 * @param zeroForm DeRhamField<Grid::dual, Space::node>, resulting 0-form \f$\tilde{u}^0\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::primal, Space::cell> const& threeForm,
                                     DeRhamField<Grid::dual, Space::node>& zeroForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(primal,cell,dual,node,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
        geom, stencilCellToNode, threeForm.m_data,
        AMREX_D_PICK(zeroForm.m_data, this->m_tempDualZeroForm, zeroForm.m_data)));
                 , (apply_1d_hodge<yDir, hodgeDegree>(
                       geom, stencilCellToNode,
                       AMREX_D_PICK((void), this->m_tempDualZeroForm, zeroForm.m_data),
                       AMREX_D_PICK((void), zeroForm.m_data, this->m_tempDualZeroForm)));
                 , (apply_1d_hodge<zDir, hodgeDegree>(geom, stencilCellToNode,
                                                      this->m_tempDualZeroForm, zeroForm.m_data)););

    zeroForm.m_data.mult(weight / GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]));

    zeroForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_2\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_2\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param twoForm DeRhamField<Grid::dual, Space::face>, 2-form \f$\tilde{u}^2\f$
 * @param oneForm DeRhamField<Grid::primal, Space::edge>, resulting 1-form \f$u^1\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::dual, Space::face> const& twoForm,
                                     DeRhamField<Grid::primal, Space::edge>& oneForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(dual,face,primal,edge,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM(
            (apply_1d_hodge<xDir, hodgeDegree>(
                geom, (comp == xDir ? stencilNodeToCell : stencilCellToNode), twoForm.m_data[comp],
                AMREX_D_PICK(oneForm.m_data[comp], this->m_tempPrimalOneForm[comp],
                             oneForm.m_data[comp])));
            , (apply_1d_hodge<yDir, hodgeDegree>(
                  geom, (comp == yDir ? stencilNodeToCell : stencilCellToNode),
                  AMREX_D_PICK((void), this->m_tempPrimalOneForm[comp], oneForm.m_data[comp]),
                  AMREX_D_PICK((void), oneForm.m_data[comp], this->m_tempPrimalOneForm[comp])));
            , (apply_1d_hodge<zDir, hodgeDegree>(
                  geom, (comp == zDir ? stencilNodeToCell : stencilCellToNode),
                  this->m_tempPrimalOneForm[comp], oneForm.m_data[comp])););

        if (comp == xDir)
        {
            oneForm.m_data[comp].mult(dr[xDir] * weight / GEMPIC_D_MULT(1, dr[yDir], dr[zDir]));
        }
        if (comp == yDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, dr[yDir], 1) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, dr[zDir]));
        }
        if (comp == zDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, 1, dr[zDir]) * weight /
                                      GEMPIC_D_MULT(dr[xDir], dr[yDir], 1));
        }
    }

    oneForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_2\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_2\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param twoForm DeRhamField<Grid::primal, Space::face>, 2-form \f$u^2\f$
 * @param oneForm DeRhamField<Grid::dual, Space::edge>, resulting 1-form \f$\tilde{u}^1\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::primal, Space::face> const& twoForm,
                                     DeRhamField<Grid::dual, Space::edge>& oneForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(primal,face,dual,edge,weight)");
    // select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM(
            (apply_1d_hodge<xDir, hodgeDegree>(
                geom, (comp == xDir ? stencilNodeToCell : stencilCellToNode), twoForm.m_data[comp],
                AMREX_D_PICK(oneForm.m_data[comp], this->m_tempDualOneForm[comp],
                             oneForm.m_data[comp])));
            , (apply_1d_hodge<yDir, hodgeDegree>(
                  geom, (comp == yDir ? stencilNodeToCell : stencilCellToNode),
                  AMREX_D_PICK((void), this->m_tempDualOneForm[comp], oneForm.m_data[comp]),
                  AMREX_D_PICK((void), oneForm.m_data[comp], this->m_tempDualOneForm[comp])));
            , (apply_1d_hodge<zDir, hodgeDegree>(
                  geom, (comp == zDir ? stencilNodeToCell : stencilCellToNode),
                  this->m_tempDualOneForm[comp], oneForm.m_data[comp])););

        if (comp == xDir)
        {
            oneForm.m_data[comp].mult(dr[xDir] * weight / GEMPIC_D_MULT(1, dr[yDir], dr[zDir]));
        }
        if (comp == yDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, dr[yDir], 1) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, dr[zDir]));
        }
        if (comp == zDir)
        {
            oneForm.m_data[comp].mult(GEMPIC_D_MULT(1, 1, dr[zDir]) * weight /
                                      GEMPIC_D_MULT(dr[xDir], dr[yDir], 1));
        }
    }

    oneForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_1\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_1\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param oneForm DeRhamField<Grid::dual, Space::edge>, 1-form \f$\tilde{u}^1\f$
 * @param twoForm DeRhamField<Grid::primal, Space::face>, resulting 2-form \f$u^2\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::dual, Space::edge> const& oneForm,
                                     DeRhamField<Grid::primal, Space::face>& twoForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(dual,edge,primal,face,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM(
            (apply_1d_hodge<xDir, hodgeDegree>(
                geom, (comp == xDir ? stencilCellToNode : stencilNodeToCell), oneForm.m_data[comp],
                AMREX_D_PICK(twoForm.m_data[comp], this->m_tempPrimalTwoForm[comp],
                             twoForm.m_data[comp])));
            , (apply_1d_hodge<yDir, hodgeDegree>(
                  geom, (comp == yDir ? stencilCellToNode : stencilNodeToCell),
                  AMREX_D_PICK((void), this->m_tempPrimalTwoForm[comp], twoForm.m_data[comp]),
                  AMREX_D_PICK((void), twoForm.m_data[comp], this->m_tempPrimalTwoForm[comp])));
            , (apply_1d_hodge<zDir, hodgeDegree>(
                  geom, (comp == zDir ? stencilCellToNode : stencilNodeToCell),
                  this->m_tempPrimalTwoForm[comp], twoForm.m_data[comp])););

        if (comp == xDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(1, dr[yDir], dr[zDir])) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, 1));
        }
        if (comp == yDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], 1, dr[zDir])) * weight /
                                      GEMPIC_D_MULT(1, dr[yDir], 1));
        }
        if (comp == zDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], dr[yDir], 1)) * weight /
                                      GEMPIC_D_MULT(1, 1, dr[zDir]));
        }
    }

    twoForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_1\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_1\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param oneForm DeRhamField<Grid::primal, Space::edge>, 1-form \f$u^1\f$
 * @param twoForm DeRhamField<Grid::dual, Space::face>, resulting 2-form \f$\tilde{u}^2\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::primal, Space::edge> const& oneForm,
                                     DeRhamField<Grid::dual, Space::face>& twoForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(primal,edge,dual,face,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    for (int comp = 0; comp < 3; ++comp)
    {
        AMREX_D_TERM(
            (apply_1d_hodge<xDir, hodgeDegree>(
                geom, (comp == xDir ? stencilCellToNode : stencilNodeToCell), oneForm.m_data[comp],
                AMREX_D_PICK(twoForm.m_data[comp], this->m_tempDualTwoForm[comp],
                             twoForm.m_data[comp])));
            , (apply_1d_hodge<yDir, hodgeDegree>(
                  geom, (comp == yDir ? stencilCellToNode : stencilNodeToCell),
                  AMREX_D_PICK((void), this->m_tempDualTwoForm[comp], twoForm.m_data[comp]),
                  AMREX_D_PICK((void), twoForm.m_data[comp], this->m_tempDualTwoForm[comp])));
            , (apply_1d_hodge<zDir, hodgeDegree>(
                  geom, (comp == zDir ? stencilCellToNode : stencilNodeToCell),
                  this->m_tempDualTwoForm[comp], twoForm.m_data[comp])););

        if (comp == xDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(1, dr[yDir], dr[zDir])) * weight /
                                      GEMPIC_D_MULT(dr[xDir], 1, 1));
        }
        if (comp == yDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], 1, dr[zDir])) * weight /
                                      GEMPIC_D_MULT(1, dr[yDir], 1));
        }
        if (comp == zDir)
        {
            twoForm.m_data[comp].mult((GEMPIC_D_MULT(dr[xDir], dr[yDir], 1)) * weight /
                                      GEMPIC_D_MULT(1, 1, dr[zDir]));
        }
    }

    twoForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\tilde{\mathbb{H}}_0\f$.
 *
 * @details
 * Computes \f$\tilde{\mathbb{H}}_0\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param zeroForm DeRhamField<Grid::dual, Space::node>, 0-form \f$\tilde{u}^0\f$
 * @param threeForm DeRhamField<Grid::primal, Space::cell>, resulting 3-form \f$u^3\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::dual, Space::node> const& zeroForm,
                                     DeRhamField<Grid::primal, Space::cell>& threeForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(dual,node,primal,cell,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
        geom, stencilNodeToCell, zeroForm.m_data,
        AMREX_D_PICK(threeForm.m_data, this->m_tempPrimalThreeForm, threeForm.m_data)));
                 , (apply_1d_hodge<yDir, hodgeDegree>(
                       geom, stencilNodeToCell,
                       AMREX_D_PICK((void), this->m_tempPrimalThreeForm, threeForm.m_data),
                       AMREX_D_PICK((void), threeForm.m_data, this->m_tempPrimalThreeForm)));
                 , (apply_1d_hodge<zDir, hodgeDegree>(
                       geom, stencilNodeToCell, this->m_tempPrimalThreeForm, threeForm.m_data)););

    threeForm.m_data.mult(GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]) * weight);

    threeForm.fill_boundary();
}

/**
 * @brief Computes the Finite Difference Hodge \f$\mathbb{H}_0\f$.
 *
 * @details
 * Computes \f$\mathbb{H}_0\f$ by multiplying the 1D stencils along
 * each direction separately
 * Supports a constant weight function of the form w(x,y,z) = w
 *
 * Boundary conditions are periodic in all directions
 *
 * For directions 1 and 2, the k-forms are taken from the "stacked" de Rham complex
 *
 * Parameters:
 * @tparam hodgeDegree int, degree of basis functions used
 * @tparam hodgeScheme HodgeScheme, determines the stencils used for the Hodge operator
 * @param zeroForm DeRhamField<Grid::primal, Space::node>, 0-form \f$u^0\f$
 * @param threeForm DeRhamField<Grid::dual, Space::cell>, resulting 3-form \f$\tilde{u}^3\f$
 * @param weight Real, constant weight function
 */
template <int hodgeDegree, HodgeScheme hodgeScheme>
void FDDeRhamComplex::hodge_operator (DeRhamField<Grid::primal, Space::node> const& zeroForm,
                                     DeRhamField<Grid::dual, Space::cell>& threeForm,
                                     amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_operator(primal,node,dual,cell,weight)");
    // Select stencil according to degree
    auto [stencilNodeToCell, stencilCellToNode] = get_hodge_stencils<hodgeDegree, hodgeScheme>();

    amrex::RealVect const dr = this->m_dr;
    amrex::Geometry const geom = this->m_geom;

    AMREX_D_TERM((apply_1d_hodge<xDir, hodgeDegree>(
        geom, stencilNodeToCell, zeroForm.m_data,
        AMREX_D_PICK(threeForm.m_data, this->m_tempDualThreeForm, threeForm.m_data)));
                 , (apply_1d_hodge<yDir, hodgeDegree>(
                       geom, stencilNodeToCell,
                       AMREX_D_PICK((void), this->m_tempDualThreeForm, threeForm.m_data),
                       AMREX_D_PICK((void), threeForm.m_data, this->m_tempDualThreeForm)));
                 , (apply_1d_hodge<zDir, hodgeDegree>(
                       geom, stencilNodeToCell, this->m_tempDualThreeForm, threeForm.m_data)););

    threeForm.m_data.mult(GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir]) * weight);

    threeForm.fill_boundary();
}

template <HodgeScheme hodgeScheme, int testDegInd>
void FDDeRhamComplex::hodge_degree_selector (anyFieldConstRef hodgeSrc,
                                            anyFieldRef hodgeDst,
                                            amrex::Real weight)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::hodge_degree_selector()")
    constexpr auto validDegreesArray = valid_degrees<hodgeScheme>();
    if constexpr (testDegInd < validDegreesArray.size())
    {
        if (this->m_hodgeDegree == validDegreesArray[testDegInd])
        {
            // Iterate over all types in anyFieldRef (all grid/space combinations) and select the
            // correct one.
            std::visit(
                [this, &validDegreesArray, weight] (auto const& hSrc, auto& hDst)
                {
                    using namespace Impl;
                    using dstFieldTypename = unwrap_ref_decay_t<decltype(hDst)>;
                    FieldTypes srcFieldTypes{hSrc};
                    // Check that the type of the source field is Hodge dual to the destination
                    if constexpr (std::is_same_v<
                                      dstFieldTypename,
                                      DeRhamField<oppositeGrid<srcFieldTypes.s_gridType>,
                                                  oppositeSpace<srcFieldTypes.s_spaceType>>&>)
                    {
                        this->hodge_operator<validDegreesArray[testDegInd], hodgeScheme>(hSrc, hDst,
                                                                                         weight);
                    }
                    else
                    {
                        std::string str =
                            "hodgeDegreeSelector must be called with field types that are deRham "
                            "duals of each other.";
                        amrex::Assert("Incompatible Hodge operator fields",
                                      "GEMPIC_FDDeRhamComplex.H", __LINE__, str.c_str());
                    }
                },
                hodgeSrc, hodgeDst);
        }
        else
        {
            hodge_degree_selector<hodgeScheme, testDegInd + 1>(hodgeSrc, hodgeDst, weight);
        }
    }
    else
    {
        std::string str =
            "No such degree " + std::to_string(m_hodgeDegree) + " for selected Hodge scheme";
        amrex::Assert("Incompatible Hodge degree and Hodge scheme", "GEMPIC_FDDeRhamComplex.H",
                      __LINE__, str.c_str());
    }
}

template <Direction dir, int testDegInd>
void FDDeRhamComplex::eval_form_degree_selector (
    anyFieldConstRef field,
    amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> const& evalShift,
    amrex::MultiFab& pointValues)
{
    BL_PROFILE("Gempic::Forms::FDDeRhamComplex::eval_form_degree_selector()")
    constexpr auto validDegreesArray = valid_degrees<HodgeScheme::FDHodge>();
    if constexpr (testDegInd < validDegreesArray.size())
    {
        if (this->m_hodgeDegree == validDegreesArray[testDegInd])
        {
            // Iterate over all types in anyFieldConstRef (all grid/space combinations) and select
            // the correct one.
            std::visit(
                [this, &validDegreesArray, &evalShift, &pointValues] (auto const& fieldArg)
                {
                    using namespace Impl;
                    FieldTypes fieldTypes{fieldArg};
                    constexpr Grid grid = fieldTypes.s_gridType;
                    constexpr Space space = fieldTypes.s_spaceType;
                    constexpr int degree = validDegreesArray[testDegInd];
                    // 0-/3-form call
                    if constexpr (fieldTypes.s_spaceType == Space::node ||
                                  fieldTypes.s_spaceType == Space::cell)
                    {
                        amrex::MultiFab const& data = fieldArg.get().m_data;
                        eval_form_component<degree, grid, space>(evalShift, data, pointValues,
                                                                 this->m_dr, this->m_geom);
                    }
                    // 1-/2-form call
                    else
                    {
                        amrex::MultiFab const& data = fieldArg.get().m_data[dir];
                        eval_form_component<degree, grid, space, dir>(evalShift, data, pointValues,
                                                                      this->m_dr, this->m_geom);
                    }
                },
                field);
        }
        else
        {
            eval_form_degree_selector<dir, testDegInd + 1>(field, evalShift, pointValues);
        }
    }
    else
    {
        std::string str = "No such degree " + std::to_string(m_hodgeDegree) + " for eval_form";
        amrex::Assert("Incompatible Hodge degree and eval_form", "GEMPIC_FDDeRhamComplex.H",
                      __LINE__, str.c_str());
    }
}
} // namespace Gempic::Forms

#endif

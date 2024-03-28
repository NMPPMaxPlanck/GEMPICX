#ifndef GEMPIC_FIELDS_H_
#define GEMPIC_FIELDS_H_

#include <memory>
#include <type_traits>
#include <variant>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Forms
{
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

// // map of pointers to the field multifabs for the diagnostics
// static std::map<std::string, amrex::MultiFab*> fieldsDiagnostics;
// // map of scaling parameters to go from degrees of freedom to point values for diagnostics
// static std::map<std::string, amrex::Real> fieldsScaling;

// Helpful for enable_if magic. @todo: If we switch to C++20, we should use concepts, as shown in
// https://gitlab.mpcdf.mpg.de/gempic/gempic/-/merge_requests/127#note_225990
template <Space space>
inline constexpr bool isZeroOrThreeForm{false};

template <>
inline constexpr bool isZeroOrThreeForm<Space::node>{true};

template <>
inline constexpr bool isZeroOrThreeForm<Space::cell>{true};

template <Space space>
inline constexpr bool isOneOrTwoForm{!isZeroOrThreeForm<space>};

class DeRhamComplex;

/** DeRhamField
 *
 *
 * Stores a specific element of a deRham complex
 *
 * @p gridType: Grid::primal or Grid::dual grid
 *
 * @p spaceType: Space::node, Space::edge, Space::face or Space::cell.
 *
 * spaceType is defined on primal grid:
 * e.g. DeRhamField<Grid::dual, node> is the cell based complex dual to DeRhamField<Grid::primal,
 * node>
 *
 * In 1D only Space::node and Space::cell exist.
 * In 2D there are 4 complexes because of the two different deRham sequences
 *
 * For node and cell only scalarData will be allocated, for edge and face only vector data
 */
// General instance. Will never be called. Here for documentation
template <Grid grid, Space space, typename enable = void>
struct DeRhamField
{
    std::shared_ptr<DeRhamComplex> m_deRham;
    /**
     * @brief actual field information.
     * Multifab for Space::node or Space::cell,
     * Array of Multifabs for Space::edge or Space::face,
     */
    amrex::Array<amrex::MultiFab, 3> m_data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    {
        static_assert(space != space, "This class should not be callable.");
    };
    /**
     * @brief Construct Space::edge/Space::face fields and add name to fieldsDiagnostics map
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const std::string& diagName) :
        DeRhamField{drc} {};
    /**
     * @brief Construct Space::edge / Space::face fields from function array
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func) :
        DeRhamField{drc} {};

    /**
     * @brief Construct Space::node / Space::cell fields from function
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func) :
        DeRhamField{drc} {};

    void fill_boundary();
    void average_sync();

    /**
     * @brief For Space::node / Space::cell fields, perform all of the synchronization operations
     * necessary after cpu and gpu particle loops
     */
    void post_particle_loop_sync();
};

// One- or Two-form, i.e., space is either edge or face
template <Grid grid, Space space>
struct DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>
{
    std::shared_ptr<DeRhamComplex> m_deRham;
    amrex::Array<amrex::MultiFab, 3> m_data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const std::string& diagName);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                const std::string& diagName);

    void fill_boundary();
    void average_sync();

    void operator*=(const amrex::Real& dt);
    void operator+=(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
};

// Zero- or Three-form, i.e., space is either node or cell
template <Grid grid, Space space>
struct DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>
{
    std::shared_ptr<DeRhamComplex> m_deRham;
    amrex::MultiFab m_data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const std::string& diagName);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func,
                const std::string& diagName);

    void fill_boundary();
    void average_sync();

    void post_particle_loop_sync();
    void operator*=(const amrex::Real& dt);
};

// Helper variants
using anyFieldRef = std::variant<std::reference_wrapper<DeRhamField<Grid::primal, Space::node>>,
                                 std::reference_wrapper<DeRhamField<Grid::primal, Space::edge>>,
                                 std::reference_wrapper<DeRhamField<Grid::primal, Space::face>>,
                                 std::reference_wrapper<DeRhamField<Grid::primal, Space::cell>>,
                                 std::reference_wrapper<DeRhamField<Grid::dual, Space::node>>,
                                 std::reference_wrapper<DeRhamField<Grid::dual, Space::edge>>,
                                 std::reference_wrapper<DeRhamField<Grid::dual, Space::face>>,
                                 std::reference_wrapper<DeRhamField<Grid::dual, Space::cell>>>;
using anyFieldConstRef =
    std::variant<std::reference_wrapper<const DeRhamField<Grid::primal, Space::node>>,
                 std::reference_wrapper<const DeRhamField<Grid::primal, Space::edge>>,
                 std::reference_wrapper<const DeRhamField<Grid::primal, Space::face>>,
                 std::reference_wrapper<const DeRhamField<Grid::primal, Space::cell>>,
                 std::reference_wrapper<const DeRhamField<Grid::dual, Space::node>>,
                 std::reference_wrapper<const DeRhamField<Grid::dual, Space::edge>>,
                 std::reference_wrapper<const DeRhamField<Grid::dual, Space::face>>,
                 std::reference_wrapper<const DeRhamField<Grid::dual, Space::cell>>>;

/**
 * @brief Base class for discrete de Rham Complexes. The member functions with the `inline`
 * specifier implement the methods of the elements of the Complex (the functions of the fields). The
 * projections are virtual functions, whose details are implemented in the inherited subclasses.
 *
 */
class DeRhamComplex
{
private:
    // Indices
    amrex::IndexType m_primalNodeIndex;
    amrex::Array<amrex::IndexType, 3> m_primalEdgeIndex;
    amrex::Array<amrex::IndexType, 3> m_primalFaceIndex;
    amrex::IndexType m_primalCellIndex;

protected:
    static constexpr std::array<int, 0> s_validDegreesNoScheme{};
    // Computational grid
    amrex::BoxArray m_grid;
    amrex::DistributionMapping m_distriMap;
    amrex::Geometry m_geom;
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> m_dxInv;
    int m_hodgeDegree;

    // Number of ghost cells along each direction
    amrex::IntVect m_nGhost;

public:
    /// Constructor for the DeRhamComplex class
    /// @param infra The computational domain needed to construct the class correctly
    /// @param hodgeDegree The degree of the Hodge scheme, used for a lower limit of ghost cells
    /// @param maxSplineDegree The degree of the splines, used for a lower limit of ghost cells
    DeRhamComplex(const ComputationalDomain& infra,
                  const int hodgeDegree,
                  const int maxSplineDegree)
    {
        Io::Parameters params("DeRhamComplex");

        m_grid = infra.m_grid;
        m_distriMap = infra.m_distriMap;
        m_geom = infra.m_geom;
        m_hodgeDegree = hodgeDegree;
        m_dxInv = infra.m_dxi;

        /* Ghost cells are decided for the fields by the degree of the Hodge.
        ** Needs to be changed for particles and decided by spline degree:
        ** to max(degx, degy, degz) + safety margin to let particles move
        **
        ** Example: for linear splines (deg_i = 1) 1 ghost cell is needed.
        ** 3 more ghost cells are added to let particles move.
        ** */
        // buffer ghost cells
        int nGhostExtra{3};
        params.get_or_set("nGhostExtra", nGhostExtra);
        int nGhostNum{std::max(m_hodgeDegree / 2, maxSplineDegree) + nGhostExtra};
        m_nGhost = amrex::IntVect{AMREX_D_DECL(nGhostNum, nGhostNum, nGhostNum)};
        std::vector<int> nGhostVector{AMREX_D_DECL(m_nGhost[xDir], m_nGhost[yDir], m_nGhost[zDir])};
        params.set("nGhost", nGhostVector);
        // Check that user isn't expecting "nGhost" to have an effect
        if (params.is_in_input_file("nGhost"))
        {
            amrex::ParmParse pp;
            std::vector<int> nGhostVectorInputFile;
            pp.getarr("nGhost", nGhostVectorInputFile);
            if (AMREX_D_TERM(nGhostVector[xDir] != nGhostVectorInputFile[xDir],
                             || nGhostVector[yDir] != nGhostVectorInputFile[yDir],
                             || nGhostVector[zDir] != nGhostVectorInputFile[zDir]))
            {
                std::cerr << "Warning: Number of ghost cells can only be influenced by specifying "
                             "the number of extra ghost cells.\n";
            }
        }

        // set index: // 0 -> cell-centered | 1 -> node-centered
        m_primalNodeIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
        m_primalEdgeIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)});
        m_primalEdgeIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)});
        m_primalEdgeIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});
        m_primalFaceIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)});
        m_primalFaceIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)});
        m_primalFaceIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)});
        m_primalCellIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)});
    }

    virtual ~DeRhamComplex() {}

    /// Getter for the number of ghost cells
    const amrex::IntVect& get_n_ghost () const { return m_nGhost; }

    /// Getter for the cell size
    const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1>& get_dx_inv () const { return m_dxInv; }

    const amrex::Periodicity get_periodicity () const { return m_geom.periodicity(); }

    // map of pointers to the field multifabs for the diagnostics
    std::map<std::string, amrex::MultiFab*> m_fieldsDiagnostics;

    // map of scaling parameters to go from degrees of freedom to point values for diagnostics
    std::map<std::string, amrex::Real> m_fieldsScaling;

    void initialize_field (DeRhamField<Grid::primal, Space::node>& field)
    {
        field.m_data.define(amrex::convert(m_grid, m_primalNodeIndex), m_distriMap, 1, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Function overloads for the initializations
    void initialize_field (DeRhamField<Grid::primal, Space::edge>& field)
    {
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::face>& field)
    {
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::cell>& field)
    {
        field.m_data.define(amrex::convert(m_grid, m_primalCellIndex), m_distriMap, 1, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Centering is always set by the primal grid
    void initialize_field (DeRhamField<Grid::dual, Space::node>& field)
    {
        field.m_data.define(amrex::convert(m_grid, m_primalCellIndex), m_distriMap, 1, m_nGhost);
        field.m_data.setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::edge>& field)
    {
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::face>& field)
    {
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap, 1,
                                  m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::cell>& field)
    {
        field.m_data.define(amrex::convert(m_grid, m_primalNodeIndex), m_distriMap, 1, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Function overloads for fill boundary
    // fillBoundary for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void fill_boundary (DeRhamField<grid, space>& field)
    {
        field.m_data[xDir].FillBoundary(m_geom.periodicity());
        field.m_data[yDir].FillBoundary(m_geom.periodicity());
        field.m_data[zDir].FillBoundary(m_geom.periodicity());
    }

    // fillBoundary for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void fill_boundary (DeRhamField<grid, space>& field)
    {
        field.m_data.FillBoundary(m_geom.periodicity());
    }

    // Function overloads for averagesync
    // averageSync for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void average_sync (DeRhamField<grid, space>& field)
    {
        field.m_data[xDir].AverageSync(m_geom.periodicity());
        field.m_data[yDir].AverageSync(m_geom.periodicity());
        field.m_data[zDir].AverageSync(m_geom.periodicity());
    }

    // averageSync for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void average_sync (DeRhamField<grid, space>& field)
    {
        field.m_data.AverageSync(m_geom.periodicity());
    }

    //---------------------
    // exterior derivatives
    //---------------------
    void grad(const DeRhamField<Grid::primal, Space::node>&,
              DeRhamField<Grid::primal, Space::edge>&);
    void curl(const DeRhamField<Grid::primal, Space::edge>&,
              DeRhamField<Grid::primal, Space::face>&);
    void div(const DeRhamField<Grid::primal, Space::face>&,
             DeRhamField<Grid::primal, Space::cell>&);
    void grad(const DeRhamField<Grid::dual, Space::node>&, DeRhamField<Grid::dual, Space::edge>&);
    void curl(const DeRhamField<Grid::dual, Space::edge>&, DeRhamField<Grid::dual, Space::face>&);
    void div(const DeRhamField<Grid::dual, Space::face>&, DeRhamField<Grid::dual, Space::cell>&);

    //------------
    // Projections
    //------------
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::node>&) = 0;
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::cell>&,
                            int gaussNodes = 6) = 0;
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::node>&) = 0;
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::cell>&,
                            int gaussNodes = 6) = 0;

    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::edge>&,
                            int gaussNodes = 6) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::face>&,
                            int gaussNodes = 6) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::edge>&,
                            int gaussNodes = 6) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::face>&,
                            int gaussNodes = 6) = 0;

    virtual void hodge(const DeRhamField<Grid::dual, Space::cell>& threeForm,
                       DeRhamField<Grid::primal, Space::node>& zeroForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::primal, Space::cell>& threeForm,
                       DeRhamField<Grid::dual, Space::node>& zeroForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::dual, Space::face>& twoForm,
                       DeRhamField<Grid::primal, Space::edge>& oneForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::primal, Space::face>& twoForm,
                       DeRhamField<Grid::dual, Space::edge>& oneForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::dual, Space::edge>& oneForm,
                       DeRhamField<Grid::primal, Space::face>& twoForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::primal, Space::edge>& oneForm,
                       DeRhamField<Grid::dual, Space::face>& twoForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::dual, Space::node>& zeroForm,
                       DeRhamField<Grid::primal, Space::cell>& threeForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(const DeRhamField<Grid::primal, Space::node>& zeroForm,
                       DeRhamField<Grid::dual, Space::cell>& threeForm,
                       amrex::Real weight = 1.) = 0;
};

// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc) :
    m_deRham{drc}
{
    m_deRham->initialize_field(*this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc) :
    m_deRham{drc}
{
    m_deRham->initialize_field(*this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, const std::string& diagName) :
    DeRhamField{drc}
{
    m_deRham->m_fieldsDiagnostics[diagName + "x"] = &(this->m_data[xDir]);
    m_deRham->m_fieldsDiagnostics[diagName + "y"] = &(this->m_data[yDir]);
    m_deRham->m_fieldsDiagnostics[diagName + "z"] = &(this->m_data[zDir]);

    if constexpr (space == Space::edge)  // edge field
    {
        auto dxInv{m_deRham->get_dx_inv()};
        m_deRham->m_fieldsScaling[diagName + "x"] = dxInv[xDir];
        m_deRham->m_fieldsScaling[diagName + "y"] = GEMPIC_D_MULT(1.0, dxInv[yDir], 1.0);
        m_deRham->m_fieldsScaling[diagName + "z"] = GEMPIC_D_MULT(1.0, 1.0, dxInv[zDir]);
    }
    if constexpr (space == Space::face)  // face field
    {
        auto dxInv{m_deRham->get_dx_inv()};
        m_deRham->m_fieldsScaling[diagName + "x"] = GEMPIC_D_MULT(1.0, dxInv[yDir], dxInv[zDir]);
        m_deRham->m_fieldsScaling[diagName + "y"] = GEMPIC_D_MULT(dxInv[xDir], 1.0, dxInv[zDir]);
        m_deRham->m_fieldsScaling[diagName + "z"] = GEMPIC_D_MULT(dxInv[xDir], dxInv[yDir], 1.0);
    }
};

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, const std::string& diagName) :
    DeRhamField{drc}
{
    m_deRham->m_fieldsDiagnostics[diagName] = &(this->m_data);
    if constexpr (space == Space::node)  // node field
    {
        m_deRham->m_fieldsScaling[diagName] = 1.0;
    }
    if constexpr (space == Space::cell)  // cell field
    {
        m_deRham->m_fieldsScaling[diagName] = m_deRham->get_dx_inv()[GEMPIC_SPACEDIM];
    }
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func) :
    DeRhamField{drc}
{
    m_deRham->projection(func, 0.0, *this);
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func) :
    DeRhamField{drc}
{
    m_deRham->projection(func, 0.0, *this);
}

/// @cond DOXYGEN_IGNORE
template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
    const std::string& diagName) :
    DeRhamField{drc, diagName}
{
    m_deRham->projection(func, 0.0, *this);
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func,
    const std::string& diagName) :
    DeRhamField{drc, diagName}
{
    m_deRham->projection(func, 0.0, *this);
}

// Synchronization necessary after cpu and gpu particle loops
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::
    post_particle_loop_sync()
{
    m_data.SumBoundary(0, 1, m_deRham->get_n_ghost(), amrex::IntVect::Zero,
                       m_deRham->get_periodicity());
    average_sync();
    fill_boundary();
}

// FillBoundary of DeRhamFields following the same logic as above
template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::fill_boundary()
{
    m_deRham->fill_boundary(*this);
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::fill_boundary()
{
    m_deRham->fill_boundary(*this);
}

// AverageSync of DeRhamFields
template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::average_sync()
{
    m_deRham->average_sync(*this);
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::average_sync()
{
    m_deRham->average_sync(*this);
}

// Multiply by scalar
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator*=(
    const amrex::Real& dt)
{
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].mult(dt, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator*=(
    const amrex::Real& dt)
{
    auto nGhost = m_deRham->get_n_ghost();
    m_data.mult(dt, nGhost[xDir]);
}

// plusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator+=(
    DeRhamField<grid, space>& form)
{
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(form.m_data[comp], 0, 1, nGhost[xDir]);
    }
}

// minusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator-=(
    DeRhamField<grid, space>& form)
{
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].minus(form.m_data[comp], 0, 1, nGhost[xDir]);
    }
}
///@endcond

}  // namespace Gempic::Forms

namespace Gempic::Forms::Impl
{
using namespace Forms;
template <class T>
struct UnwrapReference
{
    using type = T;
};
template <class U>
struct UnwrapReference<std::reference_wrapper<U>>
{
    using type = U&;
};

template <class T>
struct UnwrapRefDecay : UnwrapReference<std::decay_t<T>>
{
};

template <class T>
using unwrap_reference_t = typename UnwrapReference<T>::type;

template <class T>
using unwrap_ref_decay_t = typename UnwrapRefDecay<T>::type;

template <Grid grid, Space space>
struct FieldTypes
{
    static constexpr Grid s_gridType{grid};
    static constexpr Space s_spaceType{space};
    FieldTypes(const DeRhamField<grid, space>&){};
    FieldTypes(std::reference_wrapper<const DeRhamField<grid, space>>){};
};

template <Grid grid>
constexpr Grid get_opposite_grid ()
{
    if constexpr (grid == Grid::primal)
    {
        return Grid::dual;
    }
    else if constexpr (grid == Grid::dual)
    {
        return Grid::primal;
    }
}

template <Space space>
constexpr Space get_opposite_space ()
{
    if constexpr (space == Space::node)
    {
        return Space::cell;
    }
    else if constexpr (space == Space::cell)
    {
        return Space::node;
    }
    else if constexpr (space == Space::face)
    {
        return Space::edge;
    }
    else if constexpr (space == Space::edge)
    {
        return Space::face;
    }
}

template <Grid grid>
constexpr Grid oppositeGrid{get_opposite_grid<grid>()};

template <Space space>
constexpr Space oppositeSpace{get_opposite_space<space>()};
}  // namespace Gempic::Forms::Impl

#endif

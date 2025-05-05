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

#include "GEMPIC_BoundaryConditions.H"
#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_NumTools.H"
#include "GEMPIC_Parameters.H"

namespace Gempic::Forms
{

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
template <Grid grid, Space space, typename Enable = void>
struct DeRhamField
{
    std::shared_ptr<DeRhamComplex> m_deRham;
    /**
     * @brief actual field information.
     * Multifab of nVar variables for Space::node or Space::cell,
     * Array of three Multifabs of nVar variables each for Space::edge or Space::face
     */
    amrex::Array<amrex::MultiFab, 3> m_data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar = 1)
    {
        static_assert(space != space, "This class should not be callable.");
    };

    /**
     * @brief Construct Space::edge / Space::face fields with a 3D array of number of components
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, std::array<int, 3> ncomponents) :
        DeRhamField{drc} {};
    /**
     * @brief Construct Space::edge / Space::face fields and add name to fieldsDiagnostics map
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const std::string& diagName, int nVar = 1) :
        DeRhamField{drc} {};
    /**
     * @brief Construct Space::edge / Space::face fields of one variable from function array
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func) :
        DeRhamField{drc} {};
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                const std::string& diagName) :
        DeRhamField{drc} {};
    /**
     * @brief Construct Space::edge / Space::face fields of nVar variables from vector of nVar
     *        function arrays
     */
    DeRhamField(
        std::shared_ptr<DeRhamComplex> drc,
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs) :
        DeRhamField{drc} {};
    DeRhamField(
        std::shared_ptr<DeRhamComplex> drc,
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
        const std::string& diagName) :
        DeRhamField{drc} {};

    /**
     * @brief Construct Space::node / Space::cell fields of one variable from function
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func) :
        DeRhamField{drc} {};
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                const std::string& diagName) :
        DeRhamField{drc} {};
    /**
     * @brief Construct Space::node / Space::cell fields of nVar variables from vector of nVar
     *        functions
     */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs) :
        DeRhamField{drc} {};
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
                const std::string& diagName) :
        DeRhamField{drc} {};

    void fill_boundary();
    void fill_boundary_nowait();
    void fill_boundary_finish();
    void apply_bc();

    /**
     * @brief Perform all of the synchronization operations necessary after cpu and gpu particle
     * loops: Sum boundary, then Fill Boundary
     */
    void post_particle_loop_sync();
};
// Helper functions for field types
template <Grid grid, Space space>
struct FieldTypes
{
    static constexpr Grid s_gridType{grid};
    static constexpr Space s_spaceType{space};
    FieldTypes(const DeRhamField<grid, space>&) {};
    FieldTypes(std::reference_wrapper<const DeRhamField<grid, space>>) {};
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

// One- or Two-form, i.e., space is either edge or face
template <Grid grid, Space space>
struct DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>
{
    std::shared_ptr<DeRhamComplex> m_deRham;
    amrex::Array<amrex::MultiFab, 3> m_data;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>
        m_bcFuncX;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>
        m_bcFuncY;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>
        m_bcFuncZ;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar = 1);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, std::array<int, 3> ncomponents);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const std::string& diagName, int nVar = 1);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func);
    DeRhamField(
        std::shared_ptr<DeRhamComplex> drc,
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                const std::string& diagName);
    DeRhamField(
        std::shared_ptr<DeRhamComplex> drc,
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
        const std::string& diagName);

    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::MultiFab& mfIn1,
                const amrex::MultiFab& mfIn2,
                const amrex::MultiFab& mfIn3);

    void fill_boundary();
    void fill_boundary_nowait();
    void fill_boundary_finish();
    void apply_bc();

    void post_particle_loop_sync();
    void operator*=(amrex::Real dt);
    void operator*=(amrex::ParserExecutor<AMREX_SPACEDIM + 1> func);
    void operator+=(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
    DeRhamField<grid, space> operator-(DeRhamField<grid, space>&);
    void operator+=(amrex::Real c);
    void operator-=(amrex::Real c);
};

// Zero- or Three-form, i.e., space is either node or cell
template <Grid grid, Space space>
struct DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>
{
    std::shared_ptr<DeRhamComplex> m_deRham;
    amrex::MultiFab m_data;
    std::unique_ptr<amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>
        m_bcFunc;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc, int nVar = 1);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const std::string& diagName, int nVar = 1);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                const std::string& diagName);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc,
                const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
                const std::string& diagName);

    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::MultiFab& mfIn);

    void fill_boundary();
    void fill_boundary_nowait();
    void fill_boundary_finish();
    void apply_bc();

    void post_particle_loop_sync();
    void operator*=(amrex::Real dt);
    void operator*=(amrex::ParserExecutor<AMREX_SPACEDIM + 1> func);
    void operator+=(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
    DeRhamField<grid, space> operator-(DeRhamField<grid, space>&);
    void operator+=(amrex::Real c);
    void operator-=(amrex::Real c);
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
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dx;
    int m_hodgeDegree;

    // Number of ghost cells along each direction
    amrex::IntVect m_nGhost;
    amrex::IntVect m_nGhostRequired;

public:
    // ToDo: Temporary moved to be public until ComputationalDomain is removed
    amrex::Geometry m_geom;
    /// Constructor for the DeRhamComplex class
    /// @param infra The computational domain needed to construct the class correctly
    /// @param hodgeDegree The degree of the Hodge scheme, used for a lower limit of ghost cells
    /// @param maxSplineDegree The degree of the splines, used for a lower limit of ghost cells
    DeRhamComplex(const ComputationalDomain& infra,
                  const int hodgeDegree,
                  const int maxSplineDegree)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex:: DeRhamComplex()");
        Io::Parameters params("DeRhamComplex");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(infra.geometry().Ok(), "Uninitialized geometry");

        m_grid = infra.m_grid;
        m_distriMap = infra.m_distriMap;
        m_geom = infra.m_geom;
        m_hodgeDegree = hodgeDegree;
        m_dx = infra.geometry().CellSizeArray();

        /* Number of essential ghost cells is determined by degree of the Hodge and particle
        ** splines:
        **
        ** - Hodges: hodgeDegree / 2 - 1
        ** - Particle-Mesh-Coupling: ceil((max(degx, degy, degz) + 1) / 2)
        **   (Assuming that the particle is located inside the domain)
        ** - External Derivatives: 1 ghost cell for some dual operators, e.g., curl
        **
        ** The maximum over essential ghost cells is taken and nGhostExtra are added to allow for
        ** more particle movement.
        **
        ** Example:
        ** Tridiagonal Hodge (deg = 4) -> 1 ghost cell
        ** Linear splines (deg = 1)    -> 1 ghost cell
        ** => 1 + 3 extra = 4 ghost cells get initialized.
        ** */
        int nGhostRequired{std::max(
            std::max(m_hodgeDegree / 2 - 1, static_cast<int>(std::ceil((maxSplineDegree + 1) / 2))),
            1)};
        m_nGhostRequired =
            amrex::IntVect{AMREX_D_DECL(nGhostRequired, nGhostRequired, nGhostRequired)};
        int nGhostExtra{3};
        params.get_or_set("nGhostExtra", nGhostExtra);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nGhostExtra >= 0,
                                         "Number of extra ghost cells cannot be negative.");
        int nGhost{nGhostRequired + nGhostExtra};
        std::vector<int> nGhostVector{AMREX_D_DECL(nGhost, nGhost, nGhost)};
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
                amrex::Warning(
                    "Number of ghost cells should only be influenced by specifying the number of "
                    "extra ghost cells.\n");
            }
        }
        params.get_or_set("nGhost", nGhostVector);
        if (AMREX_D_TERM(nGhostVector[xDir] < m_nGhostRequired[xDir],
                         || nGhostVector[yDir] < m_nGhostRequired[yDir],
                         || nGhostVector[zDir] < m_nGhostRequired[zDir]))
        {
            std::string msg{"nGhost must be larger than " + std::to_string(nGhostRequired) +
                            " to accomodate spline and/or Hodge operations"};
            amrex::Assert(msg.c_str(), __FILE__, __LINE__);
        }
        m_nGhost = amrex::IntVect{
            AMREX_D_DECL(nGhostVector[xDir], nGhostVector[yDir], nGhostVector[zDir])};

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

    /// Getter for the periodicity
    const amrex::Periodicity get_periodicity () const { return m_geom.periodicity(); }

    /// Getter for prob lo
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> get_prob_lo () const
    {
        return m_geom.ProbLoArray();
    }

    /// Getter for geom
    const amrex::Geometry get_geom () const { return m_geom; }

    /// Getter for hodgeDegree
    int get_hodge_degree () const { return m_hodgeDegree; }

    // map of pointers to the field multifabs for the diagnostics
    std::map<std::string, const amrex::MultiFab> m_fieldsDiagnostics;

    // map of scaling parameters to go from degrees of freedom to point values for diagnostics
    std::map<std::string, const amrex::Real> m_fieldsScaling;

    /// @brief Initializes fields and ties them to the grid.
    ///        Called from the DeRhamField constructors.
    /// @param field The field to be initialized
    /// @param nVar  The number of variables of the field
    ///              (times one component for 0-/3-forms, times three components for 1-/2-forms)
    void initialize_field (DeRhamField<Grid::primal, Space::node>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,node,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalNodeIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Function overloads for the initializations
    void initialize_field (DeRhamField<Grid::primal, Space::edge>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::face>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::edge>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::face>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::primal, Space::cell>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(primal,cell,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalCellIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    // Centering is always set by the primal grid
    void initialize_field (DeRhamField<Grid::dual, Space::node>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,node,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalCellIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::edge>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::face>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  nVar, m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::edge>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,edge,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalFaceIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalFaceIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalFaceIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::face>& field,
                           std::array<int, 3> ncomponents)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,face,field)");
        field.m_data[xDir].define(amrex::convert(m_grid, m_primalEdgeIndex[xDir]), m_distriMap,
                                  ncomponents[xDir], m_nGhost);
        field.m_data[xDir].setVal(0.0);
        field.m_data[yDir].define(amrex::convert(m_grid, m_primalEdgeIndex[yDir]), m_distriMap,
                                  ncomponents[yDir], m_nGhost);
        field.m_data[yDir].setVal(0.0);
        field.m_data[zDir].define(amrex::convert(m_grid, m_primalEdgeIndex[zDir]), m_distriMap,
                                  ncomponents[zDir], m_nGhost);
        field.m_data[zDir].setVal(0.0);
    }

    void initialize_field (DeRhamField<Grid::dual, Space::cell>& field, int nVar = 1)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex::initialize_field(dual,cell,field)");
        field.m_data.define(amrex::convert(m_grid, m_primalNodeIndex), m_distriMap, nVar, m_nGhost);
        field.m_data.setVal(0.0);
    }

    /// Function overloads for FillBoundary. Exists for 1-/2-forms and 0-/3-forms
    // fillBoundary for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void fill_boundary (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::fill_boundary(field)");
        field.m_data[xDir].FillBoundary(m_geom.periodicity());
        field.m_data[yDir].FillBoundary(m_geom.periodicity());
        field.m_data[zDir].FillBoundary(m_geom.periodicity());
    }

    /// @cond DOXYGEN_IGNORE
    // fillBoundary for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void fill_boundary (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::fill_boundary(field)");
        field.m_data.FillBoundary(m_geom.periodicity());
    }
    /// @endcond

    /// Function overloads for FillBoundary_nowait. Exists for 1-/2-forms and 0-/3-forms
    // fillBoundary_nowait for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void fill_boundary_nowait (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::fill_boundary_nowait(field)");
        field.m_data[xDir].FillBoundary_nowait(m_geom.periodicity());
        field.m_data[yDir].FillBoundary_nowait(m_geom.periodicity());
        field.m_data[zDir].FillBoundary_nowait(m_geom.periodicity());
    }

    /// @cond DOXYGEN_IGNORE
    // fillBoundary_nowait for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void fill_boundary_nowait (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::fill_boundary_nowait(field)");
        field.m_data.FillBoundary_nowait(m_geom.periodicity());
    }
    /// @endcond
    /// Function overloads for FillBoundary_nowait. Exists for 1-/2-forms and 0-/3-forms
    // fillBoundary_nowait for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void fill_boundary_finish (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::fill_boundary_finish()");
        field.m_data[xDir].FillBoundary_finish();
        field.m_data[yDir].FillBoundary_finish();
        field.m_data[zDir].FillBoundary_finish();
    }

    /// @cond DOXYGEN_IGNORE
    // fillBoundary_nowait for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void fill_boundary_finish (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::fill_boundary_finish()");
        field.m_data.FillBoundary_finish();
    }
    /// @endcond

    /**
     * @brief Apply boundary conditions and extrapolate to ghost cells. Exists for 1-/2-forms and
     * 0-/3-forms.
     *
     * @details
     * If at least one dimension is non-periodic, primal degrees of freedom on the corresponding
     * boundaries are set to the condition specified in the input file. A polynomial extrapolation
     * of order (Hodge degree -1) is then performed to fill ghost cells. For dual forms, only the
     * extrapolation is applied.
     *
     * @param field Discrete form
     */
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void apply_bc (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isOneOrTwoForm>::apply_bc(field)");
        if (not m_geom.isAllPeriodic())
        {
            // call bcFunc with MultiFab& mf, int icomp, int ncomp, IntVect const& nghost, Real
            // time, int bccomp
            field.m_bcFuncX->operator()(field.m_data[xDir], 0, 1, m_nGhostRequired, 0., 0);
            field.m_bcFuncY->operator()(field.m_data[yDir], 0, 1, m_nGhostRequired, 0., 0);
            field.m_bcFuncZ->operator()(field.m_data[zDir], 0, 1, m_nGhostRequired, 0., 0);

            field.fill_boundary();
        }
    }

    /// @cond DOXYGEN_IGNORE
    // applyBC for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void apply_bc (DeRhamField<grid, space>& field)
    {
        BL_PROFILE("Gempic::Forms::DeRhamComplex<isZeroOrThreeForm>::apply_bc(field)");
        if (not m_geom.isAllPeriodic())
        {
            field.m_bcFunc->operator()(field.m_data, 0, 1, m_nGhostRequired, 0., 0);

            field.fill_boundary();
        }
    }
    /// @endcond

    //---------------------
    // exterior derivatives
    //---------------------
    void grad(DeRhamField<Grid::primal, Space::edge>&,
              const DeRhamField<Grid::primal, Space::node>&);
    void curl(DeRhamField<Grid::primal, Space::face>&,
              const DeRhamField<Grid::primal, Space::edge>&);
    void div(DeRhamField<Grid::primal, Space::cell>&,
             const DeRhamField<Grid::primal, Space::face>&);
    void grad(DeRhamField<Grid::dual, Space::edge>&, const DeRhamField<Grid::dual, Space::node>&);
    void curl(DeRhamField<Grid::dual, Space::face>&, const DeRhamField<Grid::dual, Space::edge>&);
    void div(DeRhamField<Grid::dual, Space::cell>&, const DeRhamField<Grid::dual, Space::face>&);

    void a_times_grad(DeRhamField<Grid::primal, Space::edge>&,
                      const DeRhamField<Grid::primal, Space::node>&,
                      amrex::Real);
    void a_times_grad(DeRhamField<Grid::dual, Space::edge>&,
                      const DeRhamField<Grid::dual, Space::node>&,
                      amrex::Real);
    void add_dt_curl(DeRhamField<Grid::primal, Space::face>&,
                     const DeRhamField<Grid::primal, Space::edge>&,
                     amrex::Real dt);
    void add_dt_curl(DeRhamField<Grid::dual, Space::face>&,
                     const DeRhamField<Grid::dual, Space::edge>&,
                     amrex::Real dt);

    //------------
    // Projections
    //------------
    // 0-/3-forms
    /**
     * @brief Projection from an AMReX compiled function to a 0-/3-form DeRhamField of one variable
     * @param func        The function to be used
     * @param t           The time step at which to evaluate the function
     * @param field       The field to project to
     * @param gaussNodes  The number of gaussNodes for the interpolation (n/a for 0-forms)
     */
    virtual void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::cell>& field,
                            int gaussNodes = 6) = 0;
    /**
     * @brief Projection from a vector of nVar AMReX compiled functions to a 0-/3-form
     *        DeRhamField of nVar variables
     * @param funcs       Vector of the functions to be used
     * @param t           The time step at which to evaluate the functions
     * @param field       The field to project to
     * @param gaussNodes  The number of gaussNodes for the interpolation
     */
    virtual void projection(const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::cell>& field,
                            int gaussNodes = 6) = 0;

    virtual void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::node>& field) = 0;
    virtual void projection(const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::node>& field) = 0;

    virtual void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::node>& field) = 0;
    virtual void projection(const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::node>& field) = 0;

    virtual void projection(amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::cell>& field,
                            int gaussNodes = 6) = 0;
    virtual void projection(const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::cell>& field,
                            int gaussNodes = 6) = 0;

    // 1-/2-forms
    /**
     * @brief Projection from a array of 3 AMReX compiled functions to a 1-/2-form DeRhamField of
     *        one variable (and three components)
     * @param func        Array of the functions to be used
     * @param t           The time step at which to evaluate the functions
     * @param field       The field to project to
     * @param gaussNodes  The number of gaussNodes for the interpolation
     */
    virtual void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::edge>& field,
                            int gaussNodes = 6) = 0;
    /**
     * @brief Projection from a vector of nVar 3-arrays of AMReX compiled functions each to a 1-/2-
     *        form DeRhamField of nVar variables (and three components)
     * @param funcs       Vector of arrays of the functions to be used
     * @param t           The time step at which to evaluate the functions
     * @param field       The field to project to
     * @param gaussNodes  The number of gaussNodes for the interpolation
     */
    virtual void projection(
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
        amrex::Real t,
        DeRhamField<Grid::dual, Space::edge>& field,
        int gaussNodes = 6) = 0;

    virtual void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::face>& field,
                            int gaussNodes = 6) = 0;
    virtual void projection(
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
        amrex::Real t,
        DeRhamField<Grid::primal, Space::face>& field,
        int gaussNodes = 6) = 0;

    virtual void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                            amrex::Real t,
                            DeRhamField<Grid::primal, Space::edge>& field,
                            int gaussNodes = 6) = 0;
    virtual void projection(
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
        amrex::Real t,
        DeRhamField<Grid::primal, Space::edge>& field,
        int gaussNodes = 6) = 0;

    virtual void projection(amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
                            amrex::Real t,
                            DeRhamField<Grid::dual, Space::face>& field,
                            int gaussNodes = 6) = 0;
    virtual void projection(
        const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
        amrex::Real t,
        DeRhamField<Grid::dual, Space::face>& field,
        int gaussNodes = 6) = 0;

    //----------------
    // Hodge operators
    //----------------
    virtual void hodge(DeRhamField<Grid::primal, Space::node>& zeroForm,
                       const DeRhamField<Grid::dual, Space::cell>& threeForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::dual, Space::node>& zeroForm,
                       const DeRhamField<Grid::primal, Space::cell>& threeForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::primal, Space::edge>& oneForm,
                       const DeRhamField<Grid::dual, Space::face>& twoForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::dual, Space::edge>& oneForm,
                       const DeRhamField<Grid::primal, Space::face>& twoForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::primal, Space::face>& twoForm,
                       const DeRhamField<Grid::dual, Space::edge>& oneForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::dual, Space::face>& twoForm,
                       const DeRhamField<Grid::primal, Space::edge>& oneForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::primal, Space::cell>& threeForm,
                       const DeRhamField<Grid::dual, Space::node>& zeroForm,
                       amrex::Real weight = 1.) = 0;
    virtual void hodge(DeRhamField<Grid::dual, Space::cell>& threeForm,
                       const DeRhamField<Grid::primal, Space::node>& zeroForm,
                       amrex::Real weight = 1.) = 0;

    /**
     * @brief Computes the \f$formOut=\mathbb{H}_{\boldsymbol M} formIn\f$.
     *
     * @details
     * Computes \f$formOut={\boldsymbol M}*weight*formIn\f$
     * tensor matrix M is the same de Rham field as formOut
     * M is 3 by 3 matrix saved as a DeRhamField with 3 components for each of the three variables
     *
     * @tparam grid primal or dual
     * @tparam space edge or node, corresponding to 1- or 2-forms
     * @param formOut DeRhamField<grid, space>
     * @param formIn DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>
     * @param matrix DeRhamField<grid, space>
     * @param weight weighting factor
     */
    template <Grid grid, Space space, std::enable_if_t<isOneOrTwoForm<space>, int> enable = 0>
    inline void hodge_dk (DeRhamField<grid, space>& formOut,
                          DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& formIn,
                          DeRhamField<grid, space>& matrix,
                          amrex::Real weight = 1.)
    {
        BL_PROFILE("Gempic::Forms::HodgeDK");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(formOut.m_data[xDir].nComp() == 1,
                                         "hodge_dk not implemented for multi-component fields");
        // Start by applying the standard Hodge operator. Needs to be stored in temporary multifab
        DeRhamField<grid, space> tempFormOut(formOut.m_deRham);
        this->hodge(tempFormOut, formIn, weight); //get tempFormOut from formIn

        // Update x component assume corresponding multifab has only 1 component
        // y and z components need to be averaged to midpoint of x edge
        for (amrex::MFIter mfi(formOut.m_data[xDir], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const& formx = (tempFormOut.m_data[xDir])[mfi].array();
            amrex::Array4<amrex::Real> const& formy = (tempFormOut.m_data[yDir])[mfi].array();
            amrex::Array4<amrex::Real> const& formz = (tempFormOut.m_data[zDir])[mfi].array();
            amrex::Array4<amrex::Real> const& matrixX = matrix.m_data[xDir][mfi].array();
            amrex::Array4<amrex::Real> const& formOutX = (formOut.m_data[xDir])[mfi].array();

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            formOutX(i, j, k) = matrixX(i, j, k, xDir) * formx(i, j, k) +
                                                matrixX(i, j, k, yDir) * 0.25 *
                                                    (formy(i - 1, j, k) + formy(i, j, k) +
                                                     formy(i - 1, j + 1, k) + formy(i, j + 1, k)) +
                                                matrixX(i, j, k, zDir) * 0.25 *
                                                    (formz(i - 1, j, k) + formz(i, j, k) +
                                                     formz(i - 1, j, k) + formz(i, j, k + 1));
                        });
        }
        // Update y component assume corresponding multifab has only 1 component
        // x and z components need to be averaged to midpoint of y edge
        for (amrex::MFIter mfi(formOut.m_data[yDir], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const& formx = tempFormOut.m_data[xDir][mfi].array();
            amrex::Array4<amrex::Real> const& formy = tempFormOut.m_data[yDir][mfi].array();
            amrex::Array4<amrex::Real> const& formz = tempFormOut.m_data[zDir][mfi].array();
            amrex::Array4<amrex::Real> const& matrixY = matrix.m_data[yDir][mfi].array();
            amrex::Array4<amrex::Real> const& formOutY = formOut.m_data[yDir][mfi].array();

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            formOutY(i, j, k) = matrixY(i, j, k, xDir) * 0.25 *
                                                    (formx(i, j - 1, k) + formx(i, j, k) +
                                                     formx(i + 1, j - 1, k) + formx(i + 1, j, k)) +
                                                matrixY(i, j, k, yDir) * formy(i, j, k) +
                                                matrixY(i, j, k, zDir) * 0.25 *
                                                    (formz(i, j, k) + formz(i, j - 1, k) +
                                                     formz(i, j, k + 1) + formz(i, j - 1, k + 1));
                        });
        }
        // Update z component assume corresponding multifab has only 1 component
        // x and y components need to be averaged to midpoint of z edge
        for (amrex::MFIter mfi(formOut.m_data[zDir], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const& formx = (tempFormOut.m_data[xDir])[mfi].array();
            amrex::Array4<amrex::Real> const& formy = (tempFormOut.m_data[yDir])[mfi].array();
            amrex::Array4<amrex::Real> const& formz = (tempFormOut.m_data[zDir])[mfi].array();
            amrex::Array4<amrex::Real> const& matrixZ = matrix.m_data[zDir][mfi].array();
            amrex::Array4<amrex::Real> const& formOutZ = (formOut.m_data[zDir])[mfi].array();

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            formOutZ(i, j, k) = matrixZ(i, j, k, xDir) * 0.25 *
                                                    (formx(i, j, k - 1) + formx(i, j, k) +
                                                     formx(i + 1, j, k - 1) + formx(i + 1, j, k)) +
                                                matrixZ(i, j, k, yDir) * 0.25 *
                                                    (formy(i, j, k) + formy(i, j, k - 1) +
                                                     formy(i, j + 1, k) + formy(i, j + 1, k - 1)) +
                                                matrixZ(i, j, k, zDir) * formz(i, j, k);
                        });
        }
        formOut.fill_boundary();
    }

    /**
     * @brief Evaluation of a 0-/3-form DeRhamField of nVar variables at points in each cell
     * given by a positive normalized shift from the lower corner of the cell
     * @param field       The field to evaluate
     * @param evalShift   The time step at which to evaluate the functions
     * @param pointValues Output cell-centered MultiFab containing nVar * nEvals components in
     * the order
     *                    var=0 eval=0,
     *                    var=0 eval=1,
     *                    ...,
     *                    var=0 shift=nEvals,
     *                    var=1 shift=0,
     *                    ...,
     *                    var=nVar shift=nEvals
     */
    virtual void eval_form(
        const DeRhamField<Grid::primal, Space::cell>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues) = 0;
    virtual void eval_form(
        const DeRhamField<Grid::dual, Space::cell>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues) = 0;
    virtual void eval_form(
        const DeRhamField<Grid::primal, Space::node>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues) = 0;
    virtual void eval_form(
        const DeRhamField<Grid::dual, Space::node>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues) = 0;
    /**
     * @brief Evaluation of a 1-/2-form DeRhamField of nVar variables (and three components) in one
     *        direction in each cell given by a positive normalized shift from the lower corner of
     *        the cell
     * @param field       The field to evaluate
     * @param evalShift   The time step at which to evaluate the functions
     * @param pointValues Output cell-centered MultiFab containing nVar * nEvals components in the
     *                    order
     *                    var=0 eval=0,
     *                    var=0 eval=1,
     *                    ...,
     *                    var=0 shift=nEvals,
     *                    var=1 shift=0,
     *                    ...,
     *                    var=nVar shift=nEvals
     * @param dir         Direction of field to evaluate
     */
    virtual void eval_form(
        const DeRhamField<Grid::primal, Space::face>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues,
        Direction dir) = 0;
    virtual void eval_form(
        const DeRhamField<Grid::dual, Space::face>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues,
        Direction dir) = 0;
    virtual void eval_form(
        const DeRhamField<Grid::primal, Space::edge>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues,
        Direction dir) = 0;
    virtual void eval_form(
        const DeRhamField<Grid::dual, Space::edge>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::MultiFab& pointValues,
        Direction dir) = 0;

    /**
     * @brief Evaluation of a 1-/2-form DeRhamField of nVar variables (and three components) in
     *        each cell given by a positive normalized shift from the lower corner of the cell
     * @param field       The field to evaluate
     * @param evalShift   The time step at which to evaluate the functions
     * @param pointValues 3D output array of cell-centered MultiFabs each containing nVar * nEvals
     *                    components in the order
     *                    var=0 eval=0,
     *                    var=0 eval=1,
     *                    ...,
     *                    var=0 shift=nEvals,
     *                    var=1 shift=0,
     *                    ...,
     *                    var=nVar shift=nEvals
     */
    inline void eval_forms (
        const DeRhamField<Grid::primal, Space::face>& field,
        const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
        amrex::Array<amrex::MultiFab*, 3>& pointValues)
    {
        eval_form(field, evalShift, *pointValues[xDir], xDir);
        eval_form(field, evalShift, *pointValues[yDir], yDir);
        eval_form(field, evalShift, *pointValues[zDir], zDir);
    }
    void eval_forms (const DeRhamField<Grid::dual, Space::face>& field,
                     const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
                     amrex::Array<amrex::MultiFab*, 3>& pointValues)
    {
        eval_form(field, evalShift, *pointValues[xDir], xDir);
        eval_form(field, evalShift, *pointValues[yDir], yDir);
        eval_form(field, evalShift, *pointValues[zDir], zDir);
    }
    void eval_forms (const DeRhamField<Grid::primal, Space::edge>& field,
                     const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
                     amrex::Array<amrex::MultiFab*, 3>& pointValues)
    {
        eval_form(field, evalShift, *pointValues[xDir], xDir);
        eval_form(field, evalShift, *pointValues[yDir], yDir);
        eval_form(field, evalShift, *pointValues[zDir], zDir);
    }
    void eval_forms (const DeRhamField<Grid::dual, Space::edge>& field,
                     const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& evalShift,
                     amrex::Array<amrex::MultiFab*, 3>& pointValues)
    {
        eval_form(field, evalShift, *pointValues[xDir], xDir);
        eval_form(field, evalShift, *pointValues[yDir], yDir);
        eval_form(field, evalShift, *pointValues[zDir], zDir);
    }
};

// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, int nVar) :
    DeRhamField{drc, std::array<int, 3>{nVar, nVar, nVar}}
{
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, std::array<int, 3> ncomponents) :
    m_deRham{drc}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc)");
    m_deRham->initialize_field(*this, ncomponents);

    if (not m_deRham->get_geom().isAllPeriodic())
    {
        amrex::Vector<amrex::BCRec> bcRec =
            load_boundary_condition(m_deRham->get_geom(), "Default");
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bcX(BoundaryConditions<grid, space>{
            m_data[xDir].ixType(), m_deRham->get_hodge_degree() - 1});
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bcY(BoundaryConditions<grid, space>{
            m_data[yDir].ixType(), m_deRham->get_hodge_degree() - 1});
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bcZ(BoundaryConditions<grid, space>{
            m_data[zDir].ixType(), m_deRham->get_hodge_degree() - 1});
        m_bcFuncX = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bcX);
        m_bcFuncY = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bcY);
        m_bcFuncZ = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bcZ);
    }
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, int nVar) :
    m_deRham{drc}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc)");
    m_deRham->initialize_field(*this, nVar);

    if (not m_deRham->get_geom().isAllPeriodic())
    {
        amrex::Vector<amrex::BCRec> bcRec =
            load_boundary_condition(m_deRham->get_geom(), "Default");
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bc(
            BoundaryConditions<grid, space>(m_data.ixType(), m_deRham->get_hodge_degree() - 1));
        m_bcFunc = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bc);
    }
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, const std::string& diagName, int nVar) :
    DeRhamField{drc, nVar}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,diagName)");
    m_deRham->m_fieldsDiagnostics.emplace(
        diagName + "x", amrex::MultiFab(this->m_data[xDir], amrex::make_alias, 0, nVar));
    m_deRham->m_fieldsDiagnostics.emplace(
        diagName + "y", amrex::MultiFab(this->m_data[yDir], amrex::make_alias, 0, nVar));
    m_deRham->m_fieldsDiagnostics.emplace(
        diagName + "z", amrex::MultiFab(this->m_data[zDir], amrex::make_alias, 0, nVar));

    if constexpr (space == Space::edge) // edge field
    {
        auto dxInv{m_deRham->m_geom.InvCellSizeArray()};
        m_deRham->m_fieldsScaling.emplace(diagName + "x", dxInv[xDir]);
        m_deRham->m_fieldsScaling.emplace(diagName + "y", GEMPIC_D_MULT(1.0, dxInv[yDir], 1.0));
        m_deRham->m_fieldsScaling.emplace(diagName + "z", GEMPIC_D_MULT(1.0, 1.0, dxInv[zDir]));
    }
    if constexpr (space == Space::face) // face field
    {
        auto dxInv{m_deRham->m_geom.InvCellSizeArray()};
        m_deRham->m_fieldsScaling.emplace(diagName + "x",
                                          GEMPIC_D_MULT(1.0, dxInv[yDir], dxInv[zDir]));
        m_deRham->m_fieldsScaling.emplace(diagName + "y",
                                          GEMPIC_D_MULT(dxInv[xDir], 1.0, dxInv[zDir]));
        m_deRham->m_fieldsScaling.emplace(diagName + "z",
                                          GEMPIC_D_MULT(dxInv[xDir], dxInv[yDir], 1.0));
    }

    if (not m_deRham->get_geom().isAllPeriodic())
    {
        amrex::Vector<amrex::BCRec> bcRec = load_boundary_condition(m_deRham->get_geom(), diagName);
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bcX(BoundaryConditions<grid, space>{
            m_data[xDir].ixType(), m_deRham->get_hodge_degree() - 1});
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bcY(BoundaryConditions<grid, space>{
            m_data[yDir].ixType(), m_deRham->get_hodge_degree() - 1});
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bcZ(BoundaryConditions<grid, space>{
            m_data[zDir].ixType(), m_deRham->get_hodge_degree() - 1});
        m_bcFuncX = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bcX);
        m_bcFuncY = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bcY);
        m_bcFuncZ = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bcZ);
    }
};

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, const std::string& diagName, int nVar) :
    DeRhamField{drc, nVar}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,diagName)");
    m_deRham->m_fieldsDiagnostics.emplace(
        diagName, amrex::MultiFab(this->m_data, amrex::make_alias, 0, nVar));

    if constexpr (space == Space::node) // node field
    {
        m_deRham->m_fieldsScaling.emplace(diagName, 1.0);
    }
    if constexpr (space == Space::cell) // cell field
    {
        double invCellVol =
            GEMPIC_D_MULT(m_deRham->m_geom.InvCellSize(xDir), m_deRham->m_geom.InvCellSize(yDir),
                          m_deRham->m_geom.InvCellSize(zDir));
        m_deRham->m_fieldsScaling.emplace(diagName, invCellVol);
    }

    if (not m_deRham->get_geom().isAllPeriodic())
    {
        amrex::Vector<amrex::BCRec> bcRec = load_boundary_condition(m_deRham->get_geom(), diagName);
        amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>> bc(
            BoundaryConditions<grid, space>(m_data.ixType(), m_deRham->get_hodge_degree() - 1));
        m_bcFunc = std::make_unique<
            amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BoundaryConditions<grid, space>>>>(
            m_deRham->get_geom(), bcRec, bc);
    }
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func) :
    DeRhamField{drc, 1}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,func)");
    m_deRham->projection(func, 0.0, *this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs) :
    DeRhamField{drc, static_cast<int>(funcs.size())}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,funcs)");
    m_deRham->projection(funcs, 0.0, *this);
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func) :
    DeRhamField{drc, 1}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,func)");
    m_deRham->projection(func, 0.0, *this);
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs) :
    DeRhamField{drc, static_cast<int>(funcs.size())}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,funcs)");
    m_deRham->projection(funcs, 0.0, *this);
}

/// @cond DOXYGEN_IGNORE
template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>& func,
    const std::string& diagName) :
    DeRhamField{drc, diagName, 1}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,func,diagName)");
    m_deRham->projection(func, 0.0, *this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::Vector<amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3>>& funcs,
    const std::string& diagName) :
    DeRhamField{drc, diagName, static_cast<int>(funcs.size())}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::DeRhamField(drc,funcs,diagName)");
    m_deRham->projection(funcs, 0.0, *this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::MultiFab& mfIn1,
    const amrex::MultiFab& mfIn2,
    const amrex::MultiFab& mfIn3) :
    DeRhamField{drc, mfIn1.nComp()}
{
    m_data[xDir] = amrex::MultiFab(mfIn1, amrex::make_alias, 0, mfIn1.nComp());
    m_data[yDir] = amrex::MultiFab(mfIn2, amrex::make_alias, 0, mfIn2.nComp());
    m_data[zDir] = amrex::MultiFab(mfIn3, amrex::make_alias, 0, mfIn3.nComp());
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    amrex::ParserExecutor<AMREX_SPACEDIM + 1>& func,
    const std::string& diagName) :
    DeRhamField{drc, diagName, 1}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,func,diagName)");
    m_deRham->projection(func, 0.0, *this);
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc,
    const amrex::Vector<amrex::ParserExecutor<AMREX_SPACEDIM + 1>>& funcs,
    const std::string& diagName) :
    DeRhamField{drc, diagName, static_cast<int>(funcs.size())}
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::DeRhamField(drc,funcs,diagName)");
    m_deRham->projection(funcs, 0.0, *this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField(
    std::shared_ptr<DeRhamComplex> drc, const amrex::MultiFab& mfIn) :
    DeRhamField{drc, mfIn.nComp()}
{
    m_data = amrex::MultiFab(mfIn, amrex::make_alias, 0, mfIn.nComp());
}

// Synchronization necessary after cpu and gpu particle loops
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::
    post_particle_loop_sync()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::post_particle_loop_sync()");
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].SumBoundary(0, m_data[comp].nComp(), m_deRham->get_n_ghost(),
                                 amrex::IntVect::Zero, m_deRham->get_periodicity());
        // IMPORTANT: SumBoundary does not guarantee the same order of summation for overlapping
        // degrees of freedom on all contributing boxes. The resulting differences in J are on the
        // order of machine precision but can accumulate in the electric field D. OverrideSync
        // ensures that the fields are consistent on all boxes. For the same reason, a fill_boundary
        // is used instead of directly filling ghost cells in SumBoundary.
        m_data[comp].OverrideSync(m_deRham->get_periodicity());
    }
    fill_boundary();
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::
    post_particle_loop_sync()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::post_particle_loop_sync()");
    m_data.SumBoundary(0, m_data.nComp(), m_deRham->get_n_ghost(), amrex::IntVect::Zero,
                       m_deRham->get_periodicity());
    // See comment on OverrideSync in function for OneOrTwoForm above
    m_data.OverrideSync(m_deRham->get_periodicity());
    fill_boundary();
}

// FillBoundary of DeRhamFields following the same logic as above
template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::fill_boundary()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::fill_boundary()");
    m_deRham->fill_boundary(*this);
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::fill_boundary()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::fill_boundary()");
    m_deRham->fill_boundary(*this);
}

// FillBoundary_nowait of DeRhamFields following the same logic as above
template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::fill_boundary_nowait()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::fill_boundary_nowait()");
    m_deRham->fill_boundary_nowait(*this);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::
    fill_boundary_nowait()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::fill_boundary()");
    m_deRham->fill_boundary_nowait(*this);
}

// FillBoundary_finish of DeRhamFields following the same logic as above
template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::fill_boundary_finish()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::fill_boundary_finish()");
    m_deRham->fill_boundary_finish(*this);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::
    fill_boundary_finish()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::fill_boundary_finish()");
    m_deRham->fill_boundary_finish(*this);
}

// ApplyBC of DeRhamFields
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::apply_bc()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::apply_bc()");
    m_deRham->apply_bc(*this);
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::apply_bc()
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::apply_bc()");
    m_deRham->apply_bc(*this);
}

// Multiply by scalar
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator*=(
    amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator*=()");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].mult(dt, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator*=(
    amrex::Real dt)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator*=()");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.mult(dt, nGhost[xDir]);
}

// Multiply by parsed function
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator*=(
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> func)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator*=(function)");

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = m_deRham->m_geom.CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_deRham->get_prob_lo();

    int nComps{m_data[xDir].nComp()};

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(m_data[comp], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const& form = (m_data[comp])[mfi].array();

            ParallelFor(
                bx, nComps,
                [r0, comp, dx, form, func] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r;
                    if constexpr ((grid == Grid::primal && space == Space::edge) ||
                                  (grid == Grid::dual && space == Space::face))
                    {
                        // midpoints of primal edge / dual face
                        r = {AMREX_D_DECL(r0[xDir] + (i + (comp == xDir ? 0.5 : 0)) * dx[xDir],
                                          r0[yDir] + (j + (comp == yDir ? 0.5 : 0)) * dx[yDir],
                                          r0[zDir] + (k + (comp == zDir ? 0.5 : 0)) * dx[zDir])};
                    }
                    else
                    {
                        // midpoints of primal face / dual edge
                        r = {AMREX_D_DECL(
                            r0[xDir] + (i + ((comp == yDir || comp == zDir) ? 0.5 : 0)) * dx[xDir],
                            r0[yDir] + (j + ((comp == xDir || comp == zDir) ? 0.5 : 0)) * dx[yDir],
                            r0[zDir] +
                                (k + ((comp == xDir || comp == yDir) ? 0.5 : 0)) * dx[zDir])};
                    }
                    form(i, j, k, n) =
                        form(i, j, k, n) * func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);
                });
        }
    }
    this->fill_boundary();
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator*=(
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> func)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator*=(function)");

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = m_deRham->m_geom.CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = m_deRham->get_prob_lo();

    int nComps{m_data.nComp()};

    for (amrex::MFIter mfi(m_data, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<amrex::Real> const& form = (m_data)[mfi].array();

        ParallelFor(bx, nComps,
                    [r0, dx, form, func] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r;
                        if constexpr ((grid == Grid::primal && space == Space::node) ||
                                      (grid == Grid::dual && space == Space::cell))
                        {
                            // primal nodes / dual cell midpoints
                            r = {AMREX_D_DECL(r0[xDir] + i * dx[xDir], r0[yDir] + j * dx[yDir],
                                              r0[zDir] + k * dx[zDir])};
                        }
                        else
                        {
                            // primal cell midpoints / dual nodes
                            r = {AMREX_D_DECL(r0[xDir] + (i + 0.5) * dx[xDir],
                                              r0[yDir] + (j + 0.5) * dx[yDir],
                                              r0[zDir] + (k + 0.5) * dx[zDir])};
                        }
                        form(i, j, k, n) =
                            form(i, j, k, n) * func(AMREX_D_DECL(r[xDir], r[yDir], r[zDir]), 0.);
                    });
    }
    this->fill_boundary();
}

// plusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator+=(
    DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator+=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(form.m_data[comp], 0, m_data[comp].nComp(), nGhost[xDir]);
    }
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator+=(
    amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator+=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(c, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator+=(
    DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator+=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.plus(form.m_data, 0, m_data.nComp(), nGhost[xDir]);
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator+=(
    amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator+=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.plus(c, nGhost[xDir]);
}

// minusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator-=(
    DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator-=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].minus(form.m_data[comp], 0, m_data[comp].nComp(), nGhost[xDir]);
    }
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator-=(
    amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator-=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        m_data[comp].plus(-c, nGhost[xDir]);
    }
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator-=(
    DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator-=(MultiFab)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.minus(form.m_data, 0, m_data.nComp(), nGhost[xDir]);
}

template <Grid grid, Space space>
inline void
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator-=(
    amrex::Real c)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator-=(amrex::Real)");
    auto nGhost = m_deRham->get_n_ghost();
    m_data.plus(-c, nGhost[xDir]);
}

// minus
template <Grid grid, Space space>
inline DeRhamField<grid, space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator-(
    DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::operator-(MultiFab)");
    DeRhamField<grid, space> result(m_deRham);
    auto nGhost = m_deRham->get_n_ghost();

    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Copy(result.m_data[comp], m_data[comp], 0, 0, m_data[comp].nComp(), nGhost);
        amrex::Subtract(result.m_data[comp], form.m_data[comp], 0, 0, form.m_data[comp].nComp(),
                        nGhost);
    }
    return result;
}

template <Grid grid, Space space>
inline DeRhamField<grid, space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator-(
    DeRhamField<grid, space>& form)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::operator-(MultiFab)");
    DeRhamField<grid, space> result(m_deRham);

    amrex::Copy(result.m_data, m_data, 0, 0, m_data.nComp(), 0);
    amrex::Subtract(result.m_data, form.m_data, 0, 0, form.m_data.nComp(), 0);

    return result;
}

///@endcond

/// Copy of fields
/// @tparam grid Grid type.
/// @tparam space Space type.
/// @param fieldOut The field to be copied into
/// @param fieldIn The field to be copied from
template <Grid grid, Space space>
inline void copy (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& fieldOut,
    const DeRhamField<grid, space>& fieldIn)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::copy");
    auto nGhost = fieldIn.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Copy(fieldOut.m_data[comp], fieldIn.m_data[comp], 0, 0, fieldIn.m_data[comp].nComp(),
                    nGhost);
    }
}

/// Copy of fields
/// @tparam grid Grid type.
/// @tparam space Space type.
/// @param fieldOut The field to be copied into
/// @param fieldIn The field to be copied from
template <Grid grid, Space space>
inline void copy (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& fieldOut,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& fieldIn)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isOneOrTwoForm>::copy(oppositeForm)");
    auto nGhost = fieldIn.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Copy(fieldOut.m_data[comp], fieldIn.m_data[comp], 0, 0, fieldIn.m_data[comp].nComp(),
                    nGhost);
    }
}

template <Grid grid, Space space>
inline void copy (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& fieldOut,
    const DeRhamField<grid, space>& fieldIn)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::copy");
    auto nGhost = fieldIn.m_deRham->get_n_ghost();
    amrex::Copy(fieldOut.m_data, fieldIn.m_data, 0, 0, fieldIn.m_data.nComp(), nGhost);
}
template <Grid grid, Space space>
inline void copy (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& fieldOut,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& fieldIn)
{
    BL_PROFILE("Gempic::Forms::DeRhamField<isZeroOrThreeForm>::copy(oppositeForm)");
    auto nGhost = fieldIn.m_deRham->get_n_ghost();
    amrex::Copy(fieldOut.m_data, fieldIn.m_data, 0, 0, fieldIn.m_data.nComp(), nGhost);
}

// compute linear combinations of fields
/**
 * Performs a linear combination operation on two DeRhamField objects for one or two forms.
 *
 * @param result The result of the linear combination operation.
 * @param a The scaling factor for the first field.
 * @param field1 The first DeRhamField object.
 * @param b The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <Grid grid, Space space>
inline void linear_combination (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& result,
    amrex::Real a,
    const DeRhamField<grid, space>& field1,
    amrex::Real b,
    const DeRhamField<grid, space>& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = result.m_deRham->get_n_ghost();
    amrex::LinComb(result.m_data, a, field1.m_data, 0, b, field2.m_data, 0, 0,
                   field1.m_data.nComp(), nGhost);
}

/**
 * Performs a linear combination operation on two DeRhamField objects for one or two forms.
 *
 * @param result The result of the linear combination operation.
 * @param a The scaling factor for the first field.
 * @param field1 The first DeRhamField object.
 * @param b The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <Grid grid, Space space>
inline void linear_combination (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& result,
    amrex::Real a,
    const DeRhamField<grid, space>& field1,
    amrex::Real b,
    const DeRhamField<grid, space>& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = result.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::LinComb(result.m_data[comp], a, field1.m_data[comp], 0, b, field2.m_data[comp], 0, 0,
                       field1.m_data[comp].nComp(), nGhost);
    }
}

/**
 * Performs a linear combination operation on two DeRhamField objects for zero or three forms.
 *
 * @param result The result of the linear combination operation.
 * @param a The scaling factor for the first field.
 * @param field1 The first DeRhamField object.
 * @param b The scaling factor for the second field.
 * @param field2 The second DeRhamField object on the opposite grid and space.
 */
template <Grid grid, Space space>
inline void linear_combination (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& result,
    amrex::Real a,
    const DeRhamField<grid, space>& field1,
    amrex::Real b,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = result.m_deRham->get_n_ghost();
    amrex::LinComb(result.m_data, a, field1.m_data, 0, b, field2.m_data, 0, 0,
                   field1.m_data.nComp(), nGhost);
}

/**
 * Performs a linear combination operation on two DeRhamField objects for one or two forms.
 *
 * @param result The result of the linear combination operation.
 * @param a The scaling factor for the first field.
 * @param field1 The first DeRhamField object.
 * @param b The scaling factor for the second field.
 * @param field2 The second DeRhamField object on the opposite grid and space.
 */
template <Grid grid, Space space>
inline void linear_combination (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& result,
    amrex::Real a,
    const DeRhamField<grid, space>& field1,
    amrex::Real b,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = result.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::LinComb(result.m_data[comp], a, field1.m_data[comp], 0, b, field2.m_data[comp], 0, 0,
                       field1.m_data[comp].nComp(), nGhost);
    }
}

// multiply and add for fields
/**
 * Multiply and add for operation on two DeRhamField objects for one or two forms.
 *
 * @param field1 The destination DeRhamField object.
 * @param a The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <Grid grid, Space space>
inline void multiply_and_add (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& field1,
    amrex::Real a,
    const DeRhamField<grid, space>& field2)
{
    BL_PROFILE("Gempic::Forms::multiply_and_add()");
    auto nGhost = field1.m_deRham->get_n_ghost();
    mult_and_add(field1.m_data, a, field2.m_data, 0, 0, field2.m_data.nComp(), nGhost);
}

/**
 * Multiply and add for operation on two DeRhamField objects for one or two forms.
 *
 * @param field1 The destination DeRhamField object.
 * @param a The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <Grid grid, Space space>
inline void multiply_and_add (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& field1,
    amrex::Real a,
    const DeRhamField<grid, space>& field2)
{
    BL_PROFILE("Gempic::Forms::multiply_and_add()");
    auto nGhost = field1.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        mult_and_add(field1.m_data[comp], a, field2.m_data[comp], 0, 0, field2.m_data[comp].nComp(),
                     nGhost);
    }
}

/**
 * Multiply and add for operation on two DeRhamField objects for one or two forms.
 *
 * @param field1 The destination DeRhamField object.
 * @param a The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <Grid grid, Space space>
inline void multiply_and_add (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& field1,
    amrex::Real a,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& field2)
{
    BL_PROFILE("Gempic::Forms::multiply_and_add()");
    auto nGhost = field1.m_deRham->get_n_ghost();
    mult_and_add(field1.m_data, a, field2.m_data, 0, 0, field2.m_data.nComp(), nGhost);
}

/**
 * Multiply and add for operation on two DeRhamField objects for one or two forms.
 *
 * @param field1 The destination DeRhamField object.
 * @param a The scaling factor for the second field.
 * @param field2 The second DeRhamField object.
 */
template <Grid grid, Space space>
inline void multiply_and_add (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& field1,
    amrex::Real a,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& field2)
{
    BL_PROFILE("Gempic::Forms::linearCombination()");
    auto nGhost = field1.m_deRham->get_n_ghost();
    for (int comp = 0; comp < 3; ++comp)
    {
        mult_and_add(field1.m_data[comp], a, field2.m_data[comp], 0, 0, field2.m_data[comp].nComp(),
                     nGhost);
    }
}
/**
 * @brief Computes the dot product of two fields on the same grid.
 *
 * @tparam grid Grid type.
 * @tparam space Space type.
 * @tparam std::enable_if_t<isOneOrTwoForm<space>> Enables the function only if space is one or
 * two form.
 * @param field1 The first field.
 * @param field2 The second field.
 *
 * @return The dot product of the fields.
 */
template <Grid grid, Space space>
inline amrex::Real dot_product (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& field1,
    const DeRhamField<grid, space>& field2)
{
    const bool skipMpiCommTrue = true;
    // Compute the dot product by summing the dot product of each component.
    amrex::Real dotProd{0};              // Initialize the dot product to 0.
    for (int comp = 0; comp < 3; ++comp) // Loop over each component.
    {
        auto mask = amrex::OwnerMask(field1.m_data[comp], field1.m_deRham->get_periodicity());
        dotProd += amrex::MultiFab::Dot(*mask, field1.m_data[comp], 0, field2.m_data[comp], 0,
                                        field1.m_data[comp].nComp(), 0, skipMpiCommTrue);
    }
    amrex::ParallelAllReduce::Sum(dotProd, amrex::ParallelContext::CommunicatorSub());
    return dotProd; // Return the computed dot product.
}

/**
 * @brief Computes the dot product of two DeRham fields on the same grid and space
 * for zero and three forms
 * @tparam grid The grid type of the DeRham field.
 * @tparam space The space type of the DeRham field.
 * @param field1 The first DeRham field.
 * @param field2 The second DeRham field.
 *
 * @return The dot product of the two fields.
 */
template <Grid grid, Space space>
inline amrex::Real dot_product (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& field1,
    const DeRhamField<grid, space>& field2)
{
    auto mask = amrex::OwnerMask(field1.m_data, field1.m_deRham->get_periodicity());
    amrex::Real dotProd =
        amrex::MultiFab::Dot(*mask, field1.m_data, 0, field2.m_data, 0, field1.m_data.nComp(), 0);
    return dotProd;
}

/**
 * Computes the dot product of two DeRham fields on opposite grid and space
 * for one and two forms
 * @tparam GridType The grid type of the first DeRham field.
 * @tparam SpaceType The space type of the first DeRham field.
 * @param field1 The first DeRham field.
 * @param field2 The second DeRham field.
 *
 * @return The dot product of the two fields.
 */
template <Grid grid, Space space>
inline amrex::Real dot_product (
    DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>& field1,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& field2)
{
    const bool skipMpiCommTrue = true;
    amrex::Real result = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        auto mask = amrex::OwnerMask(field1.m_data[comp], field1.m_deRham->get_periodicity());
        result += amrex::MultiFab::Dot(*mask, field1.m_data[comp], 0, field2.m_data[comp], 0,
                                       field1.m_data[comp].nComp(), 0, skipMpiCommTrue);
    }
    amrex::ParallelAllReduce::Sum(result, amrex::ParallelContext::CommunicatorSub());
    return result;
}

/**
 * Calculates the dot product of two DeRham fields on opposite grid and space.
 *
 * @tparam grid The grid type of the DeRham field.
 * @tparam space The space type of the DeRham field.
 *
 * @param field1 The first DeRham field.
 * @param field2 The second DeRham field.
 *
 * @return The dot product of the two fields.
 */
template <Grid grid, Space space>
inline amrex::Real dot_product (
    DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>& field1,
    const DeRhamField<oppositeGrid<grid>, oppositeSpace<space>>& field2)
{
    auto mask = amrex::OwnerMask(field1.m_data, field1.m_deRham->get_periodicity());
    amrex::Real dotProd =
        amrex::MultiFab::Dot(*mask, field1.m_data, 0, field2.m_data, 0, field1.m_data.nComp(), 0);
    return dotProd;
}

} // namespace Gempic::Forms

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
} // namespace Gempic::Forms::Impl

#endif

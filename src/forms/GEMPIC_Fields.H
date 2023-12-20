#ifndef GEMPIC_FIELDS_H_
#define GEMPIC_FIELDS_H_

#include <type_traits>

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
#include "GEMPIC_parameters.H"
#include "GEMPIC_computational_domain.H"

namespace GEMPIC_Fields
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

// Helpful for enable_if magic. @todo: If we switch to C++20, we should use concepts, as shown in
// https://gitlab.mpcdf.mpg.de/gempic/gempic/-/merge_requests/127#note_225990
template <Space space>
inline constexpr bool isZeroOrThreeForm{false};

template<>
inline constexpr bool isZeroOrThreeForm<Space::node>{true};

template<>
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
* e.g. DeRhamField<Grid::dual, node> is the cell based complex dual to DeRhamField<Grid::primal, node>
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
    std::shared_ptr<DeRhamComplex> deRham;
    /**
     * @brief actual field information.
     * Multifab for Space::node or Space::cell,
     * Array of Multifabs for Space::edge or Space::face, 
    */
    amrex::Array<amrex::MultiFab, 3> data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc) {static_assert(space != space, "This class should not be callable.");};
    /**
     * @brief Construct Space::edge / Space::face fields from function array
    */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func) : DeRhamField{drc} {};

    /**
     * @brief Construct Space::node / Space::cell fields from function
    */
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func) : DeRhamField{drc} {};

    void fillBoundary();
    void averageSync();

    /**
     * @brief For Space::node / Space::cell fields, perform all of the synchronization operations necessary after cpu and gpu particle loops
    */
    void postParticleLoopSync();
};

// One- or Two-form, i.e., space is either edge or face
template <Grid grid, Space space>
struct DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>
{
    std::shared_ptr<DeRhamComplex> deRham;
    amrex::Array<amrex::MultiFab, 3> data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func);

    void fillBoundary();
    void averageSync();

    void operator*=(const amrex::Real& dt);
    void operator+=(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
};

// Zero- or Three-form, i.e., space is either node or cell
template<Grid grid, Space space>
struct DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>
{
    std::shared_ptr<DeRhamComplex> deRham;
    amrex::MultiFab data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func);

    void fillBoundary();
    void averageSync();

    void postParticleLoopSync();
    void operator*=(const amrex::Real& dt);
};

/**
 * @brief Base class for discrete de Rham Complexes. The member functions with the `inline` specifier implement the methods of the elements of the Complex (the functions of the fields). 
 * The projections are virtual functions, whose details are implemented in the inherited subclasses.
 *
 */
class DeRhamComplex
{

private:
    // Indices
    amrex::IndexType primalNodeIndex;
    amrex::Array<amrex::IndexType, 3> primalEdgeIndex;
    amrex::Array<amrex::IndexType, 3> primalFaceIndex;
    amrex::IndexType primalCellIndex;

protected:
    // Computational grid
    amrex::BoxArray m_grid;
    amrex::DistributionMapping m_distriMap;
    amrex::Geometry m_geom;
    int m_hodgeDegree;

    // Number of ghost cells along each direction
    amrex::IntVect m_nGhost;

public:
    /// Constructor for the DeRhamComplex class
    /// @param infra The computational domain needed to construct the class correctly
    /// @param hodgeDegree The degree of the Hodge scheme, used for a lower limit of ghost cells
    /// @param maxSplineDegree The degree of the splines, used for a lower limit of ghost cells
    DeRhamComplex(const Gempic::CompDom::computational_domain& infra, const int hodgeDegree, const int maxSplineDegree)
    {
        Parameters params("DeRhamComplex");

        m_grid = infra.grid;
        m_distriMap = infra.distriMap;
        m_geom = infra.geom;
        m_hodgeDegree = hodgeDegree;

        /* Ghost cells are decided for the fields by the degree of the Hodge.
        ** Needs to be changed for particles and decided by spline degree:
        ** to max(degx, degy, degz) + safety margin to let particles move
        **
        ** Example: for linear splines (deg_i = 1) 1 ghost cell is needed.
        ** 3 more ghost cells are added to let particles move.
        ** */
        // buffer ghost cells
        int nGhostExtra{3};
        params.getOrSet("n_ghost_extra", nGhostExtra);
        int nGhostNum{std::max(m_hodgeDegree / 2, maxSplineDegree + nGhostExtra)};
        m_nGhost = amrex::IntVect{AMREX_D_DECL(nGhostNum, nGhostNum, nGhostNum)};
        std::vector<int> nGhostVector{AMREX_D_DECL(m_nGhost[xDir], m_nGhost[yDir], m_nGhost[zDir])};
        params.set("n_ghost", nGhostVector);

        // set index: // 0 -> cell-centered | 1 -> node-centered
        primalNodeIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
        primalEdgeIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)});
        primalEdgeIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)});
        primalEdgeIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});
        primalFaceIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)});
        primalFaceIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)});
        primalFaceIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)});
        primalCellIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)});
    }

    virtual ~DeRhamComplex() {}

    /// Getter for the number of ghost cells
    const amrex::IntVect& getNGhost() const
    {
        return m_nGhost;
    }

    const amrex::Periodicity getPeriodicity() const
    {
       return m_geom.periodicity(); 
    } 

    void initializeField(DeRhamField<Grid::primal, Space::node>& field)
    {
        field.data.define(amrex::convert(m_grid, primalNodeIndex) , m_distriMap, 1, m_nGhost);
        field.data.setVal(0.0);
    }

    // Function overloads for the initializations
    void initializeField(DeRhamField<Grid::primal, Space::edge>& field)
    {
        field.data[xDir].define(amrex::convert(m_grid, primalEdgeIndex[xDir]) , m_distriMap, 1, m_nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(m_grid, primalEdgeIndex[yDir]) , m_distriMap, 1, m_nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(m_grid, primalEdgeIndex[zDir]) , m_distriMap, 1, m_nGhost);
        field.data[zDir].setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::primal, Space::face>& field)
    {
        field.data[xDir].define(amrex::convert(m_grid, primalFaceIndex[xDir]) , m_distriMap, 1, m_nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(m_grid, primalFaceIndex[yDir]) , m_distriMap, 1, m_nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(m_grid, primalFaceIndex[zDir]) , m_distriMap, 1, m_nGhost);
        field.data[zDir].setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::primal, Space::cell>& field)
    {
        field.data.define(amrex::convert(m_grid, primalCellIndex) , m_distriMap, 1, m_nGhost);
        field.data.setVal(0.0);
    }

    // Centering is always set by the primal grid
    void initializeField(DeRhamField<Grid::dual, Space::node>& field)
    {
        field.data.define(amrex::convert(m_grid, primalCellIndex) , m_distriMap, 1, m_nGhost);
        field.data.setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::dual, Space::edge>& field)
    {
        field.data[xDir].define(amrex::convert(m_grid, primalFaceIndex[xDir]) , m_distriMap, 1, m_nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(m_grid, primalFaceIndex[yDir]) , m_distriMap, 1, m_nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(m_grid, primalFaceIndex[zDir]) , m_distriMap, 1, m_nGhost);
        field.data[zDir].setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::dual, Space::face>& field)
    {
        field.data[xDir].define(amrex::convert(m_grid, primalEdgeIndex[xDir]) , m_distriMap, 1, m_nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(m_grid, primalEdgeIndex[yDir]) , m_distriMap, 1, m_nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(m_grid, primalEdgeIndex[zDir]) , m_distriMap, 1, m_nGhost);
        field.data[zDir].setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::dual, Space::cell>& field)
    {
        field.data.define(amrex::convert(m_grid, primalNodeIndex) , m_distriMap, 1, m_nGhost);
        field.data.setVal(0.0);
    }

    // Function overloads for fill boundary
    // fillBoundary for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void fillBoundary(DeRhamField<grid, space>& field)
    {
        field.data[xDir].FillBoundary(m_geom.periodicity());
        field.data[yDir].FillBoundary(m_geom.periodicity());
        field.data[zDir].FillBoundary(m_geom.periodicity());
    }

    // fillBoundary for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void fillBoundary(DeRhamField<grid, space>& field)
    {
        field.data.FillBoundary(m_geom.periodicity());
    }

    // Function overloads for averagesync
    // averageSync for One- or Two-forms
    template <Grid grid, Space space, typename std::enable_if_t<isOneOrTwoForm<space>, int> = 0>
    void averageSync(DeRhamField<grid, space>& field)
    {
        field.data[xDir].AverageSync(m_geom.periodicity());
        field.data[yDir].AverageSync(m_geom.periodicity());
        field.data[zDir].AverageSync(m_geom.periodicity());
    }

    // averageSync for Zero- or Three-forms
    template <Grid grid, Space space, typename std::enable_if_t<isZeroOrThreeForm<space>, int> = 0>
    void averageSync(DeRhamField<grid, space>& field)
    {
        field.data.AverageSync(m_geom.periodicity());
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
    void grad(const DeRhamField<Grid::dual, Space::node>&,
                            DeRhamField<Grid::dual, Space::edge>&);
    void curl(const DeRhamField<Grid::dual, Space::edge>&,
                            DeRhamField<Grid::dual, Space::face>&);
    void div(const DeRhamField<Grid::dual, Space::face>&,
                            DeRhamField<Grid::dual, Space::cell>&);

    //------------
    // Projections
    //------------
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                    DeRhamField<Grid::primal, Space::node>&) = 0;
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                    DeRhamField<Grid::primal, Space::cell>&, int gaussNodes = 6) = 0;
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                    DeRhamField<Grid::dual, Space::node>&) = 0;
    virtual void projection(amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                    DeRhamField<Grid::dual, Space::cell>&, int gaussNodes = 6) = 0;

    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                    amrex::Real t, DeRhamField<Grid::dual, Space::edge>&, int gaussNodes = 6) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                    amrex::Real t, DeRhamField<Grid::primal, Space::face>&, int gaussNodes = 6) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                    amrex::Real t, DeRhamField<Grid::primal, Space::edge>&, int gaussNodes = 6) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                    amrex::Real t, DeRhamField<Grid::dual, Space::face>&, int gaussNodes = 6) = 0;
};

// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField (std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField (std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}


// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::DeRhamField (std::shared_ptr<DeRhamComplex> drc, const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func)
    : deRham{drc}
{
    deRham -> initializeField(*this);
    deRham -> projection(func, 0.0, *this);
}

template <Grid grid, Space space>
inline DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::DeRhamField (std::shared_ptr<DeRhamComplex> drc, const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func)
    : deRham{drc}
{
    deRham -> initializeField(*this);
    deRham -> projection(func, 0.0, *this);
}

/// @cond DOXYGEN_IGNORE
// Synchronization necessary after cpu and gpu particle loops
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>> ::
postParticleLoopSync () {
    data.SumBoundary(0, 1, deRham->getNGhost(), amrex::IntVect::Zero, deRham->getPeriodicity());
    averageSync();
    fillBoundary();
}

// FillBoundary of DeRhamFields following the same logic as above
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::fillBoundary ()
{
    deRham -> fillBoundary(*this);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::fillBoundary ()
{
    deRham -> fillBoundary(*this);
}

// AverageSync of DeRhamFields
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::averageSync ()
{
    deRham -> averageSync(*this);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::averageSync ()
{
    deRham -> averageSync(*this);
}

// Multiply by scalar
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator*= (const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].mult(dt, nGhost[xDir]);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isZeroOrThreeForm<space>>>::operator*= (const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    data.mult(dt, nGhost[xDir]);
}

// plusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator+= (DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].plus(form.data[comp], 0, 1, nGhost[xDir]);
}


// minusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space, typename std::enable_if_t<isOneOrTwoForm<space>>>::operator-= (DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].minus(form.data[comp], 0, 1, nGhost[xDir]);
}
///@endcond

}


// namespace GEMPIC_Fields

#endif

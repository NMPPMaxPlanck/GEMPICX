#ifndef GEMPIC_FIELDS_H_
#define GEMPIC_FIELDS_H_

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
#include "GEMPIC_Params.H"

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

template <Grid grid, Space space>
struct DeRhamField
{
    std::shared_ptr<DeRhamComplex> deRham;
    amrex::Array<amrex::MultiFab, 3> data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func);

    void fillBoundary();
    void overrideSync();
    void averageSync();
    void multEqual(const amrex::Real& dt);
    void operator*=(const amrex::Real& dt);
    void plusEqual(DeRhamField<grid, space>&);
    void operator+=(DeRhamField<grid, space>&);
    void minusEqual(DeRhamField<grid, space>&);
    void operator-=(DeRhamField<grid, space>&);
};

template <Grid grid>
struct DeRhamField <grid, Space::node>
{
    amrex::MultiFab data;
    std::shared_ptr<DeRhamComplex> deRham;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func);

    void fillBoundary();
    void overrideSync();
    void averageSync();
    void multEqual(const amrex::Real& dt);
};

template <Grid grid> 
struct DeRhamField <grid, Space::cell>
{
    amrex::MultiFab data;
    std::shared_ptr<DeRhamComplex> deRham;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    DeRhamField(std::shared_ptr<DeRhamComplex> drc, const 
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func);

    void fillBoundary();
    void averageSync();
    void multEqual(const amrex::Real& dt);
};

/**
 * @brief Base class for discrete de Rham Complexes. The member functions with the `inline` specifier implement the methods of the elements of the Complex (the functions of the fields). 
 * The projections are virtual functions, whose details are implemented in the inherited subclasses.
 *
 */
class DeRhamComplex
{

private:
    // Computational grid
    amrex::RealBox realBox;
    amrex::IntVect nCell;
    amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic;
    amrex::BoxArray grid;
    amrex::DistributionMapping distriMap;
    amrex::Geometry geom;
    int m_hodgeDegree;

    // Number of ghost cells along each direction
    amrex::IntVect nGhost;

    // Indices
    amrex::IndexType primalNodeIndex;
    amrex::Array<amrex::IndexType, 3> primalEdgeIndex;
    amrex::Array<amrex::IndexType, 3> primalFaceIndex;
    amrex::IndexType primalCellIndex;
/*
    // Masks
    amrex::iMultiFab nodeMask;
#if GEMPIC_SPACEDIM == 1
    std::unique_ptr<amrex::iMultiFab> cellMasks;
#else
    amrex::Array<amrex::iMultiFab, GEMPIC_SPACEDIM> edgeMask;
#endif
#if GEMPIC_SPACEDIM == 2
    std::unique_ptr<amrex::iMultiFab> cellMask;
#endif
#if GEMPIC_SPACEDIM == 3
    amrex::Array<amrex::iMultiFab, GEMPIC_SPACEDIM> faceMask;
    amrex::iMultiFab cellMask;
#endif
*/

public:
    /// Constructor for the DeRhamComplex class
    /// @param params The parameters needed to construct the class correctly
    DeRhamComplex(Parameters params)
    {
        realBox = params.realBox();
        nCell = params.nCell();
        isPeriodic = params.isPeriodic();
        grid = params.grid();
        distriMap = params.distriMap();
        geom = params.geometry();
        m_hodgeDegree = params.degree();

        /* Ghost cells are decided for the fields by the degree of the Hodge.
        ** Needs to be changed for particles and decided by spline degree:
        ** to max(degx, degy, degz) + safety margin to let particles move
        **
        ** Example: for linear splines (deg_i = 1) 1 ghost cell is needed.
        ** 3 more ghost cells are added to let particles move.
        ** */
        switch (m_hodgeDegree)
        {
            case 2:
                nGhost = amrex::IntVect{AMREX_D_DECL(1, 1, 1)};
                break;
            case 4:
                nGhost = amrex::IntVect{AMREX_D_DECL(2, 2, 2)};
                break;
            case 6:
                nGhost = amrex::IntVect{AMREX_D_DECL(3, 3, 3)};
                break;
        }

        // set index: // 0 -> cell-centered | 1 -> node-centered
        primalNodeIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 1)});
        primalEdgeIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 1)});
        primalEdgeIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 1)});
        primalEdgeIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 1, 0)});
        primalFaceIndex[xDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(1, 0, 0)});
        primalFaceIndex[yDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 1, 0)});
        primalFaceIndex[zDir] = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 1)});
        primalCellIndex = amrex::IndexType(amrex::IntVect{AMREX_D_DECL(0, 0, 0)});

        // Set masks. Uncommented for now as we are using averageSync()
        /*
        nodeMask.define(amrex::convert(grid, primalNodeIndex), distriMap, nComp, nGhost[xDir]); 
        edgeMask[xDir].define(amrex::convert(grid, primalEdgeIndex[xDir]), distriMap, nComp, nGhost[xDir]); 
        edgeMask[yDir].define(amrex::convert(grid, primalEdgeIndex[yDir]), distriMap, nComp, nGhost[yDir]); 
        edgeMask[zDir].define(amrex::convert(grid, primalEdgeIndex[zDir]), distriMap, nComp, nGhost[zDir]); 
        faceMask[xDir].define(amrex::convert(grid, primalFaceIndex[xDir]), distriMap, nComp, nGhost[xDir]);
        faceMask[yDir].define(amrex::convert(grid, primalFaceIndex[yDir]), distriMap, nComp, nGhost[yDir]);
        faceMask[zDir].define(amrex::convert(grid, primalFaceIndex[zDir]), distriMap, nComp, nGhost[zDir]);
        cellMask.define(amrex::convert(grid, primalCellIndex), deistriMap, nComp, nGhost[xDir]); 

        for (int comp = 0; comp < 3; ++comp)
        {
            // Sets the value of the mask vector to 1
            edgeMask[comp].setVal(0,nGhost[comp]);


            // For the edges of the BoxArray in which the MultiFab is defined, set value to zero
            for (amrex::MFIter mfi(edgeMask[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const amrex::IntVect hi = bx.bigEnd();
                amrex::Array4<int> const &maskArr = edgeMask[comp][mfi].array(); 

                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        if ((i <= (comp == xDir ? hi[xDir] : (hi[xDir] -1))) &&
                            (j <= (comp == yDir ? hi[yDir] : (hi[yDir] -1))) &&
                            (k <= (comp == zDir ? hi[zDir] : (hi[zDir] -1))))
                               maskArr(i, j, k) = 1;
                    });
            }

            edgeMask[comp].FillBoundary(params.geometry().periodicity());

        }

        for (int comp = 0; comp < 3; ++comp)
        {
            // Sets the value of the mask vector to 1
            faceMask[comp].setVal(0,nGhost[comp]);


            // For the edges of the BoxArray in which the MultiFab is defined, set value to zero
            for (amrex::MFIter mfi(faceMask[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const amrex::IntVect hi = bx.bigEnd();

                amrex::Array4<int> const &maskArr = faceMask[comp][mfi].array(); 

                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        if ((i <= (comp == xDir ? (hi[xDir] -1) : hi[xDir])) &&
                            (j <= (comp == yDir ? (hi[yDir] -1) : hi[yDir])) &&
                            (k <= (comp == zDir ? (hi[zDir] -1) : hi[zDir])))
                               maskArr(i, j, k) = 1;
                    });
            }

            faceMask[comp].FillBoundary(params.geometry().periodicity());

        }

        */
    }

    virtual ~DeRhamComplex() {}

    /// Getter for the number of ghost cells
    const amrex::IntVect& getNGhost() const
    {
        return nGhost;
    }

    void initializeField(DeRhamField<Grid::primal, Space::node>& field)
    {
        field.data.define(amrex::convert(grid, primalNodeIndex) , distriMap, 1, nGhost);
        field.data.setVal(0.0);
    }

    // Function overloads for the initializations
    void initializeField(DeRhamField<Grid::primal, Space::edge>& field)
    {
        field.data[xDir].define(amrex::convert(grid, primalEdgeIndex[xDir]) , distriMap, 1, nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(grid, primalEdgeIndex[yDir]) , distriMap, 1, nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(grid, primalEdgeIndex[zDir]) , distriMap, 1, nGhost);
        field.data[zDir].setVal(0.0);
    }


    void initializeField(DeRhamField<Grid::primal, Space::face>& field)
    {
        field.data[xDir].define(amrex::convert(grid, primalFaceIndex[xDir]) , distriMap, 1, nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(grid, primalFaceIndex[yDir]) , distriMap, 1, nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(grid, primalFaceIndex[zDir]) , distriMap, 1, nGhost);
        field.data[zDir].setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::primal, Space::cell>& field)
    {
        field.data.define(amrex::convert(grid, primalCellIndex) , distriMap, 1, nGhost);
        field.data.setVal(0.0);
    }


    // Centering is always set by the primal grid
    void initializeField(DeRhamField<Grid::dual, Space::node>& field)
    {
        field.data.define(amrex::convert(grid, primalCellIndex) , distriMap, 1, nGhost);
        field.data.setVal(0.0);
    }


    void initializeField(DeRhamField<Grid::dual, Space::edge>& field)
    {
        field.data[xDir].define(amrex::convert(grid, primalFaceIndex[xDir]) , distriMap, 1, nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(grid, primalFaceIndex[yDir]) , distriMap, 1, nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(grid, primalFaceIndex[zDir]) , distriMap, 1, nGhost);
        field.data[zDir].setVal(0.0);
    }


    void initializeField(DeRhamField<Grid::dual, Space::face>& field)
    {
        field.data[xDir].define(amrex::convert(grid, primalEdgeIndex[xDir]) , distriMap, 1, nGhost);
        field.data[xDir].setVal(0.0);
        field.data[yDir].define(amrex::convert(grid, primalEdgeIndex[yDir]) , distriMap, 1, nGhost);
        field.data[yDir].setVal(0.0);
        field.data[zDir].define(amrex::convert(grid, primalEdgeIndex[zDir]) , distriMap, 1, nGhost);
        field.data[zDir].setVal(0.0);
    }

    void initializeField(DeRhamField<Grid::dual, Space::cell>& field)
    {
        field.data.define(amrex::convert(grid, primalNodeIndex) , distriMap, 1, nGhost);
        field.data.setVal(0.0);
    }




    // Function overloads for fill boundary
    void fillBoundary(DeRhamField<Grid::primal, Space::edge>& field)
    {
        field.data[xDir].FillBoundary(geom.periodicity());
        field.data[yDir].FillBoundary(geom.periodicity());
        field.data[zDir].FillBoundary(geom.periodicity());
    }
   

    void fillBoundary(DeRhamField<Grid::primal, Space::face>& field)
    {
        field.data[xDir].FillBoundary(geom.periodicity());
        field.data[yDir].FillBoundary(geom.periodicity());
        field.data[zDir].FillBoundary(geom.periodicity());
    }


    void fillBoundary(DeRhamField<Grid::dual, Space::edge>& field)
    {
        field.data[xDir].FillBoundary(geom.periodicity());
        field.data[yDir].FillBoundary(geom.periodicity());
        field.data[zDir].FillBoundary(geom.periodicity());
    }
   

    void fillBoundary(DeRhamField<Grid::dual, Space::face>& field)
    {
        field.data[xDir].FillBoundary(geom.periodicity());
        field.data[yDir].FillBoundary(geom.periodicity());
        field.data[zDir].FillBoundary(geom.periodicity());
    }

    void fillBoundary(DeRhamField<Grid::primal, Space::node>& field)
    {
        field.data.FillBoundary(geom.periodicity());
    }


    void fillBoundary(DeRhamField<Grid::primal, Space::cell>& field)
    {
        field.data.FillBoundary(geom.periodicity());
    }


    void fillBoundary(DeRhamField<Grid::dual, Space::node>& field)
    {
        field.data.FillBoundary(geom.periodicity());
    }


    void fillBoundary(DeRhamField<Grid::dual, Space::cell>& field)
    {
        field.data.FillBoundary(geom.periodicity());
    }
    /*
    // Function overloads for overridesync
    void overrideSync(DeRhamField<Grid::primal, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[xDir].OverrideSync(edgeMask[xDir], geom.periodicity());
        field.data[yDir].OverrideSync(edgeMask[yDir], geom.periodicity());
        field.data[zDir].OverrideSync(edgeMask[zDir], geom.periodicity());
#endif
    }


    void overrideSync(DeRhamField<Grid::primal, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[xDir].OverrideSync(faceMask[xDir], geom.periodicity());
        field.data[yDir].OverrideSync(faceMask[yDir], geom.periodicity());
        field.data[zDir].OverrideSync(faceMask[zDir], geom.periodicity());
#endif
    }

    void overrideSync(DeRhamField<Grid::dual, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[xDir].OverrideSync(edgeMask[xDir], geom.periodicity());
        field.data[yDir].OverrideSync(edgeMask[yDir], geom.periodicity());
        field.data[zDir].OverrideSync(edgeMask[zDir], geom.periodicity());
#endif
    }


    void overrideSync(DeRhamField<Grid::dual, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[xDir].OverrideSync(faceMask[xDir], geom.periodicity());
        field.data[yDir].OverrideSync(faceMask[yDir], geom.periodicity());
        field.data[zDir].OverrideSync(faceMask[zDir], geom.periodicity());
#endif
    }

    void overrideSync(DeRhamField<Grid::primal, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.OverrideSync(nodeMask, geom.periodicity());
#endif
    }
    */

    // Function overloads for averagesync
    void averageSync(DeRhamField<Grid::primal, Space::node>& field)
    {
        field.data.AverageSync(geom.periodicity());
    }

    void averageSync(DeRhamField<Grid::primal, Space::edge>& field)
    {
        field.data[xDir].AverageSync(geom.periodicity());
        field.data[yDir].AverageSync(geom.periodicity());
        field.data[zDir].AverageSync(geom.periodicity());
    }

    void averageSync(DeRhamField<Grid::primal, Space::face>& field)
    {
        field.data[xDir].AverageSync(geom.periodicity());
        field.data[yDir].AverageSync(geom.periodicity());
        field.data[zDir].AverageSync(geom.periodicity());
    }

    void averageSync(DeRhamField<Grid::primal, Space::cell>& field)
    {
        field.data.AverageSync(geom.periodicity());
    }

    void averageSync(DeRhamField<Grid::dual, Space::node>& field)
    {
        field.data.AverageSync(geom.periodicity());
    }

    void averageSync(DeRhamField<Grid::dual, Space::edge>& field)
    {
        field.data[xDir].AverageSync(geom.periodicity());
        field.data[yDir].AverageSync(geom.periodicity());
        field.data[zDir].AverageSync(geom.periodicity());
    }


    void averageSync(DeRhamField<Grid::dual, Space::face>& field)
    {
        field.data[xDir].AverageSync(geom.periodicity());
        field.data[yDir].AverageSync(geom.periodicity());
        field.data[zDir].AverageSync(geom.periodicity());
    }


    void averageSync(DeRhamField<Grid::dual, Space::cell>& field)
    {
        field.data.AverageSync(geom.periodicity());
    }

//---------------------
// exterior derivatives
//---------------------
/*
#if GEMPIC_SPACEDIM == 1
    void grad(const DeRhamField<Grid::primal, Space::node>&,
                            DeRhamField<Grid::primal, Space::cell>&);
    void grad(const DeRhamField<Grid::dual, Space::cell>&,
                            DeRhamField<Grid::dual, Space::node>&);
#endif
#if GEMPIC_SPACEDIM == 2
    // grad-curl complex
    void grad(const DeRhamField<Grid::primal, Space::node>&,
                            DeRhamField<Grid::primal, Space::edge>&);
    void curl(const DeRhamField<Grid::primal, Space::edge>&,
                            DeRhamField<Grid::primal, Space::cell>&);
    void rot(const DeRhamField<Grid::dual, Space::cell>&,
                            DeRhamField<Grid::dual, Space::edge>&);
    void div(const DeRhamField<Grid::dual, Space::edge>&,
                            DeRhamField<Grid::dual, Space::node>&);
    // rot-div complex
    void rot(const DeRhamField<Grid::primal, Space::node>&,
                            DeRhamField<Grid::primal, Space::face>&);
    void div(const DeRhamField<Grid::primal, Space::face>&,
                            DeRhamField<Grid::primal, Space::cell>&);
    void grad(const DeRhamField<Grid::dual, Space::node>&,
                            DeRhamField<Grid::dual, Space::edge>&);
    void curl(const DeRhamField<Grid::dual, Space::face>&,
                            DeRhamField<Grid::dual, Space::node>&);
#endif
*/
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


    //------------------
    // Getter for the Masks 
    // -----------------
/*
#if GEMPIC_SPACEDIM == 3 
    amrex::iMultiFab getEdgeMask(const int comp) const
    {
        amrex::iMultiFab alias_edgeMask(edgeMask[comp], amrex::make_alias, 0, 3);
        return alias_edgeMask;
    }

#endif
*/
};

// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
inline DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

template <Grid grid>
inline DeRhamField<grid, Space::node>::DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

template <Grid grid>
inline DeRhamField<grid, Space::cell>::DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <Grid grid, Space space>
inline DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc, const amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func)
    : deRham{drc}
{
    deRham -> initializeField(*this);
    deRham->projection(func, 0.0, *this);
}

template <Grid grid>
inline DeRhamField<grid, Space::node>::DeRhamField(std::shared_ptr<DeRhamComplex> drc, const 
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func)
    : deRham{drc}
{
    deRham -> initializeField(*this);
    deRham -> projection(func, 0.0, *this);
}

template <Grid grid>
inline DeRhamField<grid, Space::cell>::DeRhamField(std::shared_ptr<DeRhamComplex> drc, const 
    amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func)
    : deRham{drc}
{
    deRham -> initializeField(*this);
    deRham -> projection(func, 0.0, *this);
}

// FillBoundary of DeRhamFields following the same logic as above
template <Grid grid, Space space>
inline void DeRhamField<grid, space>::fillBoundary()
{
    deRham -> fillBoundary(*this);
}

template <Grid grid>
inline void DeRhamField<grid, Space::node>::fillBoundary()
{
    deRham -> fillBoundary(*this);
}

template <Grid grid>
inline void DeRhamField<grid, Space::cell>::fillBoundary()
{
    deRham -> fillBoundary(*this);
}

/*
// OverrideSync of DeRhamFields
template <Grid grid, Space space>
inline void DeRhamField<grid, space>::overrideSync()
{
    deRham -> overrideSync(*this);
}

template <Grid grid>
inline void DeRhamField<grid, Space::node>::overrideSync()
{
    deRham -> overrideSync(*this);
}
*/


// AverageSync of DeRhamFields
template <Grid grid, Space space>
inline void DeRhamField<grid, space>::averageSync()
{
    deRham -> averageSync(*this);
}

template <Grid grid>
inline void DeRhamField<grid, Space::node>::averageSync()
{
    deRham -> averageSync(*this);
}

template <Grid grid>
inline void DeRhamField<grid, Space::cell>::averageSync()
{
    deRham -> averageSync(*this);
}

// Multiply by scalar
template <Grid grid, Space space>
inline void DeRhamField<grid, space>::multEqual(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].mult(dt, nGhost[xDir]);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space>::operator*=(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].mult(dt,nGhost[xDir]);
}

template <Grid grid>
inline void DeRhamField<grid, Space::node>::multEqual(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    data.mult(dt, nGhost[xDir]);
}

template <Grid grid>
inline void DeRhamField<grid, Space::cell>::multEqual(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    data.mult(dt, nGhost[xDir]);
}

// plusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space>::plusEqual(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].plus(form.data[comp], 0, 1, nGhost[xDir]);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space>::operator+=(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].plus(form.data[comp], 0, 1, nGhost[xDir]);
}


// minusEqual
template <Grid grid, Space space>
inline void DeRhamField<grid, space>::minusEqual(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].minus(form.data[comp], 0, 1, nGhost[xDir]);
}

template <Grid grid, Space space>
inline void DeRhamField<grid, space>::operator-=(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        data[comp].minus(form.data[comp], 0, 1, nGhost[xDir]);
}

}


// namespace GEMPIC_Fields

#endif

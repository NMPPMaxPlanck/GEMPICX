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
#include <GEMPIC_Params.H>

namespace GEMPIC_Fields
{
enum Grid
{
    primal,
    dual
};

enum Space
{
    node,
    edge,
    face,
    cell
};

class DeRhamComplex;
/// DeRhamField
///
/// Stores a specific element of a deRham complex
///
/// @p gridType: Grid::primal or Grid::dual grid
///
/// @p spaceType: Space::node, Space::edge, Space::face or Space::cell.
///
/// spaceType is defined on primal grid:
/// e.g. DeRhamField<dual, node> is the cell based complex dual to DeRhamField<primal, node>
///
/// In 1D only Space::node and Space::cell exist.
///
/// In 2D there are 4 complexes because of the two different deRham sequences
///
/// For node and cell only scalarData will be allocated, for edge and face only vector data

template <int grid, int space>
struct DeRhamField
{
    std::shared_ptr<DeRhamComplex> deRham;
    amrex::Array<amrex::MultiFab, GEMPIC_SPACEDIM> data;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);

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

template <int grid>
struct DeRhamField <grid, Space::node>
{
    amrex::MultiFab data;
    std::shared_ptr<DeRhamComplex> deRham;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
    void fillBoundary();
    void overrideSync();
    void averageSync();
    void multEqual(const amrex::Real& dt);
};

template <int grid> 
struct DeRhamField <grid, Space::cell>
{
    amrex::MultiFab data;
    std::shared_ptr<DeRhamComplex> deRham;

    DeRhamField(std::shared_ptr<DeRhamComplex> drc);
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
    amrex::Array<int, GEMPIC_SPACEDIM> nGhost;

    // Index and Mask
    amrex::IndexType primalNodeIndex;
    amrex::iMultiFab nodeMask;
#if GEMPIC_SPACEDIM == 1
    amrex::IndexType primalCellIndex;
    std::unique_ptr<amrex::iMultiFab> cellMasks;
#else
    amrex::Array<amrex::IndexType, GEMPIC_SPACEDIM> primalEdgeIndex;
    amrex::Array<amrex::iMultiFab, GEMPIC_SPACEDIM> edgeMask;
#endif
#if GEMPIC_SPACEDIM == 2
    amrex::IndexType(amrex::IntVect) primalCellIndex;
    std::unique_ptr<amrex::iMultiFab> cellMask;
#endif
#if GEMPIC_SPACEDIM == 3
    amrex::Array<amrex::IndexType, GEMPIC_SPACEDIM> primalFaceIndex;
    amrex::Array<amrex::iMultiFab, GEMPIC_SPACEDIM> faceMask;
    amrex::IndexType primalCellIndex;
    amrex::iMultiFab cellMask;
#endif

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

        // set index: // 0 -> cell-centered | 1 -> node-centered
#if GEMPIC_SPACEDIM == 1
        primalNodeIndex = amrex::IndexType(amrex::IntVect(1));
        primalCellIndex = amrex::IndexType(amrex::IntVect(0));
#endif
#if GEMPIC_SPACEDIM == 2
        primalNodeIndex = amrex::IndexType(amrex::IntVect(1, 1));
        primalEdgeIndex[0] = amrex::IndexType(amrex::IntVect(0, 1));
        primalEdgeIndex[1] = amrex::IndexType(amrex::IntVect(1, 0));
        primalFaceIndex[0] = amrex::IndexType(amrex::IntVect(1, 0));
        primalFaceIndex[1] = amrex::IndexType(amrex::IntVect(0, 1));
        primalCellIndex = amrex::IndexType(amrex::IntVect(0, 0));
#endif
#if GEMPIC_SPACEDIM == 3
        switch (m_hodgeDegree)
        {
            case 2:
                nGhost = {1,1,1}; // one ghost cell always needed for differential operators
                break;
            case 4:
                nGhost = {1,1,1};
                break;
            case 6:
                nGhost = {2,2,2};
                break;
        }

        primalNodeIndex = amrex::IndexType(amrex::IntVect(1, 1, 1));
        primalEdgeIndex[0] = amrex::IndexType(amrex::IntVect(0, 1, 1));
        primalEdgeIndex[1] = amrex::IndexType(amrex::IntVect(1, 0, 1));
        primalEdgeIndex[2] = amrex::IndexType(amrex::IntVect(1, 1, 0));
        primalFaceIndex[0] = amrex::IndexType(amrex::IntVect(1, 0, 0));
        primalFaceIndex[1] = amrex::IndexType(amrex::IntVect(0, 1, 0));
        primalFaceIndex[2] = amrex::IndexType(amrex::IntVect(0, 0, 1));
        primalCellIndex = amrex::IndexType(amrex::IntVect(0, 0, 0));

        // Set masks. Uncommented for now as we are using averageSync()
        /*
        nodeMask.define(amrex::convert(grid, primalNodeIndex), distriMap, nComp, nGhost[0]); 
        edgeMask[0].define(amrex::convert(grid, primalEdgeIndex[0]), distriMap, nComp, nGhost[0]); 
        edgeMask[1].define(amrex::convert(grid, primalEdgeIndex[1]), distriMap, nComp, nGhost[1]); 
        edgeMask[2].define(amrex::convert(grid, primalEdgeIndex[2]), distriMap, nComp, nGhost[2]); 
        faceMask[0].define(amrex::convert(grid, primalFaceIndex[0]), distriMap, nComp, nGhost[0]);
        faceMask[1].define(amrex::convert(grid, primalFaceIndex[1]), distriMap, nComp, nGhost[1]);
        faceMask[2].define(amrex::convert(grid, primalFaceIndex[2]), distriMap, nComp, nGhost[2]);
        cellMask.define(amrex::convert(grid, primalCellIndex), deistriMap, nComp, nGhost[0]); 

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
                        if ((i <= (comp == 0 ? hi[0] : (hi[0] -1))) &&
                            (j <= (comp == 1 ? hi[1] : (hi[1] -1))) &&
                            (k <= (comp == 2 ? hi[2] : (hi[2] -1))))
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
                        if ((i <= (comp == 0 ? (hi[0] -1) : hi[0])) &&
                            (j <= (comp == 1 ? (hi[1] -1) : hi[1])) &&
                            (k <= (comp == 2 ? (hi[2] -1) : hi[2])))
                               maskArr(i, j, k) = 1;
                    });
            }

            faceMask[comp].FillBoundary(params.geometry().periodicity());

        }

        */
#endif
    }

    virtual ~DeRhamComplex() {}

    /// Getter for the number of ghost cells
    const amrex::Array<int, GEMPIC_SPACEDIM>& getNGhost() const
    {
        return nGhost;
    }

    void initializeField(DeRhamField<Grid::primal, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data.define(amrex::convert(grid, primalNodeIndex) , distriMap, nComp, nGhost[0]);
#endif
    }

    // Function overloads for the initializations
    void initializeField(DeRhamField<Grid::primal, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data[0].define(amrex::convert(grid, primalEdgeIndex[0]) , distriMap, nComp, nGhost[0]);
        field.data[1].define(amrex::convert(grid, primalEdgeIndex[1]) , distriMap, nComp, nGhost[1]);
        field.data[2].define(amrex::convert(grid, primalEdgeIndex[2]) , distriMap, nComp, nGhost[2]);
#endif
    }


    void initializeField(DeRhamField<Grid::primal, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data[0].define(amrex::convert(grid, primalFaceIndex[0]) , distriMap, nComp, nGhost[0]);
        field.data[1].define(amrex::convert(grid, primalFaceIndex[1]) , distriMap, nComp, nGhost[1]);
        field.data[2].define(amrex::convert(grid, primalFaceIndex[2]) , distriMap, nComp, nGhost[2]);
#endif
    }

    void initializeField(DeRhamField<Grid::primal, Space::cell>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data.define(amrex::convert(grid, primalCellIndex) , distriMap, nComp, nGhost[0]);
#endif
    }


    // Centering is always set by the primal grid
    void initializeField(DeRhamField<Grid::dual, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data.define(amrex::convert(grid, primalCellIndex) , distriMap, nComp, nGhost[0]);
#endif
    }


    void initializeField(DeRhamField<Grid::dual, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data[0].define(amrex::convert(grid, primalFaceIndex[0]) , distriMap, nComp, nGhost[0]);
        field.data[1].define(amrex::convert(grid, primalFaceIndex[1]) , distriMap, nComp, nGhost[1]);
        field.data[2].define(amrex::convert(grid, primalFaceIndex[2]) , distriMap, nComp, nGhost[2]);
#endif
    }


    void initializeField(DeRhamField<Grid::dual, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data[0].define(amrex::convert(grid, primalEdgeIndex[0]) , distriMap, nComp, nGhost[0]);
        field.data[1].define(amrex::convert(grid, primalEdgeIndex[1]) , distriMap, nComp, nGhost[1]);
        field.data[2].define(amrex::convert(grid, primalEdgeIndex[2]) , distriMap, nComp, nGhost[2]);
#endif
    }

    void initializeField(DeRhamField<Grid::dual, Space::cell>& field)
    {
#if GEMPIC_SPACEDIM == 3
        const int nComp = 1;

        field.data.define(amrex::convert(grid, primalNodeIndex) , distriMap, nComp, nGhost[0]);
#endif
    }

    


    // Function overloads for fill boundary
    void fillBoundary(DeRhamField<Grid::primal, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].FillBoundary(geom.periodicity());
        field.data[1].FillBoundary(geom.periodicity());
        field.data[2].FillBoundary(geom.periodicity());
#endif
    }
   

    void fillBoundary(DeRhamField<Grid::primal, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].FillBoundary(geom.periodicity());
        field.data[1].FillBoundary(geom.periodicity());
        field.data[2].FillBoundary(geom.periodicity());
#endif
    }


    void fillBoundary(DeRhamField<Grid::dual, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].FillBoundary(geom.periodicity());
        field.data[1].FillBoundary(geom.periodicity());
        field.data[2].FillBoundary(geom.periodicity());
#endif
    }
   

    void fillBoundary(DeRhamField<Grid::dual, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].FillBoundary(geom.periodicity());
        field.data[1].FillBoundary(geom.periodicity());
        field.data[2].FillBoundary(geom.periodicity());
#endif
    }

    void fillBoundary(DeRhamField<Grid::primal, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.FillBoundary(geom.periodicity());
#endif
    }


    void fillBoundary(DeRhamField<Grid::primal, Space::cell>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.FillBoundary(geom.periodicity());
#endif
    }


    void fillBoundary(DeRhamField<Grid::dual, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.FillBoundary(geom.periodicity());
#endif
    }


    void fillBoundary(DeRhamField<Grid::dual, Space::cell>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.FillBoundary(geom.periodicity());
#endif
    }
    /*
    // Function overloads for overridesync
    void overrideSync(DeRhamField<primal, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].OverrideSync(edgeMask[0], geom.periodicity());
        field.data[1].OverrideSync(edgeMask[1], geom.periodicity());
        field.data[2].OverrideSync(edgeMask[2], geom.periodicity());
#endif
    }


    void overrideSync(DeRhamField<primal, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].OverrideSync(faceMask[0], geom.periodicity());
        field.data[1].OverrideSync(faceMask[1], geom.periodicity());
        field.data[2].OverrideSync(faceMask[2], geom.periodicity());
#endif
    }

    void overrideSync(DeRhamField<dual, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].OverrideSync(edgeMask[0], geom.periodicity());
        field.data[1].OverrideSync(edgeMask[1], geom.periodicity());
        field.data[2].OverrideSync(edgeMask[2], geom.periodicity());
#endif
    }


    void overrideSync(DeRhamField<dual, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].OverrideSync(faceMask[0], geom.periodicity());
        field.data[1].OverrideSync(faceMask[1], geom.periodicity());
        field.data[2].OverrideSync(faceMask[2], geom.periodicity());
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
    void averageSync(DeRhamField<primal, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.AverageSync(geom.periodicity());
#endif
    }

    void averageSync(DeRhamField<primal, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].AverageSync(geom.periodicity());
        field.data[1].AverageSync(geom.periodicity());
        field.data[2].AverageSync(geom.periodicity());
#endif
    }


    void averageSync(DeRhamField<primal, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].AverageSync(geom.periodicity());
        field.data[1].AverageSync(geom.periodicity());
        field.data[2].AverageSync(geom.periodicity());
#endif
    }

    void averageSync(DeRhamField<primal, Space::cell>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.AverageSync(geom.periodicity());
#endif
    }

    void averageSync(DeRhamField<dual, Space::node>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.AverageSync(geom.periodicity());
#endif
    }

    void averageSync(DeRhamField<dual, Space::edge>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].AverageSync(geom.periodicity());
        field.data[1].AverageSync(geom.periodicity());
        field.data[2].AverageSync(geom.periodicity());
#endif
    }


    void averageSync(DeRhamField<dual, Space::face>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data[0].AverageSync(geom.periodicity());
        field.data[1].AverageSync(geom.periodicity());
        field.data[2].AverageSync(geom.periodicity());
#endif
    }


    void averageSync(DeRhamField<dual, Space::cell>& field)
    {
#if GEMPIC_SPACEDIM == 3
        field.data.AverageSync(geom.periodicity());
#endif
    }


//---------------------
// exterior derivatives
//---------------------
#if GEMPIC_SPACEDIM == 1
    void grad(const DeRhamField<primal, Space::node>&,
                            DeRhamField<primal, Space::cell>&);
    void grad(const DeRhamField<dual, Space::cell>&,
                            DeRhamField<dual, Space::node>&);
#endif
#if GEMPIC_SPACEDIM == 2
    // grad-curl complex
    void grad(const DeRhamField<primal, Space::node>&,
                            DeRhamField<primal, Space::edge>&);
    void curl(const DeRhamField<primal, Space::edge>&,
                            DeRhamField<primal, Space::cell>&);
    void rot(const DeRhamField<dual, Space::cell>&,
                            DeRhamField<dual, Space::edge>&);
    void div(const DeRhamField<dual, Space::edge>&,
                            DeRhamField<dual, Space::node>&);
    // rot-div complex
    void rot(const DeRhamField<primal, Space::node>&,
                            DeRhamField<primal, Space::face>&);
    void div(const DeRhamField<primal, Space::face>&,
                            DeRhamField<primal, Space::cell>&);
    void grad(const DeRhamField<dual, Space::node>&,
                            DeRhamField<dual, Space::edge>&);
    void curl(const DeRhamField<dual, Space::face>&,
                            DeRhamField<dual, Space::node>&);
#endif
#if GEMPIC_SPACEDIM == 3
    void grad(const DeRhamField<primal, Space::node>&,
                            DeRhamField<primal, Space::edge>&);
    void curl(const DeRhamField<primal, Space::edge>&,
                            DeRhamField<primal, Space::face>&);
    void div(const DeRhamField<primal, Space::face>&,
                            DeRhamField<primal, Space::cell>&);
    void grad(const DeRhamField<dual, Space::node>&,
                            DeRhamField<dual, Space::edge>&);
    void curl(const DeRhamField<dual, Space::edge>&,
                            DeRhamField<dual, Space::face>&);
    void div(const DeRhamField<dual, Space::face>&,
                            DeRhamField<dual, Space::cell>&);
#endif

    //------------
    // Projections
    //------------
    
    virtual void projection(amrex::ParserExecutor<4> func, amrex::Real t,
                    DeRhamField<primal, Space::node>&) = 0;
    virtual void projection(amrex::ParserExecutor<4> func, amrex::Real t,
                    DeRhamField<primal, Space::cell>&) = 0;
    virtual void projection(amrex::ParserExecutor<4> func, amrex::Real t,
                    DeRhamField<dual, Space::node>&) = 0;
    virtual void projection(amrex::ParserExecutor<4> func, amrex::Real t,
                    DeRhamField<dual, Space::cell>&) = 0;
                    
#if GEMPIC_SPACEDIM > 1
    virtual void projection(amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                    amrex::Real t, DeRhamField<dual, Space::edge>&) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                    amrex::Real t, DeRhamField<primal, Space::face>&) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                    amrex::Real t, DeRhamField<primal, Space::edge>&) = 0;
    virtual void projection(amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                    amrex::Real t, DeRhamField<dual, Space::face>&) = 0;
                    
#endif

    //------------------
    // Getter for the Masks 
    // -----------------

#if GEMPIC_SPACEDIM == 3 
    amrex::iMultiFab getEdgeMask(const int comp) const
    {
        amrex::iMultiFab alias_edgeMask(edgeMask[comp], amrex::make_alias, 0, 3);
        return alias_edgeMask;
    }

#endif
};


// Constructors of DeRhamFields after DeRhamComplex to avoid using a member of an incomplete type
template <int grid, int space>
inline DeRhamField<grid, space>::DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

template <int grid>
inline DeRhamField<grid, Space::node>::DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

template <int grid>
inline DeRhamField<grid, Space::cell>::DeRhamField(std::shared_ptr<DeRhamComplex> drc)
    : deRham{drc}
{
    deRham -> initializeField(*this);
}

// FillBoundary of DeRhamFields following the same logic as above
template <int grid, int space>
inline void DeRhamField<grid, space>::fillBoundary()
{
    deRham -> fillBoundary(*this);
}

template <int grid>
inline void DeRhamField<grid, Space::node>::fillBoundary()
{
    deRham -> fillBoundary(*this);
}

template <int grid>
inline void DeRhamField<grid, Space::cell>::fillBoundary()
{
    deRham -> fillBoundary(*this);
}

/*
// OverrideSync of DeRhamFields
template <int grid, int space>
inline void DeRhamField<grid, space>::overrideSync()
{
    deRham -> overrideSync(*this);
}

template <int grid>
inline void DeRhamField<grid, Space::node>::overrideSync()
{
    deRham -> overrideSync(*this);
}
*/


// AverageSync of DeRhamFields
template <int grid, int space>
inline void DeRhamField<grid, space>::averageSync()
{
    deRham -> averageSync(*this);
}

template <int grid>
inline void DeRhamField<grid, Space::node>::averageSync()
{
    deRham -> averageSync(*this);
}

template <int grid>
inline void DeRhamField<grid, Space::cell>::averageSync()
{
    deRham -> averageSync(*this);
}

// Multiply by scalar
template <int grid, int space>
inline void DeRhamField<grid, space>::multEqual(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
        data[comp].mult(dt, nGhost[0]);
}

template <int grid, int space>
inline void DeRhamField<grid, space>::operator*=(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
        data[comp].mult(dt,nGhost[0]);
}

template <int grid>
inline void DeRhamField<grid, Space::node>::multEqual(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    data.mult(dt, nGhost[0]);
}

template <int grid>
inline void DeRhamField<grid, Space::cell>::multEqual(const amrex::Real& dt)
{
    auto nGhost = deRham -> getNGhost();
    data.mult(dt, nGhost[0]);
}

// plusEqual
template <int grid, int space>
inline void DeRhamField<grid, space>::plusEqual(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        ((this) -> data[comp]).plus(form.data[comp], 0, 1, nGhost[comp]);
}

template <int grid, int space>
inline void DeRhamField<grid, space>::operator+=(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        ((this) -> data[comp]).plus(form.data[comp], 0, 1, nGhost[comp]);
}


// minusEqual
template <int grid, int space>
inline void DeRhamField<grid, space>::minusEqual(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        ((this) -> data[comp]).minus(form.data[comp], 0, 1, nGhost[comp]);
}

template <int grid, int space>
inline void DeRhamField<grid, space>::operator-=(DeRhamField<grid, space>& form)
{
    auto nGhost = deRham -> getNGhost();
    for (int comp = 0; comp < 3; ++comp)
        ((this) -> data[comp]).minus(form.data[comp], 0, 1, nGhost[comp]);
}

}  


// namespace GEMPIC_Fields

#endif

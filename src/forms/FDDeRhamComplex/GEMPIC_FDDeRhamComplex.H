#ifndef GEMPIC_FDDERHAMCOMPLEX_H_
#define GEMPIC_FDDERHAMCOMPLEX_H_

#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>

using namespace GEMPIC_Fields;

namespace GEMPIC_FDDeRhamComplex {

template <int degree, int scheme, int stencilLength>
inline std::tuple<amrex::GpuArray<amrex::Real, stencilLength>, amrex::GpuArray<amrex::Real, stencilLength>> getHodgeStencils()
{
    /**
    * @brief Fills the arrays of Hodge stencils.
    * 
    * Description:
    * ----------
    * Computes stencils for periodic boundary conditions for either
    * finite difference of Galerkin Hodge
    * 
    * 
    * Parameters:
    * @tparam degree : int, degree of basis functions used
    * @tparam scheme : int, 0 for FDHodge, 1 for Galerkin Hodge
    * @tparam stencilLength : int, total length of the Hodge stencil
    * 
    * Returns:
    * ----------
    * Returns std::tuple : contains the stencils in the following order:
    *                      stencilNodeToCell, stencilCellToNode
    */
    amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
    amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
    switch (scheme)
    {
        case 0:
            switch (degree)
            {
                case 2:
                    stencilNodeToCell[0] = 1.0;
                    stencilCellToNode[0] = 1.0;
                    break;
                case 4:
                    stencilNodeToCell[0] = 1./8.;
                    stencilNodeToCell[1] = 6./8.;
                    stencilNodeToCell[2] = 1./8.;
                    stencilCellToNode[0] = -1./24.;
                    stencilCellToNode[1] = 13./12.;
                    stencilCellToNode[2] = -1./24.;
                    break;
                case 6:
                    stencilNodeToCell[0] = -7./384.;
                    stencilNodeToCell[1] = 44./384.;
                    stencilNodeToCell[2] = 310./384.;
                    stencilNodeToCell[3] = 44./384.; 
                    stencilNodeToCell[4] = -7./384; 
                    stencilCellToNode[0] = 3./640.;
                    stencilCellToNode[1] = -29./480.;
                    stencilCellToNode[2] = 1067./960.;
                    stencilCellToNode[3] = -29./480.;
                    stencilCellToNode[4] = 3./640.;
                    break;
                default:
                    amrex::Print() << "Degree not implemented" << std::endl;
                    break;
            }
            break;
        case 1:
            switch (degree)
            {
                case 2:
                    stencilCellToNode[0] = 1./8.;
                    stencilCellToNode[1] = 3./4.;
                    stencilCellToNode[2] = 1./8.;
                    stencilNodeToCell = stencilCellToNode;
                    break;
                case 4:
                    stencilCellToNode[0] = 121./46080.;
                    stencilCellToNode[1] = -959./23040.;
                    stencilCellToNode[2] = 4663./46080.;
                    stencilCellToNode[3] = 10087./11520.;
                    stencilCellToNode[4] = 4663./46080.;
                    stencilCellToNode[5] = -959./23040.;
                    stencilCellToNode[6] = 121./46080.;
                    stencilNodeToCell = stencilCellToNode;
                    break;
                case 6:
                    stencilCellToNode[0] = 365053./3715891200.;
                    stencilCellToNode[1] = -2981533./1857945600.;
                    stencilCellToNode[2] = 55805753./3715891200.;
                    stencilCellToNode[3] = -27050951./464486400.;
                    stencilCellToNode[4] = 147416293./1857945600.;
                    stencilCellToNode[5] = 864656441./928972800.;
                    stencilCellToNode[6] = 147416293./1857945600.;
                    stencilCellToNode[7] = -27050951./464486400.;
                    stencilCellToNode[8] = 55805753./3715891200.;
                    stencilCellToNode[9] = -2981533./1857945600.;
                    stencilCellToNode[10] = 365053./3715891200.;
                    stencilNodeToCell = stencilCellToNode;
                    break;
                default:
                    amrex::Print() << "Degree not implemented" << std::endl;
            }
            break;
        default:
            amrex::Print() << "Hodge not implemented" << std::endl;
            break;
    }

    return std::make_tuple(stencilNodeToCell, stencilCellToNode);
}


template <int stencilLength, int dimension>
inline void matrixMult(amrex::Geometry geom, amrex::GpuArray<amrex::Real, stencilLength> stencil,
                        const amrex::MultiFab& MultiInput, amrex::MultiFab& MultiOutput)
{
    /**
    * @brief Computes a matrix-vector product along 1 spatial direction.
    * 
    * Description:
    * ----------
    * Computes M*u for all edges in a given direction, where M is given
    * by a stencil and u are the values of the input along one edge.
    * 
    * 
    * Parameters:
    * @tparam stencilLength : int, length of the stencil array defining the matrix 
    * @tparam dimension : int, direction along which matrix vector product is calculated
    * @param geom : Geometry, needed for FillBoundary
    * @param stencil : GpuArray, defining the matrix
    * @param MultiInput : MultiFab, containing the values of the vector being multiplied
    * @param MultiOutput : MultiFab, containing the result
    * 
    * Returns:
    * ----------
    * Returns void
    */
    for (amrex::MFIter mfi(MultiOutput); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real const> const &vecIn = MultiInput[mfi].const_array();
        amrex::Array4<amrex::Real> const &vecOut = MultiOutput[mfi].array();
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real val = 0.0;

            for (int d = 0; d < stencilLength; d++)
            {
                int shift = d - (stencilLength + 1) / 2 + 1;
                val += stencil[d] * vecIn(i + (dimension == 0 ? shift : 0),
                                          j + (dimension == 1 ? shift : 0),
                                          k + (dimension == 2 ? shift : 0));
            }
            vecOut(i, j, k) = val;
        });
    }
    MultiOutput.FillBoundary(geom.periodicity());
}

/// FDDeRhamComplex
class FDDeRhamComplex : public GEMPIC_Fields::DeRhamComplex
{
    private:
        amrex::RealVect m_dr;
        amrex::Geometry m_geom;
        amrex::BoxArray m_grid;
        amrex::DistributionMapping m_distriMap;
        int m_nGhost;

        // Temporary 3D Array of MultiFab with the centeredness of the primal oneForm
        amrex::MultiFab m_tempPrimalZeroForm;
        amrex::MultiFab m_tempDualZeroForm;
        amrex::Array<amrex::MultiFab, 3> m_tempPrimalOneForm;
        amrex::Array<amrex::MultiFab, 3> m_tempDualOneForm;
        amrex::Array<amrex::MultiFab, 3> m_tempPrimalTwoForm;
        amrex::Array<amrex::MultiFab, 3> m_tempDualTwoForm;
        amrex::MultiFab m_tempPrimalThreeForm;
        amrex::MultiFab m_tempDualThreeForm;
        
    public:
        /// FDDeRhamComplex constructor
        /// @param Parameters
        FDDeRhamComplex(Parameters params);

        ~FDDeRhamComplex();

        void projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                        DeRhamField<primal, Space::node>& field) override;
        void projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                        DeRhamField<primal, Space::cell>& field) override;
        void projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                        DeRhamField<dual, Space::node>& field) override;
        void projection (amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::Real t,
                        DeRhamField<dual, Space::cell>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                        amrex::Real t, DeRhamField<dual, Space::edge>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                        amrex::Real t, DeRhamField<primal, Space::face>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                        amrex::Real t, DeRhamField<primal, Space::edge>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<GEMPIC_SPACEDIM + 1>, 3> func,
                        amrex::Real t, DeRhamField<dual, Space::face>& field) override;




        // Hodge_3 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::cell>& threeForm, DeRhamField<primal, Space::node>& zeroForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

#if (GEMPIC_SPACEDIM == 1)
            matrixMult<stencilLength, 0>(geom, stencilCellToNode, threeForm.data,zeroForm.data);
#endif
#if (GEMPIC_SPACEDIM == 2)
            matrixMult<stencilLength, 0>(geom, stencilCellToNode, threeForm.data,m_tempPrimalZeroForm);
            matrixMult<stencilLength, 1>(geom, stencilCellToNode, m_tempPrimalZeroForm, zeroForm.data);
#endif
#if (GEMPIC_SPACEDIM == 3)
            matrixMult<stencilLength, 0>(geom, stencilCellToNode, threeForm.data,zeroForm.data);
            matrixMult<stencilLength, 1>(geom, stencilCellToNode, zeroForm.data, m_tempPrimalZeroForm);
            matrixMult<stencilLength, 2>(geom, stencilCellToNode, m_tempPrimalZeroForm, zeroForm.data);
#endif
            
            zeroForm.data.mult(1 / (GEMPIC_D_MULT(dr[0], dr[1], dr[2]) * weight));

            zeroForm.averageSync();
            zeroForm.fillBoundary();
        }


        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::cell>& threeForm, DeRhamField<dual, Space::node>& zeroForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

#if (GEMPIC_SPACEDIM == 1)
            matrixMult<stencilLength, 0>(geom, stencilCellToNode, threeForm.data,zeroForm.data);
#endif
#if (GEMPIC_SPACEDIM == 2)
            matrixMult<stencilLength, 0>(geom, stencilCellToNode, threeForm.data,m_tempDualZeroForm);
            matrixMult<stencilLength, 1>(geom, stencilCellToNode, m_tempDualZeroForm, zeroForm.data);
#endif
#if (GEMPIC_SPACEDIM == 3)
            matrixMult<stencilLength, 0>(geom, stencilCellToNode, threeForm.data,zeroForm.data);
            matrixMult<stencilLength, 1>(geom, stencilCellToNode, zeroForm.data, m_tempDualZeroForm);
            matrixMult<stencilLength, 2>(geom, stencilCellToNode, m_tempDualZeroForm, zeroForm.data);
#endif
            
            zeroForm.data.mult(1 / (GEMPIC_D_MULT(dr[0], dr[1], dr[2]) * weight));

            zeroForm.averageSync();
            zeroForm.fillBoundary();
        }

        // Hodge_2 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::face>& twoForm, DeRhamField<primal, Space::edge>& oneForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

            for (int comp = 0; comp < 3; ++comp)
            {
#if (GEMPIC_SPACEDIM == 1)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilNodeToCell : stencilCellToNode), twoForm.data[comp], oneForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 2)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilNodeToCell : stencilCellToNode), twoForm.data[comp], m_tempPrimalOneForm[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilNodeToCell : stencilCellToNode), m_tempPrimalOneForm[comp], oneForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 3)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilNodeToCell : stencilCellToNode), twoForm.data[comp], oneForm.data[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilNodeToCell : stencilCellToNode), oneForm.data[comp], m_tempPrimalOneForm[comp]);
                matrixMult<stencilLength, 2>(geom, (comp == 2 ? stencilNodeToCell : stencilCellToNode), m_tempPrimalOneForm[comp], oneForm.data[comp]);
#endif
            
                if(comp == 0)
                    oneForm.data[comp].mult(dr[0] / (GEMPIC_D_MULT(1, dr[1], dr[2]) * weight));
                if(comp == 1)
                    oneForm.data[comp].mult(GEMPIC_D_MULT(1, dr[1], 1) / (GEMPIC_D_MULT(dr[0], 1, dr[2]) * weight));
                if(comp == 2)
                    oneForm.data[comp].mult(GEMPIC_D_MULT(1, 1, dr[2]) / (GEMPIC_D_MULT(dr[0], dr[1], 1) * weight));
            }

            oneForm.averageSync();
            oneForm.fillBoundary();
        }

        

        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::face>& twoForm, DeRhamField<dual, Space::edge>& oneForm, amrex::Real weight = 1.)
        {
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;

            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

            for (int comp = 0; comp < 3; ++comp)
            {
#if (GEMPIC_SPACEDIM == 1)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilNodeToCell : stencilCellToNode), twoForm.data[comp], oneForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 2)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilNodeToCell : stencilCellToNode), twoForm.data[comp], m_tempDualOneForm[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilNodeToCell : stencilCellToNode), m_tempDualOneForm[comp], oneForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 3)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilNodeToCell : stencilCellToNode), twoForm.data[comp], oneForm.data[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilNodeToCell : stencilCellToNode), oneForm.data[comp], m_tempDualOneForm[comp]);
                matrixMult<stencilLength, 2>(geom, (comp == 2 ? stencilNodeToCell : stencilCellToNode), m_tempDualOneForm[comp], oneForm.data[comp]);
#endif
            
                if(comp == 0)
                    oneForm.data[comp].mult(dr[0] / (GEMPIC_D_MULT(1, dr[1], dr[2]) * weight));
                if(comp == 1)
                    oneForm.data[comp].mult(GEMPIC_D_MULT(1, dr[1], 1)/(GEMPIC_D_MULT(dr[0], 1, dr[2]) * weight));
                if(comp == 2)
                    oneForm.data[comp].mult(GEMPIC_D_MULT(1, 1, dr[2]) / (GEMPIC_D_MULT(dr[0], dr[1], 1) * weight));
            }

            oneForm.averageSync();
            oneForm.fillBoundary();
        }


        // Hodge_1 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::edge>& oneForm, DeRhamField<primal, Space::face>& twoForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

            for (int comp = 0; comp < 3; ++comp)
            {
#if (GEMPIC_SPACEDIM == 1)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilCellToNode : stencilNodeToCell), oneForm.data[comp], twoForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 2)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilCellToNode : stencilNodeToCell), oneForm.data[comp], m_tempPrimalTwoForm[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilCellToNode : stencilNodeToCell), m_tempPrimalTwoForm[comp], twoForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 3)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilCellToNode : stencilNodeToCell), oneForm.data[comp], twoForm.data[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilCellToNode : stencilNodeToCell), twoForm.data[comp], m_tempPrimalTwoForm[comp]);
                matrixMult<stencilLength, 2>(geom, (comp == 2 ? stencilCellToNode : stencilNodeToCell), m_tempPrimalTwoForm[comp], twoForm.data[comp]);
#endif
            
                if(comp == 0)
                    twoForm.data[comp].mult((GEMPIC_D_MULT(1, dr[1], dr[2])) / (GEMPIC_D_MULT(dr[0], 1, 1) * weight));
                if(comp == 1)
                    twoForm.data[comp].mult((GEMPIC_D_MULT(dr[0], 1, dr[2])) / (GEMPIC_D_MULT(1, dr[1], 1) * weight));
                if(comp == 2)
                    twoForm.data[comp].mult((GEMPIC_D_MULT(dr[0], dr[1], 1)) / (GEMPIC_D_MULT(1, 1, dr[2]) * weight));
            }

            twoForm.averageSync();
            twoForm.fillBoundary();
        }


        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::edge>& oneForm, DeRhamField<dual, Space::face>& twoForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

            for (int comp = 0; comp < 3; ++comp)
            {
#if (GEMPIC_SPACEDIM == 1)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilCellToNode : stencilNodeToCell), oneForm.data[comp], twoForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 2)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilCellToNode : stencilNodeToCell), oneForm.data[comp], m_tempDualTwoForm[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilCellToNode : stencilNodeToCell), m_tempDualTwoForm[comp], twoForm.data[comp]);
#endif
#if (GEMPIC_SPACEDIM == 3)
                matrixMult<stencilLength, 0>(geom, (comp == 0 ? stencilCellToNode : stencilNodeToCell), oneForm.data[comp], twoForm.data[comp]);
                matrixMult<stencilLength, 1>(geom, (comp == 1 ? stencilCellToNode : stencilNodeToCell), twoForm.data[comp], m_tempDualTwoForm[comp]);
                matrixMult<stencilLength, 2>(geom, (comp == 2 ? stencilCellToNode : stencilNodeToCell), m_tempDualTwoForm[comp], twoForm.data[comp]);
#endif
            
                if(comp == 0)
                    twoForm.data[comp].mult((GEMPIC_D_MULT(1, dr[1], dr[2])) / (GEMPIC_D_MULT(dr[0], 1, 1) * weight));
                if(comp == 1)
                    twoForm.data[comp].mult((GEMPIC_D_MULT(dr[0], 1, dr[2])) / (GEMPIC_D_MULT(1, dr[1], 1) * weight));
                if(comp == 2)
                    twoForm.data[comp].mult((GEMPIC_D_MULT(dr[0], dr[1], 1)) / (GEMPIC_D_MULT(1, 1, dr[2]) * weight));
            }

            twoForm.averageSync();
            twoForm.fillBoundary();
        }


        // hodge_0 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::node>& zeroForm, DeRhamField<primal, Space::cell>& threeForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

#if (GEMPIC_SPACEDIM == 1)
            matrixMult<stencilLength, 0>(geom, stencilNodeToCell, zeroForm.data, threeForm.data);
#endif
#if (GEMPIC_SPACEDIM == 2)
            matrixMult<stencilLength, 0>(geom, stencilNodeToCell, zeroForm.data, m_tempPrimalThreeForm);
            matrixMult<stencilLength, 1>(geom, stencilNodeToCell, m_tempPrimalThreeForm, threeForm.data);
#endif
#if (GEMPIC_SPACEDIM == 3)
            matrixMult<stencilLength, 0>(geom, stencilNodeToCell, zeroForm.data, threeForm.data);
            matrixMult<stencilLength, 1>(geom, stencilNodeToCell, threeForm.data, m_tempPrimalThreeForm);
            matrixMult<stencilLength, 2>(geom, stencilNodeToCell, m_tempPrimalThreeForm, threeForm.data);
#endif
            
            threeForm.data.mult(GEMPIC_D_MULT(dr[0], dr[1], dr[2]) / weight);

            threeForm.averageSync();
            threeForm.fillBoundary();
        }


        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::node>& zeroForm, DeRhamField<dual, Space::cell>& threeForm, amrex::Real weight = 1.)
        {
            // Select stencil according to degree
            const int stencilLength = degree - 1;
            amrex::GpuArray<amrex::Real, stencilLength> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, stencilLength> stencilCellToNode;
            
            std::tie(stencilNodeToCell, stencilCellToNode) = getHodgeStencils<degree, 0, stencilLength>();

            const amrex::RealVect dr = m_dr;
            const amrex::Geometry geom = m_geom;

#if (GEMPIC_SPACEDIM == 1)
            matrixMult<stencilLength, 0>(geom, stencilNodeToCell, zeroForm.data, threeForm.data);
#endif
#if (GEMPIC_SPACEDIM == 2)
            matrixMult<stencilLength, 0>(geom, stencilNodeToCell, zeroForm.data, m_tempDualThreeForm);
            matrixMult<stencilLength, 1>(geom, stencilNodeToCell, m_tempDualThreeForm, threeForm.data);
#endif
#if (GEMPIC_SPACEDIM == 3)
            matrixMult<stencilLength, 0>(geom, stencilNodeToCell, zeroForm.data, threeForm.data);
            matrixMult<stencilLength, 1>(geom, stencilNodeToCell, threeForm.data, m_tempDualThreeForm);
            matrixMult<stencilLength, 2>(geom, stencilNodeToCell, m_tempDualThreeForm, threeForm.data);
#endif
            
            threeForm.data.mult(GEMPIC_D_MULT(dr[0], dr[1], dr[2]) / weight);

            threeForm.averageSync();
            threeForm.fillBoundary();
        }



};
}

#endif

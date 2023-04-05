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

        template<int degree>
        AMREX_GPU_DEVICE amrex::Real evalLagrange(const int polyIndex, const amrex::Real xk, const amrex::Real dx, const amrex::Real y)
        {
            /**
            * @brief Evaluates a 1D Lagrange polynomial
            * 
            * Description:
            * ----------
            * Evaluates a Lagrange polynomial $L_i$ on the interval $[x_k, x_{k+1}], where $L_0$ is
            * the Lagrange polynomial associated to node $x_k$, $L_{-1}$ to $x_{k-1}$, $L_1$ to $x_{k+1}$,
            * and so forth.
            * The whole grid is assumed to be uniform, evaluation is done using Neville's algorithm.
            * 
            * 
            * Parameters:
            * @tparam degree : int, degree of basis functions used
            * @param polyIndex : int, indicates the polynomial w.r.t. $x_k$
            * @param xk : Real, node $x_k$ which defines the interval of evaluation
            * @param dx : (uniform) grid size
            * @param y : Real, point in $[x_k, x_{k+1}]$, where $L_i$ is to be evaluated
            * 
            * Returns:
            * ----------
            * Returns Real, $L_{polyIndex} (y)$
            */
            amrex::GpuArray<amrex::Real, degree> vals;
            amrex::GpuArray<amrex::Real, degree> nodes;
            for (int j = 0; j < degree; ++j)
            {
                if (j == polyIndex + degree/2 - 1){
                    vals[j] = 1;
                } else {
                    vals[j] = 0;
                }
                nodes[j] = xk + (1 - degree/2 + j) * dx;
            }
            for (int i = 1; i <= degree-1; ++i)
            {
                for (int j = 0; j < degree-i; ++j)
                {
                    vals[j] = vals[j+1] + (y - nodes[j+i]) * (vals[j+1] - vals[j]) / (nodes[j+i] - nodes[j]);
                }
            }
            return vals[0];
        }

        template<int degree>
        AMREX_GPU_DEVICE amrex::Real evalEdge(const int polyIndex, const amrex::Real xk, const amrex::Real dx, const amrex::Real y)
        {
            /**
            * @brief Evaluates a 1D edge function
            * 
            * Description:
            * ----------
            * Evaluates an edge function $e_i$ on the interval $[x_k, x_{k+1}]$, where $e_0$ is
            * the edge function associated to the cell $[x_k,  x_{k+1}]$, $e_{-1}$ to $[x_{k-1},  x_k]$,
            * $e_1$ to $[x_{k+1},  x_{k+2}]$, and so forth.
            * The whole grid is assumed to be uniform, only edge functions of degree 0, 2, 4 can be evaluated.
            * 
            * 
            * Parameters:
            * @tparam degree : int, degree of basis functions used
            * @param polyIndex : int, indicates the edge function w.r.t. $[x_k, x_{k+1}]$
            * @param xk : Real, node $x_k$ which defines the interval of evaluation
            * @param dx : (uniform) grid size
            * @param y : Real, point in $[x_k, x_{k+1}]$, where $e_i$ is to be evaluated
            * 
            * Returns:
            * ----------
            * Returns Real, $e_{polyIndex} (y)$
            */
            amrex::Real val = 0;
            amrex:: Real diff = y - xk;
            switch (degree)
            {
            case 2:
                val =  1. / dx;
                break;
            case 4:
                switch (polyIndex)
                {
                case -1:
                    val = ((diff / (2.*dx) - 1.) * diff / dx + 1./3.) / dx;
                    break;
                case 0:
                    val = ((-diff / dx + 1.) * diff / dx + 5./6.) / dx;
                    break;
                case 1:
                    val = (std::pow(diff,2) / (2*std::pow(dx,2)) - 1./6.) / dx;
                    break;
                default:
                    break;
                }
                break;
            case 6:
                switch (polyIndex)
                {
                case -2:
                    val = ((((1./24. * diff / dx -  1./6.) * diff / dx +  1./8.) * diff / dx + 1./12.) * diff / dx - 1./20.) / dx;
                    break;
                case -1:
                    val = (((((-1./6. * diff) / dx + 1./2.) * diff / dx + 1./4.) * diff / dx - 5./4.) *  diff / dx + 9./20.) / dx;
                    break;
                case 0:
                    val = (((((1./4. * diff) / dx - 1./2.) * diff / dx - 1.) * diff / dx + 5./4.) * diff / dx + 47./60.) / dx;
                    break;
                case 1:
                    val = (((((-1./6. * diff) / dx + 1./6.) * diff / dx + 3./4.) * diff / dx - 1./12.) * diff / dx - 13./60.) / dx;
                    break;
                case 2:
                    val = ((1./24. * std::pow(diff, 2) / std::pow(dx, 2) - 1./8.) * std::pow(diff, 2) / std::pow(dx, 2) + 1./30.) / dx;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }

            return val;
        }

        template<int degree>
        amrex::Real maxErrorMidpoint(amrex::Geometry geom, amrex::ParserExecutor<GEMPIC_SPACEDIM + 1> func, amrex::MultiFab &func_h,
                            amrex::RealVect dr, int form, bool dual, int comp = 0, amrex::Real t = 0.)
        {
            /**
            * @brief Computes the maximum norm w.r.t. the values at the cell midpoints
            * 
            * Description:
            * ----------
            * Interpolates the field, evaluates it a the cell midpoint and computes the max error
            * 
            * 
            * Parameters:
            * @tparam degree : int, degree of basis functions used
            * @param geom : Geometry, needed for physical coordinates
            * @param func : ParserExecutor, provides the exact function to be evaluated
            * @param func_h : MultiFab, contains the DOF values of the numeric approximation
            * @param dr : RealVect, contains the spatial grid sizes
            * @param form : int, specifies k with func_h in C_k
            * @param dual : bual, true if func_h is defined on dual grid
            * @param comp : int, specifies which component func_h belongs to
            * @param t : Real, time point at which func is to be evaluated
            * 
            * Returns:
            * ----------
            * Returns void
            */

            const amrex::BoxArray& ba = func_h.boxArray();
            const amrex::DistributionMapping& dm = func_h.DistributionMap();
            int nghost = func_h.nGrow();
            amrex::MultiFab values(ba, dm, 1, nghost);

            for (amrex::MFIter mfi(func_h); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                amrex::Array4<amrex::Real> const &funcMF = func_h[mfi].array();
                amrex::Array4<amrex::Real> const &valuesMF = values[mfi].array();

                const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = geom.ProbLoArray();

                const amrex::Real drHalfX = dr[0]/2;
#if (GEMPIC_SPACEDIM > 1)
                const amrex::Real drHalfY = dr[1]/2;
#endif
#if (GEMPIC_SPACEDIM > 2)
                const amrex::Real drHalfZ = dr[2]/2;
#endif
                
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Compute the position of the point i, j, k
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> rr = 
                    {
                        AMREX_D_DECL(rr[0] = r0[0] + i*dr[0] - (dual ? drHalfX : 0),
                                    rr[1] = r0[1] + j*dr[1] - (dual ? drHalfY : 0),
                                    rr[2] = r0[2] + k*dr[2] - (dual ? drHalfZ : 0))
                    };

                    amrex::Real error = 0.0;

                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = 
                    {
                        AMREX_D_DECL(rr[0] + drHalfX, rr[1] + drHalfY, rr[2] + drHalfZ)
                    };

                    amrex::Real fval = 0;
                    switch (form)
                    {
                    case 0:
                        for (int a = 1 - degree/2; a < degree/2 + 1; ++a)
                        {
#if (GEMPIC_SPACEDIM > 1)
                            for (int b = 1 - degree/2; b < degree/2 + 1; ++b)
#else
                            int b = 0;
#endif
                            {
#if (GEMPIC_SPACEDIM > 2)
                                for (int c = 1 - degree/2; c < degree/2 + 1; ++c)
#else
                                int c = 0;
#endif
                                {
                                    fval += funcMF(i+a,j+b,k+c) * GEMPIC_D_MULT(evalLagrange<degree>(a, rr[0] + (dual ? dr[0] : 0), dr[0], r[0] + (dual ? dr[0] : 0)),
                                                                                evalLagrange<degree>(b, rr[1] + (dual ? dr[1] : 0), dr[1], r[1] + (dual ? dr[1] : 0)),
                                                                                evalLagrange<degree>(c, rr[2] + (dual ? dr[2] : 0), dr[2], r[2] + (dual ? dr[2] : 0)));
                                }
                            }
                        }
                        break;
                    case 1:
                        for (int a = 1 - degree/2; a < degree/2 + (comp == 0 ? 0 : 1); ++a)
                        {
#if (GEMPIC_SPACEDIM > 1)
                            for (int b = 1 - degree/2; b < degree/2 + (comp == 1 ? 0 : 1); ++b)
#else
                            int b = 0;
#endif
                            {
#if (GEMPIC_SPACEDIM > 2)
                                for (int c = 1 - degree/2; c < degree/2 + (comp == 2 ? 0 : 1); ++c)
#else
                                int c = 0;
#endif
                                {
                                    fval += funcMF(i+a,j+b,k+c) * GEMPIC_D_MULT(comp == 0 ? evalEdge<degree>(a, rr[0], dr[0], r[0]) : evalLagrange<degree>(a, rr[0], dr[0], r[0]),
                                                                                comp == 1 ? evalEdge<degree>(b, rr[1], dr[1], r[1]) : evalLagrange<degree>(b, rr[1], dr[1], r[1]),
                                                                                comp == 2 ? evalEdge<degree>(c, rr[2], dr[2], r[2]) : evalLagrange<degree>(c, rr[2], dr[2], r[2]));
                                }
                            }
                        }
                        break;
                    case 2:
                        for (int a = 1 - degree/2; a < degree/2 + (comp == 0 ? 1 : 0); ++a)
                        {
#if (GEMPIC_SPACEDIM > 1)
                            for (int b = 1 - degree/2; b < degree/2 + (comp == 1 ? 1 : 0); ++b)
#else
                            int b = 0;
#endif
                            {
#if (GEMPIC_SPACEDIM > 2)
                                for (int c = 1 - degree/2; c < degree/2 + (comp == 2 ? 1 : 0); ++c)
#else
                                int c = 0;
#endif
                                {
                                    fval += funcMF(i+a,j+b,k+c) * GEMPIC_D_MULT(comp == 0 ? evalLagrange<degree>(a, rr[0], dr[0], r[0]) : evalEdge<degree>(a, rr[0], dr[0], r[0]),
                                                                                comp == 1 ? evalLagrange<degree>(b, rr[1], dr[1], r[1]) : evalEdge<degree>(b, rr[1], dr[1], r[1]),
                                                                                comp == 2 ? evalLagrange<degree>(c, rr[2], dr[2], r[2]) : evalEdge<degree>(c, rr[2], dr[2], r[2]));
                                }
                            }
                        }
                        break;
                    case 3:
                        for (int a = 1 - degree/2; a < degree/2; ++a)
                        {
#if (GEMPIC_SPACEDIM > 1)
                            for (int b = 1 - degree/2; b < degree/2; ++b)
#else
                            int b = 0;
#endif
                            {
#if (GEMPIC_SPACEDIM > 2)
                                for (int c = 1 - degree/2; c < degree/2; ++c)
#else
                                int c = 0;
#endif
                                {
                                    fval += funcMF(i+a,j+b,k+c) * GEMPIC_D_MULT(evalEdge<degree>(a, rr[0], dr[0], r[0]), evalEdge<degree>(b, rr[1], dr[1], r[1]),
                                                                                evalEdge<degree>(c, rr[2], dr[2], r[2]));
                                }
                            }
                        }
                        break;
                    
                    default:
                        break;
                    }
                    
                    switch (form)
                    {
                    case 0:
                        error += func(AMREX_D_DECL(r[0] + (dual ? dr[0] : 0),
                                                    r[1] + (dual ? dr[1] : 0),
                                                    r[2] + (dual ? dr[2] : 0)), t) - fval;
                        break;
                    case 1:
                        error += func(AMREX_D_DECL(r[0] + (dual && comp != 0 ? dr[0] : 0),
                                                    r[1] + (dual && comp != 1 ? dr[1] : 0),
                                                    r[2] + (dual && comp != 2 ? dr[2] : 0)), t) - fval;
                        break;
                    case 2:
                        error += func(AMREX_D_DECL(r[0] + (dual && comp == 0 ? dr[0] : 0),
                                                    r[1] + (dual && comp == 1 ? dr[1] : 0),
                                                    r[2] + (dual && comp == 2 ? dr[2] : 0)), t) - fval;
                        break;
                    case 3:
                        error += func(AMREX_D_DECL(r[0], r[1], r[2]), t) - fval;
                        break;
                    default:
                        break;
                    }
                    valuesMF(i,j,k) = error;

                });
            }
            return values.norm0();
        }


};
}

#endif

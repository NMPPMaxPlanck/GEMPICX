#ifndef GEMPIC_FDDERHAMCOMPLEX_H_
#define GEMPIC_FDDERHAMCOMPLEX_H_

#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>

using namespace GEMPIC_Fields;

namespace GEMPIC_FDDeRhamComplex {
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
        amrex::Array<amrex::MultiFab, GEMPIC_SPACEDIM> m_tempPrimalOneForm;
        amrex::Array<amrex::MultiFab, GEMPIC_SPACEDIM> m_tempDualOneForm;
        amrex::Array<amrex::MultiFab, GEMPIC_SPACEDIM> m_tempPrimalTwoForm;
        amrex::Array<amrex::MultiFab, GEMPIC_SPACEDIM> m_tempDualTwoForm;
        amrex::MultiFab m_tempPrimalThreeForm;
        amrex::MultiFab m_tempDualThreeForm;
        
    public:
        /// FDDeRhamComplex constructor
        /// @param Parameters
        FDDeRhamComplex(Parameters params);

        ~FDDeRhamComplex();

        void projection (amrex::ParserExecutor<4> func, amrex::Real t,
                        DeRhamField<primal, Space::node>& field) override;
        void projection (amrex::ParserExecutor<4> func, amrex::Real t,
                        DeRhamField<primal, Space::cell>& field) override;
        void projection (amrex::ParserExecutor<4> func, amrex::Real t,
                        DeRhamField<dual, Space::node>& field) override;
        void projection (amrex::ParserExecutor<4> func, amrex::Real t,
                        DeRhamField<dual, Space::cell>& field) override;
        
#if GEMPIC_SPACEDIM > 1
        void projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                        amrex::Real t, DeRhamField<dual, Space::edge>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                        amrex::Real t, DeRhamField<primal, Space::face>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                        amrex::Real t, DeRhamField<primal, Space::edge>& field) override;
        void projection (amrex::Array<amrex::ParserExecutor<4>, GEMPIC_SPACEDIM> func,
                        amrex::Real t, DeRhamField<dual, Space::face>& field) override;

#endif

#if GEMPIC_SPACEDIM > 1

        // Hodge_0 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::cell>& threeForm, DeRhamField<primal, Space::node>& zeroForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (amrex::MFIter mfi(zeroForm.data); mfi.isValid(); ++mfi)
            {
                amrex::Box bx = mfi.validbox();

                amrex::Array4<amrex::Real const> const &threeFormMF = (threeForm.data)[mfi].const_array();
                amrex::Array4<amrex::Real> const &zeroFormMF = (zeroForm.data)[mfi].array();
                amrex::Array4<amrex::Real> const &tempMF = (m_tempPrimalZeroForm)[mfi].array();
                // Finite differences hodge operator algorithm
                               
                // t_1
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilCellToNode)
                    {
                        val += elem * (threeFormMF(i, j, k + shift));
                        shift++;
                    }

                    zeroFormMF(i, j, k) = val;
                });

                // t_2 
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilCellToNode)
                    {
                        val += elem * (zeroFormMF(i, j + shift, k));
                        shift++;
                    }

                    tempMF(i, j, k) = val;
                });

                // t_3
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilCellToNode)
                    {
                        val += elem * (tempMF(i + shift, j, k));
                        shift++;
                    }

                    zeroFormMF(i, j, k) = val;
                });
                
            }
            
            zeroForm.data.mult(1/(dr[0]*dr[1]*dr[2]));

            zeroForm.averageSync();
            zeroForm.fillBoundary();
        }


        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::cell>& threeForm, DeRhamField<dual, Space::node>& zeroForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (amrex::MFIter mfi(zeroForm.data); mfi.isValid(); ++mfi)
            {
                amrex::Box bx = mfi.validbox();

                amrex::Array4<amrex::Real const> const &threeFormMF = (threeForm.data)[mfi].const_array();
                amrex::Array4<amrex::Real> const &zeroFormMF = (zeroForm.data)[mfi].array();
                amrex::Array4<amrex::Real> const &tempMF = (m_tempDualZeroForm)[mfi].array();
                // Finite differences hodge operator algorithm
                               
                // t_1
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilCellToNode)
                    {
                        val += elem * (threeFormMF(i, j, k + shift));
                        shift++;
                    }

                    zeroFormMF(i, j, k) = val;
                });

                // t_2 
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilCellToNode)
                    {
                        val += elem * (zeroFormMF(i, j + shift, k));
                        shift++;
                    }

                    tempMF(i, j, k) = val;
                });

                // t_3
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilCellToNode)
                    {
                        val += elem * (tempMF(i + shift, j, k));
                        shift++;
                    }

                    zeroFormMF(i, j, k) = val;
                });
                
            }
            
            zeroForm.data.mult(1/(dr[0]*dr[1]*dr[2]));
            
            zeroForm.averageSync();
            zeroForm.fillBoundary();
        }

        // Hodge_1 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::face>& twoForm, DeRhamField<primal, Space::edge>& oneForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (int comp = 0; comp < 3; ++comp)
            {
                for (amrex::MFIter mfi(oneForm.data[comp]); mfi.isValid(); ++mfi)
                {
                    amrex::Box bx = mfi.validbox();

                    amrex::Array4<amrex::Real const> const &twoFormMF = (twoForm.data[comp])[mfi].const_array();
                    amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.data[comp])[mfi].array();
                    amrex::Array4<amrex::Real> const &tempMF = (m_tempPrimalOneForm[comp])[mfi].array();
                    // Finite differences hodge operator algorithm
                                   
                    // t_1
                    //bx.grow(2, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 2 ? stencilNodeToCell : stencilCellToNode))
                        {
                            val += elem * (twoFormMF(i, j, k + shift));
                            shift++;

                        }

                        oneFormMF(i, j, k) = val;
                    });

                    //(m_tempPrimalOneForm[comp]).FillBoundary(m_geom.periodicity());

                    // t_2 
                    //bx.grow(1, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 1 ? stencilNodeToCell : stencilCellToNode))
                        {
                            val += elem * (oneFormMF(i, j + shift, k));
                            shift++;

                        }

                        tempMF(i, j, k) = val;
                    });


                    // t_3
                   // bx.grow(0, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 0 ? stencilNodeToCell : stencilCellToNode))
                        {
                            val += elem * (tempMF(i + shift, j, k));
                            shift++;

                        }

                        oneFormMF(i, j, k) = val;
                    });
                    
                }
            
                if(comp == 0)
                    oneForm.data[comp].mult(dr[0]/(dr[1]*dr[2]));
                if(comp == 1)
                    oneForm.data[comp].mult(dr[1]/(dr[0]*dr[2]));
                if(comp == 2)
                    oneForm.data[comp].mult(dr[2]/(dr[0]*dr[1]));
            }

            oneForm.averageSync();
            oneForm.fillBoundary();
        }

        

        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::face>& twoForm, DeRhamField<dual, Space::edge>& oneForm)
        {
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (int comp = 0; comp < 3; ++comp)
            {
                for (amrex::MFIter mfi(oneForm.data[comp]); mfi.isValid(); ++mfi)
                {
                    amrex::Box bx = mfi.validbox();

                    amrex::Array4<amrex::Real const> const &twoFormMF = (twoForm.data[comp])[mfi].const_array();
                    amrex::Array4<amrex::Real> const &oneFormMF = (oneForm.data[comp])[mfi].array();
                    amrex::Array4<amrex::Real> const &tempMF = (m_tempDualOneForm[comp])[mfi].array();
                    // Finite differences hodge operator algorithm
                                   
                    // t_1
                   // bx.grow(2, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 2 ? stencilNodeToCell : stencilCellToNode))
                        {
                            val += elem * (twoFormMF(i, j, k + shift));
                            shift++;

                        }

                        oneFormMF(i, j, k) = val;
                    });
                    
                    // t_2
                   // bx.grow(1, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 1 ? stencilNodeToCell : stencilCellToNode))
                        {
                            val += elem * (oneFormMF(i, j + shift, k));
                            shift++;

                        }

                        tempMF(i, j, k) = val;
                    });

                    // t_3
                   // bx.grow(0, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 0 ? stencilNodeToCell : stencilCellToNode))
                        {
                            val += elem * (tempMF(i + shift, j, k));
                            shift++;

                        }

                        oneFormMF(i, j, k) = val;
                    });

                }
            
                if(comp == 0)
                    oneForm.data[comp].mult(dr[0]/(dr[1]*dr[2]));
                if(comp == 1)
                    oneForm.data[comp].mult(dr[1]/(dr[0]*dr[2]));
                if(comp == 2)
                    oneForm.data[comp].mult(dr[2]/(dr[0]*dr[1]));
            }

            oneForm.averageSync();
            oneForm.fillBoundary();
        }


        // Hodge_2 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::edge>& oneForm, DeRhamField<primal, Space::face>& twoForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (int comp = 0; comp < 3; ++comp)
            {
                for (amrex::MFIter mfi(twoForm.data[comp]); mfi.isValid(); ++mfi)
                {
                    amrex::Box bx = mfi.validbox();

                    amrex::Array4<amrex::Real const> const &oneFormMF = (oneForm.data[comp])[mfi].const_array();
                    amrex::Array4<amrex::Real> const &twoFormMF = (twoForm.data[comp])[mfi].array();
                    amrex::Array4<amrex::Real> const &tempMF = (m_tempPrimalTwoForm[comp])[mfi].array();
                    // Finite differences hodge operator algorithm
                                   
                    // t_1
                    //bx.grow(2, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 2 ? stencilCellToNode : stencilNodeToCell))
                        {
                            val += elem * (oneFormMF(i, j, k + shift));
                            shift++;

                        }

                        twoFormMF(i, j, k) = val;
                    });

                    //(m_tempPrimalOneForm[comp]).FillBoundary(m_geom.periodicity());

                    // t_2 
                    //bx.grow(1, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 1 ? stencilCellToNode : stencilNodeToCell))
                        {
                            val += elem * (twoFormMF(i, j + shift, k));
                            shift++;

                        }

                        tempMF(i, j, k) = val;
                    });


                    // t_3
                   // bx.grow(0, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 0 ? stencilCellToNode : stencilNodeToCell))
                        {
                            val += elem * (tempMF(i + shift, j, k));
                            shift++;

                        }

                        twoFormMF(i, j, k) = val;
                    });
                    
                }
            
                if(comp == 0)
                    twoForm.data[comp].mult((dr[1]*dr[2])/dr[0]);
                if(comp == 1)
                    twoForm.data[comp].mult((dr[0]*dr[2])/dr[1]);
                if(comp == 2)
                    twoForm.data[comp].mult((dr[0]*dr[1])/dr[2]);
            }

            twoForm.averageSync();
            twoForm.fillBoundary();
        }


        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::edge>& oneForm, DeRhamField<dual, Space::face>& twoForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (int comp = 0; comp < 3; ++comp)
            {
                for (amrex::MFIter mfi(twoForm.data[comp]); mfi.isValid(); ++mfi)
                {
                    amrex::Box bx = mfi.validbox();

                    amrex::Array4<amrex::Real const> const &oneFormMF = (oneForm.data[comp])[mfi].const_array();
                    amrex::Array4<amrex::Real> const &twoFormMF = (twoForm.data[comp])[mfi].array();
                    amrex::Array4<amrex::Real> const &tempMF = (m_tempPrimalTwoForm[comp])[mfi].array();
                    // Finite differences hodge operator algorithm
                                   
                    // t_1
                    //bx.grow(2, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 2 ? stencilCellToNode : stencilNodeToCell))
                        {
                            val += elem * (oneFormMF(i, j, k + shift));
                            shift++;

                        }

                        twoFormMF(i, j, k) = val;
                    });

                    //(m_tempPrimalOneForm[comp]).FillBoundary(m_geom.periodicity());

                    // t_2 
                    //bx.grow(1, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 1 ? stencilCellToNode : stencilNodeToCell))
                        {
                            val += elem * (twoFormMF(i, j + shift, k));
                            shift++;

                        }

                        tempMF(i, j, k) = val;
                    });


                    // t_3
                   // bx.grow(0, boxGrow);
                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real val = 0.0;
                        int shift = 1 - degree / 2;
                        for (const auto& elem : (comp == 0 ? stencilCellToNode : stencilNodeToCell))
                        {
                            val += elem * (tempMF(i + shift, j, k));
                            shift++;

                        }

                        twoFormMF(i, j, k) = val;
                    });
                    
                }
            
                if(comp == 0)
                    twoForm.data[comp].mult((dr[1]*dr[2])/dr[0]);
                if(comp == 1)
                    twoForm.data[comp].mult((dr[0]*dr[2])/dr[1]);
                if(comp == 2)
                    twoForm.data[comp].mult((dr[0]*dr[1])/dr[2]);
            }

            twoForm.averageSync();
            twoForm.fillBoundary();
        }


        // hodge_3 primal and dual
        template <int degree>
        void hodgeFD (const DeRhamField<dual, Space::node>& zeroForm, DeRhamField<primal, Space::cell>& threeForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (amrex::MFIter mfi(zeroForm.data); mfi.isValid(); ++mfi)
            {
                amrex::Box bx = mfi.validbox();

                amrex::Array4<amrex::Real const> const &zeroFormMF = (zeroForm.data)[mfi].const_array();
                amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.data)[mfi].array();
                amrex::Array4<amrex::Real> const &tempMF = (m_tempPrimalThreeForm)[mfi].array();
                // Finite differences hodge operator algorithm
                               
                // t_1
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilNodeToCell)
                    {
                        val += elem * (zeroFormMF(i, j, k + shift));
                        shift++;
                    }

                    threeFormMF(i, j, k) = val;
                });

                // t_2 
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilNodeToCell)
                    {
                        val += elem * (threeFormMF(i, j + shift, k));
                        shift++;
                    }

                    tempMF(i, j, k) = val;
                });

                // t_3
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilNodeToCell)
                    {
                        val += elem * (tempMF(i + shift, j, k));
                        shift++;
                    }

                    threeFormMF(i, j, k) = val;
                });
                
            }
            
            threeForm.data.mult(dr[0]*dr[1]*dr[2]);

            threeForm.averageSync();
            threeForm.fillBoundary();
        }


        template <int degree>
        void hodgeFD (const DeRhamField<primal, Space::node>& zeroForm, DeRhamField<dual, Space::cell>& threeForm)
        {
            // Select stencil according to degree
            amrex::GpuArray<amrex::Real, degree - 1> stencilNodeToCell;
            amrex::GpuArray<amrex::Real, degree - 1> stencilCellToNode;
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
            }

            const amrex::RealVect dr = m_dr;

            // Calculate box growth depending on the degree. bx.grow(direction, growth)
            //int boxGrow = degree/2 - 1;

            for (amrex::MFIter mfi(zeroForm.data); mfi.isValid(); ++mfi)
            {
                amrex::Box bx = mfi.validbox();

                amrex::Array4<amrex::Real const> const &zeroFormMF = (zeroForm.data)[mfi].const_array();
                amrex::Array4<amrex::Real> const &threeFormMF = (threeForm.data)[mfi].array();
                amrex::Array4<amrex::Real> const &tempMF = (m_tempDualThreeForm)[mfi].array();
                // Finite differences hodge operator algorithm
                               
                // t_1
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilNodeToCell)
                    {
                        val += elem * (zeroFormMF(i, j, k + shift));
                        shift++;
                    }

                    threeFormMF(i, j, k) = val;
                });

                // t_2 
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilNodeToCell)
                    {
                        val += elem * (threeFormMF(i, j + shift, k));
                        shift++;
                    }

                    tempMF(i, j, k) = val;
                });

                // t_3
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real val = 0.0;
                    int shift = 1 - degree / 2;
                    for (const auto& elem : stencilNodeToCell)
                    {
                        val += elem * (tempMF(i + shift, j, k));
                        shift++;
                    }

                    threeFormMF(i, j, k) = val;
                });
                
            }
            
            threeForm.data.mult(dr[0]*dr[1]*dr[2]);

            threeForm.averageSync();
            threeForm.fillBoundary();
        }


#endif

};
}

#endif

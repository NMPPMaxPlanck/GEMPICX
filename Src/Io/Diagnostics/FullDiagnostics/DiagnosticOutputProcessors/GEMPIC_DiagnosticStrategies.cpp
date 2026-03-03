/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX.H>
#include <AMReX_BaseFwd.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_DiagnosticStrategies.H"

namespace Gempic::Io
{
ComputeDiagOutputProcessor::ComputeDiagOutputProcessor(int ncomp, amrex::IntVect crseRatio) :
    m_crseRatio(crseRatio), m_ncomp(ncomp)
{
    BL_PROFILE("ComputeDiagOutputProcessor::ComputeDiagOutputProcessor()");
}

namespace Impl
{

RawOutputProcessor::RawOutputProcessor(AnyFieldPtr const& dataSrc, amrex::IntVect const crseRatio) :
    ComputeDiagOutputProcessor(dataSrc.n_comp(), crseRatio), m_dataSrc(&dataSrc)
{
    BL_PROFILE("RawOutputProcessor<drc>::RawOutputProcessor()");
}

amrex::BoxArray RawOutputProcessor::get_box_array (amrex::IndexType& t)
{
    amrex::BoxArray bx = convert(m_compDom.m_grid, t);
    return bx;
};

void RawOutputProcessor::operator ()(amrex::MultiFab& mfDst, int dcomp) const
{
    this->m_dataSrc->any_field_to_rawdata_multifabs(mfDst, dcomp);
}

CellCenterOutputProcessor::CellCenterOutputProcessor(AnyFieldPtr const& dataSrc,
                                                     amrex::IntVect const crseRatio) :
    ComputeDiagOutputProcessor(dataSrc.n_comp(), crseRatio), m_dataSrc(&dataSrc)
{
    BL_PROFILE("CellCenterOutputProcessor<drc>::CellCenterOutputProcessor()");
    amrex::IntVect const indexTypeSrc = dataSrc.box_array().ixType().toIntVect();
    AMREX_D_TERM(m_ishift = indexTypeSrc[xDir];, m_jshift = indexTypeSrc[yDir];
                 , m_kshift = indexTypeSrc[zDir];)
}

amrex::BoxArray CellCenterOutputProcessor::get_box_array (amrex::IndexType& t)
{
    amrex::IndexType tNew{t};
    tNew.clear(); // cell-centered plotter;
    amrex::BoxArray bx = convert(m_compDom.m_grid, tNew);
    return bx;
};

void CellCenterOutputProcessor::operator ()(amrex::MultiFab& mfDst, int dcomp) const
{
    BL_PROFILE("CellCenterOutputProcessor<drc>::operator()");
    this->m_dataSrc->any_field_to_cellcentered_multifabs(mfDst, dcomp);
}

} //namespace Impl

} //namespace Gempic::Io

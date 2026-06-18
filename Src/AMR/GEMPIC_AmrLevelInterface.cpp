/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <filesystem>

#include <AMReX_MultiFabUtil.H>

#include "GEMPIC_AmrLevelInterface.H"

namespace Gempic::Amr::Impl
{
void simplify (amrex::BoxList& bl, bool best = false, int maxIter = 5)
{
    // Repeatedly apply AMReX's BoxList::simplify() until no more boxes can be merged or maxIter
    // reached
    int count = 0;
    while (bl.simplify(best) > 0 && count < maxIter)
    {
        count++;
    }
    if (count == maxIter)
    {
        amrex::Warning("Warning: Did not fully simplify boxList");
    }
}

void remove_duplicates (amrex::BoxList& bl)
{
    // O(N²) implementation; only use on small BoxLists from single box surfaces
    auto& boxes = bl.data(); // reference to underlying vector

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        for (size_t j = i + 1; j < boxes.size();)
        {
            if (boxes[i] == boxes[j])
            {
                boxes.erase(boxes.begin() + j);
            }
            else
            {
                ++j;
            }
        }
    }
}

void remove_overlaps (amrex::BoxList& bl)
{
    // O(N²) implementation; only use on small BoxLists from single box surfaces
    auto& boxes = bl.data(); // reference to underlying vector

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        for (size_t j = 0; j < boxes.size(); ++j)
        {
            if (j != i and boxes[i].contains(boxes[j]))
            {
                boxes[j].setRange(xDir, 0, -1); // make box invalid to remove later
            }
        }
    }
    bl.removeEmpty();
}

bool refinement_touches_boundary (amrex::BoxArray ba, amrex::Geometry const& geom)
{
    ba.convert(amrex::IndexType::TheCellType());
    amrex::Box const& domain = geom.Domain();

    for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
    {
        for (auto const& bx : ba.boxList())
        {
            if (bx.smallEnd(dir) <= domain.smallEnd(dir))
            {
                return true;
            }
            if (bx.bigEnd(dir) >= domain.bigEnd(dir))
            {
                return true;
            }
        }
    }
    return false;
}

std::array<amrex::IntVect, 1 << AMREX_SPACEDIM> get_nodal_offsets (amrex::IndexType const idxType)
{
    // only need to shift and look in node-centered directions
    // prevents looking too far for small boxes close to refinement corners
    // selection done by componentwise multiplication with idxType (0=cell, 1=node)
    constexpr int numVertices{1 << AMREX_SPACEDIM};
    auto idxTypeVect = idxType.ixType();
    std::array<amrex::IntVect, numVertices> offset = {
        amrex::IntVect(AMREX_D_DECL(-1, -1, -1)) * idxTypeVect,
        amrex::IntVect(AMREX_D_DECL(1, -1, -1)) * idxTypeVect,
#if AMREX_SPACEDIM > 1
        amrex::IntVect(AMREX_D_DECL(-1, 1, -1)) * idxTypeVect,
        amrex::IntVect(AMREX_D_DECL(1, 1, -1)) * idxTypeVect,
#if AMREX_SPACEDIM == 3
        amrex::IntVect(AMREX_D_DECL(-1, -1, 1)) * idxTypeVect,
        amrex::IntVect(AMREX_D_DECL(1, -1, 1)) * idxTypeVect,
        amrex::IntVect(AMREX_D_DECL(-1, 1, 1)) * idxTypeVect,
        amrex::IntVect(AMREX_D_DECL(1, 1, 1)) * idxTypeVect,
#endif
#endif
    };
    return offset;
}

std::vector<BoxOrientation> get_orientation (amrex::BoxList const& bl,
                                             amrex::BoxArray const& ba,
                                             amrex::IndexType const idxType)
{
    BL_PROFILE("Amr::Impl::getOrientation()");
    std::vector<BoxOrientation> result;
    result.reserve(bl.size());

    constexpr int numVertices{1 << AMREX_SPACEDIM};
    auto const offsets{get_nodal_offsets(idxType)};

    std::array<std::array<std::array<int, 2>, 4>, 3> const edges = {{
        {{{0, 1}, {2, 3}, {4, 5}, {6, 7}}},
        {{{0, 2}, {1, 3}, {4, 6}, {5, 7}}},
        {{{0, 4}, {1, 5}, {2, 6}, {3, 7}}},
    }};

    for (auto const& box : bl)
    {
        auto smallEnd = box.smallEnd();
        auto bigEnd = box.bigEnd();
        auto middle = (smallEnd + bigEnd) / 2;

        std::array<int, numVertices> v;

        for (int n = 0; n < numVertices; n++)
        {
            v[n] = ba.contains(middle + offsets[n]);
        }

        amrex::IntVect diff = amrex::IntVect::TheZeroVector();
        for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
        {
            if (idxType.nodeCentered(dir))
            {
                // compute differences in dir
                for (int e = 0; e < numVertices / 2; e++)
                {
                    diff[dir] += v[edges[dir][e][0]] - v[edges[dir][e][1]];
                }
            }
        }
        int coveredNeighbors = std::accumulate(v.begin(), v.end(), 0);
        amrex::IntVect outwardNormal{
            AMREX_D_DECL(sign(diff[xDir]), sign(diff[yDir]), sign(diff[zDir]))};

        result.push_back(BoxOrientation{outwardNormal, coveredNeighbors});
    }
    return result;
}

amrex::BoxArray compute_communication_boxes (amrex::BoxArray const& ba,
                                             std::vector<BoxOrientation> orientation,
                                             amrex::IntVect refRatio,
                                             int growNum,
                                             int growDir)
{
    // Grows boxes outward/inward based on refinement ratio and box orientation
    // Growth only applied in directions with refRatio==2
    AMREX_ALWAYS_ASSERT(growDir == 1 or growDir == -1);
    auto bl = ba.boxList();
    amrex::BoxList blResult;
    blResult.reserve(bl.size());

    for (int n = 0; n < bl.size(); n++)
    {
        auto box = bl.data()[n];
        for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
        {
            if (orientation[n].m_outwardNormal[dir] * growDir > 0 and refRatio[dir] == 2)
            {
                box.growHi(dir, growNum);
            }
            else if (orientation[n].m_outwardNormal[dir] * growDir < 0 and refRatio[dir] == 2)
            {
                box.growLo(dir, growNum);
            }
        }
        blResult.push_back(box);
    }
    return amrex::BoxArray{std::move(blResult)};
}

amrex::BoxList split_edge (amrex::Box box,
                           amrex::BoxArray const& fba,
                           amrex::IndexType const idxType,
                           amrex::IntVect const refRatio,
                           bool isCoarse,
                           int dir)
{
#if AMREX_SPACEDIM == 3
    // iteratively split edge according to surrounding boxes
    amrex::BoxList subEdges;
    if (isCoarse)
    {
        subEdges = amrex::intersect(amrex::coarsen(fba, refRatio).boxList(), box);
    }
    else
    {
        subEdges = amrex::intersect(fba.boxList(), box);
    }
    remove_duplicates(subEdges);

    amrex::BoxList oldBl(idxType), newBl(idxType);
    oldBl.push_back(box);

    auto chopAndUpdateBls = [&] (amrex::IntVect chopPnt)
    {
        for (auto bx : oldBl)
        {
            if (bx.contains(chopPnt) and chopPnt != bx.smallEnd() and chopPnt != bx.bigEnd())
            {
                auto newbx = bx.chop(dir, chopPnt[dir]);
                newBl.push_back(bx);
                newBl.push_back(newbx);
            }
            else
            {
                newBl.push_back(bx);
            }
        }
        oldBl.clear();
        oldBl.catenate(newBl); // clears newBl
    };

    for (auto const& subEdge : subEdges)
    {
        newBl.clear();
        // make sure that we do not take the full edge or single vertices
        if (subEdge == box or subEdge.smallEnd() == subEdge.bigEnd()) continue;
        if (subEdge.bigEnd() != box.bigEnd())
        {
            // need to split box at subEdge.bigEnd()
            auto chopPnt = subEdge.bigEnd();
            if (idxType.cellCentered(dir))
            {
                // for cell-centered boxes, the chopPnt is included in the high end but in this
                // branch we want to chop after chopPnt
                chopPnt[dir] += 1;
            }
            chopAndUpdateBls(chopPnt);
        }
        if (subEdge.smallEnd() != box.smallEnd())
        {
            // need to split box at subEdge.smallEnd()
            auto chopPnt = subEdge.smallEnd();
            chopAndUpdateBls(chopPnt);
        }
    }
    newBl.join(oldBl); // write final results in newBl
    if (newBl.size() == 0)
    {
        newBl.push_back(box);
    }
#else
    amrex::BoxList newBl;
    newBl.push_back(box);
#endif

    amrex::BoxList resultBl(idxType);
    // split into interior edge and two vertices in node centered direction
    if (idxType.nodeCentered(dir))
    {
        for (auto bx : newBl)
        {
            auto smallEnd = bx.smallEnd();
            auto bigEnd = bx.bigEnd();

            resultBl.push_back(amrex::Box(smallEnd, smallEnd, idxType));
            resultBl.push_back(amrex::Box(bigEnd, bigEnd, idxType));
            bx.growLo(dir, -1);
            bx.growHi(dir, -1);
            resultBl.push_back(bx);
        }
    }
    else
    {
        resultBl.catenate(newBl);
    }
    return resultBl;
}

/// @brief Splits interface boxes into component parts (vertices, edges, faces) based on
///        surrounding fine-level geometry
amrex::BoxList split_boxes (amrex::BoxList const& inputBl,
                            amrex::BoxArray const& fba,
                            amrex::IndexType const idxType,
                            amrex::IntVect const refRatio,
                            bool isCoarse)
{
    BL_PROFILE("Amr::Impl::splitBoxes()");
    amrex::BoxList outputBl(idxType);
    outputBl.reserve(inputBl.size());

    for (int i = 0; i < inputBl.size(); i++)
    {
        amrex::Box box = inputBl.data()[i];

        // check type of the interface box (vertex, edge, face) and store its extent (i.e., 0, 1, or
        // 2 directions, respectively)
        auto length = box.length();
        int boxDimension = 0;
        Direction extent1{noDir}, extent2{noDir};
        for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
        {
            if (length[dir] > 1)
            {
                boxDimension++;
                if (extent1 != noDir)
                {
                    extent2 = dir;
                }
                else
                {
                    extent1 = dir;
                }
            }
        }

        switch (boxDimension)
        {
            case 0:
            {
                // vertex: do not need to do anything
                outputBl.push_back(box);
                break;
            }

            case 1:
            {
                outputBl.join(split_edge(box, fba, idxType, refRatio, isCoarse, extent1));
                break;
            }

            case 2:
            {
                // face: split into interior, edges, and vertices
                auto smallEnd = box.smallEnd();
                auto bigEnd = box.bigEnd();
                auto length = box.length();

                // create unit vectors for the two extent directions
                auto extent1Vect{amrex::IntVect::TheDimensionVector(extent1) * (length - 1)};
                auto extent2Vect{amrex::IntVect::TheDimensionVector(extent2) * (length - 1)};

                if (idxType.nodeCentered(extent1) and idxType.cellCentered(extent2))
                {
                    box.growLo(extent1, -1);
                    box.growHi(extent1, -1);
                    // edges (might have to be split further)
                    outputBl.join(split_edge(amrex::Box(smallEnd, smallEnd + extent2Vect, idxType),
                                             fba, idxType, refRatio, isCoarse, extent2));
                    outputBl.join(split_edge(amrex::Box(bigEnd - extent2Vect, bigEnd, idxType), fba,
                                             idxType, refRatio, isCoarse, extent2));
                }
                else if (idxType.nodeCentered(extent2) and idxType.cellCentered(extent1))
                {
                    box.growLo(extent2, -1);
                    box.growHi(extent2, -1);
                    // edges (might have to be split further)
                    outputBl.join(split_edge(amrex::Box(smallEnd, smallEnd + extent1Vect, idxType),
                                             fba, idxType, refRatio, isCoarse, extent1));
                    outputBl.join(split_edge(amrex::Box(bigEnd - extent1Vect, bigEnd, idxType), fba,
                                             idxType, refRatio, isCoarse, extent1));
                }
                else if (idxType.nodeCentered(extent1) and idxType.nodeCentered(extent2))
                {
                    // interior surface
                    box.growLo(extent1, -1);
                    box.growHi(extent1, -1);
                    box.growLo(extent2, -1);
                    box.growHi(extent2, -1);

                    // edges (might have to be split further)
                    outputBl.join(split_edge(amrex::Box(smallEnd, smallEnd + extent2Vect, idxType),
                                             fba, idxType, refRatio, isCoarse, extent2));
                    outputBl.join(split_edge(amrex::Box(bigEnd - extent2Vect, bigEnd, idxType), fba,
                                             idxType, refRatio, isCoarse, extent2));
                    outputBl.join(split_edge(amrex::Box(smallEnd, smallEnd + extent1Vect, idxType),
                                             fba, idxType, refRatio, isCoarse, extent1));
                    outputBl.join(split_edge(amrex::Box(bigEnd - extent1Vect, bigEnd, idxType), fba,
                                             idxType, refRatio, isCoarse, extent1));
                }

                outputBl.push_back(box);
                break;
            }

            default:
            {
                amrex::Error("Boundary box is not a vertex, edge, or face");
                break;
            }
        }
    }
    remove_overlaps(outputBl);

    // try to unify boxes back together if they have the same orientation
    // the orientation has to always be computed on the fine indices to prevent edge cases with
    // small boxes (single dof but not a corner) correct orientations are necessary to prevent
    // unification of incompatible boxes in this step
    if (isCoarse)
    {
        outputBl.refine(refRatio);
    }
    auto orientation = get_orientation(outputBl, fba, idxType);
    if (isCoarse)
    {
        outputBl.coarsen(refRatio);
    }
    std::map<BoxOrientation, amrex::BoxList> grouped;
    int n = outputBl.size();
    for (int i = 0; i < n; ++i)
    {
        grouped[orientation[i]].push_back(outputBl.data()[i]);
    }
    outputBl.clear();
    for (auto& pair : grouped)
    {
        amrex::BoxList tmp = pair.second;
        simplify(tmp);
        outputBl.catenate(tmp);
    }

    return outputBl;
}

void print_ba_to_file (amrex::BoxArray const& ba, std::filesystem::path name)
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream ofs(name, std::ios::out | std::ios::trunc);
        if (!ofs.is_open())
        {
            amrex::Error("Could not open " + name.string() + "\n");
            return;
        }
        ofs << ba << std::endl;
        ofs.close();
    }
}

std::shared_ptr<amrex::iMultiFab> build_covered_mask (amrex::FabArrayBase const& finefa,
                                                      amrex::FabArrayBase const& coarsefa,
                                                      amrex::IntVect const refRatio,
                                                      amrex::IndexType const idxType)
{
    BL_PROFILE("Amr::Impl::LevelInterface::build_covered_mask()");

    auto coarseCoveredMask =
        std::make_shared<amrex::iMultiFab>(coarsefa.boxArray(), coarsefa.DistributionMap(), 1, 0);

    // Mask value: 0 where coarse is covered by fine, 1 otherwise (interface points have value 0)
    /// @todo periodicity is hard-coded
    amrex::iMultiFab mask = amrex::makeFineMask(coarsefa.boxArray(), coarsefa.DistributionMap(),
                                                amrex::IntVect::TheUnitVector(), finefa.boxArray(),
                                                refRatio, amrex::Periodicity::NonPeriodic(), 0, 1);

    auto const offsets{get_nodal_offsets(idxType)};

    for (amrex::MFIter mfi(*coarseCoveredMask); mfi.isValid(); ++mfi)
    {
        amrex::Array4<int const> const& fineMask = mask.const_array(mfi);
        amrex::Array4<int> const& resultMask = coarseCoveredMask->array(mfi);

        ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        resultMask(i, j, k) = 0;
                        for (auto offset : offsets)
                        {
                            if (fineMask(i + offset[0], j AMREX_D_PICK(, +offset[1], +offset[1]),
                                         k AMREX_D_PICK(, , +offset[2])) == 0)
                            {
                                resultMask(i, j, k) = 1;
                                break;
                            }
                        }
                    });
    }
    return coarseCoveredMask;
}

LevelInterface::LevelInterface(amrex::FabArrayBase const& finefa,
                               amrex::FabArrayBase const& coarsefa,
                               amrex::IntVect refRatio,
                               amrex::Geometry fineGeom,
                               bool buildCoveredMask)
{
    BL_PROFILE("Amr::Impl::LevelInterface::LevelInterface()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(finefa.boxArray().ixType() == coarsefa.boxArray().ixType(),
                                     "Coarse and fine level need the same index type!");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(not refinement_touches_boundary(finefa.boxArray(), fineGeom),
                                     "Refinement touches domain boundary!");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(refRatio.allGE(1) && refRatio.allLE(2),
                                     "Refinement ratio can only be 1 or 2!");

    m_idxType = finefa.boxArray().ixType(); // 0 -> cell, 1 -> node
    m_refRatio = refRatio;
    m_fba = finefa.boxArray();
    m_cba = coarsefa.boxArray();

    // coarsened exterior of finefa, needs to be done on cell centered and
    // then converted back afterwards
    auto ba = amrex::coarsen(amrex::convert(finefa.boxArray(), amrex::IndexType::TheCellType()),
                             refRatio);
    m_blCoarseOutside = ba.complementIn(
        amrex::convert(coarsefa.boxArray().minimalBox(), amrex::IndexType::TheCellType()));
    m_blCoarseOutside.convert(m_idxType);
    simplify(m_blCoarseOutside);

    build_fine_interface(finefa);
    build_coarse_interface(finefa, coarsefa);
    // communication boxes needed for moment-preservation in conforming projections
    m_baCommFine = compute_communication_boxes(m_baFine, m_orientationFine, refRatio, 4, 1);
    m_baCommCoarse = compute_communication_boxes(m_baCoarse, m_orientationCoarse, refRatio, 1, -1);

    if (buildCoveredMask)
    {
        m_coarseCoveredMask = build_covered_mask(finefa, coarsefa, m_refRatio, m_idxType);
    }
}

void LevelInterface::write_to_file (std::string folderName)
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // create directory if it does not exist
        std::filesystem::path folder = folderName;
        if (!folder.empty())
        {
            try
            {
                std::filesystem::create_directories(folder);
            }
            catch (std::filesystem::filesystem_error const& e)
            {
                amrex::Error("Failed to create directory " + folder.string() + ": " + e.what());
                return;
            }
        }
        print_ba_to_file(m_baFine, folder / "FineInterface.txt");
        print_ba_to_file(m_fba, folder / "FineDomain.txt");
        print_ba_to_file(m_baCoarse, folder / "CoarseInterface.txt");
        print_ba_to_file(m_cba, folder / "CoarseDomain.txt");
        {
            std::ofstream ofs(folder / "RefinementRatio.txt", std::ios::out | std::ios::trunc);
            if (!ofs.is_open())
            {
                amrex::Error("Could not open RefinementRatio.txt\n");
            }
            ofs << m_refRatio << std::endl;
            ofs.close();
        }
        {
            std::ofstream ofs(folder / "IndexType.txt", std::ios::out | std::ios::trunc);
            if (!ofs.is_open())
            {
                amrex::Error("Could not open IndexType.txt\n");
            }
            ofs << m_idxType << std::endl;
            ofs.close();
        }
    }
}

void LevelInterface::build_fine_interface (amrex::FabArrayBase const& finefa)
{
    BL_PROFILE("Amr::Impl::LevelInterface::build_fine_interface()");
    amrex::BoxArray const& fba = amrex::convert(finefa.boxArray(), amrex::IndexType::TheCellType());
    // REMARK: Needs to be CellType. Otherwise the complementIn removes dof on the interface
    amrex::DistributionMapping const& fdm = finefa.DistributionMap();

    amrex::BoxList bl(m_idxType);
    amrex::Vector<int> iprocs;
    int const myproc = amrex::ParallelDescriptor::MyProc();

    for (int i = 0, N = static_cast<int>(fba.size()); i < N; ++i)
    {
        amrex::Box bx = fba[i];
        bx.grow(m_idxType.toIntVect());
        /// @todo correctly handle boundaries
        //bx &= m_fine_domain;

        amrex::BoxList noncovered = fba.complementIn(bx);
        noncovered.convert(m_idxType);
        noncovered.intersect(finefa.boxArray()[i]);

        if (noncovered.isNotEmpty())
        {
            simplify(noncovered);
            auto blSplit = split_boxes(noncovered, finefa.boxArray(), m_idxType, m_refRatio, false);
            iprocs.insert(iprocs.end(), blSplit.size(), fdm[i]);
            if (fdm[i] == myproc)
            {
                m_fineIdx.insert(m_fineIdx.end(), blSplit.size(), i);
            }
            bl.catenate(blSplit);
        }
    }
    if (bl.isNotEmpty())
    {
        m_orientationFine = get_orientation(bl, finefa.boxArray(), m_idxType);
        m_baFine.define(bl);
        m_dmFine.define(std::move(iprocs));
    }
}

void LevelInterface::build_coarse_interface (amrex::FabArrayBase const& finefa,
                                            amrex::FabArrayBase const& coarsefa)
{
    BL_PROFILE("Amr::Impl::LevelInterface::build_coarse_interface()");
    amrex::BoxArray const& fba = amrex::coarsen(
        amrex::convert(finefa.boxArray(), amrex::IndexType::TheCellType()), m_refRatio);
    amrex::BoxArray const& cba =
        amrex::convert(coarsefa.boxArray(), amrex::IndexType::TheCellType());
    amrex::DistributionMapping const& cdm = coarsefa.DistributionMap();

    amrex::BoxList bl(m_idxType);
    amrex::BoxList blLocal(m_idxType);
    amrex::Vector<int> iprocs;
    amrex::Vector<int> iprocsLocal;
    int const myproc = amrex::ParallelDescriptor::MyProc();
    amrex::BoxList fine = fba.boxList();
    simplify(fine);

    amrex::IntVect nGhost = m_idxType.toIntVect();

    for (int i = 0, N = static_cast<int>(cba.size()); i < N; ++i)
    {
        // keep operations for one box together to be able to remove duplicates
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            amrex::IntVect nGhostDir{AMREX_D_DECL((dir == 0) * nGhost[0], (dir == 1) * nGhost[1],
                                                  (dir == 2) * nGhost[2])};
            amrex::Box bx = cba[i];

            // cut away fine domain from box resulting in box list
            auto bxs = complementIn(amrex::grow(bx, amrex::IntVect::TheUnitVector()), fine);
            for (auto& b : bxs)
            {
                b.grow(nGhostDir);
            }
            /// @todo correctly handle boundaries
            //bx &= m_fine_domain;

            bxs.intersect(fine);
            bxs.convert(m_idxType);
            bxs.intersect(amrex::convert(bx, m_idxType));

            bxs.intersect(m_blCoarseOutside);
            simplify(bxs);

            for (amrex::Box const& b : bxs)
            {
                // make sure that intersections in the wrong direction are ignored
                // needed because we have to do it direction by direction
                if (not(b.length(dir) > 1))
                {
                    blLocal.push_back(b);
                }
            }
        }
        if (blLocal.isNotEmpty())
        {
            // cut with boxes of fine level to ensure that every geometry is resolved
            // split_boxes calls get_orientation internally which needs to work on fine boxes
            auto blSplit = split_boxes(blLocal, finefa.boxArray(), m_idxType, m_refRatio, true);
            blLocal.clear();

            iprocs.insert(iprocs.end(), blSplit.size(), cdm[i]);
            if (cdm[i] == myproc)
            {
                m_coarseIdx.insert(m_coarseIdx.end(), blSplit.size(), i);
            }
            bl.catenate(blSplit);
        }
    }
    amrex::BoxArray baCfb;
    amrex::DistributionMapping dmCfb;
    if (bl.isNotEmpty())
    {
        // use fine grid to compute orientation to prevent edge cases for single dof boxes that are
        // not a corner
        auto blRefined = bl;
        blRefined.refine(m_refRatio);
        m_orientationCoarse = get_orientation(blRefined, finefa.boxArray(), m_idxType);
        m_baCoarse.define(bl);
        m_dmCoarse.define(std::move(iprocs));
    }
}
} // namespace Gempic::Amr::Impl
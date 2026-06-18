/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "GEMPIC_Config.H"

namespace Gempic::Amr::Impl
{
inline int sign (int x) { return (x > 0) - (x < 0); }

/// @brief Stores the topological information of a box relative to its surroundings
/// @details Used to classify interface boxes and group them by their orientation
///          relative to refined and coarse domains
struct BoxOrientation
{
    /// Outward normal direction from refined domain; each component is -1, 0, or 1
    amrex::IntVect m_outwardNormal;
    /// Number of neighbors (boxes from the refined domain) that cover this box
    int m_coveredNeighbors = 0;

    auto operator<=>(BoxOrientation const& other) const
    {
        // amrex::IntVect does not have operator<=> so we have to manually define an ordering
        // compare outwardNormal lexicographically
        for (int i = 0; i < AMREX_SPACEDIM; ++i)
        {
            if (m_outwardNormal[i] < other.m_outwardNormal[i])
            {
                return std::strong_ordering::less;
            }
            if (m_outwardNormal[i] > other.m_outwardNormal[i])
            {
                return std::strong_ordering::greater;
            }
        }
        // if outwardNormal is equal, compare coveredNeighbors
        return m_coveredNeighbors <=> other.m_coveredNeighbors;
    }
};

/// @brief Stores all topological and geometric information about the interface between a single
///        pair of adjacent AMR refinement levels (one coarse level and one fine level)
/// @details This class computes and maintains interface boxes for both the fine and coarse levels
///          of a single refinement pair. Each LevelInterface manages exactly one coarse-fine level
///          pair; multiple LevelInterfaces are used in a full AMR hierarchy (one per adjacent level
///          pair). The interface is specific to a given index type (cell or node centered).
class LevelInterface
{
public:
    /// @brief Constructs a LevelInterface for a single coarse-fine level pair
    /// @details Computes interface boxes and their properties by analyzing the geometry of
    ///          the fine and coarse levels and their refinement relationship. Interface boxes
    ///          are identified at the refinement boundary, then decomposed into vertices, edges,
    ///          and faces, and organized by their topological orientation.
    ///          Must be called with FabArrays of the same index type.
    /// @param finefa Fine-level FabArray defining the refined domain
    /// @param coarsefa Coarse-level FabArray defining the unrefined domain
    /// @param refRatio Refinement ratio between coarse and fine levels in each direction
    /// @param fineGeom Geometry object for fine level
    /// @param buildCoveredMask Construct m_coarseCoveredMask, otherwise no memory is allocated
    LevelInterface(amrex::FabArrayBase const& finefa,
                   amrex::FabArrayBase const& coarsefa,
                   amrex::IntVect refRatio,
                   amrex::Geometry fineGeom,
                   bool buildCoveredMask = true);

    /// @brief Writes interface geometry data to disk for visualization and debugging
    /// @details Creates text files containing BoxArrays, index type, and refinement ratio
    ///          in the specified folder. Files include:
    ///          - FineInterface.txt: Interface boxes at fine level
    ///          - FineDomain.txt: Domain boxes at fine level
    ///          - CoarseInterface.txt: Interface boxes at coarse level
    ///          - CoarseDomain.txt: Domain boxes at coarse level
    ///          - IndexType.txt: Index type used
    ///          - RefinementRatio.txt: Refinement ratio
    /// @param folderName Directory where files will be written (default: "LevelInterface/")
    void write_to_file(std::string folderName = "LevelInterface/");

private:
    void build_fine_interface(amrex::FabArrayBase const& finefa);

    void build_coarse_interface(amrex::FabArrayBase const& finefa,
                                amrex::FabArrayBase const& coarsefa);

public:
    /// @brief Interface boxes at the fine level (degrees of freedom at the refinement boundary)
    amrex::BoxArray m_baFine;
    /// @brief Communication boxes at the fine level (interface boxes grown to include necessary
    /// DOF for moment-preserving conforming projection)
    amrex::BoxArray m_baCommFine;
    /// @brief Processor distribution of fine-level interface boxes
    amrex::DistributionMapping m_dmFine;
    /// @brief Indices mapping fine-level interface boxes to their parent fine-level boxes
    amrex::Vector<int> m_fineIdx;
    /// @brief Topology information (orientation, covered neighbors) for fine-level interface boxes
    std::vector<BoxOrientation> m_orientationFine;

    /// @brief Interface boxes at the coarse level (degrees of freedom at the refinement boundary)
    amrex::BoxArray m_baCoarse;
    /// @brief Communication boxes at the coarse level (interface boxes grown to include necessary
    /// DOF for moment-preserving conforming projection)
    amrex::BoxArray m_baCommCoarse;
    /// @brief Processor distribution of coarse-level interface boxes
    amrex::DistributionMapping m_dmCoarse;
    /// @brief Indices mapping coarse-level interface boxes to their parent coarse-level boxes
    amrex::Vector<int> m_coarseIdx;
    /// @brief Topology information (orientation, covered neighbors) for coarse-level interface
    /// boxes
    std::vector<BoxOrientation> m_orientationCoarse;
    /// @brief Mask that contains zeros for dof that are covered by fine level excluding the
    /// interface
    std::shared_ptr<amrex::iMultiFab> m_coarseCoveredMask;

    /// @brief Refinement ratio between the coarse and fine levels in each spatial direction
    amrex::IntVect m_refRatio;
    /// @brief Index type (cell-centered or node-centered) common to both levels
    amrex::IndexType m_idxType;
    /// @brief Fine-level domain boxes (stored for diagnostic output)
    amrex::BoxArray m_fba;
    /// @brief Coarse-level domain boxes (stored for diagnostic output)
    amrex::BoxArray m_cba;

private:
    amrex::BoxList m_blCoarseOutside;
};
} //namespace Gempic::Amr::Impl
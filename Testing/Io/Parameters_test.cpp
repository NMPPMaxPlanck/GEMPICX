#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic::Io;

class ParametersTest : public testing::Test
{
protected:
    static std::vector<int>& file_n_cell_vector ()
    {
        static std::vector<int> nCellVec{AMREX_D_DECL(128, 8, 8)};
        return nCellVec;
    }

    static std::vector<int>& file_is_periodic_vector ()
    {
        static std::vector<int> isPeriodicVector{AMREX_D_DECL(1, 1, 1)};
        return isPeriodicVector;
    }

    static std::string& file_particle_species0_density ()
    {
        static std::string particleSpecies0Density{"1.0 + 0.01 * cos(kvarx * x)"};
        return particleSpecies0Density;
    };

    ParametersTest()
    {
        amrex::ParmParse pp;

        pp.addarr("ComputationalDomain.nCell", file_n_cell_vector());
        pp.addarr("ComputationalDomain.isPeriodic", file_is_periodic_vector());

        pp.add("Particle.species0.density", file_particle_species0_density());
    }
};

TEST_F(ParametersTest, InitMainInstanceAndGetSet)
{
    Parameters params{};

    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};
    params.get("ComputationalDomain.isPeriodic", isPeriodic);
    EXPECT_THAT(isPeriodic, ::testing::ElementsAreArray(file_is_periodic_vector()));

    // check for one vector. If the size is too small, there is no problem (and no warning!)
    // If the size is too large, the vector doesn't get resized, and amrex simply fills in until it
    // runs out of values.
    std::vector<int> something{0};
    params.get_or_set("ComputationalDomain.nCell", something);
    EXPECT_THAT(something, ::testing::ElementsAreArray(file_n_cell_vector()));

    amrex::Array<double, AMREX_SPACEDIM> k{AMREX_D_DECL(1.25, 1.25, 1.25)};
    params.set("k", k);
    EXPECT_THAT(k, ::testing::ElementsAreArray({AMREX_D_DECL(1.25, 1.25, 1.25)}));

    amrex::Array<double, AMREX_SPACEDIM> laterK{AMREX_D_DECL(0, 0, 0)};
    params.get("k", laterK);
    EXPECT_THAT(laterK, ::testing::ElementsAreArray({AMREX_D_DECL(1.25, 1.25, 1.25)}));
}

TEST_F(ParametersTest, ClassFunctionality)
{
    std::string outputFile{"np_test_class.output"};
    amrex::ParmParse pp;
    pp.add("outputFile", outputFile);
    // Actual tests, reading, writing, etc.
    {
        Parameters::set_print_output();
        Parameters parameters{};
        Parameters params{"Particle"};

        std::vector<double> something{1.0, 2.0, 3.0};
        params.get_or_set("doesntExist", something);
        EXPECT_THAT(something, ::testing::ElementsAreArray({1, 2, 3}));

        std::string densitystring;
        params.get("species0.density", densitystring);
        EXPECT_EQ(densitystring, file_particle_species0_density());
    }
    // Check that output file created just now is readable as input file
    {
        // Make sure the quantities that are read are not from elsewhere
        pp.remove("Particle.species0.density");
        pp.remove("Particle.doesntExist");
        pp.addfile(outputFile);
        // stop the warning about unused variables
        pp.remove("outputFile");

        // This time we don't want to print
        Parameters::set_print_output(false);
        Parameters params{};

        Parameters params2{"Particle"};
        std::vector<double> somethingElse{};
        params2.get("doesntExist", somethingElse);

        EXPECT_THAT(somethingElse, ::testing::ElementsAreArray({1, 2, 3}));

        std::string densitystring;
        params2.get("species0.density", densitystring);
        EXPECT_EQ(densitystring, "1.0 + 0.01 * cos(kvarx * x)");

        std::string simulationName;
        params2.get("simName", simulationName);
        EXPECT_EQ(simulationName, "unnamed simulation");
    }
}

// "Death" tests, but we have changed AMReX behaviour to throw an exception first
TEST(ParametersErrorTests, NoOutputFile)
{
    Parameters::set_print_output();
    EXPECT_THROW(Parameters{}, std::runtime_error);
    // special circumstances because Parameters was technically not created and
    // therefore not destroyed, so m_printOutput was not reset automatically.
    Parameters::set_print_output(false);
}

TEST(ParametersErrorTests, NoSuchParameter)
{
    int a;
    Parameters params{};
    EXPECT_THROW(params.get("NonexistentVar", a), std::runtime_error);
}

TEST(ParametersErrorTests, AttemptedChange)
{
    int ghostCellBuffer{3};
    Parameters params{};
    params.get_or_set("nGhostExtra", ghostCellBuffer);
    int ghostCellInParams;
    params.get("nGhostExtra", ghostCellInParams);
    EXPECT_EQ(ghostCellBuffer, ghostCellInParams);
    EXPECT_THROW(params.set("nGhostExtra", ghostCellBuffer), std::runtime_error);
}

TEST(ParametersErrorTests, NoGeneralInstance)
{
    EXPECT_THROW(Parameters("ClassPrefix"), std::runtime_error);
}

TEST(ParametersErrorTests, TwoGeneralInstances)
{
    Parameters params{};
    EXPECT_THROW(Parameters::set_print_output(), std::runtime_error);
}
} // namespace

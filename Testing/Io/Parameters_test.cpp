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
    static std::vector<int> &file_n_cell_vector ()
    {
        static std::vector<int> nCellVec{AMREX_D_DECL(128, 8, 8)};
        return nCellVec;
    }

    static std::vector<int> &file_is_periodic_vector ()
    {
        static std::vector<int> isPeriodicVector{AMREX_D_DECL(1, 1, 1)};
        return isPeriodicVector;
    }

    static std::string &file_particle_species0_density ()
    {
        static std::string particleSpecies0Density{"1.0 + 0.01 * cos(kvarx * x)"};
        return particleSpecies0Density;
    };

    static void SetUpTestSuite ()
    {
        amrex::ParmParse pp;

        pp.addarr("nCellVector", file_n_cell_vector());
        pp.addarr("isPeriodicVector", file_is_periodic_vector());

        pp.add("Particle.species0.density", file_particle_species0_density());
    }
};

TEST_F(ParametersTest, InitMainInstanceAndGetSet)
{
    Parameters params{};

    amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};
    params.get("isPeriodicVector", isPeriodic);
    EXPECT_THAT(isPeriodic, ::testing::ElementsAreArray(file_is_periodic_vector()));

    // check for one vector. If the size is too small, there is no problem (and no warning!)
    // If the size is too large, the vector doesn't get resized, and amrex simply fills in until it
    // runs out of values.
    std::vector<int> something{0};
    params.get_or_set("nCellVector", something);
    EXPECT_THAT(something, ::testing::ElementsAreArray(file_n_cell_vector()));

    amrex::Array<double, GEMPIC_SPACEDIM> k{AMREX_D_DECL(1.25, 1.25, 1.25)};
    params.set("k", k);
    EXPECT_THAT(k, ::testing::ElementsAreArray({AMREX_D_DECL(1.25, 1.25, 1.25)}));

    amrex::Array<double, GEMPIC_SPACEDIM> laterK{AMREX_D_DECL(0, 0, 0)};
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

// Death tests should be named as such in the testing suite
TEST(ParametersDeathTests, NoOutputFile)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    Parameters::set_print_output();
    EXPECT_EXIT(Parameters{}, testing::ExitedWithCode(Error::InvalidInput),
                "Error: Parameter outputFile not in input file!");
    Parameters::set_print_output(
        false);  // special circumstances because Parameters was technically not created and
                 // therefore not destroyed, so m_printOutput was not reset automatically.
}

// Death tests should be named as such in the testing suite
TEST(ParametersDeathTests, NoSuchParameter)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    int a;
    Parameters params{};
    EXPECT_EXIT(params.get("NonexistentVar", a), testing::ExitedWithCode(Error::InvalidInput),
                "Error: Parameter NonexistentVar not in input file!");
}

// Death tests should be named as such in the testing suite
TEST(ParametersDeathTests, AttemptedChange)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    int ghostCellBuffer{3};
    Parameters params{};
    params.get_or_set("nGhostExtra", ghostCellBuffer);
    int ghostCellInParams;
    params.get("nGhostExtra", ghostCellInParams);
    EXPECT_EQ(ghostCellBuffer, ghostCellInParams);

    EXPECT_EXIT(params.set("nGhostExtra", ghostCellBuffer),
                testing::ExitedWithCode(Error::AttemptedParameterChange),
                "Error: Parameters class attempted to set parameter nGhostExtra already set by "
                "Parameters class");
}

// Death tests should be named as such in the testing suite
TEST(ParametersDeathTests, NoGeneralInstance)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(Parameters("ClassPrefix"), testing::ExitedWithCode(Error::ParametersNotInitialized),
                "Error: Parameters class not previously initialized!");
}

// Death tests should be named as such in the testing suite
TEST(ParametersDeathTests, TwoGeneralInstances)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    Parameters params{};
    EXPECT_EXIT(Parameters::set_print_output(),
                testing::ExitedWithCode(Error::ParametersAlreadyInitialized),
                "Error: Parameters class already previously initialized!");
}
}  // namespace

/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <git.h>

#include <AMReX_Print.H>

#include "GEMPIC_Version.H"

void Gempic::Utils::print_gempicx_version ()
{
    if (git::IsPopulated())
    {
        if (git::AnyUncommittedChanges())
        {
            amrex::Warning("Warning: There were uncommitted changes at build-time");
        }
        amrex::Print() << "GEMPICX commit " << git::CommitSHA1() << '\n';
        amrex::Print() << "GEMPICX branch " << git::Branch() << '\n';
        // Split string <tag>[-<#commits>]-...
        auto gitDescribe{git::Describe()};
        auto tagEnd{gitDescribe.find("-")};
        amrex::Print() << "Tag: " << gitDescribe.substr(0, tagEnd) << '\n';
        ++tagEnd;
        auto commitNumberEnd{gitDescribe.find("-", tagEnd)};
        if (commitNumberEnd == gitDescribe.npos)
        {
            // -<#commits> not found, meaning we're exactly on a tag
            amrex::Print() << "0 commits since tag\n";
        }
        else
        {
            amrex::Print() << gitDescribe.substr(tagEnd, commitNumberEnd - tagEnd)
                           << " commits since tag\n";
        }
    }
    else
    {
        amrex::Print() << "Unknown version (repository was not fetched/installed with git?)\n";
    }
}
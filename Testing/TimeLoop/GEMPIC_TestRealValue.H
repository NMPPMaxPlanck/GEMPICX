/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/

#ifndef REALVALUE_H_
#define REALVALUE_H_

#include <filesystem>
#include <fstream>
#include <iostream>

/** Example of a "physical object" class
 *
 * In general holds some physically meaningful data structure, along with IO functionality.
 * For this simple example, a single real number
 */

class RealValue
{
private:
    double m_value;

public:
    void set (double const newValue) { m_value = newValue; }
    double value () const { return m_value; }

    int read (std::filesystem::path const fname)
    {
        std::ifstream iFile;
        iFile.open(fname, std::ios::in);
        iFile >> m_value;
        iFile.close();
        return EXIT_SUCCESS;
    }
    int write (std::filesystem::path const fname) const
    {
        std::ofstream oFile;
        oFile.open(fname, std::ios::out);
        oFile << m_value;
        oFile.close();
        return EXIT_SUCCESS;
    }
    int write_diagnostics (std::filesystem::path const fname) const
    {
        std::cout << "current sum is " << m_value << std::endl;
        std::ofstream oFile;
        oFile.open(fname, std::ios::out);
        oFile << m_value;
        oFile.close();
        return EXIT_SUCCESS;
    }
};

class RealValueFreeForm
{
private:
    double m_value;

public:
    void set (double const newValue) { m_value = newValue; }
    double value () const { return m_value; }
};

int free_form_read (RealValueFreeForm v, std::filesystem::path const fname)
{
    double valueIn;
    std::ifstream iFile;
    iFile.open(fname, std::ios::in);
    iFile >> valueIn;
    iFile.close();
    v.set(valueIn);
    return EXIT_SUCCESS;
}
int free_form_write (RealValueFreeForm v, std::filesystem::path const fname)
{
    std::ofstream oFile;
    oFile.open(fname, std::ios::out);
    oFile << v.value();
    oFile.close();
    return EXIT_SUCCESS;
}
int free_form_write_diagnostics (RealValueFreeForm v, std::filesystem::path const fname)
{
    std::cout << "current sum is " << v.value() << std::endl;
    std::ofstream oFile;
    oFile.open(fname, std::ios::out);
    oFile << v.value();
    oFile.close();
    return EXIT_SUCCESS;
}

#endif // REALVALUE_H_

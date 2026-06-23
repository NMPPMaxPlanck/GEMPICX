/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef PLACEHOLDER_IO_H_
#define PLACEHOLDER_IO_H_

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

/** Basic IO handler independent of GEMPICX
 *
 * Physical models may use any number of arbitrary data structures to describe their "state".
 * Each state data structure should provide a "read", "write" and "write_diagnostics" method.
 * These three methods are required such that each data structure object can be registered for the
 * global read/write calls made within `TimeStepperBase`.
 *
 * Individual data structure implementations may offer additional methods to configure what their
 * specific diagnostics output (i.e. "cell average" vs "interpolation" vs "L2 norm" etc).
 */
namespace PlaceholderIO
{

inline int handle_io_call (int const callResult, std::string const message)
{
    if (callResult == EXIT_SUCCESS) return EXIT_SUCCESS;
    std::cerr << "GEMPICX IO error: " << message << std::endl;
    return callResult;
}

class RegistryElementBase
{
public:
    virtual int read(std::filesystem::path const) = 0;
    virtual int write(std::filesystem::path const) = 0;
    virtual int write_diagnostics(std::filesystem::path const) = 0;
    virtual ~RegistryElementBase() {}
};

template <typename T>
concept RegistryElement = requires(T t, std::filesystem::path const fname) {
    { t.read(fname) } -> std::same_as<int>;
    { t.write(fname) } -> std::same_as<int>;
    { t.write_diagnostics(fname) } -> std::same_as<int>;
};

template <typename T>
concept FreeFormIOCapable = requires(T t, std::filesystem::path const fname) {
    { free_form_read(t, fname) } -> std::same_as<int>;
    { free_form_write(t, fname) } -> std::same_as<int>;
    { free_form_write_diagnostics(t, fname) } -> std::same_as<int>;
};

template <RegistryElement T>
class RegistryElementImpl : public RegistryElementBase
{
private:
    T* const m_dataStructure;

public:
    RegistryElementImpl(T* const inDs) : m_dataStructure(inDs) {}
    int read (std::filesystem::path const fname) final { return m_dataStructure->read(fname); }
    int write (std::filesystem::path const fname) final { return m_dataStructure->write(fname); }
    int write_diagnostics (std::filesystem::path const fname) final
    {
        return m_dataStructure->write_diagnostics(fname);
    }
};

template <FreeFormIOCapable T>
class RegistryElementFreeFormImpl : public RegistryElementBase
{
private:
    T* m_dataStructure;

public:
    RegistryElementFreeFormImpl(T* inDs) : m_dataStructure(inDs) {}
    int read (std::filesystem::path const fname) final
    {
        return free_form_read(*m_dataStructure, fname);
    }
    int write (std::filesystem::path const fname) final
    {
        return free_form_write(*m_dataStructure, fname);
    }
    int write_diagnostics (std::filesystem::path const fname) final
    {
        return free_form_write_diagnostics(*m_dataStructure, fname);
    }
};

class Registry
{
private:
    std::vector<std::unique_ptr<RegistryElementBase>> m_ds;
    std::vector<std::string> m_name;

public:
    Registry() {}
    ~Registry()
    {
        m_ds.clear();
        m_name.clear();
    }

    template <RegistryElement T>
    int add_data_structure (T* const inDs, std::string const name)
    {
        std::unique_ptr<RegistryElementBase> obj(new RegistryElementImpl<T>(inDs));
        m_ds.push_back(std::move(obj));
        m_name.push_back(name);
        return EXIT_SUCCESS;
    }

    template <FreeFormIOCapable T>
    int add_data_structure (T* inDs, std::string const name)
    {
        std::unique_ptr<RegistryElementBase> obj(new RegistryElementFreeFormImpl<T>(inDs));
        m_ds.push_back(std::move(obj));
        m_name.push_back(name);
        return EXIT_SUCCESS;
    }

    int read (std::filesystem::path const baseName)
    {
        int globalResult = EXIT_SUCCESS;
        for (unsigned int i = 0; i < m_ds.size(); i++)
        {
            auto result = m_ds[i]->read(baseName / m_name[i]);
            handle_io_call(result, "read " + (baseName / m_name[i]).string());
            if (result != EXIT_SUCCESS) globalResult = EXIT_FAILURE;
        }
        return globalResult;
    }

    int write (std::filesystem::path const baseName)
    {
        int globalResult = EXIT_SUCCESS;
        for (unsigned int i = 0; i < m_ds.size(); i++)
        {
            auto result = m_ds[i]->write(baseName / m_name[i]);
            handle_io_call(result, "write " + (baseName / m_name[i]).string());
            if (result != EXIT_SUCCESS) globalResult = EXIT_FAILURE;
        }
        return globalResult;
    }

    int write_diagnostics (std::filesystem::path const baseName)
    {
        int globalResult = EXIT_SUCCESS;
        for (unsigned int i = 0; i < m_ds.size(); i++)
        {
            auto result = m_ds[i]->write_diagnostics(baseName / m_name[i]);
            handle_io_call(result, "write_diagnostics " + (baseName / m_name[i]).string());
            if (result != EXIT_SUCCESS) globalResult = EXIT_FAILURE;
        }
        return globalResult;
    }
};
} //namespace PlaceholderIO

#endif // PLACEHOLDER_IO_H_

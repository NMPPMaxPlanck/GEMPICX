#ifndef PLACEHOLDER_IO_H_
#define PLACEHOLDER_IO_H_

#include <cassert>
#include <cstdint>
#include <filesystem>
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

class RegistryElement
{
public:
    virtual int read(std::filesystem::path const) = 0;
    virtual int write(std::filesystem::path const) = 0;
    virtual int write_diagnostics(std::filesystem::path const) = 0;
    virtual ~RegistryElement() {}
};

template <typename T>
concept fitsRegistryElement = requires(T t, std::filesystem::path const fname) {
    { t.read(fname) } -> std::same_as<int>;
    { t.write(fname) } -> std::same_as<int>;
    { t.write_diagnostics(fname) } -> std::same_as<int>;
    { t.extension() } -> std::same_as<std::string>;
};

template <fitsRegistryElement T>
class RegistryElementImpl : public RegistryElement
{
private:
    T* const m_dataStructure;

public:
    RegistryElementImpl(T* const inDs) : m_dataStructure(inDs) {}
    int read (std::filesystem::path const fname) final
    {
        std::filesystem::path tempPath(fname);
        tempPath += "." + m_dataStructure->extension();
        return m_dataStructure->read(tempPath);
    }
    int write (std::filesystem::path const fname) final
    {
        std::filesystem::path tempPath(fname);
        tempPath += "." + m_dataStructure->extension();
        return m_dataStructure->write(tempPath);
    }
    int write_diagnostics (std::filesystem::path const fname) final
    {
        std::filesystem::path tempPath(fname);
        tempPath += "." + m_dataStructure->extension();
        return m_dataStructure->write_diagnostics(tempPath);
    }
};

class Registry
{
private:
    std::vector<RegistryElement*> m_ds;
    std::vector<std::string> m_name;

public:
    Registry() {}
    ~Registry()
    {
        for (auto* ds : m_ds) delete ds;
        m_ds.clear();
        m_name.clear();
    }
    template <fitsRegistryElement T>
    int add_data_structure (T* const inDs, std::string const name)
    {
        RegistryElement* obj = new RegistryElementImpl<T>(inDs);
        m_ds.push_back(obj);
        m_name.push_back(name);
        return EXIT_SUCCESS;
    }

    int read (std::filesystem::path const baseName)
    {
        int globalResult = EXIT_SUCCESS;
        for (auto i = 0; i < m_ds.size(); i++)
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
        for (auto i = 0; i < m_ds.size(); i++)
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
        for (auto i = 0; i < m_ds.size(); i++)
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

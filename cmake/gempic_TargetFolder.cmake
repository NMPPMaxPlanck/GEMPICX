function(get_all_targets_recursive out_var dir)
    # Get targets in current directory
    get_property(local_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)

    set(all_targets ${local_targets})

    # Get subdirectories
    get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)

    foreach(subdir ${subdirs})
        get_all_targets_recursive(sub_targets ${subdir})
        list(APPEND all_targets ${sub_targets})
    endforeach()

    # Return result
    set(${out_var} ${all_targets} PARENT_SCOPE)
endfunction()


function(assign_folders_by_origin)
    get_all_targets_recursive(all_targets ${CMAKE_SOURCE_DIR})
    # List of known meta-targets or interface-only targets to skip warnings for
    set(SKIP_FOLDER_WARNINGS
        GEMPICX
        gempic_linalg
    )

    foreach(t ${all_targets})
        get_target_property(_imported ${t} IMPORTED)
        get_target_property(_alias ${t} ALIAS)
        if(_imported OR _alias)
            message(STATUS "Skipping target ${t} (IMPORTED or ALIAS)")
            continue()
        endif()

        get_target_property(_src ${t} SOURCE_DIR)

        if(NOT _src)
            if(NOT t IN_LIST SKIP_FOLDER_WARNINGS)
                message(WARNING "Target ${t} doesn't match any pattern and has no source dir")
            endif()
            continue()
        endif()

        # Normalize path
        file(TO_CMAKE_PATH "${_src}" _src_norm)

        # --- Third party ---
        if(_src_norm MATCHES "third_party/amrex(-.*)?$")
            set_target_properties(${t} PROPERTIES FOLDER "ThirdParty/AMReX")
        elseif(_src_norm MATCHES "third_party/hdf5(-.*)?$")
            set_target_properties(${t} PROPERTIES FOLDER "ThirdParty/HDF5")
        elseif(_src_norm MATCHES "third_party/lapack(-.*)?$")
            set_target_properties(${t} PROPERTIES FOLDER "ThirdParty/LAPACK")

        elseif(_src_norm MATCHES "third_party/googletest(-.*)?$")
            set_target_properties(${t} PROPERTIES FOLDER "ThirdParty/GTest")

        elseif(_src_norm MATCHES "third_party")
            set_target_properties(${t} PROPERTIES FOLDER "ThirdParty/Other")

        # --- Your project ---
        elseif(_src_norm MATCHES "/Src(/|$)")
            set_target_properties(${t} PROPERTIES FOLDER "GEMPIC")

        elseif(_src_norm MATCHES "/Examples/")
            set_target_properties(${t} PROPERTIES FOLDER "GEMPIC/Examples")

        elseif(_src_norm MATCHES "/Testing(/|$)")
            set_target_properties(${t} PROPERTIES FOLDER "GEMPIC/Tests")
        else()
            if(NOT t IN_LIST SKIP_FOLDER_WARNINGS)
                message(WARNING "Target ${t} doesn't match any pattern, from ${_src}")
                set_target_properties(${t} PROPERTIES FOLDER "GEMPIC/Misc")
                message(STATUS "Assigning target ${t} to GEMPIC/Misc")
            endif()
        endif()
    endforeach()
    # main library
    set_target_properties(GEMPICX PROPERTIES FOLDER "GEMPIC")
endfunction()
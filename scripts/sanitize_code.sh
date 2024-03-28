#!/bin/bash
set -e

# --------------------------------------------------------------------------------------------------
# Read build directory and include libraries from scripts/build_dir.txt
# --------------------------------------------------------------------------------------------------
read_build_dir_file ()
{
    if [[ $BASH_SOURCE = */* ]]; then
        BUILD_DIR=$(head -n 1 ${BASH_SOURCE%/*}/build_dir.txt)
        inclLibs=$(sed -n '2s/--/-/gp' ${BASH_SOURCE%/*}/build_dir.txt )
    else
        BUILD_DIR=$(head -n 1 ./build_dir.txt)
        inclLibs=$(sed -n '2s/--/-/gp' ./build_dir.txt )
    fi
}

# --------------------------------------------------------------------------------------------------
# Enforces consistent folder and C++ file naming
# --------------------------------------------------------------------------------------------------
fix_folder_names ()
{
    # Ensure folders are snake_case:
    # Find all folders
    for folder in $(find $1 -maxdepth 1 -type d -print)
    do
        # ... starting with lowercase or with underscores in their name?
        if [[ $(echo $folder | grep -E "/[a-z]|_") ]]
        then
            # replace snake_case with CamelCase
            # \u is GNU sed, doesn't work on mac
            # newName=$(echo $folder | sed -E -e 's#/([a-z])#/\u\1#g' -e 's#_+([[:alnum:]])#\u\1#g')
            # perl is default installed on mac and linux
            newName=$(echo $folder | sed 's#/\([a-z]\)#/_\1#g' | perl -pe 's#_+([\dA-z])#\u$1#g')
            
            # Indirection hack because MacOS names ignore case
            git mv $folder "TMPCOPY_$folder$folder"

            if [[ -d $newName ]]
            then
              echo "Cannot fix incorrect folder name '$folder' because '$newName' already exists!"
              echo "Moved '$folder' to 'TMPCOPY_$folder$folder'"
              exit 1
            fi

            git mv "TMPCOPY_$folder$folder" $newName
            
            echo "Fixed $folder to $newName because folders are CamelCase."

            # An automatic solution would be hard to look at and prone to bugs
            if [[ $(grep -nR "\b${folder##*/}\b" . --include=CMakeLists.txt --exclude-dir=third_party --exclude-dir=*build*) ]]
            then
                echo "Remember to update CMakeLists.txt files pointing to this folder:"
                grep -nR "\b${folder##*/}\b" . --include=CMakeLists.txt --exclude-dir=third_party --exclude-dir=*build*
            fi
            
            # Launch subfolder search with changed folder name
            fix_folder_names $newName
        # Folder conforming to name scheme
        else
            if [[ $folder != $1 ]] # Avoid infinite recursion ...
            then
                # Folder name was fine, so we simply search subfolders
                fix_folder_names $folder
            fi
        fi

    done
}


GEMPIC_fix_folder_file_names ()
{
    # Ensure code folders are CamelCase
    fix_folder_names "./Src"
    fix_folder_names "./Testing"
    #fix_folder_names "./Examples"

    # Ensure all code files follow one of three patterns:
    # A) GEMPIC_CamelCase.H or GEMPIC_CamelCase.cpp
    # B) CamelCase_test.cpp
    # C) test_aNy_StYlE.cpp (deprecated)
    # for file in $(find ./Src/ ./Testing/ ./Examples/ -name "*.H" -or -name "*.cpp" -type f |
    for file in $(find ./Src/ ./Testing/ -name "*.H" -or -name "*.cpp" -type f |
                  grep -Ev -- '/GEMPIC_[A-Z][a-zA-Z]*\.|/[A-Z][a-zA-Z]*_test\.cpp$|/test\w*\.cpp$')
    do
        # Remove underscores and convert following letters to uppercase
        newName=$(echo $file | sed -E -e 's#(.*/)(.)#\1\u\2#' -e 's/_(\w)/\u\1/g')
        # \u is GNU sed, doesn't work on mac
        # newName=$(echo $file | sed -E -e 's#/([a-z])#/\u\1#g' -e 's#_+([[:alnum:]])#\u\1#g')
        # perl is default installed on mac and linux
        newName=$(echo $file | sed 's#\(.*/\)#\1_#g' | perl -pe 's#_+([\dA-z])#\u$1#g')
        # Is it likely to be a test file?
        if [[ $(echo $file | grep '\./Testing/.*\.cpp') ]]
        then
            # Add _test postfix
            newName=$(echo $newName | sed -E 's#(Test)?(\.cpp$)#_test\1#I')
        else
            # Add GEMPIC_ prefix
            newName=$(echo $newName | sed -E 's#(.*/)(GEMPIC)?#\1GEMPIC_#I')

        fi
        # Rename file
        # Indirection hack because MacOS names ignore case
        git mv $file "TMPCOPY_$file$file"
        git mv "TMPCOPY_$file$file" $newName

        # Old file name (with extension, without path) and extension
        oldFileName="${file##*/}"
        oldFileExt="${oldFileName##*.}"
        # Source or header file?
        if [[ "$oldFileExt" == "cpp" ]]
        then
            # For source (.cpp) files, update CMakeLists.txt files accordingly
            newFileName="${newName##*/}"
            grep -lnR "$oldFileName" . --include=CMakeLists.txt --exclude-dir=third_party |
            xargs sed -i'' "s/$oldFileName/$newFileName/g"
        else
            # For header (.H) files, update #include statements accordingly
            newFileName="${newName##*/}"
            grep -lnR "^#include .[^\">]*$oldFileName" . --include=*.H --include=*.cpp --exclude-dir=third_party |
            xargs sed -i'' -E "s/(^#include ).([^\">]*)$oldFileName[\">]/\1\"\2$newFileName\"/"
        fi

        echo "Attempted to fix the name of $file to $newName because the style was not one of:"
        echo "A) GEMPIC_CamelCase.H or GEMPIC_CamelCase.cpp"
        echo "B) CamelCase_test.cpp"
        echo "C) test_aNy_StYlE.cpp (deprecated)"
    done
}

# --------------------------------------------------------------------------------------------------
# Sanitizes code using clang-tidy
# --------------------------------------------------------------------------------------------------
GEMPIC_run_clang_tidy ()
{

    # Avoid clang-tidy checking third_party files at all, speeding up the process by a factor of ~5
    # Do this by removing all compilation instructions for third_party source files in compile_commands.json.
    # Remove lines where the file to be compiled is in the third_party folder
    sed -E 's#^\s*.file.: .[[:alnum:]/\_\-]*/third_party/[[:alnum:]/\_.\-]*\.(cpp|cc).,?\s*$##g' $BUILD_DIR/compile_commands.json > $BUILD_DIR/compile_commandsNoThirdParty.json
    # Remove translation units where the file specification is missing.
    # perl is default installed on mac and linux
    perl -i'' -p0e 's#,?[\r\n]\{\s*.directory[^\n\r]*\s*"command[^\r\n]*\s*("output[^\r\n]*\s*)?\}##g' $BUILD_DIR/compile_commandsNoThirdParty.json
    # sed -i'' -zE 's#,?\n\{\s*.directory[^\n]*\s*"command[^\n]*\s*("output[^\n]*\s*)?\}##g' $BUILD_DIR/compile_commandsNoThirdParty.json
    # Remove the initial comma in case a third_party translation unit was the first one in compile_commands.json
    sed -E -i'' 's/\[,/[/' $BUILD_DIR/compile_commandsNoThirdParty.json
    # Pry out nvcc compiler arguments (--nvcc-flags) and other stuff clang doesn't understand
    sed -E -i'' -e 's# --[^ ]*##g' -e 's# -(Xcudafe|ccbin|forward-unknown-to-host|maxrregcount|rdc)[^ ]*##g' $BUILD_DIR/compile_commandsNoThirdParty.json
    # Explicitly convert googletest modules to normal includes. Needed for newer cmake versions.
    sed -E -i'' 's#-isystem=([^ ]*google)#-I\1#g' $BUILD_DIR/compile_commandsNoThirdParty.json
    cp $BUILD_DIR/compile_commands.json $BUILD_DIR/_full_compile_commands.json
    cp $BUILD_DIR/compile_commandsNoThirdParty.json $BUILD_DIR/compile_commands.json

    # Run clang tidy on all files
    run-clang-tidy -quiet $inclLibs -fix -p $BUILD_DIR 2>&1 | tee tidyOutput.txt

    ## Fix the destroyed lambda captures
    # clang-tidy has the unfortunate side effect that lambdas containing variables whose names are fixed have their capture sections destroyed.
    # [=] (...) { return 2*bad_variable; } -> [badVariable] (...) { return 2 * badVariable; }
    # This undoes the change for AMREX_GPU lambdas.
    # Use git diff --name-only --cached for staged changes
    grep -inIR "\[[,\&[:alnum:]]*\] AMREX_GPU" $(git diff --name-only) --include=*.H --include=*.cpp --exclude-dir=third_party | \
    while read -r line ; do
        #echo "Repairing $line"
        fileName=$(echo "$line" | cut -d':' -f1)
        lineNum=$(echo "$line" | cut -d':' -f2)
        sed -E -i'' "$lineNum s/\[[^]]*\] AMREX_GPU/[=] AMREX_GPU/" $fileName
    done
}

# --------------------------------------------------------------------------------------------------
# Sanitizes code using clang-format
# --------------------------------------------------------------------------------------------------
GEMPIC_run_clang_format ()
{
    #for file in $(find ./Src/ ./Testing/ ./Examples/ -name "*.H" -or -name "*.cpp" -type f)
    for file in $(find ./Src/ ./Testing/ -name "*.H" -or -name "*.cpp" -type f)
    do
        # Enforce style: #include "GEMPIC_File.H" or #include <notAGempicFile>
        sed -E -i'' 's/(^#include )"([^"]*)"/\1<\2>/g' $file
        sed -E -i'' 's/(^#include )<([^>]*GEMPIC_[^>]*\.H)>/\1"\2"/g' $file

        clang-format -i -style=file $file
        # This is a hotfix to a bug in clang-format around the formatting of lambdas inside function calls.
        # This _sometimes_ happens, and it's difficult to say when:
        # function(..., [whatever] (...)     ->    function(..., [whatever] (...) {
        # {                                  ->    
        # The hotfix introduces the correct formatting.
        #
        # The simplified call without possibility of (one level of) nested (...) inside lambda arguments is:
        # sed -Ei 's#^((\s*)\[[^\)]*\)+) \{#\1\n\2{#' $file
        # The more complicated version is needed for cases such as
        # function(..., [whatever] (..., AMREX_D_DECL(x, y, z), ...) {
        #
        #           space '[  ... (' >= 0 nested '(...)' pairs (one level of nesting only) '...) {'
        sed -E -i'' 's#^((\s*)\[[^(]*\(([^)({]*\([^()]*\))*[^(){]*\)) \{$#\1\n\2{#' $file

        # Yet another hotfix to the bug that clang-format doesn't recognize scoped function definitions and thus removes the spaceBeforeParens.
        # The fix accidentally adds a space to multiline scoped function declarations, i.e.
        # void nameSpaceName::fnc (..., ..., ...,
        #                          ...);
        # which are, however, rare enough as to be ignored here.
        #
        #               _return type_  _class/namespace_::_funcName_ _args_
        sed -E -i'' 's/^(\s*[^ =\(\)/]{1,} \w*(::\w*){1,})(\([^;]*)$/\1 \3/' $file
        # Template specification declarations:
        # void className<tVar1, tVar2>::fnc(...
        sed -E -i'' 's/^(\s*[^ =\(\)/]{1,} \w*(<(\w|[, :])*>{1,})?(::\w*(<(\w|[, :])*>{1,})?){1,})(\([^;]*)$/\1 \7/' $file
    done
}

# --------------------------------------------------------------------------------------------------
# Clean up the mess
# --------------------------------------------------------------------------------------------------
cleanup ()
{
    # Undo the changes to the compile_commands.json file
    if test -f "$BUILD_DIR/_full_compile_commands.json"; then
        mv $BUILD_DIR/_full_compile_commands.json $BUILD_DIR/compile_commands.json
    fi
    # Otherwise, sourcing the script makes the terminal crash upon next error
    set +e
}


# --------------------------------------------------------------------------------------------------
# Main begins
# --------------------------------------------------------------------------------------------------
read_build_dir_file
trap cleanup EXIT
# --------------------------------------------------------------------------------------------------
# Actual sanitation. Comment or uncomment desired parts. It's recommended to do -format after -tidy
# It's also recommended to run this after setting the CMAKE configure settings to 3D debug.

GEMPIC_fix_folder_file_names

GEMPIC_run_clang_tidy

GEMPIC_run_clang_format

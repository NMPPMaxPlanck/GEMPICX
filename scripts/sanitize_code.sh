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
# Sanitizes code using clang-tidy
# --------------------------------------------------------------------------------------------------
GEMPIC_run_clang_tidy ()
{

    # Avoid clang-tidy checking third_party files at all, speeding up the process by a factor of ~5
    # Do this by removing all compilation instructions for third_party source files in compile_commands.json.
    # Remove lines where the file to be compiled is in the third_party folder
    sed -E 's#^\s*.file.: .[[:alnum:]/\_\-]*/third_party/[[:alnum:]/\_.\-]*\.(cpp|cc).,?\s*$##g' $BUILD_DIR/compile_commands.json > $BUILD_DIR/compile_commandsNoThirdParty.json
    # Remove translation units where the file specification is missing.
    sed -zEi 's#,?\n\{\s*.directory[^\n]*\s*"command[^\n]*\s*("output[^\n]*\s*)?\}##g' $BUILD_DIR/compile_commandsNoThirdParty.json
    # Remove the initial comma in case a third_party translation unit was the first one in compile_commands.json
    sed -Ei 's/\[,/[/' $BUILD_DIR/compile_commandsNoThirdParty.json
    # Pry out nvcc compiler arguments (--nvcc-flags) and other stuff clang doesn't understand
    sed -Ei -e 's# --[^ ]*##g' -e 's# -(Xcudafe|ccbin|forward-unknown-to-host|maxrregcount|rdc)[^ ]*##g' $BUILD_DIR/compile_commandsNoThirdParty.json
    # Explicitly convert googletest modules to normal includes. Needed for newer cmake versions.
    sed -Ei 's#-isystem=([^ ]*google)#-I\1#g' $BUILD_DIR/compile_commandsNoThirdParty.json
    cp $BUILD_DIR/compile_commands.json $BUILD_DIR/_full_compile_commands.json
    cp $BUILD_DIR/compile_commandsNoThirdParty.json $BUILD_DIR/compile_commands.json

    # Run clang tidy on all files
    run-clang-tidy -quiet $inclLibs -fix -p $BUILD_DIR 2>&1 | tee tidyOutput.txt

    ## Fix the destroyed lambda captures
    # clang-tidy has the unfortunate side effect that lambdas containing variables whose names are fixed have their capture sections destroyed.
    # [=] (...) { return 2*bad_variable; } -> [badVariable] (...) { return 2 * badVariable}
    # This undoes the change for AMREX_GPU lambdas.
    # Use git diff --name-only --cached for staged changes
    grep -inIR "\[[,\&[:alnum:]]*\] AMREX_GPU" $(git diff --name-only) --include=*.H --include=*.cpp --exclude-dir=third_party | \
    while read -r line ; do
        #echo "Repairing $line"
        fileName=$(echo "$line" | cut -d':' -f1)
        lineNum=$(echo "$line" | cut -d':' -f2)
        sed -Ei "$lineNum s/\[[^]]*\] AMREX_GPU/[=] AMREX_GPU/" $fileName
    done

    ## Attempt to fix the destroyed gempic_ prefix
    # Use git diff --name-only --cached for staged changes
    grep -nIR "[Gg]empic[A-Z]" $(git diff --name-only) --include=*.H --include=*.cpp --exclude-dir=third_party | \
    while read -r line ; do
        echo "clang-tidy fixed a gempic_ prefix because the rest of the name was in the wrong case."
        fileName=$(echo "$line" | cut -d':' -f1)
        lineNum=$(echo "$line" | cut -d':' -f2)
        echo "Please check that the correct fix was applied to $fileName:$lineNum."
        # This fix fails where we invoke the namespace literally ( GEMPIC_Something::aFunction(); )
        sed -Ei "$lineNum s/namespace Gempic/namespace GEMPIC_/" $fileName # namespace GEMPIC_Name
        sed -Ei "$lineNum s/(Gempic)([A-Z])/\1_\2/" $fileName # class Gempic_ClassName
        sed -Ei "$lineNum s/(gempic)([A-Z])/\1_\L\2/" $fileName # function gempic_doSomething
    done
}

# --------------------------------------------------------------------------------------------------
# Sanitizes code using clang-format
# --------------------------------------------------------------------------------------------------
GEMPIC_run_clang_format ()
{
    for file in $(find ./src/ ./Testing/ -name "*.H" -or -name "*.cpp" -type f)
    do
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
        sed -Ei 's#^((\s*)\[[^(]*\(([^)({]*\([^()]*\))*[^(){]*\)) \{$#\1\n\2{#' $file

        # Yet another hotfix to the bug that clang-format doesn't recognize scoped function definitions and thus removes the spaceBeforeParens.
        # The fix accidentally adds a space to multiline scoped function declarations, i.e.
        # void nameSpaceName::fnc (..., ..., ...,
        #                          ...);
        # which are, however, rare enough as to be ignored here.
        #
        #               _return type_  _class/namespace_::_funcName_ _args_
        sed -Ei 's/^(\s*[^ =\(\)/]{1,} \w*(::\w*){1,})(\([^;]*)$/\1 \3/' $file
        # Template specification declarations:
        # void className<tVar1, tVar2>::fnc(...
        sed -Ei 's/^(\s*[^ =\(\)/]{1,} \w*(<(\w|[, :])*>{1,})?(::\w*(<(\w|[, :])*>{1,})?){1,})(\([^;]*)$/\1 \7/' $file
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
# Actual sanitation. Comment or uncomment desired parts. It's recommended to do -tidy before -format
# It's also recommended to run this after setting the CMAKE configure settings to 3D debug.
GEMPIC_run_clang_tidy

GEMPIC_run_clang_format

import argparse
import fileinput
import json
import os
import pathlib
import re
import subprocess
import warnings

def warning_format_no_traceback(message, category, filename, lineno, line=None):
    return f'{category.__name__}: {message}\n'

warnings.formatwarning = warning_format_no_traceback

def conservative_capitalize(arg: str) -> str:
    """
    Capitalizes but does not convert other characters to lowercase.
    """
    strlen = len(arg)
    if strlen == 0:
        return arg
    elif strlen == 1:
        return arg.upper()
    else:
        return arg[0].upper() + arg[1:]

def convert_to_camel_case(arg: str) -> str:
    """
    Converts a string to CamelCase, maintaining capitalised letters
    """
    return ''.join([conservative_capitalize(word) for word in arg.split('_')])

def convert_to_gempic_name(arg: str) -> str:
    """
    Converts filenames to 'GEMPIC_CamelCase'
    """
    gempicPrefix = re.compile('^gempic_', re.IGNORECASE)
    arg = re.sub(gempicPrefix, '', arg)
    
    arg = 'GEMPIC_' + convert_to_camel_case(arg)
    return arg

def convert_to_test_name(arg: str) ->str:
    """
    Converts filenames to 'CamelCase_test'
    """
    testSuffix = re.compile('_test$', re.IGNORECASE)
    arg = re.sub(testSuffix, '', arg)
    
    arg = convert_to_camel_case(arg) + '_test'
    return arg

def execute_git_mv(source, destination, forceMove=False):
    """
    Moves a file or folder (source) to destination using git.
    If forceMove is on, 
      the file is added before being renamed so git knows it
      the folder has an empty file added for the same purpose
        the empty file is removed afterwards.
    Returns
      a message detailing the operation performed
      a boolean describing whether or not (source) was an empty folder and thus removed
    """
    source = pathlib.Path(source).resolve()
    destination = pathlib.Path(destination).resolve()

    created_file_name=''
    if forceMove:
        if source.is_file():
            subprocess.run(f'git add {str(source)}', shell=True)
        else:
            created_file_name = '__madeBySanitizeWithPython__.txt'
            created_file=pathlib.Path(source, created_file_name)
            created_file.touch()
            subprocess.run(f'git add {str(created_file)}', shell=True)

    # The actual call
    gitmvout = subprocess.run(f'git mv {str(source)} {str(destination)}', shell=True, capture_output=True, text=True)
    if gitmvout.returncode == 0:
        msg = f'{str(source)} was moved to {str(destination)}.'
    else:
        msg = f'The command failed with the message:\t{gitmvout.stderr}'
        msg += "(You probably haven't staged your changes before running the script. This is highly recommended!)\n"
        msg += '(Alternately, you can try using the --forceMove option)'
        raise RuntimeError(msg) 

    # Clean up
    removedDir=False
    if created_file_name:
        new_created_file = destination / created_file_name
        new_created_file.unlink()
        if not os.listdir(str(destination)):
            # Directory was empty
            destination.rmdir()
            msg = f'The badly named directory {str(source)} was empty and should be gone now.'
            removedDir=True
        subprocess.run(f'git add -u {str(new_created_file)}', shell=True)
        
    return msg, removedDir

def correct_in_files(oldPattern, newPattern, fileList, linenums=[]):
    """
    Converts occurrences of oldPattern in a given file (list) into newPattern.
    linenums marks the line number(s) to be checked if present. Otherwise, all lines are checked.
    """
    if isinstance(oldPattern, str):
        oldPattern = re.compile(oldPattern)
    if isinstance(fileList, str) or isinstance(fileList, pathlib.Path):
        fileList = [pathlib.Path(fileList).resolve()]
    if linenums:
        if not isinstance(linenums, list):
            linenums = [linenums]
        linenums = [number if isinstance(number, int) else int(number) for number in linenums]

    with fileinput.input(files=fileList, inplace=True) as input:
        if linenums:
            for line in input:
                if fileinput.lineno() in linenums:
                    changedLine = re.sub(oldPattern, newPattern, line)
                    print(changedLine, end='')
                else:
                    print(line, end='')
        else:
            for line in input:
                changedLine = re.sub(oldPattern, newPattern, line)
                print(changedLine, end='')

def find_files(globPatterns, folders=['Src', 'Testing', 'Examples'], searchMainDir=True) -> list[pathlib.Path]:
    """
    Searches folders recursively for files matching one or more glob patterns.
    Calls correct_in_file to replace oldPattern with newPattern.
    If searchMainDir is True, also searches the main directory non-recursively.
    """
    if isinstance(globPatterns, str):
        globPatterns = [globPatterns]

    if isinstance(folders, str):
        folders = [folders]
    # Collect all files matching the glob patterns
    filelist = []
    for globPattern in globPatterns:
        if searchMainDir:
            filelist += list(pathlib.Path.cwd().resolve().glob(globPattern))
        for subfolder in folders:
            filelist += list(pathlib.Path(subfolder).resolve().rglob(globPattern))

    return filelist 

def fix_folders_syntax(folderNames, forceMove=False):
    """
    Ensure all subfolders of folderName are CamelCase
    """
    if isinstance(folderNames, str) or isinstance(folderNames, pathlib.Path):
        folderNames = [pathlib.Path(folderNames).resolve()]

    for folderName in folderNames:
        for root, dirs, _ in os.walk(folderName):
            for dirName in dirs:
                # Construct CamelCase name
                newName = convert_to_camel_case(dirName)
                # Restores case of the path if it was lost
                dirName = pathlib.Path(root, dirName).resolve()
                if dirName.name != newName: # The casing is wrong
                    newName = dirName.with_name(newName)
                    if newName.name in os.listdir(root): # The correctly cased folder already exists
                        tmpName = newName / ('Duplicated' + newName.name)
                        
                        # Cover the eventuality that more duplicated folders exist.
                        counter = 1
                        while tmpName.name in os.listdir(newName):
                            counter += 1
                            tmpName = newName / ('Duplicated' + newName.name + str(counter))
                        
                        gitmvout, removedDir = execute_git_mv(dirName, tmpName, forceMove)
                        if (not removedDir):
                            # A non-empty folder exists and so does the correct name. User action required.
                            msg = f'Tried to fix folder {str(dirName)} but a folder with the correct casing already exists: {str(newName)}.\n{gitmvout}'
                            raise IsADirectoryError(msg)
                    else:
                        # Two calls necessary for systems that usually don't care about case
                        tmpName = dirName.with_name('_' + dirName.name)
                        gitmvout, removedDir = execute_git_mv(dirName, tmpName, forceMove)
                        print(gitmvout)

                        if not removedDir:
                            # The wrongly cased folder was not empty and we need to continue our changes
                            gitmvout, _ = execute_git_mv(tmpName, newName, forceMove)
                            print(gitmvout)

                            # Point out the corresponding occurrences in CMakeLists.txt files
                            # Fixing these automatically is too complicated
                            occurrences = []
                            dirNamePattern = re.compile(rf'\b{dirName.name}\b')
                            for fileName in pathlib.Path.cwd().rglob('CMakeList.txt'):
                                with open(fileName) as file:
                                    for lineNum, line in enumerate(file):
                                        if dirNamePattern.match(line):
                                            occurrences.append({'lineNum': lineNum, 'fileName': str(fileName)})
                                            break
                            if occurrences:
                                warningMsg = "Remember to change the folder names the following places (if applicable):\n"
                                for occurence in occurrences:
                                    warningMsg += f"{occurence['fileName']}:{occurence['lineNum']}\n"
                                warnings.warn(warningMsg)

                            # Recurse with changed folder name
                            fix_folders_syntax(newName, forceMove)

def fix_files_syntax(folders, forceMove=False):
    """
    Ensure code files have one of the following styles
    A) GEMPIC_CamelCase.H or GEMPIC_CamelCase.cpp
    B) CamelCase_test.cpp
    C) test_aNy_StYlE.cpp (deprecated)
    """
    if isinstance(folders, str) or isinstance(folders, pathlib.Path):
        folders = [folders]
    
    srcFolder = pathlib.Path('Src').resolve()
    testingFolder = pathlib.Path('Testing').resolve()

    replacementMade = False
    nameCamelCase = re.compile('^[A-Z][a-zA-Z]*$')
    gempicCamelCase = re.compile('^GEMPIC_[A-Z][a-zA-Z]*$')
    for folderName in folders:
        folderName = pathlib.Path(folderName).resolve()
        for file in folderName.rglob('*.H'):
            if gempicCamelCase.fullmatch(file.stem):
                continue
            # Filename does not match gempic file name pattern and is in source folders
            elif (folderName.resolve().is_relative_to(srcFolder) or folderName.resolve().is_relative_to(testingFolder)):
                newName = convert_to_gempic_name(file.stem) + file.suffix
            # Filename is outside gempic source folders and CamelCase
            elif nameCamelCase.fullmatch(file.stem):
                continue
            # Filename is outside gempic source folders and not CamelCase
            else:
                newName = convert_to_camel_case(file.stem) + file.suffix

            gitmvout, _, = execute_git_mv(file, file.with_name(newName), forceMove)
            print(gitmvout)
            replacementMade = True

            # Update #include statements accordingly
            codeFiles = find_files(['*.H', '*.cpp'])
            correct_in_files(rf'(^#include ).([^">]*){file.name}[">]', rf'\1"\2{newName}"', codeFiles)
        
        testCase = re.compile('^[A-Z][a-zA-Z]*_test$')
        outdatedTestCase = re.compile(r'^test\w*$')
        for file in folderName.rglob('*.cpp'):
            if gempicCamelCase.fullmatch(file.stem) or testCase.fullmatch(file.stem):
                continue
            elif outdatedTestCase.fullmatch(file.stem):
                warnings.warn(f"Warning: The naming (and probably type) of the test file '{str(file)}' is deprecated.")
                continue
            # Make an educated guess as to the correct format
            elif folderName.resolve().is_relative_to(srcFolder):
                newName = convert_to_gempic_name(file.stem) + file.suffix
            elif folderName.resolve().is_relative_to(testingFolder):
                newName = convert_to_test_name(file.stem) + file.suffix
            # Filename is outside gempic source folders and CamelCase
            elif nameCamelCase.fullmatch(file.stem):
                continue
            # Filename is outside gempic source folders and not CamelCase
            else:
                newName = convert_to_camel_case(file.stem) + file.suffix

            gitmvout, _, = execute_git_mv(file, file.with_name(newName), forceMove)
            print(gitmvout)
            replacementMade = True
            
            # Update CMakeLists.txt files accordingly
            cmakelistFiles = find_files(['CMakeLists.txt'])
            correct_in_files(file.name, newName, cmakelistFiles)
    
    if replacementMade:
        print("Attempted to fix the name of one or more files because the style was not one of:")
        print("A) GEMPIC_CamelCase.H or GEMPIC_CamelCase.cpp ('Src' or 'Testing' folder)")
        print("B) CamelCase_test.cpp ('Testing' folder)")
        print("C) test_aNy_StYlE.cpp (deprecated)")
        print("D) CamelCase.H or CamelCase.cpp (other folders)")

def exclude_subproject_clang_tools(subProjectFolders) -> tuple:
    """
    Renames .clang-tidy and .clang-format files in certain folders into unused.clang-...
    This is done to avoid them being used by our clang-tidy/-format processes, because the writers
    of these tools couldn't be bothered to include a mechanism to ignore specific directories ...
    Returns a tuple of two lists:
    1) The original file names
    2) The renamed file names
    """
    if not isinstance(subProjectFolders, list):
        subProjectFolders = [subProjectFolders]
    # Collect all files matching the glob patterns
    filelist = []
    for subfolder in subProjectFolders:
        filelist += list(pathlib.Path(subfolder).resolve().rglob('.clang-*'))

    oldNames = []
    newNames = []
    for file in filelist:
        oldNames += [file.absolute()]
        newFileName = file.with_name('unused'+file.name) 
        file.rename(newFileName)
        newNames += [newFileName]
    
    return oldNames, newNames
    

def undo_exclude_subproject_clang_tools(changedFiles, originalNames):
    """
    Undoes the changes made by exclude_subproject_clang_tools
    """
    if not isinstance(changedFiles, list):
        changedFiles = [changedFiles]
    if not isinstance(originalNames, list):
        originalNames = [originalNames]
    if len(changedFiles) != len(originalNames):
        raise RuntimeError("undo_exclude_subproject_clang_tools needs the same number of changed filenames as original filenames.")
    for file, original in zip(changedFiles, originalNames):
        pathlib.Path(file).resolve().rename(original)

def clang_version_is_ok(tidyOrFormat: str, minVersion=14, maxVersion=17) -> bool:
    """
    Checks if Clang-Tidy or Clang-Format is
    A) Available
    B) An acceptable version
    """
    try:
        output=subprocess.run("clang-tidy --version", shell=True, capture_output=True, text=True)
        version=int(re.sub(r'^.*?version (\d+)\.\d+\..*$', r'\g<1>', output.stdout.replace('\n','')))
        if minVersion <= version <= maxVersion:
            return True
        else:
            msg=f'clang-{tidyOrFormat} version is {version}, but accepted versions are {minVersion}-{maxVersion}. Skipping run.'
    except (FileNotFoundError):
        msg=f'clang-{tidyOrFormat} was not found. Did you install it?'
    warnings.warn(msg)
    return False

def read_build_info() -> tuple[pathlib.Path, str]:
    """
    Get build directory and implicitly included libraries necessary for clang.
    The scripts/build_dir.txt file is created at CMake configuration time.
    """
    buildInfoFile = (pathlib.Path('scripts') / 'build_dir.txt').resolve()
    try:
        with open(buildInfoFile) as file:
            buildDir = pathlib.Path(file.readline().rstrip('\r\n')).resolve()
            includeLibs = file.readline().replace('--', '-')
    except FileNotFoundError:
        raise RuntimeError("scripts/build_dir.txt doesn't exist. Have you configured with CMake?")
    return buildDir, includeLibs

def curate_compile_commands(buildDir: pathlib.Path, folders):
    """
    Temporarily remove all compile objects not relative to the folders from the compile_commands.json file.
    Simply removing third party compile objects speeds up Clang-Tidy by a factor of ~5.
    """
    compileCommandsFile = buildDir / 'compile_commands.json'
    try:
        with open(compileCommandsFile) as file:
            compileCommands = json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"{str(compileCommandsFile)} doesn't exist in the build folder. Have you configured with CMake?")
    
    if isinstance(folders, str):
        folders = [folders]
    # Keep only compile commands for given folders
    compileCommandsNoThirdParty = []
    for folder in folders:
        folderPath = pathlib.Path(folder).resolve()
        compileCommandsNoThirdParty += [command for command in compileCommands if pathlib.Path(command['file']).resolve().is_relative_to(folderPath)]

    nvccFlags = re.compile(' --[^ ]*')
    nonClangFlags = re.compile(r' -(Xcudafe|ccbin|forward-unknown-to-host|maxrregcount|rdc)[^ ]*')
    googleIncludes = re.compile('(-isystem)=([^ ]*?google)')
    for command in compileCommandsNoThirdParty:
        # Pry out nvcc compiler arguments (--nvcc-flags) and other stuff clang doesn't understand
        command['command'] = re.sub(nvccFlags, '', command['command'])
        command['command'] = re.sub(nonClangFlags, '', command['command'])
        # Explicitly convert googletest modules to space separated. Needed for newer cmake versions.
        command['command'] = re.sub(googleIncludes, r'\1 \2', command['command'])
    
    compileCommandsFile.rename(compileCommandsFile.with_stem('_full_compile_commands'))

    with open(compileCommandsFile, 'w', encoding='utf-8') as file:
        json.dump(compileCommandsNoThirdParty, file, indent=2)

def parse_git_word_diff(gitDiffOut, word='', change=''):
    """
    Reads a git diff --word-diff, trying to find places where 'word' was removed/added
    yields tuples of (filename, lineNumbers)
    """
    if not word or not change:
        warnings.warn("keyword arguments 'word' and 'change' MUST be passed to parse_git_word_diff")
        return
    if change=='removed' or change=='-':
        keyword = '-' + word
    elif change=='added' or change=='+':
        keyword = '+' + word
    else:
        warnings.warn(f"Unrecognised 'change' argument, '{change}'")
        return

    lineNumbersToCorrect = []
    for line in gitDiffOut.stdout.split('\n'):
        if line.startswith('+++'): # starting new file
            if lineNumbersToCorrect: # yield changes (if any) for previous file
                yield file, lineNumbersToCorrect
                lineNumbersToCorrect = []
            file = pathlib.Path(line.split('+++ ')[-1])
        elif line.startswith('@@'): # the line numbers of a change
            lineNumStr = line.split(' +')[1].split(' @')[0] # get line number(s)
            if ',' in lineNumStr: # several lines
                start, length = lineNumStr.split(',')
                lineNumbers = list(range(int(start), int(length)))
            else: # one line
                lineNumbers = [int(lineNumStr)]
        elif line.startswith(keyword): # The change we're actually looking for
            lineNumbersToCorrect += lineNumbers
    if lineNumbersToCorrect: # yield changes (if any) for last file
        yield file, lineNumbersToCorrect


def fix_destroyed_lambda_captures():
    """
    Fix the destroyed lambda captures
    Clang-tidy has the unfortunate side effect that lambdas containing variables whose names are fixed have their capture sections destroyed.
      [=] (...) { return 2*bad_variable; } -> [badVariable] (...) { return 2 * badVariable; }
    This undoes the change for AMREX_GPU lambdas.
    Use git diff --cached for staged changes
    """
    gitDiffOut = subprocess.run(r'git diff -S"\[=\]" --pickaxe-regex --no-prefix --word-diff-regex="\[=\]" --word-diff=porcelain -I"=[^\]]" -U0', shell=True, capture_output=True, text=True)
    # Finds every instance where '[=]' was removed
    for file, lineNumbers in parse_git_word_diff(gitDiffOut, word='[=]', change='removed'):
        correct_in_files(r'\[[^\]]*\] AMREX_GPU', '[=] AMREX_GPU', file, linenums=lineNumbers)

def restore_compile_commands(buildDir: pathlib.Path):
    compileCommandsFile = buildDir / 'compile_commands.json'
    compileCommandsFile.with_stem('_full_compile_commands').replace(compileCommandsFile)

def gempic_run_clang_tidy(folders):
    """
    Run Clang-Tidy and fix the mistakes introduced thereby.
    Uses the compile_commands.json in the build directory to look for targets,
    and folders to exclude translation units not inside these folders.
    The compile_commands.json file, as well as the scripts/build_dir.txt file
    pointing to it, are created at cmake configuration time.
    """
    if not clang_version_is_ok('tidy'):
        return

    buildDir, includeLibs = read_build_info()
    
    curate_compile_commands(buildDir, folders)

    print("Running Clang-Tidy ...")
    try:
        # Run Clang-Tidy on all files in folders
        clangTidyOut = subprocess.run(f'run-clang-tidy -quiet {includeLibs} -fix -p {str(buildDir)}', shell=True, capture_output=True, text=True)
        outfile = buildDir / 'tidyOutput.out'
        with open(outfile, 'w') as file:
            file.write(clangTidyOut.stdout)
        if clangTidyOut.returncode == 0:
            print(f"Clang-Tidy ran succesfully. See '{str(outfile)}' for the output.")
            fix_destroyed_lambda_captures()
        else:
            warnings.warn(f"Clang-Tidy failed with the message:\n{clangTidyOut.stderr}")
    except (FileNotFoundError):
        warnings.warn("clang-tidy-run not found! Did you install Clang-Tidy?")

    restore_compile_commands(buildDir)

def fix_includes(file):
    """
    Enforces the include style:
    #include "GEMPIC_Library.H"
    for internal libraries and
    #include <otherstuff>
    for external libraries
    """
    correct_in_files('(^#include )"([^"]*)"', r'\1<\2>', file)
    correct_in_files(r'(^#include )<([^>]*GEMPIC_[^>]*\.H)>', r'\1"\2"', file)

def fix_lambda_brackets(file):
    r"""
    This is a hotfix to a bug in Clang-Format around the formatting of lambdas inside function calls.
    This _sometimes_ happens, and it's difficult to say when:
    function(..., [whatever] (...)     ->    function(..., [whatever] (...) {
    {                                  ->    
    The hotfix (re-)introduces the correct formatting.
    
    The simplified call without possibility of (one level of) nested (...) inside lambda arguments is:
    sed -Ei 's#^((\s*)\[[^\)]*\)+) \{#\1\n\2{#' $file
    The more complicated version is needed for cases such as
    function(..., [whatever] (..., AMREX_D_DECL(x, y, z), ...) {
    
              space '[  ... (' >= 0 nested '(...)' pairs (one level of nesting only) '...) {'
    """
    correct_in_files(r'^((\s*)\[[^(]*\(([^)({]*\([^()]*\))*[^(){]*\)) \{$', r'\1\n\2{', file)

def fix_scoped_definition_spacing(file):
    """
    Yet another hotfix to the bug that Clang-Format doesn't recognize scoped function definitions and thus removes the spaceBeforeParens.
    The fix accidentally adds a space to multiline scoped function declarations, i.e.
    void nameSpaceName::fnc (..., ..., ...,
                             ...);
    which are, however, rare enough as to be ignored here.
    """
    
    #            _return type_  _class/namespace_::_funcName_ _args_
    correct_in_files(r'^(\s*[^ =\(\)/]{1,} \w*(::\w*){1,})(\([^;]*)$', r'\1 \3', file)
    # Template specification declarations:
    #                                  void className<tVar1, tVar2>::fnc(...
    correct_in_files(r'^(\s*[^ =\(\)/]{1,} \w*(<(\w|[, :])*>{1,})?(::\w*(<(\w|[, :])*>{1,})?){1,})(\([^;]*)$', r'\1 \7', file)

def gempic_run_clang_format(folders):
    """
    Run Clang-Format and fix the mistakes introduced thereby.
    """
    if not clang_version_is_ok('format'):
        return

    print("Running Clang-Format ... ", end='')
    for file in find_files(["*.cpp", "*.H"], folders=folders):
        fix_includes(file)
        try:
            clangFormatOut = subprocess.run(f'clang-format -i -style=file {file}', shell=True, capture_output=True, text=True)

            if clangFormatOut.returncode == 0:
                fix_lambda_brackets(file)
                fix_scoped_definition_spacing(file)
            else:
                warnings.warn(clangFormatOut.stderr)
        except (FileNotFoundError):
            warnings.warn("clang-format not found! Did you install Clang-Format?")
            return
    print("done.")

def main():
    #folders = ['bad']
    #forceMove = False
    parser=argparse.ArgumentParser(description="Cleanup script for GEMPIC that should be "
                                   "independent of OS. Requires (run-)clang-tidy and clang-format"
                                   "in path as well as Python 3.7+.")
    parser.add_argument('-forcemove', '-force-move', '--forceMove', action='store_true',
                        help='Git add files or create fake files to facilitate git move when fixing'
                        ' folder and file syntax. Fake files are automatically removed again.')
    #parser.add_argument('-folders', nargs='*', action='append', dest='folders', default=['Src', 'Testing'], help='Folders to check folder and file syntax and run clang-tidy and -format on.')
    parser.add_argument('-folders', '--folders', nargs='+', default=['Src', 'Testing', 'Examples'], metavar='FOLDER',
                        help='Folders to sanitize, (re)creating the folder list (default: %(default)s)')
    parser.add_argument('-add-folders', '--add-folders', nargs='+', action='extend', dest='folders',metavar='FOLDER',
                        help='Add folders to sanitize, extending the list of folders')
    args = parser.parse_args()
    initDir = pathlib.Path.cwd().resolve()
    if (initDir.stem == 'scripts'):
        os.chdir('..')
    originalNames, changedFiles = exclude_subproject_clang_tools('third_party')
    try:
        # In case you didn't care about case before, you do now.
        subprocess.run('git config --local core.ignorecase false', shell=True)

        fix_folders_syntax(args.folders, args.forceMove)
        fix_files_syntax(args.folders, args.forceMove)

        gempic_run_clang_tidy(args.folders)
        gempic_run_clang_format(args.folders)
    finally:
        undo_exclude_subproject_clang_tools(changedFiles, originalNames)
        os.chdir(initDir)

if __name__=='__main__':
    main()

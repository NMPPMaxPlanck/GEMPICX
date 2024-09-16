from sphinx.cmd.build import main
import sys
from io import StringIO
import re

def runSphinxSilenced() -> tuple[int, StringIO]:
  """ Hacks stderr because sphinx will not be silenced, and you can't make it. """
  err = StringIO()
  sys.stderr = err
  try:
    return main(sys.argv[3:]), err
  finally:
    sys.stderr = sys.__stderr__

def removeDuplicateNamespaceWarnings(err: str) -> tuple[str, int]:
  """
  Removes the three warning types known to occur because of a namespace declared in multiple places.
  The bug is very specific to doxygen + sphinx + breathe.
  See https://github.com/breathe-doc/breathe/issues/772#issuecomment-1906104911
  """
  # an exhaustive list of namespaces in the project
  namespaces = ['Gempic', 'FieldSolvers', 'Filter', 'Forms', 'Io', 'Particle', 'ParticleMeshCoupling', 'TimeLoop', 'Utils', 'Impl']
  # ... and their corresponding regex OR pattern
  nsRegex = '('+'|'.join(namespaces)+')'
  
  duplicateIDPattern = re.compile(rf'\n?[^\n]*CRITICAL: Duplicate ID: "namespace[^"]*{nsRegex}"\.[^\n]*')
  duplicateTargetNamePattern = re.compile(rf'\n?[^\n]*WARNING: Duplicate explicit target name: "namespace[^"]*?{nsRegex}"\.[^\n]*')
  duplicateCPPTypePattern = re.compile(rf"\n?[^\n]*WARNING: Duplicate C\+\+ declaration[^\n]*\nDeclaration is '\.\. cpp:type:: {nsRegex}'\.[^\n]*")

  err, numNamespaceWarnings = re.subn(duplicateCPPTypePattern, '', err)
  err, numNamespaceWarnings2 = re.subn(duplicateTargetNamePattern, '', err)
  err, numNamespaceWarnings3 = re.subn(duplicateIDPattern, '', err)
  
  totalNamespaceWarnings = numNamespaceWarnings + numNamespaceWarnings2 + numNamespaceWarnings3

  return err, totalNamespaceWarnings

exitCode, err = runSphinxSilenced()
cleanedErr, totalNamespaceWarnings = removeDuplicateNamespaceWarnings(err.getvalue())

if (totalNamespaceWarnings):
  print(totalNamespaceWarnings, 'warnings were from "duplicated" namespaces and have been ignored.')
print(cleanedErr, file=sys.stderr)
sys.exit(exitCode)
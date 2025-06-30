
# Original attempts to read data using yt.
# This failed, hence the ad-hoc reader below.
#
# .. code::
#
#   import yt
#   file_object = yt.load('ParticleAddOutput/particles')
#
#   from yt.frontends.amrex.data_structures import AMReXDataset
#   pds = AMReXDataset('ParticleAddOutput/particles')

import pathlib
import numpy as np

def AMReX_ASCII_format(dimension = 3):
    """
    Returns numpy data type based on ASCII format for AMReX particles with one
    extra real component (assumed to be the weight).
    """
    list_of_directions = [('x', np.float64),
                          ('y', np.float64),
                          ('z', np.float64)]
    return np.dtype(list_of_directions[:dimension]
                  + [('int1', np.int32),
                     ('int2', np.int32),
                     ('weight', np.float64)])

def AMReX_binary_format_real(dimension = 3, names = ['weight']):
    """
    Returns numpy data type corresponding to AMReX particles with extra real
    components named in the `names` list.
    Assumes double precision.
    """
    list_of_names = [('x', np.float64),
                     ('y', np.float64),
                     ('z', np.float64)][:dimension]
    for nn in names:
        list_of_names.append((nn, np.float64))
    return np.dtype(list_of_names)

def AMReX_binary_format_int():
    """
    **Warning**:
    I assume there are two 32bit integers because the ASCII
    format has two integers.
    Within this sanity check script these 8 bytes are ignored.
    """
    return np.dtype([('int1', np.int32),
                     ('int2', np.int32)])

def read_AMReX_particle_header(base_dir):
    fname = pathlib.PurePath(base_dir, 'Header')
    with open(fname, 'r') as header:
        # confirm version
        fversion = header.readline().strip()
        assert(fversion == 'Version_Two_Dot_Zero_double')
        # read space dimension(?)
        dimension = int(header.readline())
        # read number of extra real components
        nreals = int(header.readline())
        names = []
        for rr in range(nreals):
            names.append(header.readline().strip())

        # ignore 2 lines
        zz = int(header.readline())
        assert(zz == 0)
        oo = int(header.readline())
        assert(oo == 1)

        # get number of particles
        nparticles = int(header.readline())

        # ignore 2 more lines
        oo = int(header.readline())
        # assert(oo == 38) --- As of 2025-06-18 the file reads "38" here,
        #                      but I don't know why (number of particles is 37).
        zz = int(header.readline())
        assert(zz == 0)

        # get number of chunks
        nchunks = int(header.readline())
        # get number of particles per chunk
        nparticles_per_chunk = []
        for cc in range(nchunks):
            nparticles_per_chunk.append(int(header.readline().split()[1]))
    assert(nparticles == sum(nparticles_per_chunk))
    return { 'dimension'            : dimension,
             'nreals'               : nreals,
             'names'                : names,
             'nparticles'           : nparticles,
             'nchunks'              : nchunks,
             'nparticles_per_chunk' : nparticles_per_chunk
            }

def read_AMReX_binary_particles(base_dir):
    """
    Reads AMReX particle data output by the `WritePlotFile` method of
    `ParticleContainer`.

    **Warning**: the file format used here has been reversed engineered, I
    didn't find an obvious description quickly enough
    (comment valid as of 2025-06-18).

    For future reference:
      * particle data is stored in separate "chunks" (my terminology).
      * each chunk begins with "packed" (AMReX terminology) integers,
        and continues with "packed" "reals" (double precision in our case).
      * "packed integers" are (in our case) a sequence of pairs of 32bit integers
        The "pair of 32bit integers" part is guessed based on the ASCII format
        (in principle it could be a single 64bit integer).
        In other words: AoS integers for the current "chunk".
      * "packed reals" are AoS reals for the current "chunk".

    I **assume** chunks correspond to the geometric grid (it would make sense
    for visualisation-oriented postprocessing)
    """
    info = read_AMReX_particle_header(base_dir)
    fname = pathlib.PurePath(base_dir, 'Level_0', 'DATA_00000')
    with open(fname, 'rb') as fobject:
        # allocate data
        data = np.zeros(
                shape = (info['nparticles'],),
                dtype = [(nn, np.int32)
                         for nn in AMReX_binary_format_int().names]
                      + [(nn, np.float64)
                         for nn in AMReX_binary_format_real(
                             dimension = info['dimension'],
                             names = info['names']).names])

        ## test file size
        file_size = pathlib.Path(fname).stat().st_size
        assert(file_size == data.itemsize*data.size)

        # start particle index at 0
        pp = 0
        for n in info['nparticles_per_chunk']:
            dint = np.fromfile(
                        fobject,
                        dtype = AMReX_binary_format_int(),
                        count = n)
            dreal = np.fromfile(
                        fobject,
                        dtype = AMReX_binary_format_real(
                            dimension = info['dimension'],
                            names = info['names']),
                        count = n)
            for kk in dint.dtype.names:
                data[kk][pp:pp+n] = dint[kk]
            for kk in dreal.dtype.names:
                data[kk][pp:pp+n] = dreal[kk]
            pp += n
    return data

def generate_prescribed_data(nparticles):
    """
    Generates data consistent with `ParticleAdd_test.cpp`.
    Other than the prescribed particle locations, this means the data has 0
    velocity components.
    """
    data = np.zeros(
            shape = (nparticles,),
            dtype = AMReX_ASCII_format())
    base_fraction = 1.0 / nparticles
    # **Warning**: pL and p0 have been read once from c++ test output
    # TODO: if there is a systematic way to ensure correct values are used,
    #       use that instead.
    pL = [2*np.pi, 2*np.pi, 2*np.pi]
    p0 = [-(np.pi - 0.3), -(np.pi - 0.6), -(np.pi - 0.4)]
    # **Warning**: the formulas in this for loop must agree with the code
    # used in `ParticleAdd_test.cpp`
    for pp in range(nparticles):
        data['weight'][pp] = pp + 1.1
        data['x'][pp] = p0[0] + ((3*pp) % nparticles) * base_fraction * pL[0]
        data['y'][pp] = p0[1] + ((4*pp) % nparticles) * base_fraction * pL[1]
        data['z'][pp] = p0[2] + ((5*pp) % nparticles) * base_fraction * pL[2]
    return data

def compare_ascii_and_binary():
    base_dir = 'ParticleAddOutput/particles'
    info = read_AMReX_particle_header(base_dir)

    ascii_data = np.loadtxt(
            'ParticleAddASCII.txt',
            dtype = AMReX_ASCII_format(dimension = info['dimension']),
            skiprows = 5)
    binary_data = read_AMReX_binary_particles(base_dir)

    assert(ascii_data.size == binary_data.size)

    # confirm weights are ok
    assert(np.all(
            np.abs(ascii_data['weight'] - binary_data['weight']) < 0.01))
    for kk in ['int1', 'int2']:
        assert(np.all(ascii_data[kk] == binary_data[kk]))
    for kk in ['x', 'y', 'z'][:info['dimension']]:
        assert(np.allclose(ascii_data[kk], binary_data[kk], rtol = 1e-10))
    print('SUCCESS: ASCII data agrees with binary data.')

def compare_prescribed_and_binary():
    base_dir = 'ParticleAddOutput/particles'
    info = read_AMReX_particle_header(base_dir)

    prescribed_data = generate_prescribed_data(info['nparticles'])
    binary_data = read_AMReX_binary_particles(base_dir)

    assert(prescribed_data.size == binary_data.size)
    for pp in range(info['nparticles']):
        binary_index = pp
        prescribed_index = np.argmin(np.abs(
            binary_data['weight'][pp] - prescribed_data['weight']))
        for kk in ['x', 'y', 'z'][:info['dimension']]:
            assert(np.abs(binary_data[kk][binary_index]
                        - prescribed_data[kk][prescribed_index]) < 1e-10)
    print('SUCCESS: prescribed data agrees with binary data.')

if __name__ == '__main__':
    compare_ascii_and_binary()
    compare_prescribed_and_binary()


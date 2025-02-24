#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "GEMPIC_SobolSampler.H"

using namespace std;

//****************************************************************************80

int Gempic::Particle::sobol_bit_hi1 (long long int n)

//****************************************************************************80
//
//  Purpose:
//
//    SOBOL_BIT_HI1 returns the position of the high 1 bit base 2 in an integer.
//
//  Example:
//
//       N    Binary    Hi 1
//    ----    --------  ----
//       0           0     0
//       1           1     1
//       2          10     2
//       3          11     2
//       4         100     3
//       5         101     3
//       6         110     3
//       7         111     3
//       8        1000     4
//       9        1001     4
//      10        1010     4
//      11        1011     4
//      12        1100     4
//      13        1101     4
//      14        1110     4
//      15        1111     4
//      16       10000     5
//      17       10001     5
//    1023  1111111111    10
//    1024 10000000000    11
//    1025 10000000001    11
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    12 May 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, long long int N, the integer to be measured.
//    N should be nonnegative.  If N is nonpositive, I8_BIT_HI1
//    will always be 0.
//
//    Output, int SOBOL_BIT_HI1, the number of bits base 2.
//
{
    int bit;

    bit = 0;

    while (0 < n)
    {
        bit = bit + 1;
        n = n / 2;
    }

    return bit;
}
//****************************************************************************80

int Gempic::Particle::sobol_bit_lo0 (long long int n)

//****************************************************************************80
//
//  Purpose:
//
//    SOBOL_BIT_LO0 returns the position of the low 0 bit base 2 in an integer.
//
//  Example:
//
//       N    Binary    Lo 0
//    ----    --------  ----
//       0           0     1
//       1           1     2
//       2          10     1
//       3          11     3
//       4         100     1
//       5         101     2
//       6         110     1
//       7         111     4
//       8        1000     1
//       9        1001     2
//      10        1010     1
//      11        1011     3
//      12        1100     1
//      13        1101     2
//      14        1110     1
//      15        1111     5
//      16       10000     1
//      17       10001     2
//    1023  1111111111    11
//    1024 10000000000     1
//    1025 10000000001     2
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    12 May 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, long long int N, the integer to be measured.
//    N should be nonnegative.
//
//    Output, int SOBOL_BIT_LO0, the position of the low 1 bit.
//
{
    int bit;
    long long int n2;

    bit = 0;

    while (true)
    {
        bit = bit + 1;
        n2 = n / 2;

        if (n == 2 * n2)
        {
            break;
        }

        n = n2;
    }

    return bit;
}
//****************************************************************************80

void Gempic::Particle::sobol (int dimNum, long long int *seed, double quasi[])

//****************************************************************************80
//
//  Purpose:
//
//    SOBOL generates a new quasirandom Sobol vector with each call.
//
//  Discussion:
//
//    The routine adapts the ideas of Antonov and Saleev.
//
//    This routine uses LONG LONG INT for integers and DOUBLE for real values.
//
//    Thanks to Steffan Berridge for supplying (twice) the properly
//    formatted V data needed to extend the original routine's dimension
//    limit from 40 to 1111, 05 June 2007.
//
//    Thanks to Francis Dalaudier for pointing out that the range of allowed
//    values of DIM_NUM should start at 1, not 2!  17 February 2009.
//
//  Licensing:
//
//    This code is distributed under the MIT license.
//
//  Modified:
//
//    17 February 2009
//
//  Author:
//
//    FORTRAN77 original version by Bennett Fox.
//    C++ version by John Burkardt
//
//  Reference:
//
//    IA Antonov, VM Saleev,
//    An Economic Method of Computing LP Tau-Sequences,
//    USSR Computational Mathematics and Mathematical Physics,
//    Volume 19, 1980, pages 252 - 256.
//
//    Paul Bratley, Bennett Fox,
//    Algorithm 659:
//    Implementing Sobol's Quasirandom Sequence Generator,
//    ACM Transactions on Mathematical Software,
//    Volume 14, Number 1, pages 88-100, 1988.
//
//    Bennett Fox,
//    Algorithm 647:
//    Implementation and Relative Efficiency of Quasirandom
//    Sequence Generators,
//    ACM Transactions on Mathematical Software,
//    Volume 12, Number 4, pages 362-376, 1986.
//
//    Stephen Joe, Frances Kuo
//    Remark on Algorithm 659:
//    Implementing Sobol's Quasirandom Sequence Generator,
//    ACM Transactions on Mathematical Software,
//    Volume 29, Number 1, pages 49-57, March 2003.
//
//    Ilya Sobol,
//    USSR Computational Mathematics and Mathematical Physics,
//    Volume 16, pages 236-242, 1977.
//
//    Ilya Sobol, YL Levitan,
//    The Production of Points Uniformly Distributed in a Multidimensional
//    Cube (in Russian),
//    Preprint IPM Akad. Nauk SSSR,
//    Number 40, Moscow 1976.
//
//  Parameters:
//
//    Input, int DIM_NUM, the number of spatial dimensions.
//    DIM_NUM must satisfy 1 <= DIM_NUM <= 1111.
//
//    Input/output, long long int *SEED, the "seed" for the sequence.
//    This is essentially the index in the sequence of the quasirandom
//    value to be generated.  On output, SEED has been set to the
//    appropriate next value, usually simply SEED+1.
//    If SEED is less than 0 on input, it is treated as though it were 0.
//    An input value of 0 requests the first (0-th) element of the sequence.
//
//    Output, double QUASI[DIM_NUM], the next quasirandom vector.
//
{
#define DIM_MAX 10
#define LOG_MAX 62
    //
    //  Here, we have commented out the definition of ATMOST, because
    //  in some cases, a compiler was complaining that the value of ATMOST could not
    //  seem to be properly stored.  We only need ATMOST in order to specify MAXCOL,
    //  so as long as we set MAXCOL (below) to what we expect it should be, we
    //  may be able to get around this difficulty.
    //  JVB, 24 January 2006.
    //
    //static long long int atmost = 4611686018427387903;
    //
    static int dimNumSave = 0;
    long long int i;
    bool includ[LOG_MAX];
    static bool initialized = false;
    long long int j;
    long long int j2;
    long long int k;
    long long int l;
    static long long int lastq[DIM_MAX];
    long long int m;
    static long long int maxcol;
    long long int newv;
    static long long int poly[DIM_MAX] = {1, 3, 7, 11, 13, 19, 25, 37, 59, 47};
    static double recipd;
    static long long int seedSave = -1;
    long long int seedTemp;
    static long long int v[DIM_MAX][LOG_MAX];

    if (!initialized || dimNum != dimNumSave)
    {
        initialized = true;
        for (i = 0; i < DIM_MAX; i++)
        {
            for (j = 0; j < LOG_MAX; j++)
            {
                v[i][j] = 0;
            }
        }
        //
        //  Initialize (part of) V.
        //
        v[0][0] = 1;
        v[1][0] = 1;
        v[2][0] = 1;
        v[3][0] = 1;
        v[4][0] = 1;
        v[5][0] = 1;
        v[6][0] = 1;
        v[7][0] = 1;
        v[8][0] = 1;
        v[9][0] = 1;

        v[2][1] = 1;
        v[3][1] = 3;
        v[4][1] = 1;
        v[5][1] = 3;
        v[6][1] = 1;
        v[7][1] = 3;
        v[8][1] = 3;
        v[9][1] = 1;

        v[3][2] = 7;
        v[4][2] = 5;
        v[5][2] = 1;
        v[6][2] = 3;
        v[7][2] = 3;
        v[8][2] = 7;
        v[9][2] = 5;

        v[5][3] = 1;
        v[6][3] = 7;
        v[7][3] = 9;
        v[8][3] = 13;
        v[9][3] = 11;

        v[7][4] = 9;
        v[8][4] = 3;
        v[9][4] = 27;
        //
        //  Check parameters.
        //
        if (dimNum < 1 || DIM_MAX < dimNum)
        {
            cout << "\n";
            cout << "I8_SOBOL - Fatal error!\n";
            cout << "  The spatial dimension DIM_NUM should satisfy:\n";
            cout << "    1 <= DIM_NUM <= " << DIM_MAX << "\n";
            cout << "  But this input value is DIM_NUM = " << dimNum << "\n";
            exit(1);
        }

        dimNumSave = dimNum;
        //
        //  Find the number of bits in ATMOST.
        //
        //  Here, we have short-circuited the computation of MAXCOL from ATMOST, because
        //  in some cases, a compiler was complaining that the value of ATMOST could not
        //  seem to be properly stored.  We only need ATMOST in order to specify MAXCOL,
        //  so if we know what the answer should be we can try to get it this way!
        //  JVB, 24 January 2006.
        //
        //  maxcol = i8_bit_hi1 ( atmost );
        //
        maxcol = 62;
        //
        //  Initialize row 1 of V.
        //
        for (j = 0; j < maxcol; j++)
        {
            v[0][j] = 1;
        }
        //
        //  Initialize the remaining rows of V.
        //
        for (i = 1; i < dimNum; i++)
        {
            //
            //  The bit pattern of the integer POLY(I) gives the form
            //  of polynomial I.
            //
            //  Find the degree of polynomial I from binary encoding.
            //
            j = poly[i];
            m = 0;

            while (true)
            {
                j = j / 2;
                if (j <= 0)
                {
                    break;
                }
                m = m + 1;
            }
            //
            //  We expand this bit pattern to separate components
            //  of the logical array INCLUD.
            //
            j = poly[i];
            for (k = m - 1; 0 <= k; k--)
            {
                j2 = j / 2;
                includ[k] = (j != (2 * j2));
                j = j2;
            }
            //
            //  Calculate the remaining elements of row I as explained
            //  in Bratley and Fox, section 2.
            //
            for (j = m; j < maxcol; j++)
            {
                newv = v[i][j - m];
                l = 1;

                for (k = 0; k < m; k++)
                {
                    l = 2 * l;

                    if (includ[k])
                    {
                        newv = (newv ^ (l * v[i][j - k - 1]));
                    }
                }
                v[i][j] = newv;
            }
        }
        //
        //  Multiply columns of V by appropriate power of 2.
        //
        l = 1;
        for (j = maxcol - 2; 0 <= j; j--)
        {
            l = 2 * l;
            for (i = 0; i < dimNum; i++)
            {
                v[i][j] = v[i][j] * l;
            }
        }
        //
        //  RECIPD is 1/(common denominator of the elements in V).
        //
        recipd = 1.0E+00 / ((double)(2 * l));
    }

    if (*seed < 0)
    {
        *seed = 0;
    }

    if (*seed == 0)
    {
        l = 1;
        for (i = 0; i < dimNum; i++)
        {
            lastq[i] = 0;
        }
    }
    else if (*seed == seedSave + 1)
    {
        l = sobol_bit_lo0(*seed);
    }
    // modification to compute the first value directly before switching to the efficient recursion
    else
    {
        long long int graycode = *seed ^ (*seed >> 1);
        for (i = 0; i < dimNum; i++)
        {
            lastq[i] = 0;
            long long int g = graycode;
            for (j = 0; j < LOG_MAX; j++)
            {
                if (g & 1)
                {
                    lastq[i] = lastq[i] ^ v[i][j];
                }
                g >>= 1;
            }
        }
        l = sobol_bit_lo0(*seed);
    }
    //
    //  Check that the user is not calling too many times!
    //
    if (maxcol < l)
    {
        cout << "\n";
        cout << "SOBOL - Fatal error!\n";
        cout << "  The value of SEED seems to be too large!\n";
        cout << "  SEED =   " << *seed << "\n";
        cout << "  MAXCOL = " << maxcol << "\n";
        cout << "  L =      " << l << "\n";
        exit(2);
    }
    //
    //  Calculate the new components of QUASI.
    //  The caret indicates the bitwise exclusive OR.
    //
    for (i = 0; i < dimNum; i++)
    {
        quasi[i] = ((double)lastq[i]) * recipd;

        lastq[i] = (lastq[i] ^ v[i][l - 1]);
    }

    seedSave = *seed;
    *seed = *seed + 1;

    return;
#undef DIM_MAX
#undef LOG_MAX
}

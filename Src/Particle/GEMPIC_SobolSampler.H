#ifndef SOBOL_SAMPLER_H_
#define SOBOL_SAMPLER_H_

namespace Gempic::Particle
{

int sobol_bit_hi1(long long int n);
int sobol_bit_lo0(long long int n);
void sobol(int dimNum, long long int *seed, double quasi[]);

} // namespace Gempic::Particle
#endif
/*****************************************************************/
/*
\brief Header file to config.cpp
       Create configuration structures for different components
       of the bsl6d-Framework.
       The configuration is initialized by the 

\author Nils Schild
\department NMPP
\email  nils.schild@ipp.mpg.de
\date   01.09.2021
*/
/*****************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <fstream>

namespace Gempic{

struct t_params_propagator {
  int n_steps;
};

class gempic_param_config {

  public : 
    gempic_param_config() = default;
    ~gempic_param_config() = default;
    void print_gempic_param_config(std::ofstream& ofs);
    
    t_params_propagator propagator;

};

}

#endif /*CONFIG_H*/

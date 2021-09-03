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

struct t_ConfigTest {
  int testKey;
};

class ConfigGempic {

  public : 
    ConfigGempic() = default;
    ~ConfigGempic() = default;
    void printConfigGempic(std::ofstream& ofs);
    
    t_ConfigTest test;

};

}

#endif /*CONFIG_H*/

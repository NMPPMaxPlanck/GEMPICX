/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <stdexcept>
#include <string>

#include "GEMPIC_Config.H"

std::string direction_to_string (Direction d)
{
    switch (d)
    {
        case Direction::xDir:
            return "x";
        case Direction::yDir:
            return "y";
        case Direction::zDir:
            return "z";
        default:
            throw std::runtime_error("Unknown direction given to direction_to_string");
    }
}

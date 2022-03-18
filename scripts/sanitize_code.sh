#!/bin/bash

# Sanitizes code using clang-tidy

#find ./src/ -name *.H -or -name *.cpp -type f | xargs clang-tidy --fix -fix-errors -p /home/bdealbuq/gempic_obj/gempic/compile_commands.json

# Sanitizes code using clang-format
find ./src/ -name "*.H" -or -name "*.cpp" -type f | xargs clang-format -i -style=file

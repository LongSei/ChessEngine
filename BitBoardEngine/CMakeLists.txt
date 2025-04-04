cmake_minimum_required(VERSION 3.4)
project(BitBoardEngine)

add_subdirectory(pybind11)
pybind11_add_module(BitBoardEngine main.cpp)

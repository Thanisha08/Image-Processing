"""
.. module:: 3D array operations
   :platform: Windows
   :synopsis: This module does simple details

.. module author:: Thanisha
.. copyrights: Karomi Technology Private Limited
.. date created: 08/08/2018

"""

import numpy as np


def add(d, e):
    """
    this function adds two 3D array
    :param d, e: Two 3-D array
    :return: returns resultant array
    """
    output = np.add(d, e)
    return output


def subtract(g, h):
    """
    this function Subtract two 3D array
    :param d, e: Two 3-D array
    :return: returns resultant array
    """
    output = np.subtract(g, h)
    return output


def multiply(j, k):
    """
    this function multiply two 3D array
    :param d, e: Two 3-D array
    :return: returns resultant array
    """
    output = np.dot(j, k)
    return output


def divide(a, b):
    """
    this function divide two 3D array
    :param d, e: Two 3-D array
    :return: returns resultant array
    """
    output = np.divide(x, y)
    return output


if __name__ == "__main__":

    x = np.array([[1, 2, 4], [3, 4, 1], [2, 9, 5]], dtype=np.float64)
    y = np.array([[5, 6, 2], [7, 8, 4], [1, 3, 7]], dtype=np.float64)

out_array = add(x, y)
print(out_array)

out_array = subtract(x, y)
print(out_array)

out_array = multiply(x, y)
print(out_array)

out_array = divide(x, y)
print(out_array)



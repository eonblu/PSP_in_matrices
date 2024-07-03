#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math
import random

import mysql_connection
from matrices import *
import MinMaxHeapTriplets
from MinMaxHeapTriplets import *
from QuickselectMinTriplets import *

class Comparisons:
    def __init__(self, initial_value=0):
        self.value = initial_value
    def increment(self, amount=1):
        self.value += amount

def retrieve_matrix(matrix_id):
    conn = mysql_connection.new_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT MatrixSeed, MRows FROM Matrices WHERE MatrixID = %s", (matrix_id,))
    result = cursor.fetchone()

    if result:
        seed, rows = result
        matrix = create_matrix_with_ssp(seed, rows)

    cursor.close()
    mysql_connection.close_connection(conn)
    return matrix, rows, matrix_id

# This simplified randomized algorithm takes 3 elements per row, picks the middle one and adds it to a list, from this list the minimum is picked as pivot
# any column with an element smaller than the pivot in the same row cannot contain a PSP
# Using a similar approach as Median of Medians we start with 3 and expand possibly to 5 and 7

def SimpleFindHorizontalPivot(matrix, CompsObj):
    setof_q_i = []
    # select a "representative" for each row
    if len(matrix[0]) == 1:
        return 0, []
    elif len(matrix[0]) == 2:
        for row in range(len(matrix)):
            if matrix[row, 0] > matrix[row, 1]:
                setof_q_i.append([matrix[row, 1], row, 1])
            else:
                setof_q_i.append([matrix[row, 0], row, 0])
    else:
        for row in range(len(matrix)):
            sample_pick = random.sample(range(len(matrix[0])), 3)
            # possibly replace this with Quickselect for larger than 3
            if matrix[row, sample_pick[0]] > matrix[row, sample_pick[1]] and matrix[row, sample_pick[1]] < matrix[row, sample_pick[2]]:
                setof_q_i.append([matrix[row, sample_pick[1]], row, sample_pick[1]])
            elif matrix[row, sample_pick[1]] > matrix[row, sample_pick[0]] and matrix[row, sample_pick[0]] < matrix[row, sample_pick[2]]:
                setof_q_i.append([matrix[row, sample_pick[0]], row, sample_pick[0]])
            else:
                setof_q_i.append([matrix[row, sample_pick[2]], row, sample_pick[2]])
    # pick minimum of medians
    min_q_i = setof_q_i[0]
    for q_i in setof_q_i:
        if q_i[0] < min_q_i[0]:
            min_q_i = q_i
    columns_to_remove = []
    # find all columns that contain a smaller element in the same row as min_q_i
    for column in range(len(matrix[0])):
        if matrix[min_q_i[1], column] < min_q_i[0]:
            columns_to_remove.append(column)
    return min_q_i, columns_to_remove

# equivalent to Horizontal with swapped min/max and row/column
def SimpleFindVerticalPivot(matrix, CompsObj):
    setof_q_i = []
    if len(matrix) == 1:
        return 0, []
    elif len(matrix) == 2:
        for column in range(len(matrix[0])):
            if matrix[0, column] > matrix[1, column]:
                setof_q_i.append([matrix[1, column], 1, column])
            else:
                setof_q_i.append([matrix[0, column], 0, column])
    else:
        for column in range(len(matrix[0])):
            sample_pick = random.sample(range(len(matrix)), 3)
            if matrix[sample_pick[0], column] > matrix[sample_pick[1], column] and matrix[sample_pick[1], column] < matrix[sample_pick[2], column]:
                setof_q_i.append([matrix[sample_pick[1], column], sample_pick[1], column])
            elif matrix[sample_pick[1], column] > matrix[sample_pick[0], column] and matrix[sample_pick[0], column] < matrix[sample_pick[2], column]:
                setof_q_i.append([matrix[sample_pick[0], column], sample_pick[0], column])
            else:
                setof_q_i.append([matrix[sample_pick[2], column], sample_pick[2], column])
    max_q_i = setof_q_i[0]
    for q_i in setof_q_i:
        if q_i[0] > max_q_i[0]:
            max_q_i = q_i
    rows_to_remove = []
    for row in range(len(matrix)):
        if matrix[row, max_q_i[2]] > max_q_i[0]:
            rows_to_remove.append(row)
    return max_q_i, rows_to_remove

def SimpleReduceMatrix(matrix, s, CompsObj):
    while len(matrix) > s or len(matrix[0]) > s:
        print(matrix.shape)
        print(matrix)
        p_hor, columns_to_remove = SimpleFindHorizontalPivot(matrix, CompsObj)
        p_ver, rows_to_remove = SimpleFindVerticalPivot(matrix, CompsObj)
        for col in reversed(columns_to_remove):
            matrix = np.delete(matrix, col, 1)
        for row in reversed(rows_to_remove):
            matrix = np.delete(matrix, row, 0)
    return matrix

if __name__ == '__main__':
    MID = 20
    CompsObjRandom = Comparisons()
    matrix, rows, matrixid = retrieve_matrix(MID)
    print(matrix)
    print(SimpleReduceMatrix(matrix, 3, CompsObjRandom))
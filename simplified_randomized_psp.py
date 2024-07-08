#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math
import random

import mysql_connection
from matrices import *
from QuickselectMinTriplets import *
from deterministic_psp import *

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

def SimpleFindHorizontalPivot(matrix, s, CompsObj):
    setof_q_i = []
    # select a "representative" for each row
    for row in range(len(matrix)):
        sample_pick = random.sample(range(len(matrix[0])), s)
        R = []
        for sample in sample_pick:
            R.append([matrix[row, sample], row, sample])
        setof_q_i.append(select_kth_triplet(R, math.ceil(s/2), CompsObj)) # s-1
    # pick minimum of medians
    min_q_i = setof_q_i[0]
    for q_i in setof_q_i:
        CompsObj.increment()
        if q_i[0] < min_q_i[0]:
            min_q_i = q_i
    columns_to_remove = []
    # find all columns that contain a smaller element in the same row as min_q_i
    for column in range(len(matrix[0])):
        CompsObj.increment()
        if matrix[min_q_i[1], column] < min_q_i[0]:
            columns_to_remove.append(column)
    return min_q_i, columns_to_remove

# equivalent to Horizontal with swapped min/max and row/column
def SimpleFindVerticalPivot(matrix, s, CompsObj):
    setof_q_i = []
    for column in range(len(matrix[0])):
        sample_pick = random.sample(range(len(matrix)), s)
        R = []
        for sample in sample_pick:
            R.append([matrix[sample, column], sample, column])
        setof_q_i.append(select_kth_triplet(R, math.ceil(s/2), CompsObj)) # 2
    max_q_i = setof_q_i[0]
    for q_i in setof_q_i:
        CompsObj.increment()
        if q_i[0] > max_q_i[0]:
            max_q_i = q_i
    rows_to_remove = []
    for row in range(len(matrix)):
        CompsObj.increment()
        if matrix[row, max_q_i[2]] > max_q_i[0]:
            rows_to_remove.append(row)
    return max_q_i, rows_to_remove

def SimpleReduceMatrix(matrix, samplesize, CompsObj):
    while len(matrix) > samplesize or len(matrix[0]) > samplesize:
        print(matrix.shape)
        if len(matrix[0]) > samplesize: # Prevent unecessary comparisons if one dimension is already minimal size
            p_hor, columns_to_remove = SimpleFindHorizontalPivot(matrix, samplesize, CompsObj)
        if len(matrix) > samplesize:
            p_ver, rows_to_remove = SimpleFindVerticalPivot(matrix, samplesize, CompsObj)
        for col in reversed(columns_to_remove):
            if len(matrix[0]) > samplesize: # Prevent Non square final matrix
                matrix = np.delete(matrix, col, 1)
        for row in reversed(rows_to_remove):
            if len(matrix) > samplesize:
                matrix = np.delete(matrix, row, 0)
    return BienstockBase(matrix, samplesize, CompsObj)[0]

if __name__ == '__main__':
    MID = 9
    CompsObjRandom = Comparisons()
    matrix, rows, matrixid = retrieve_matrix(MID)
    print(SimpleReduceMatrix(matrix, 5, CompsObjRandom))
    print(CompsObjRandom.value)
#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math
import copy

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

# this is just a prototype, there are issues with the way memory is handled - deepcopies are expensive to handle
# also removed rows/columns will need to be removed again, this could be circumvented by keeping track of rows and columns to be removed before actually removing them
# handling matrices with 2 columns or 2 rows also seems to cause issues

def FindHorizontalPivot(matrix, CompsObj):
    t = 1000000000 # maximum entry within matrix generated by matrices.py
    ignored_rows = set() # need to keep track of deleted rows to correctly return p at the end
    columns_to_remove = [] 
    # Phase 1
    while len(matrix) - len(ignored_rows) > math.floor(len(matrix)**(19/20)):
        setof_q_i = []
        for i in range(len(matrix)):
            if not i in ignored_rows: # if row was removed in earlier run of this loop
                rand_k = np.random.randint(0, len(matrix[0]))
                setof_q_i.insert(len(setof_q_i), ([matrix[i][rand_k], i, rand_k]))
        # the array in select_kth_triplet is modified, so there needs to be an auxiliary array as the ordered array is still required
        aux_setof_q_i = copy.deepcopy(setof_q_i)
        q_selected = select_kth_triplet(aux_setof_q_i, math.ceil((3/4) * len(setof_q_i)), CompsObj)
        t = min(t, q_selected[0])
        for j in reversed(setof_q_i):
            CompsObj.increment()
            if j[0] > t:
                ignored_rows.add(j[1]) # add row number to ignore in phase 2
       
    # Phase 2
    setof_q_r = []
    for rem in range(len(matrix)):
        if not rem in ignored_rows: # if row was removed in earlier step, ignore
            rand_k = np.random.randint(0, len(matrix[0])) # this should pick |m|^1/20 elements, realistically always 1 for matrices generated with numpy
            setof_q_r.append([matrix[rem][rand_k], rem, rand_k])
    p = setof_q_r[0]
    for q_r in setof_q_r: # select minimum
        CompsObj.increment()
        if q_r[0] < p[0]:
            p = q_r
    larger_count = 0
    for j in range(len(matrix[p[1]])):
        CompsObj.increment()
        if p[0] > matrix[p[1], j]:
            larger_count += 1
            columns_to_remove.append(j)
    # if larger_count < math.floor(len(matrix[p[1]])/4):
    #    return False
    # else:
    return p, columns_to_remove

# equivalent with min/max reversed to HorizontalPivot
def FindVerticalPivot(matrix, CompsObj):
    t = 0
    ignored_columns = set()
    rows_to_remove = []
    # Phase 1
    while len(matrix[0]) - len(ignored_columns) > math.floor(len(matrix[0])**(19/20)):
        setof_q_i = []
        for i in range(len(matrix[0])):
            if not i in ignored_columns:
                rand_k = np.random.randint(0, len(matrix))
                setof_q_i.insert(len(setof_q_i), ([matrix[rand_k][i], rand_k, i]))
        aux_setof_q_i = copy.deepcopy(setof_q_i)
        q_selected = select_kth_triplet(aux_setof_q_i, max(math.floor((1/4) * len(setof_q_i)), 1), CompsObj)
        t = max(t, q_selected[0])
        for j in reversed(setof_q_i):
            CompsObj.increment()
            if j[0] < t:
                ignored_columns.add(j[2])

    # Phase 2
    setof_q_r = []
    for rem in range(len(matrix[0])):
        if not rem in ignored_columns:
            rand_k = np.random.randint(0, len(matrix))
            setof_q_r.append([matrix[rand_k][rem], rand_k, rem])
    p = setof_q_r[0]
    for q_r in setof_q_r:
        CompsObj.increment()
        if q_r[0] > p[0]:
            p = q_r
    smaller_count = 0
    for j in range(len(matrix[:,p[2]])):
        CompsObj.increment()
        if p[0] < matrix[j, p[2]]:
            smaller_count += 1
            rows_to_remove.append(j)
    # if smaller_count < math.floor(len(matrix[p[2]])/4):
    #    return False
    # else:
    return p, rows_to_remove

def ReduceMatrix(matrix, s, CompsObj):
    while len(matrix) > s or len(matrix[0]) > s:
        print(matrix.shape)
        if len(matrix[0]) > s: # Prevent unecessary comparisons if one dimension is already minimal size
            p_hor, columns_to_remove = FindHorizontalPivot(matrix, CompsObj)
        if len(matrix) > s:
            p_ver, rows_to_remove = FindVerticalPivot(matrix, CompsObj)
        if p_hor and p_ver:
            for col in reversed(columns_to_remove):
                if len(matrix[0]) > s: # Prevent Non square final matrix
                    matrix = np.delete(matrix, col, 1)
            for row in reversed(rows_to_remove):
                if len(matrix) > s:
                    matrix = np.delete(matrix, row, 0)
        else:
            return "Failed"
    return BienstockBase(matrix, s, CompsObj)[0]


if __name__ == '__main__':
    MID = 9
    CompsObjRandom = Comparisons()
    matrix, rows, matrixid = retrieve_matrix(MID)
    print(ReduceMatrix(matrix, 7, CompsObjRandom))
    print(CompsObjRandom.value)

    
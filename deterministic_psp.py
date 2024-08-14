#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math

import mysql_connection
from matrices import *
import MinMaxHeapTriplets
from MinMaxHeapTriplets import *

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

def BienstockBase(matrix, rows, CompsObj):
    heap = MinMaxHeap(rows)
    for i in range(rows):
        heap.insert_triplet([matrix[i][i], i, i], CompsObj)
    while heap.size > 1: # done heapsize-1 times
        currentmax = heap.peekmax(CompsObj)
        currentmin = heap.peekmin(CompsObj)
        # Check if A_il is smaller or equal than minimum
        if matrix[currentmin[1]][currentmax[2]] <= currentmin[0]:
            CompsObj.increment()
            heap.popmax(CompsObj)
        # Check if A_il is larger or equal than maximum
        elif currentmax[0] <= matrix[currentmin[1]][currentmax[2]]:
            CompsObj.increment(2)
            heap.popmin(CompsObj)
        # A_il needs be to in between min and max now, no need for a new comparison - this else if is just a sanity check
        elif matrix[currentmin[1]][currentmin[2]] < matrix[currentmin[1]][currentmax[2]] and matrix[currentmin[1]][currentmax[2]] < matrix[currentmax[1]][currentmax[2]]:
            CompsObj.increment(2)
            heap.popmax(CompsObj)
            heap.popmin(CompsObj)
            heap.insert_triplet([matrix[currentmin[1]][currentmax[2]], currentmin[1], currentmax[2]], CompsObj)
        else:
            raise ValueError('There was an error in the heap structure')
        assert minmaxheapproperty(heap.a, len(heap))
    return heap.peekmin(CompsObj)

def BienstockAlgorithm(matrix, rows, matrixid, CompsObj):
    BienstockBase(matrix, rows, CompsObj)
    # TODO: Confirm that final value is a saddlepoint
    UpdateResult(matrixid, "BienstockRes", CompsObj.value)

def RecursiveBase(matrix, rows, CompsObj, MultiRecursion): # MultiRecursion 0 : No extra Recursion, 1 : One extra Recursion, 2 : Recursion until block size is 8
    # define the minimum block size for recursion base
    min_l = 10
    # define the block size
    l = math.ceil(math.log(rows, 2))
    # calculate the amount of rows in the auxiliary matrix, if case for no overlapping
    if rows % l == 0:
        aux_rows = math.floor(rows/l)
    else:
        aux_rows = math.floor(rows/l) + 1
    aux_heap = MinMaxHeap(aux_rows)

    original_position_dict = {}
    # add PSP along the diagonal besides the last block that may be overlapping
    for i in range(0, (aux_rows - 1)*l, l):
        matrix_block = []
        matrix_block = matrix[i:i+l, i:i+l]
        # find PSP of submatrix along the diagonal
        # if case depending on level of recursions
        if MultiRecursion == 0 or l <= min_l:
            PSP = BienstockBase(matrix_block, len(matrix_block), CompsObj)
        elif MultiRecursion == 1:
            PSP = RecursiveBase(matrix_block, len(matrix_block), CompsObj, 0)
        elif MultiRecursion == 2:
            PSP = RecursiveBase(matrix_block, len(matrix_block), CompsObj, 2)
        # store the auxiliary position in the heap, store the real position in the dictionary
        aux_heap.insert_triplet([PSP[0], i//l, i//l], CompsObj)
        original_position_dict[(PSP[0], i//l, i//l)] = [PSP[0], PSP[1] + i, PSP[2] + i]

    # add the final, possibly overlapping element of the original matrix, if there are leftover rows/columns, store aux position and real position
    m_rows, m_cols = matrix.shape
    # if case depending on level of recursion
    if MultiRecursion == 0 or l <= min_l:
        PSP = BienstockBase(matrix[m_rows-l:m_rows, m_cols-l:m_cols], l, CompsObj)
    elif MultiRecursion == 1:
        PSP = RecursiveBase(matrix[m_rows-l:m_rows, m_cols-l:m_cols], l, CompsObj, 0)
    elif MultiRecursion == 2:
        PSP = RecursiveBase(matrix[m_rows-l:m_rows, m_cols-l:m_cols], l, CompsObj, 2)
    aux_heap.insert_triplet([PSP[0], aux_rows-1, aux_rows-1], CompsObj)
    original_position_dict[(PSP[0], aux_rows-1, aux_rows-1)] = [PSP[0], m_rows - l + PSP[1], m_rows - l + PSP[2]]
    
    while aux_heap.size > 1: # done heapsize-1 times
        currentmax = aux_heap.peekmax(CompsObj)
        currentmin = aux_heap.peekmin(CompsObj)
        # now need to evaluate PSP of aux matrix at position A[i,l]
        matrix_block = []
        aux_i = currentmin[1]
        aux_l = currentmax[2]
        # check if the min or the max are in the final block, that requires a different calculation of the real positions
        if aux_i == aux_rows-1:
            matrix_block = matrix[m_rows-l:m_rows, aux_l*l:(aux_l+1)*l]
        elif aux_l == aux_rows-1:
            matrix_block = matrix[(aux_i*l):(aux_i+1)*l, m_cols-l:m_cols]
        # aux_i and aux_l cannot both be the maximum value as that is already in the heap from the start, can ignore that case -> else is that it is neither max row nor max column
        else:
            matrix_block = matrix[(aux_i*l):(aux_i+1)*l, (aux_l*l):(aux_l+1)*l]
        
        # if case depending on level of recursion
        if MultiRecursion == 0 or l <= min_l:
            currentquery = BienstockBase(matrix_block, len(matrix_block), CompsObj)
        elif MultiRecursion == 1:
            currentquery = RecursiveBase(matrix_block, len(matrix_block), CompsObj, 0)
        elif MultiRecursion == 2:
            currentquery = RecursiveBase(matrix_block, len(matrix_block), CompsObj, 2)

        # Apply same properties of H
        # Check if A_il in the auxiliary matrix is smaller or equal than the minimum (at A_ij)
        if currentquery[0] <= currentmin[0]:
            CompsObj.increment()
            aux_heap.popmax(CompsObj)
            del original_position_dict[(currentmax[0], currentmax[1], currentmax[2])]
        # Check if A_il in the aux matrix is larger or equal than the maximum (at A_kl)
        elif currentmax[0] <= currentquery[0]:
            CompsObj.increment(2)
            aux_heap.popmin(CompsObj)
            del original_position_dict[(currentmin[0], currentmin[1], currentmin[2])]
        # A_il has to be between min and max, comparison here is only a sanity check
        elif currentmin[0] < currentquery[0] and currentquery[0] < currentmax[0]:
            CompsObj.increment(2)
            aux_heap.popmax(CompsObj)
            del original_position_dict[(currentmax[0], currentmax[1], currentmax[2])]
            aux_heap.popmin(CompsObj)
            del original_position_dict[(currentmin[0], currentmin[1], currentmin[2])]
            # add original position to dictionary (depends on )
            if aux_i == aux_rows-1:
                new_dict_entry = [currentquery[0], m_rows - l + currentquery[1], aux_l*l + currentquery[2]]
            elif aux_l == aux_rows-1:
                new_dict_entry = [currentquery[0], aux_i*l + currentquery[1], m_cols - l + currentquery[2]]
            else: # this else doesnt consider max row and max column since we should never consider that case
                new_dict_entry = [currentquery[0], aux_i*l + currentquery[1], aux_l*l + currentquery[2]]
            original_position_dict[(currentquery[0], aux_i, aux_l)] = new_dict_entry
            aux_heap.insert_triplet([currentquery[0], aux_i, aux_l], CompsObj)
        else:
            raise ValueError('There was an error in the heap structure')
    res = aux_heap.peekmin(CompsObj)
    return original_position_dict[res[0], res[1], res[2]]

def RecursiveAlgorithm(matrix, rows, matrixid, CompsObj):
    result = RecursiveBase(matrix, rows, CompsObj, 0)
    print(result[1])
    # TODO: Confirm that final value is a saddlepoint
    UpdateResult(matrixid, "RecursiveRes", CompsObj.value)

def TwoLevelRecursionAlgorithm(matrix, rows, matrixid, CompsObj):
    result = RecursiveBase(matrix, rows, CompsObj, 1)
    print(result[1])
    # TODO: Confirm that final value is a saddlepoint
    UpdateResult(matrixid, "TwoLevelRes", CompsObj.value)

def MultiLevelRecursionAlgorithm(matrix, rows, matrixid, CompsObj):
    result = RecursiveBase(matrix, rows, CompsObj, 2)
    print(result[1])
    # TODO: Confirm that final value is a saddlepoint
    UpdateResult(matrixid, "MultiLevelRes", CompsObj.value)

def UpdateResult(matrixid, field_name, result):
    if matrixid == 0:
        return
    try:
        conn = mysql_connection.new_connection()
        cursor = conn.cursor()
        update_query = f"UPDATE Matrices SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
        print("Record updated successfully")
    except mysql.connector.Error as error:
        print("Failed to update record: {}".format(error))
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
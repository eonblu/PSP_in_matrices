#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import pickle
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
        print (result)
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
        if matrix[currentmin[1]][currentmax[2]] <= matrix[currentmin[1]][currentmin[2]] and matrix[currentmin[1]][currentmin[2]] <= matrix[currentmax[1]][currentmax[2]]:
            CompsObj.increment(2)
            heap.popmax(CompsObj)
        elif matrix[currentmin[1]][currentmin[2]] <= matrix[currentmax[1]][currentmax[2]] and matrix[currentmax[1]][currentmax[2]] <= matrix[currentmin[1]][currentmax[2]]:
            if matrix[currentmin[1]][currentmax[2]] <= matrix[currentmin[1]][currentmin[2]]: # required increment if considering lazy evaluation for first if case
                CompsObj.increment()
            CompsObj.increment(3)
            heap.popmin(CompsObj)
        elif matrix[currentmin[1]][currentmin[2]] < matrix[currentmin[1]][currentmax[2]] and matrix[currentmin[1]][currentmax[2]] < matrix[currentmax[1]][currentmax[2]]:
            if matrix[currentmin[1]][currentmax[2]] <= matrix[currentmin[1]][currentmin[2]]: # required increment if considering lazy evaluation for first if case
                CompsObj.increment()
            if matrix[currentmin[1]][currentmin[2]] <= matrix[currentmax[1]][currentmax[2]]: # required increment if considering lazy evaluation for second if case
                CompsObj.increment()
            CompsObj.increment(4)
            heap.popmin(CompsObj)
            heap.popmax(CompsObj)
            heap.insert_triplet([matrix[currentmin[1]][currentmax[2]], currentmin[1], currentmax[2]], CompsObj)
        else:
            raise ValueError('There was an error in the heap structure')
        assert minmaxheapproperty(heap.a, len(heap))
    return heap.peekmin(CompsObj)

def BienstockAlgorithm(matrix, rows, matrixid, CompsObj):
    BienstockBase(matrix, rows, CompsObj)
    # TODO: Confirm that final value is a saddlepoint
    UpdateResult(matrixid, "BienstockRes", CompsObj.value)

def RecursiveAlgorithm(matrix, rows, matrixid, CompsObj):
    # define the block size
    l = math.ceil(math.log(rows, 2))
    # calculate the amount of rows in the auxiliry matrix, if case for no overlapping
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
        PSP = BienstockBase(matrix_block, len(matrix_block), CompsObj)
        # store the original position instead of the auxilary position in the heap
        aux_heap.insert_triplet([PSP[0], i//l, i//l], CompsObj)
        original_position_dict[(PSP[0], i//l, i//l)] = [PSP[0], PSP[1] + i, PSP[2] + i]

    # add the final, possibly overlapping element of the original matrix, if necessary
    m_rows, m_cols = matrix.shape
    if rows % l == 0:
        PSP = BienstockBase(matrix[m_rows-l:m_rows, m_cols-l:m_cols], l, CompsObj)
        aux_heap.insert_triplet([PSP[0], aux_rows-1, aux_rows-1], CompsObj)
        original_position_dict[(PSP[0], aux_rows-1, aux_rows-1)] = [PSP[0], m_rows - l + PSP[1], m_rows - l + PSP[2]]

    while aux_heap.size > 1: # done heapsize-1 times
        print(aux_heap.a)
        currentmax = aux_heap.peekmax(CompsObj)
        currentmin = aux_heap.peekmin(CompsObj)
        # now need to evaluate PSP of aux matrix at position A[i,l]
        matrix_block = []
        aux_i = currentmin[1]
        aux_l = currentmax[2]
        if aux_i == aux_rows-1:
            matrix_block = matrix[m_rows-l:m_rows, aux_l*l:(aux_l+1)*l]
            currentquery = BienstockBase(matrix_block, len(matrix_block), CompsObj)
            new_value = [currentquery[0], m_rows - l + currentquery[1], aux_l*l + currentquery[2]]
        elif aux_l == aux_rows-1:
            matrix_block = matrix[(aux_i*l):(aux_i+1)*l, m_cols-l:m_cols]
            currentquery = BienstockBase(matrix_block, len(matrix_block), CompsObj)
            new_value = [currentquery[0], aux_i*l + currentquery[1], m_cols - l + currentquery[2]]
        # aux_i and aux_l cannot both be the maximum value as that is already in the heap from the start
        else:
            matrix_block = matrix[(aux_i*l):(aux_i+1)*l, (aux_l*l):(aux_l+1)*l]
            currentquery = BienstockBase(matrix_block, len(matrix_block), CompsObj)
            new_value = [currentquery[0], aux_i*l + currentquery[1], aux_l*l + currentquery[2]]
        
        # Apply same properties of H
        if currentquery[0] <= currentmin[0]:
            aux_heap.popmax(CompsObj)
            del original_position_dict[(currentmax[0], currentmax[1], currentmax[2])]
        elif currentmax[0] <= currentquery[0]:
            aux_heap.popmin(CompsObj)
            del original_position_dict[(currentmin[0], currentmin[1], currentmin[2])]
        elif currentmin[0] < currentquery[0] and currentquery[0] < currentmax[0]:
            aux_heap.popmax(CompsObj)
            del original_position_dict[(currentmax[0], currentmax[1], currentmax[2])]
            aux_heap.popmin(CompsObj)
            del original_position_dict[(currentmin[0], currentmin[1], currentmin[2])]
            original_position_dict[(currentquery[0], aux_i, aux_l)] = new_value
            aux_heap.insert_triplet([currentquery[0], aux_i, aux_l], CompsObj)
    # UpdateResult(matrixid, "RecursiveRes", CompsObj.value)
    print(original_position_dict)

def UpdateResult(matrixid, field_name, result):
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

if __name__ == '__main__':
    # CompsObjB = Comparisons()
    # CompsObjR = Comparisons()
    # matrix, rows, matrixid = retrieve_matrix(15)
    # BienstockAlgorithm(matrix, rows, 15, CompsObjB)
    # RecursiveAlgorithm(matrix, rows, 15, CompsObjR)
    
    # print(BienstockBase([[5,5],[3,3]], 2, Comparisons()))
    # RecursiveAlgorithm(np.array([[6,5,5,5],[3,4,3,3],[3,5,4,4],[3,5,4,4]]), 4, 0, Comparisons())

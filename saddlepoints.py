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
        print("Retrieved matrix: \n", matrix)

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
    # calculate the  amount of rows in the auxiliry matrix
    aux_rows = math.floor(rows/l) + 1
    heap = MinMaxHeap(aux_rows)
    for i in range(0, aux_rows*l, l):
        matrix_block = []
        matrix_block = matrix[i:i+l, i:i+l]
        # find PSP of submatrix along the diagonal
        PSP = BienstockBase(matrix_block, len(matrix_block), CompsObj)
        # store the original position instead of the auxilary position in the heap
        heap.insert_triplet([PSP[0], PSP[1] + i, PSP[2] + i], CompsObj)
    while heap.size > 1: # done heapsize-1 times
        currentmax = heap.peekmax(CompsObj)
        currentmin = heap.peekmin(CompsObj)
        # now need to evaluate PSP of aux matrix at position A[i,l]
        matrix_block = []
        aux_i = math.floor(currentmin[1]/aux_rows)
        aux_l = math.floor(currentmax[2]/aux_rows)
        matrix_block = matrix[aux_i*aux_rows:aux_i*aux_rows+l, aux_l*aux_rows:aux_l*aux_rows+l]
        print(matrix_block)
        heap.popmin(CompsObj)

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
    CompsObj = Comparisons()
    matrix, rows, matrixid = retrieve_matrix(4)
    print(matrix)
    RecursiveAlgorithm(matrix, rows, 4, CompsObj)


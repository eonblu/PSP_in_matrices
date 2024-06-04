#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import pickle

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

def Bienstock(matrix, rows, matrixid, CompsObj):
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
            heap.size = 0
            print ("what")
        assert minmaxheapproperty(heap.a, len(heap))
    # TODO: Confirm that final value is a saddlepoint
    UpdateResult(matrixid, "BienstockRes", CompsObj.value)

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
    matrix, rows, matrixid = retrieve_matrix(1)
    Bienstock(matrix, rows, matrixid, CompsObj)
    print (CompsObj.value)

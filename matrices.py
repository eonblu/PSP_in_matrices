#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import pickle

import mysql_connection

def create_matrix_with_ssp(random_seed, rows):
    #set seed, to create consistent results, as the matrix itself wont be stored
    np.random.seed(random_seed+rows)
    matrix = np.random.randint(0, 2*rows, size=(rows, rows))
    #pick (pseudo) random position for saddle point
    ssp_row = np.random.randint(0, rows)
    ssp_column = np.random.randint(0, rows)
    print("Planting SSP at ["+str(ssp_row)+"] ["+str(ssp_column)+"]")

    ssp_value = round((2*rows)/2)
    matrix[ssp_row][ssp_column] = ssp_value

    for i in range(rows):
        # sets the seed to a predictable value such that matrices can be generated again just based on seed
        np.random.seed(i)
        if i != ssp_column and matrix[ssp_row][i] >= ssp_value:
            matrix[ssp_row][i] = np.random.randint(0, ssp_value)

    for i in range(rows):
        np.random.seed(i)
        if i != ssp_row and matrix[i][ssp_column] <= ssp_value:
            matrix[i][ssp_column] = np.random.randint(ssp_value, 2*rows)
    
    return matrix

def add_to_mysql_table(matrix):
    binary_data = pickle.dumps(matrix)
    rows = len(matrix)
    columns = len(matrix[0]) if rows > 0 else 0

    conn = mysql_connection.new_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO Matrices (MatrixBlob, MRows, MColumns, BienstockRes, RecursiveRes, RandomizedRes) VALUES (%s, %s, %s, %s, %s, %s)",
        (binary_data, rows, columns, 0, 0, 0)
    )
    conn.commit()
    print("Matrix stored with ID:", cursor.lastrowid)
    cursor.close()
    mysql_connection.close_connection(conn)


if __name__ == '__main__':
    seed = 216
    r = 20
    matrix1 = create_matrix_with_ssp(seed, r)
    print (matrix1)
    add_to_mysql_table(matrix1)
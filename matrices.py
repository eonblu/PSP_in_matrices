#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql

import mysql_connection

def create_matrix_with_ssp(random_seed, rows):
    #set seed, to create consistent results, as the matrix itself wont be stored
    np.random.seed(random_seed+rows)
    matrix = np.random.randint(0, 2*rows, size=(rows, rows))
    #pick (pseudo) random position for saddle point
    ssp_row = np.random.randint(0, rows)
    ssp_column = np.random.randint(0, rows)
    print("Planting SSP at ["+str(ssp_row)+"] ["+str(ssp_column)+"]")

    #set ssp_value to the middle of maximum value and mimimum value within matrix
    ssp_value = rows
    matrix[ssp_row][ssp_column] = ssp_value

    for i in range(rows):
        # sets the seed to a predictable value such that matrices can be generated again just based on seed
        np.random.seed(random_seed + i)
        if i != ssp_column and matrix[ssp_row][i] >= ssp_value:
            matrix[ssp_row][i] = np.random.randint(0, ssp_value - 1)

    for i in range(rows):
        np.random.seed(random_seed + i)
        if i != ssp_row and matrix[i][ssp_column] <= ssp_value:
            matrix[i][ssp_column] = np.random.randint(ssp_value + 1, 2*rows)
    
    return matrix

def add_to_mysql_table(matrix, seed):
    rows = len(matrix)
    columns = len(matrix[0]) if rows > 0 else 0
    if columns != rows:
        print ("Cannot add matrices with different size of columns and rows")
        return

    conn = mysql_connection.new_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO Matrices (MatrixSeed, MRows, BienstockRes, RecursiveRes) VALUES (%s, %s, %s, %s)",
        (seed, rows, 0, 0)
    )
    conn.commit()
    print("Matrix stored with ID:", cursor.lastrowid)
    cursor.close()
    mysql_connection.close_connection(conn)


if __name__ == '__main__':
    seed = 21592
    r = 24
    print(create_matrix_with_ssp(seed, r))
    # matrix1 = create_matrix_with_ssp(seed, r)
    # add_to_mysql_table(matrix1, seed)
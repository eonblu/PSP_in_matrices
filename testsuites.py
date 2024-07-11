#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math

import mysql_connection
from matrices import *
from deterministic_psp import *

def Testsuite2():
    for seed in range(151, 200, 1):
        create_in_result_tables("FindingMinL", 5, seed)
        create_in_result_tables("FindingMinL", 6, seed)
        create_in_result_tables("FindingMinL", 7, seed) 
        create_in_result_tables("FindingMinL", 8, seed)
        create_in_result_tables("FindingMinL", 9, seed)
        create_in_result_tables("FindingMinL", 10, seed)
        create_in_result_tables("FindingMinL", 11, seed)
        create_in_result_tables("FindingMinL", 12, seed)
        create_in_result_tables("FindingMinL", 13, seed)
    result = FetchTestMatrices("FindingMinL")
    if result:
        for matrix_info in result:
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            CompsObjB = Comparisons()
            CompsObjR = Comparisons()
            PSP = BienstockBase(matrix, rows, CompsObjB)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "FindingMinL", "BienstockRes", CompsObjB.value)
            PSP = RecursiveBase(matrix, rows, CompsObjR, 0)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "FindingMinL", "RecursiveRes", CompsObjR.value)

def FetchTestMatrices(table):
    conn = mysql_connection.new_connection()
    cursor = conn.cursor()

    if table == "FindingMinL":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM FindingMinL WHERE BienstockRes = 0 OR RecursiveRes = 0",)
        result = cursor.fetchall()

    cursor.close()
    mysql_connection.close_connection(conn)
    return result

def UpdateTestResult(matrixid, table, field_name, result):
    conn = mysql_connection.new_connection()
    cursor = conn.cursor()
    if table == "FindingMinL":
        update_query = f"UPDATE FindingMinL SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()

    cursor.close()
    conn.close()

if __name__ == '__main__':
    Testsuite2()
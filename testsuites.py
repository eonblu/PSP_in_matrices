#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math
import matplotlib.pyplot as plt
import statistics

import mysql_connection
from matrices import *
from deterministic_psp import *

def Testsuite1():
    for rows in range(3, 13, 1):
        AmountOfTestedMatrices = 500000
        RandomMatricesWithSSP = 0
        for seed in range(AmountOfTestedMatrices):
            np.random.seed(seed)
            matrix = np.random.randint(0, 10000*rows, size=(rows, rows))
            PSP = BienstockBase(matrix, rows, Comparisons())
            # confirm if PSP is SSP
            PSP_is_SSP = True
            for column in range(rows):
                if column != PSP[2] and matrix[PSP[1]][column] >= PSP[0]:
                    PSP_is_SSP = False
            for row in range(rows):
                if row != PSP[1] and matrix[row][PSP[2]] <= PSP[0]:
                    PSP_is_SSP = False
            if PSP_is_SSP:
                RandomMatricesWithSSP += 1
        
        conn = mysql_connection.new_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO SSPinRandomMatrices (MRows, MatricesWithSSP, NumberOfTestedMatrices) VALUES (%s, %s, %s)",
            (rows, RandomMatricesWithSSP, AmountOfTestedMatrices)
        )
        conn.commit()
        cursor.close()
        mysql_connection.close_connection(conn)       

def Testsuite1Graph():
    result = FetchTestMatrices("SSPinRandomMatrices")
    if result:
        MRows = [item[0] for item in result]
        x_values = [item[1] / item[2] for item in result]
        conjecture_values = [math.factorial(entry)*math.factorial(entry)/math.factorial(2*entry-1) for entry in MRows]

        plt.figure(figsize=(6, 6))
        line1 = plt.plot(MRows, x_values, marker='o', linestyle=':', color='r')
        line2 = plt.plot(MRows, conjecture_values, marker='x', linestyle=':', color='b')

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Matrices with SSP / # Matrices tested')
        plt.grid(True)
        plt.ylim(bottom=-0.001)
        plt.xlim(left=MRows[0], right=MRows[-1])
        plt.legend([line1[0], line2[0]], ["Test Results", "n! * m! / (n + m -1)!"])

        plt.savefig('RandomMatricesWithSSP.png')

def Testsuite2():
    for seed in range(1, 200, 1):
        for rows in range(5, 19, 1):
            create_in_result_tables("FindingMinL", rows, seed)
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

def Testsuite2Graph():
    result = FetchTestMatrices("FindingMinL_results")
    if result:
        comp1_dict = {}
        comp2_dict = {}
        for row_count, comp1, comp2 in result:
            if row_count not in comp1_dict:
                comp1_dict[row_count] = []
            if row_count not in comp2_dict:
                comp2_dict[row_count] = []
            comp1_dict[row_count].append(comp1)
            comp2_dict[row_count].append(comp2)

        sorted_keys = sorted(comp1_dict.keys())
        comp1_data = [comp1_dict[key] for key in sorted_keys]
        comp2_data = [comp2_dict[key] for key in sorted_keys]

        fig, ax = plt.subplots(figsize=(10, 7))
        positions1 = np.arange(len(sorted_keys)) * 2.0
        positions2 = positions1 + 0.7

        bp1 = ax.boxplot(comp1_data, positions=positions1, widths=0.6, patch_artist=True, boxprops=dict(facecolor="C0"), notch=True)
        bp2 = ax.boxplot(comp2_data, positions=positions2, widths=0.6, patch_artist=True, boxprops=dict(facecolor="C1"), notch=True)

        ax.set_xticks(positions1 + 0.35)
        ax.set_xticklabels(sorted_keys)
        ax.set_xlabel('Amount of Rows')
        ax.set_ylabel('Comparisons')
        ax.set_title('Boxplot of Comparisons by Amount of Rows')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Bienstock", "Submatrices"], loc='upper right', bbox_to_anchor=(0.2, 1))

        plt.savefig('FindingMinL_results.png')


def FetchTestMatrices(table):
    conn = mysql_connection.new_connection()
    cursor = conn.cursor()

    if table == "FindingMinL":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM FindingMinL WHERE BienstockRes = 0 OR RecursiveRes = 0",)
        result = cursor.fetchall()
    if table == "FindingMinL_results":
        cursor.execute("SELECT MRows, BienstockRes, RecursiveRes FROM FindingMinL WHERE NOT BienstockRes = 0 AND NOT RecursiveRes = 0",)
        result = cursor.fetchall()
    if table == "SSPinRandomMatrices":
        cursor.execute("SELECT * FROM SSPinRandomMatrices",)
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
    Testsuite1Graph()
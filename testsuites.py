#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math
import matplotlib.pyplot as plt
import statistics
from collections import defaultdict

import mysql_connection
from matrices import *
from deterministic_psp import *
from randomized_psp import *
from adjusted_randomized_psp import *
from sadjusted_randomized_psp import *
from tadjusted_randomized_psp import *

def Testsuite1(): # Chance of SSP in Random Matrices
    for rows in range(3, 13, 1):
        AmountOfTestedMatrices = 500000
        RandomMatricesWithSSP = 0
        for seed in range(AmountOfTestedMatrices):
            np.random.seed(seed)
            matrix = np.random.randint(0, 10000*rows, size=(rows, rows), dtype=uint32)
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

        plt.figure(figsize=(8, 4))
        line1 = plt.plot(MRows, x_values, marker='o', linestyle=':', color='r')
        line2 = plt.plot(MRows, conjecture_values, marker='x', linestyle=':', color='b')

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Matrices with SSP / # Matrices tested')
        plt.grid(True)
        plt.ylim(bottom=-0.001)
        plt.xlim(left=MRows[0], right=MRows[-1])
        plt.legend([line1[0], line2[0]], ["Test Results", "n! * m! / (n + m -1)!"])

        plt.savefig('ResultGraphs/RandomMatricesWithSSP.png')

def Testsuite2(): # Bienstock vs Dallant small matrices
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

        fig, ax = plt.subplots(figsize=(10, 5))
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

        plt.savefig('ResultGraphs/FindingMinL_results.png')

def Testsuite3(): # Bienstock vs Dallant general
    # Prerequisite: CREATE TABLE BienstockDallantGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int);
    for seed in range(1, 201, 1):
        for rows in range(500, 10001, 500):
            create_in_result_tables("BienstockDallantGeneral", rows, seed)
    result = FetchTestMatrices("BienstockDallantGeneral")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            CompsObjB = Comparisons()
            CompsObjR = Comparisons()
            PSP = BienstockBase(matrix, rows, CompsObjB)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "BienstockDallantGeneral", "BienstockRes", CompsObjB.value)
            PSP = RecursiveBase(matrix, rows, CompsObjR, 0)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "BienstockDallantGeneral", "RecursiveRes", CompsObjR.value)

def Testsuite3Graph():
    result = FetchTestMatrices("BienstockDallantGeneral_results")
    if result:
        grouped_data = defaultdict(list)
        for row, res1, res2 in result:
            grouped_data[row].append((res1, res2))

        rows = sorted(grouped_data.keys())
        result1 = [np.mean([res1 for res1, res2 in grouped_data[row]]) for row in rows]
        result2 = [np.mean([res2 for res1, res2 in grouped_data[row]]) for row in rows]

        plt.figure(figsize=(8, 4))

        line1 = plt.plot(rows, result1, linestyle='-', color='r', marker='')
        line2 = plt.plot(rows, result2, linestyle='-', color='b', marker='')

        plt.ylim(bottom=0)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,100000,200000,300000,400000,500000],["0","10⁵","2*10⁵","3*10⁵","4*10⁵","5*10⁵"])
        plt.xticks([500,2000,4000,6000,8000,10000],["500","2000","4000","6000","8000","10000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Comparisons made to find PSP')
        plt.grid(False)
        plt.legend([line1[0], line2[0]], ["Bienstock", "Dallant"])

        plt.savefig('ResultGraphs/BienstockDallantGeneral.png')

def Testsuite4(): # Bienstock vs Dallant vs TwoLevelRecursion general
    # Prerequisite: CREATE TABLE BienstockDallantTwoLevelGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int, TwoLevelRes int);
    for seed in range(201, 221, 1):
        for rows in range(500, 10001, 500):
            create_in_result_tables("BienstockDallantTwoLevelGeneral", rows, seed)
    result = FetchTestMatrices("BienstockDallantTwoLevelGeneral")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            CompsObjB = Comparisons()
            CompsObjR = Comparisons()
            CompsObjT = Comparisons()
            PSP = BienstockBase(matrix, rows, CompsObjB)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "BienstockDallantTwoLevelGeneral", "BienstockRes", CompsObjB.value)
            PSP = RecursiveBase(matrix, rows, CompsObjR, 0)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "BienstockDallantTwoLevelGeneral", "RecursiveRes", CompsObjR.value)
            PSP = RecursiveBase(matrix, rows, CompsObjT, 2)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "BienstockDallantTwoLevelGeneral", "TwoLevelRes", CompsObjT.value)

def Testsuite4Graph():
    result = FetchTestMatrices("BienstockDallantTwoLevelGeneral_results")
    if result:
        grouped_data = defaultdict(list)
        for row, res1, res2, res3 in result:
            grouped_data[row].append((res1, res2, res3))

        rows = sorted(grouped_data.keys())
        result1 = [np.mean([res1 for res1, res2, res3 in grouped_data[row]]) for row in rows]
        result2 = [np.mean([res2 for res1, res2, res3 in grouped_data[row]]) for row in rows]
        result3 = [np.mean([res3 for res1, res2, res3 in grouped_data[row]]) for row in rows]

        plt.figure(figsize=(8, 4))

        line1 = plt.plot(rows, result1, linestyle='-', color='r', marker='')
        line2 = plt.plot(rows, result2, linestyle='-', color='b', marker='')
        line3 = plt.plot(rows, result3, linestyle='-', color='g', marker='')

        plt.ylim(bottom=0)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,100000,200000,300000,400000,500000],["0","10⁵","2*10⁵","3*10⁵","4*10⁵","5*10⁵"])
        plt.xticks([500,2000,4000,6000,8000,10000,12000,14000,16000,18000],["500","2000","4000","6000","8000","10000","12000","14000","16000","18000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Comparisons made to find PSP')
        plt.grid(False)
        plt.legend([line1[0], line2[0], line3[0]], ["Bienstock", "Dallant", "Two Level Dallant"])

        plt.savefig('ResultGraphs/BienstockDallantTwoLevelGeneral.png')

def Testsuite5():
    # Prerequisite: CREATE TABLE RandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float);
    for seed in range(1, 41, 1):
        for rows in range(500, 10001, 500):
            create_in_result_tables("RandomizedPSPLargerSmaller", rows, seed)
    result = FetchTestMatrices("RandomizedPSPLargerSmaller")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            larger_counts = []
            smaller_counts = []
            for i in range(10):
                testresult = ReduceMatrixTestsuite(matrix, Comparisons())
                larger_counts.append(len(testresult[0]))
                smaller_counts.append(len(testresult[1]))
            UpdateTestResult(matrixid, "RandomizedPSPLargerSmaller", "LargerCount", (sum(larger_counts)/len(larger_counts)))
            UpdateTestResult(matrixid, "RandomizedPSPLargerSmaller", "SmallerCount", (sum(smaller_counts)/len(smaller_counts)))     

def Testsuite5Graph():
    result = FetchTestMatrices("RandomizedPSPLargerSmaller_results")
    if result:
        grouped_data = defaultdict(list)
        for row, res1, res2 in result:
            grouped_data[row].append([res1, res2])

        rows = sorted(grouped_data.keys())
        result = [np.mean([grouped_data[row]]) for row in rows]

        plt.figure(figsize=(6, 4))

        line1 = plt.plot(rows, result, linestyle='-', color='r', marker='')

        plt.ylim(bottom=0, top=3)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,1,2,3],["0","1","2","3"])
        plt.xticks([500,2000,4000,6000,8000,10000],["500","2000","4000","6000","8000","10000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Rows/Columns to be\nremoved based on pivot')
        plt.grid(False)

        plt.savefig('ResultGraphs/RandomizedPSPLargerSmaller.png')

def Testsuite6():
    # Prerequisite: CREATE TABLE AdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
    for seed in range(200, 301, 1):
        for rows in range(200, 10401, 200):
            create_in_result_tables("AdjustedRandomizedPSPLargerSmaller", rows, seed)      
    result = FetchTestMatrices("AdjustedRandomizedPSPLargerSmaller")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            larger_counts = []
            smaller_counts = []
            hard_failures = 0
            for i in range(20):
                testresult = AdjustedReduceMatrixTestsuite(matrix, Comparisons())
                if len(testresult[0]) == 1:
                    if testresult[0][0] == -1:
                        hard_failures += 1
                elif len(testresult[1]) == 1:
                    if testresult[1][0] == -1:
                        hard_failures += 1
                else:
                    larger_counts.append(len(testresult[0]))
                    smaller_counts.append(len(testresult[1]))
            UpdateTestResult(matrixid, "AdjustedRandomizedPSPLargerSmaller", "LargerCount", (sum(larger_counts)/len(larger_counts)))
            UpdateTestResult(matrixid, "AdjustedRandomizedPSPLargerSmaller", "SmallerCount", (sum(smaller_counts)/len(smaller_counts)))
            UpdateTestResult(matrixid, "AdjustedRandomizedPSPLargerSmaller", "HardFailures", hard_failures)

def Testsuite6Graph():
    result = FetchTestMatrices("AdjustedRandomizedPSPLargerSmaller_results")
    if result:
        grouped_data = defaultdict(list)
        failure_data = defaultdict(int)
        for row, res1, res2, res3 in result:
            grouped_data[row].append([res1, res2])
            failure_data[row] =+ res3

        rows = sorted(grouped_data.keys())
        failure_rate_line = []
        for n in rows:
            failure_rate_line.append((failure_data[n])/(40*len(grouped_data[n]))) # 20 runs for each matrix, 2 results per run

        plt.figure(figsize=(8, 4))

        rows = [row for row in rows]
        koverfour_rows = [row/4 for row in rows]
        result = [np.mean([grouped_data[row]]) for row in rows]
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(rows, result, linestyle='-', color='r', marker='')
        fill_between = ax1.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0, color='r')
        line2 = ax1.plot(rows, koverfour_rows, linestyle='-', color='b', marker='')
        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Rows/Columns')
        plt.xlim(left=rows[0], right=rows[-1])
        plt.ylim(bottom=50)

        ax2 = ax1.twinx()
        line3 = ax2.plot(rows, failure_rate_line, linestyle='-', color='g', marker='')
        plt.ylabel('Hard failure rate')
        plt.ylim(bottom=0, top=1)
        plt.legend([line1[0], line2[0], line3[0]], ["to be removed based on pivot", "minimum to pass soft failure", "% hard failures"], loc='upper left')
        plt.savefig('ResultGraphs/AdjustedRandomizedPSPLargerSmaller.png')
        plt.clf()

def Testsuite7():
    # Prerequisite: CREATE TABLE SAdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
    for seed in range(100, 201, 1):          
        for rows in range(200, 10401, 200):
            create_in_result_tables("SAdjustedRandomizedPSPLargerSmaller", rows, seed)      
    result = FetchTestMatrices("SAdjustedRandomizedPSPLargerSmaller")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            larger_counts = []
            smaller_counts = []
            hard_failures = 0
            for i in range(20):
                testresult = SAdjustedReduceMatrixTestsuite(matrix, Comparisons())
                if len(testresult[0]) == 1:
                    if testresult[0][0] == -1:
                        hard_failures += 1
                elif len(testresult[1]) == 1:
                    if testresult[1][0] == -1:
                        hard_failures += 1
                else:
                    larger_counts.append(len(testresult[0]))
                    smaller_counts.append(len(testresult[1]))
            UpdateTestResult(matrixid, "SAdjustedRandomizedPSPLargerSmaller", "LargerCount", (sum(larger_counts)/len(larger_counts)))
            UpdateTestResult(matrixid, "SAdjustedRandomizedPSPLargerSmaller", "SmallerCount", (sum(smaller_counts)/len(smaller_counts)))
            UpdateTestResult(matrixid, "SAdjustedRandomizedPSPLargerSmaller", "HardFailures", hard_failures)

def Testsuite7Graph():
    result = FetchTestMatrices("SAdjustedRandomizedPSPLargerSmaller_results")
    if result:
        grouped_data = defaultdict(list)
        failure_data = defaultdict(int)
        for row, res1, res2, res3 in result:
            grouped_data[row].append([res1, res2])
            failure_data[row] += res3

        rows = sorted(grouped_data.keys())
        failure_rate_line = []
        for n in rows:
            failure_rate_line.append((failure_data[n])/(40*len(grouped_data[n]))) # 20 runs for each matrix, 2 results per run

        plt.figure(figsize=(8, 4))

        rows = [row for row in rows]
        koverfour_rows = [row/4 for row in rows]
        result = [np.mean([grouped_data[row]]) for row in rows]
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(rows, result, linestyle='-', color='r', marker='')
        fill_between = ax1.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0)
        line2 = ax1.plot(rows, koverfour_rows, linestyle='-', color='b', marker='')
        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Rows/Columns')
        plt.xlim(left=rows[0], right=rows[-1])
        plt.ylim(bottom=50)

        ax2 = ax1.twinx()
        line3 = ax2.plot(rows, failure_rate_line, linestyle='-', color='g', marker='')
        plt.ylabel('Hard failure rate')
        plt.ylim(bottom=0, top=0.1)
        plt.legend([line1[0], line2[0], line3[0]], ["to be removed based on pivot", "minimum to pass soft failure", "% hard failures"], loc='upper left')
        plt.savefig('ResultGraphs/SAdjustedRandomizedPSPLargerSmaller.png')
        plt.clf()

def Testsuite8():
    # Prerequisite: CREATE TABLE TAdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
    for seed in range(300, 401, 1):          
        for rows in range(20, 401, 20):
            create_in_result_tables("TAdjustedRandomizedPSPLargerSmaller", rows, seed)      
    result = FetchTestMatrices("TAdjustedRandomizedPSPLargerSmaller")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            larger_counts = []
            smaller_counts = []
            hard_failures = 0
            for i in range(20):
                testresult = TAdjustedReduceMatrixTestsuite(matrix, Comparisons())
                if len(testresult[0]) == 1:
                    if testresult[0][0] == -1:
                        hard_failures += 1
                elif len(testresult[1]) == 1:
                    if testresult[1][0] == -1:
                        hard_failures += 1
                else:
                    larger_counts.append(len(testresult[0]))
                    smaller_counts.append(len(testresult[1]))
            UpdateTestResult(matrixid, "TAdjustedRandomizedPSPLargerSmaller", "LargerCount", (sum(larger_counts)/len(larger_counts)))
            UpdateTestResult(matrixid, "TAdjustedRandomizedPSPLargerSmaller", "SmallerCount", (sum(smaller_counts)/len(smaller_counts)))
            UpdateTestResult(matrixid, "TAdjustedRandomizedPSPLargerSmaller", "HardFailures", hard_failures)

def Testsuite8Graph():
    result = FetchTestMatrices("TAdjustedRandomizedPSPLargerSmaller_results")
    if result:
        grouped_data = defaultdict(list)
        failure_data = defaultdict(int)
        for row, res1, res2, res3 in result:
            grouped_data[row].append([res1, res2])
            failure_data[row] += res3

        rows = sorted(grouped_data.keys())
        failure_rate_line = []
        for n in rows:
            failure_rate_line.append((failure_data[n])/(40*len(grouped_data[n]))) # 20 runs for each matrix, 2 results per run

        plt.figure(figsize=(8, 4))

        rows = [row for row in rows]
        koverfour_rows = [row/4 for row in rows]
        result = [np.mean([grouped_data[row]]) for row in rows]
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(rows, result, linestyle='-', color='r', marker='')
        fill_between = ax1.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0)
        line2 = ax1.plot(rows, koverfour_rows, linestyle='-', color='b', marker='')
        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Rows/Columns')
        plt.xlim(left=rows[0], right=rows[-1])
        plt.ylim(bottom=0)

        ax2 = ax1.twinx()
        line3 = ax2.plot(rows, failure_rate_line, linestyle='-', color='g', marker='')
        plt.ylabel('Hard failure rate')
        plt.ylim(bottom=0, top=0.1)
        plt.legend([line1[0], line2[0], line3[0]], ["to be removed based on pivot", "minimum to pass soft failure", "% hard failures"], loc='upper left')
        plt.savefig('ResultGraphs/TAdjustedRandomizedPSPLargerSmaller.png')
        plt.clf()

def FetchTestMatrices(table):
    conn = mysql_connection.new_connection()
    cursor = conn.cursor()

    if table == "FindingMinL":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM FindingMinL WHERE BienstockRes = 0 OR RecursiveRes = 0",)
        result = cursor.fetchall()
    elif table == "FindingMinL_results":
        cursor.execute("SELECT MRows, BienstockRes, RecursiveRes FROM FindingMinL WHERE NOT BienstockRes = 0 AND NOT RecursiveRes = 0",)
        result = cursor.fetchall()
    elif table == "BienstockDallantGeneral":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM BienstockDallantGeneral WHERE BienstockRes = 0 OR RecursiveRes = 0",)
        result = cursor.fetchall()
    elif table == "BienstockDallantGeneral_results":
        cursor.execute("SELECT MRows, BienstockRes, RecursiveRes FROM BienstockDallantGeneral WHERE NOT BienstockRes = 0 AND NOT RecursiveRes = 0",)
        result = cursor.fetchall()
    elif table == "BienstockDallantTwoLevelGeneral":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM BienstockDallantTwoLevelGeneral WHERE BienstockRes = 0 OR RecursiveRes = 0 OR TwoLevelRes = 0",)
        result = cursor.fetchall()
    elif table == "BienstockDallantTwoLevelGeneral_results":
        cursor.execute("SELECT MRows, BienstockRes, RecursiveRes, TwoLevelRes FROM BienstockDallantTwoLevelGeneral WHERE NOT BienstockRes = 0 AND NOT RecursiveRes = 0 AND NOT TwoLevelRes = 0",)
        result = cursor.fetchall()
    elif table == "SSPinRandomMatrices":
        cursor.execute("SELECT * FROM SSPinRandomMatrices",)
        result = cursor.fetchall()
    elif table == "RandomizedPSPLargerSmaller":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM RandomizedPSPLargerSmaller WHERE LargerCount = 0 OR SmallerCount = 0")
        result = cursor.fetchall()
    elif table == "RandomizedPSPLargerSmaller_results":
        cursor.execute("SELECT MRows, LargerCount, SmallerCount FROM RandomizedPSPLargerSmaller WHERE NOT LargerCount = 0 AND NOT SmallerCount = 0")
        result = cursor.fetchall()
    elif table == "AdjustedRandomizedPSPLargerSmaller":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM AdjustedRandomizedPSPLargerSmaller WHERE LargerCount = 0 OR SmallerCount = 0")
        result = cursor.fetchall()
    elif table == "AdjustedRandomizedPSPLargerSmaller_results":
        cursor.execute("SELECT MRows, LargerCount, SmallerCount, HardFailures FROM AdjustedRandomizedPSPLargerSmaller WHERE NOT LargerCount = 0 AND NOT SmallerCount = 0")
        result = cursor.fetchall()    
    elif table == "SAdjustedRandomizedPSPLargerSmaller":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM SAdjustedRandomizedPSPLargerSmaller WHERE LargerCount = 0 OR SmallerCount = 0")
        result = cursor.fetchall()
    elif table == "SAdjustedRandomizedPSPLargerSmaller_results":
        cursor.execute("SELECT MRows, LargerCount, SmallerCount, HardFailures FROM SAdjustedRandomizedPSPLargerSmaller WHERE NOT LargerCount = 0 AND NOT SmallerCount = 0")
        result = cursor.fetchall()
    elif table == "TAdjustedRandomizedPSPLargerSmaller":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM TAdjustedRandomizedPSPLargerSmaller WHERE LargerCount = 0 OR SmallerCount = 0")
        result = cursor.fetchall()
    elif table == "TAdjustedRandomizedPSPLargerSmaller_results":
        cursor.execute("SELECT MRows, LargerCount, SmallerCount, HardFailures FROM TAdjustedRandomizedPSPLargerSmaller WHERE NOT LargerCount = 0 AND NOT SmallerCount = 0")
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
    elif table == "BienstockDallantGeneral":
        update_query = f"UPDATE BienstockDallantGeneral SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
    elif table == "BienstockDallantTwoLevelGeneral":
        update_query = f"UPDATE BienstockDallantTwoLevelGeneral SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()    
    elif table == "RandomizedPSPLargerSmaller":
        update_query = f"UPDATE RandomizedPSPLargerSmaller SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
    elif table == "AdjustedRandomizedPSPLargerSmaller":
        update_query = f"UPDATE AdjustedRandomizedPSPLargerSmaller SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
    elif table == "SAdjustedRandomizedPSPLargerSmaller":
        update_query = f"UPDATE SAdjustedRandomizedPSPLargerSmaller SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
    elif table == "TAdjustedRandomizedPSPLargerSmaller":
        update_query = f"UPDATE TAdjustedRandomizedPSPLargerSmaller SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    Testsuite8Graph()
#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import statistics
from collections import defaultdict

import mysql_connection
from matrices import *
from deterministic_psp import *
from randomized_psp import *
from adjusted_randomized_psp import *
from sadjusted_randomized_psp import *
from tadjusted_randomized_psp import *

# These are all the tables in database Matrices that need to be created if you want to run all tests, adjust final line to run TestsuiteX()
# CREATE TABLE SSPinRandomMatrices (MRows int, MatricesWithSSP int, NumberOfTestedMatrices int);
# CREATE TABLE FindingMinL (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int);
# CREATE TABLE BienstockDallantGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int);
# CREATE TABLE DallantTwoLevelGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, RecursiveRes int, TwoLevelRes int);
# CREATE TABLE RandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float);
# CREATE TABLE AdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
# CREATE TABLE SAdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
# CREATE TABLE TAdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
# CREATE TABLE FinalGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int, TwoLevelRes int, RandomizedRes int);

main_color = "xkcd:royal blue"
second_color = "xkcd:red"
third_color = "xkcd:green"
fourth_color = "xkcd:bright blue"

graph_size = (10, 4)
graph_withsubplot_size = (20, 4)

def ExampleGraph():
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')
    X = np.arange(1, 10, 1)
    Y = np.arange(1, 10, 1)
    X, Y = np.meshgrid(X, Y)
    matrix = np.matrix([[192, 255, 298, 321, 324, 307, 270, 213, 156], [131, 194, 237, 260, 263, 246, 209, 152, 75], [90, 153, 196, 219, 222, 205, 168, 111, 34], [69, 132, 175, 198, 201, 184, 147, 90, 13], [68, 131, 174, 197, 200, 183, 146, 89, 12], [87, 150, 193, 216, 219, 202, 165, 108, 31], [126, 189, 232, 255, 258, 241, 204, 147, 70], [185, 248, 291, 314, 317, 300, 263, 206, 129], [264, 327, 370, 393, 396, 379, 342, 285, 208]])
    ax.view_init(elev=25, azim=35)
    point = ax.scatter(5, 5, 201, color='xkcd:black', s=50, alpha=1)
    surf = ax.plot_surface(X, Y, matrix, cmap=cm.plasma, linewidth=0, antialiased=True, alpha=.7)
    points = ax.scatter(X, Y, matrix, color='xkcd:black', s=1)
    plt.yticks([])
    plt.xticks([])
    plt.savefig('ResultGraphs/ExampleGraph.svg',bbox_inches='tight')

def Testsuite1(): # Chance of SSP in Random Matrices
    # Prerequisite: CREATE TABLE SSPinRandomMatrices (MRows int, MatricesWithSSP int, NumberOfTestedMatrices int);
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

        plt.figure(figsize=graph_size)
        line1 = plt.plot(MRows, x_values, marker='o', linestyle=':', color=main_color)
        line2 = plt.plot(MRows, conjecture_values, marker='x', linestyle=':', color=second_color)

        plt.xlabel('# Matrix Rows')
        plt.ylabel('Probability of SSP')
        plt.grid(False)
        plt.ylim(bottom=-0.001)
        plt.xlim(left=MRows[0], right=MRows[-1])
        plt.yticks([0,0.05,0.1,0.15,0.2,0.25,0.3],["0%","5%","10%","15%","20%","25%","30%"])

        plt.legend([line1[0], line2[0]], ["Test results", "n! * m! / (n + m -1)!"])

        plt.savefig('ResultGraphs/RandomMatricesWithSSP.svg')

def Testsuite2(): # Bienstock vs Dallant small matrices
    # Prerequisite: CREATE TABLE FindingMinL (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int);
    for seed in range(1, 251, 1):
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

        fig, ax = plt.subplots(figsize=graph_size)
        positions1 = np.arange(len(sorted_keys)) * 2.0
        positions2 = positions1 + 0.7

        bp1 = ax.boxplot(comp1_data, positions=positions1, widths=0.6, patch_artist=True, boxprops={'facecolor': main_color}, notch=True, meanline=False, showfliers=False, medianprops={'color': 'none'})
        bp2 = ax.boxplot(comp2_data, positions=positions2, widths=0.6, patch_artist=True, boxprops={'facecolor': second_color}, notch=True, meanline=False, showfliers=False, medianprops={'color': 'none'})

        ax.set_xticks(positions1 + 0.35)
        ax.set_xticklabels(sorted_keys)
        ax.set_xlabel('# Matrix Rows')
        ax.set_ylabel('# Comparisons between Matrix entries')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Bienstock", "Dallant"], loc='upper right', bbox_to_anchor=(0.2, 1))

        plt.savefig('ResultGraphs/FindingMinL_results.svg')

def Testsuite3(): # Bienstock vs Dallant general
    # Prerequisite: CREATE TABLE BienstockDallantGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int);
    for seed in range(251, 351, 1):
        for rows in range(200, 10001, 200):
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

        plt.figure(figsize=graph_size)

        line1 = plt.plot(rows, result1, linestyle='-', color=main_color, marker='')
        line2 = plt.plot(rows, result2, linestyle='-', color=second_color, marker='')

        plt.ylim(bottom=0)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,100000,200000,300000,400000,500000],["0","10⁵","2*10⁵","3*10⁵","4*10⁵","5*10⁵"])
        plt.xticks([200,2000,4000,6000,8000,10000],["200","2000","4000","6000","8000","10000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Comparisons between Matrix entries')
        plt.grid(False)
        plt.legend([line1[0], line2[0]], ["Bienstock", "Dallant"])

        plt.savefig('ResultGraphs/BienstockDallantGeneral.svg')

def Testsuite4(): # Dallant vs TwoLevelRecursion general
    # Prerequisite: CREATE TABLE DallantTwoLevelGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, RecursiveRes int, TwoLevelRes int);
    for seed in range(351, 451, 1):
        for rows in range(200, 18001, 200):
            create_in_result_tables("DallantTwoLevelGeneral", rows, seed)
    result = FetchTestMatrices("DallantTwoLevelGeneral")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            CompsObjR = Comparisons()
            CompsObjT = Comparisons()
            PSP = RecursiveBase(matrix, rows, CompsObjR, 0)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "DallantTwoLevelGeneral", "RecursiveRes", CompsObjR.value)
            PSP = RecursiveBase(matrix, rows, CompsObjT, 2)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "DallantTwoLevelGeneral", "TwoLevelRes", CompsObjT.value)

def Testsuite4Graph():
    result = FetchTestMatrices("DallantTwoLevelGeneral_results")
    if result:
        grouped_data = defaultdict(list)
        for row, res1, res2 in result:
            grouped_data[row].append((res1, res2))

        rows = sorted(grouped_data.keys())
        result1 = [np.mean([res1 for res1, res2 in grouped_data[row]]) for row in rows]
        result2 = [np.mean([res2 for res1, res2 in grouped_data[row]]) for row in rows]

        plt.figure(figsize=graph_size)

        line1 = plt.plot(rows, result1, linestyle='-', color=second_color, marker='')
        line2 = plt.plot(rows, result2, linestyle='-', color=main_color, marker='')

        plt.ylim(bottom=0)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,100000,200000,300000,400000,500000],["0","10⁵","2*10⁵","3*10⁵","4*10⁵","5*10⁵"])
        plt.xticks([2000,4000,6000,8000,10000,12000,14000,16000,18000],["2000","4000","6000","8000","10000","12000","14000","16000","18000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Comparisons between matrix entries')
        plt.grid(False)
        plt.legend([line1[0], line2[0]], ["Dallant", "Dallant recursively applied"])

        plt.savefig('ResultGraphs/DallantTwoLevelGeneral.svg')

def Testsuite5():
    # Prerequisite: CREATE TABLE RandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float);
    for seed in range(451, 551, 1):
        for rows in range(200, 10001, 200):
            create_in_result_tables("RandomizedPSPLargerSmaller", rows, seed)
    result = FetchTestMatrices("RandomizedPSPLargerSmaller")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            larger_counts = []
            smaller_counts = []
            for i in range(20):
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
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        plt.figure(figsize=graph_size)

        line1 = plt.plot(rows, result, linestyle='-', color=main_color, marker='')
        fill_between = plt.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0, color=main_color)

        plt.ylim(bottom=0, top=3)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,1,2,3,4,5],["0","1","2","3","4","5"])
        plt.xticks([200,2000,4000,6000,8000,10000],["200","2000","4000","6000","8000","10000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Rows/Columns to be\nremoved based on pivot')
        plt.legend([line1[0]], ["to be removed based on pivot"], loc='upper left')
        plt.grid(False)

        plt.savefig('ResultGraphs/RandomizedPSPLargerSmaller.svg')

def Testsuite6():
    # Prerequisite: CREATE TABLE AdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
    for seed in range(551, 651, 1):
        for rows in range(200, 10001, 200):
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

        rows = [row for row in rows]
        koverfour_rows = [row/4 for row in rows]
        result = [np.mean([grouped_data[row]]) for row in rows]
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        fig, ax1 = plt.subplots(1, 1, figsize=(graph_size))
        line1 = ax1.plot(rows, result, linestyle='-', color=main_color, marker='')
        fill_between = ax1.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0, color=main_color)
        line2 = ax1.plot(rows, koverfour_rows, linestyle='-', color=second_color, marker='')
        plt.xlabel('# Matrix Rows')
        plt.xticks([200,2000,4000,6000,8000,10000],["200","2000","4000","6000","8000","10000"])
        plt.ylabel('# Rows/Columns to be\nremoved based on pivot')
        plt.xlim(left=rows[0], right=rows[-1])
        plt.ylim(bottom=50)

        ax2 = ax1.twinx()
        line3 = ax2.plot(rows, failure_rate_line, linestyle='-', color=third_color, marker='')
        plt.ylabel('Hard failure rate')
        plt.ylim(bottom=0, top=0.05)
        plt.yticks([0,0.01,0.02,0.03,0.04,0.05],["0%","1%","2%","3%","4%","5%"])
        plt.legend([line1[0], line2[0], line3[0]], ["to be removed based on pivot", "minimum to pass soft failure", "hard failures"], loc='upper left')
        plt.savefig('ResultGraphs/AdjustedRandomizedPSPLargerSmaller.svg')
        plt.clf()

def Testsuite7():
    # Prerequisite: CREATE TABLE SAdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
    for seed in range(651, 751, 1):          
        for rows in range(200, 10001, 200):
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

        rows = [row for row in rows]
        koverfour_rows = [row/4 for row in rows]
        result = [np.mean([grouped_data[row]]) for row in rows]
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        fig, ax1 = plt.subplots(1, 1, figsize=(graph_size))
        line1 = ax1.plot(rows, result, linestyle='-', color=main_color, marker='')
        fill_between = ax1.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0, color=main_color)
        line2 = ax1.plot(rows, koverfour_rows, linestyle='-', color=second_color, marker='')
        plt.xlabel('# Matrix Rows')
        plt.xticks([200,2000,4000,6000,8000,10000],["200","2000","4000","6000","8000","10000"])
        plt.ylabel('# Rows/Columns to be\nremoved based on pivot')
        plt.xlim(left=rows[0], right=rows[-1])
        plt.ylim(bottom=50)

        ax2 = ax1.twinx()
        line3 = ax2.plot(rows, failure_rate_line, linestyle='-', color=third_color, marker='')
        plt.ylabel('Hard failure rate')
        plt.yticks([0,0.01,0.02,0.03,0.04,0.05],["0%","1%","2%","3%","4%","5%"])
        plt.ylim(bottom=0, top=0.05)
        plt.legend([line1[0], line2[0], line3[0]], ["to be removed based on pivot", "minimum to pass soft failure", "hard failures"], loc='upper left')
        plt.savefig('ResultGraphs/SAdjustedRandomizedPSPLargerSmaller.svg')
        plt.clf()

def Testsuite8():
    # Prerequisite: CREATE TABLE TAdjustedRandomizedPSPLargerSmaller (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, LargerCount float, SmallerCount float, HardFailures int);
    for seed in range(751, 851, 1):          
        for rows in range(20, 401, 5):
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

        rows = [row for row in rows]
        koverfour_rows = [row/4 for row in rows]
        result = [np.mean([grouped_data[row]]) for row in rows]
        max_res = [np.max([grouped_data[row]]) for row in rows]
        min_res = [np.min([grouped_data[row]]) for row in rows]

        fig, ax1 = plt.subplots(1, 1, figsize=(graph_size))
        line1 = ax1.plot(rows, result, linestyle='-', color=main_color, marker='')
        fill_between = ax1.fill_between(rows, max_res, min_res, alpha=.3, linewidth=0, color=main_color)
        line2 = ax1.plot(rows, koverfour_rows, linestyle='-', color=second_color, marker='')
        plt.xlabel('# Matrix Rows')
        plt.xticks([20,100,200,300,400],["20","100","200","300","400"])
        plt.ylabel('# Rows/Columns to be\nremoved based on pivot')
        plt.xlim(left=rows[0], right=rows[-1])
        plt.ylim(bottom=0)

        ax2 = ax1.twinx()
        line3 = ax2.plot(rows, failure_rate_line, linestyle='-', color=third_color, marker='')
        plt.ylabel('Hard failure rate')
        plt.yticks([0,0.01,0.02,0.03,0.04,0.05],["0%","1%","2%","3%","4%","5%"])
        plt.ylim(bottom=0, top=0.05)
        plt.legend([line1[0], line2[0], line3[0]], ["to be removed based on pivot", "minimum to pass soft failure", "hard failures"], loc='upper left')
        plt.savefig('ResultGraphs/TAdjustedRandomizedPSPLargerSmaller.svg')
        plt.clf()

def Testsuite9(): # Final comparison between Bienstock, Dallant 1 & 2, Adjusted Randomized
    # Prerequisite: CREATE TABLE FinalGeneral (MatrixID int AUTO_INCREMENT PRIMARY KEY, MatrixSeed int, MRows int, BienstockRes int, RecursiveRes int, TwoLevelRes int, RandomizedRes int);
    for seed in range(851, 951, 1):
        for rows in range(200, 10001, 200):
            create_in_result_tables("FinalGeneral", rows, seed)
    result = FetchTestMatrices("FinalGeneral")
    if result:
        for matrix_info in result:
            sys.stdout.flush()
            matrixid, seed, rows = matrix_info
            matrix = create_matrix_with_ssp(seed, rows)
            CompsObjB = Comparisons()
            CompsObjD = Comparisons()
            CompsObjT = Comparisons()
            CompsObjR = Comparisons()
            PSP = BienstockBase(matrix, rows, CompsObjB)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "FinalGeneral", "BienstockRes", CompsObjB.value)
            PSP = RecursiveBase(matrix, rows, CompsObjD, 0)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "FinalGeneral", "RecursiveRes", CompsObjD.value)
            PSP = RecursiveBase(matrix, rows, CompsObjT, 2)
            if PSP[0] != rows:
                print("Error in calculation of PSP")
            else:
                UpdateTestResult(matrixid, "FinalGeneral", "TwoLevelRes", CompsObjT.value)
            
            error = False
            CompsList = []
            for i in range(20):
                CompsObjR = Comparisons()
                PSP = TAdjustedReduceMatrix(matrix, CompsObjR)
                if PSP != rows:
                    print("Error in calculation of PSP")
                    error = True
                else:
                    CompsList.append(CompsObjR.value)
            if not error:
                UpdateTestResult(matrixid, "FinalGeneral", "RandomizedRes", sum(CompsList)/len(CompsList))

def Testsuite9Graph():
    result = FetchTestMatrices("FinalGeneral_results")
    if result:
        grouped_data = defaultdict(list)
        for row, res1, res2, res3, res4 in result:
            grouped_data[row].append((res1, res2, res3, res4))

        rows = sorted(grouped_data.keys())
        result1 = [np.mean([res1 for res1, res2, res3, res4 in grouped_data[row]]) for row in rows]
        result2 = [np.mean([res2 for res1, res2, res3, res4 in grouped_data[row]]) for row in rows]
        result3 = [np.mean([res3 for res1, res2, res3, res4 in grouped_data[row]]) for row in rows]
        result4 = [np.mean([res4 for res1, res2, res3, res4 in grouped_data[row]]) for row in rows]

        plt.figure(figsize=graph_size)

        line1 = plt.plot(rows, result1, linestyle='-', color=main_color, marker='')
        line2 = plt.plot(rows, result2, linestyle='-', color=second_color, marker='')
        line3 = plt.plot(rows, result3, linestyle='-', color=fourth_color, marker='')
        line4 = plt.plot(rows, result4, linestyle='-', color=third_color, marker='')
        fill_between4 = plt.fill_between(rows, [np.max([res4 for res1, res2, res3, res4 in grouped_data[row]]) for row in rows], [np.min([res4 for res1, res2, res3, res4 in grouped_data[row]]) for row in rows], alpha=.2, linewidth=0, color=third_color)

        plt.ylim(bottom=0)
        plt.xlim(left=rows[0], right=rows[-1])
        plt.yticks([0,100000,200000,300000,400000,500000],["0","10⁵","2*10⁵","3*10⁵","4*10⁵","5*10⁵"])
        plt.xticks([200,2000,4000,6000,8000,10000],["200","2000","4000","6000","8000","10000"])

        plt.xlabel('# Matrix Rows')
        plt.ylabel('# Comparisons between Matrix entries')
        plt.grid(False)
        plt.legend([line1[0], line2[0], line3[0], line4[0]], ["Bienstock", "Dallant", "Dallant recursively applied", "Adjusted Randomized"])

        plt.savefig('ResultGraphs/FinalGeneral.svg') 

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
    elif table == "DallantTwoLevelGeneral":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM DallantTwoLevelGeneral WHERE RecursiveRes = 0 OR TwoLevelRes = 0",)
        result = cursor.fetchall()
    elif table == "DallantTwoLevelGeneral_results":
        cursor.execute("SELECT MRows, RecursiveRes, TwoLevelRes FROM DallantTwoLevelGeneral WHERE NOT RecursiveRes = 0 AND NOT TwoLevelRes = 0",)
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
    elif table == "FinalGeneral":
        cursor.execute("SELECT MatrixID, MatrixSeed, MRows FROM FinalGeneral WHERE BienstockRes = 0 OR RecursiveRes = 0 OR TwoLevelRes = 0 OR RandomizedRes = 0",)
        result = cursor.fetchall()
    elif table == "FinalGeneral_results":
        cursor.execute("SELECT MRows, BienstockRes, RecursiveRes, TwoLevelRes, RandomizedRes FROM FinalGeneral WHERE NOT BienstockRes = 0 AND NOT RecursiveRes = 0 AND NOT TwoLevelRes = 0 AND NOT RandomizedRes = 0",)
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
    elif table == "DallantTwoLevelGeneral":
        update_query = f"UPDATE DallantTwoLevelGeneral SET {field_name} = %s WHERE MatrixID = %s"
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
    elif table == "FinalGeneral":
        update_query = f"UPDATE FinalGeneral SET {field_name} = %s WHERE MatrixID = %s"
        cursor.execute(update_query, (result, matrixid))
        conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    Testsuite1()
    Testsuite1Graph()
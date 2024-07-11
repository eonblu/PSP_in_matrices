#!/usr/bin/env python3
import sys
import numpy as np
import mysql.connector as mysql

def load_auth(access_rights):
    file_path = 'auth.txt'
    auth_info = {}
    current_section = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('[creator]') and access_rights == 'creator':
                username, password = line.replace('[creator]','').split(':', 1)
                return username, password
            elif line.startswith('[reader]') and access_rights == "reader":
                username, password = line.replace('[reader]','').split(':', 1)
                return username, password
    return "No correct user selected", " "

def new_connection():
    try:
        username, password = load_auth("creator")
        # Establish the connection
        connection = mysql.connect(
            host='localhost',
            user=username,
            password=password,
            database='Matrices'
        )

        if connection.is_connected():
            db_Info = connection.get_server_info()
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
    except Error as e:
        print("Error while connecting to MySQL: {e}")
    return connection

def close_connection(connection):
    if connection.is_connected():
        connection.close()

if __name__ == '__main__':
    conn = new_connection()
    close_connection(conn)

const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const fs = require('fs');

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

function readAuthFile() {
    const data = fs.readFileSync('../auth.txt', 'utf-8');
    const lines = data.split('\n');
    const pattern = /\[reader\](\w+):(.+)/;
    for (let line of lines) {
        const match = line.match(pattern);
        if (match) {
            const username = match[1];
            const password = match[2];
            return { username, password };
        }
    }
    
    return null;
}
const credentials = readAuthFile();
const db = mysql.createConnection({
    host:'127.0.0.1',
    user: credentials.username,
    password: credentials.password,
    database:'Matrices'
});

db.connect((err) => {
    if (err) {
        console.error('Error connecting to the database:', err);
        return;
    }
    console.log('Connected to the MySQL database.');
});

let cachedData = [];

const updateData = () => {
    const query = 'SELECT MatrixID, MRows, BienstockRes, RecursiveRes, TwoLevelRes, MultiLevelRes FROM Matrices';
    db.query(query, (err, results) => {
        if (err) {
            console.error('Error fetching data:', err);
            return;
        }
        cachedData = results;
    });
};

// Initial data fetch
updateData();

// Polling interval set to 1 minute (60000 milliseconds)
setInterval(updateData, 60000);

app.get('/data', (req, res) => {
    res.json(cachedData);
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
    const [data, setData] = useState([]);

    useEffect(() => {
        axios.get('http://161.35.206.159:5000/data')
            .then(response => {
                setData(response.data);
            })
            .catch(error => {
                console.error('Error fetching data:', error);
            });
    }, []);

    const ScatterPlot = () => (
        <ResponsiveContainer width="100%" height={400}>
            <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
                <CartesianGrid />
                <XAxis type="number" dataKey="MRows" name="MRows" />
                <YAxis type="number" dataKey="BienstockRes" name="BienstockRes" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Data Points" data={data} fill="#8884d8" />
            </ScatterChart>
        </ResponsiveContainer>
    );

    return (
        <div className="App">
            <ScatterPlot />
        </div>
    );
}

export default App;
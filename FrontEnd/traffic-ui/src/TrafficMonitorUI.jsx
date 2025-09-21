import React, { useEffect, useState } from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Type "npm start" in the terminal to run the FrontEnd
const TrafficMonitorUI = () => {
  // Dummy datas for presenting, need to replace them with actual data from Backend
  const [lastCar, setLastCar] = useState(null);
  const [power, setPower] = useState(80);
  const [totalCommercial, setTotalCommercial] = useState(8);
  const [totalNormal, setTotalNormal] = useState(12);
  const [totalVehicles, setTotalVehicles] = useState(20);

  const [historicalPower, setHistoricalPower] = useState([]);
  const [historicalCars, setHistoricalCars] = useState([]);

  // -------------------- Fetching data from Backend-------------------------
  // useEffect(() => {
  //   const fetchLastCar = async () => {
  //     try {
  //       const res = await fetch("http://localhost:3000/api/lastcar");
  //       const data = await res.json();
  //       setLastCar(data.lastCar);
  //     } catch (err) {
  //       console.error("Failed to fetch lastCar:", err);
  //     }
  //   };
  //   const fetchPower = async () => {
  //   try {
  //       const res = await fetch("http://localhost:3000/api/power");
  //       const data = await res.json();
  //       setPower(data.power);
  //     } catch (err) {
  //       console.error("Failed to fetch power:", err);
  //     }
  //   };

  //   fetchLastCar();
  //   fetchPower();

  //   // Control the refresh rate of fetching data
  //   const interval = setInterval(() => {
  //     fetchLastCar();
  //     fetchPower();
  //   }, 1000); // Refresh every 1 second
  //   return () => clearInterval(interval);
  // }, []);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:3001/ws");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastCar(data.lastCar);
      setPower(data.power);
      setTotalNormal(data.totalNormal);
      setTotalCommercial(data.totalCommercial);
      setTotalVehicles(data.totalVehicles);

      // Update historical data and keep only the last 60 seconds
      setHistoricalPower(prevData => {
        const newData = [...prevData, { power: data.power }];
        return newData.slice(Math.max(newData.length - 60, 0));
      });

      setHistoricalCars(prevData => {
        const newData = [...prevData, { normal: data.totalNormal, commercial: data.totalCommercial }];
        return newData.slice(Math.max(newData.length - 60, 0));
      });
    };

    ws.onopen = () => console.log("WebSocket connected");
    ws.onclose = () => console.log("WebSocket disconnected");
    ws.onerror = (error) => console.error("WebSocket error:", error);

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  const formattedPowerData = historicalPower.map((point, index) => ({
    time: `${historicalPower.length - index}s`,
    power: point.power,
  })).reverse();

  const formattedCarsData = historicalCars.map((point, index) => ({
    time: `${historicalCars.length - index}s`,
    normal: point.normal,
    commercial: point.commercial,
  })).reverse();

  // ------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-black text-white p-6 rounded-3xl flex flex-col gap-6 w-full max-w-7xl mx-auto">
      <div className="text-3xl font-bold text-center">Ultra-low-power acoustic trafficMonitoring</div>

      <div className="flex flex-col md:flex-row gap-6 justify-between">
        {/* Car status */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-1/5 flex flex-col items-center">
          <h2 className="text-4xl mb-4">Last Car</h2>
          <div className={`w-40 py-3 mb-4 rounded-xl text-center font-semibold ${lastCar === 'Commercial' ? 'bg-yellow-300 text-black' : 'bg-zinc-700 text-gray-400'}`}>
            Commercial
          </div>
          <div className={`w-40 py-3 rounded-xl text-center font-semibold ${lastCar === 'Normal' ? 'bg-yellow-300 text-black' : 'bg-zinc-700 text-gray-400'}`}>
            Normal
          </div>
        </div>

        {/* Line chart for power consumption */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-3/5">
          <h2 className="text-4xl mb-4">Power Consumption Over Time</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={formattedPowerData}>
              <XAxis dataKey="time" stroke="#aaa" />
              <YAxis
                stroke="#aaa"
                domain={[0, 'auto']}
                tickCount={5}
                tickFormatter={(value) => `${value}mAh`}
              />
              <Tooltip />
              <Line type="monotone" dataKey="power" stroke="#f6ad55" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Battery Ring */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-1/5 flex flex-col items-center justify-center">
          <h2 className="text-2xl mb-4">Current Power Consumption</h2>
          <div className="w-32 h-32">
            <CircularProgressbar
              value={power}
              maxValue={100}
              text={`${power}mAh`}
              styles={buildStyles({
                pathColor: 'red',
                textColor: 'white',
                trailColor: '#555'
              })}
            />
          </div>
        </div>
      </div>

      {/* Line chart for car numbers */}
      <div className="flex flex-col md:flex-row gap-6">
        <div className="bg-zinc-800 rounded-2xl p-6 flex-1 text-base space-y-20">
          <div>Total Normal: {String(totalNormal).padStart(3, '0')}</div>
          <div>Total Commercial: {String(totalCommercial).padStart(3, '0')}</div>
          <div>Total Vehicles: {String(totalVehicles).padStart(3, '0')}</div>
        </div>

        <div className="bg-zinc-800 rounded-2xl p-6 flex-[5]">
          <h2 className="text-4xl mb-4">Cars Passed Over Time</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={formattedCarsData}>
              <XAxis dataKey="time" stroke="#aaa" />
              <YAxis stroke="#aaa" domain={[0, 'auto']} tickCount={5} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="normal" stroke="#f6ad55" name="Normal" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="commercial" stroke="#a78bfa" name="Commercial" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default TrafficMonitorUI;

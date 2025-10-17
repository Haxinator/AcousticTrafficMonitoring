import React, { useEffect, useState } from "react";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const TrafficMonitorUI = () => {
  const [lastCar, setLastCar] = useState(null);
  const [power, setPower] = useState(80);
  const [totalCommercial, setTotalCommercial] = useState(8);
  const [totalNormal, setTotalNormal] = useState(12);
  const [totalVehicles, setTotalVehicles] = useState(20);
  const [historicalPower, setHistoricalPower] = useState([]);
  const [historicalCars, setHistoricalCars] = useState([]);

  const WINDOW_MINUTES = 10;
  const TICK_COUNT = 6;

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:3001/ws");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const now = Date.now();

      setLastCar(data.lastCar);
      setPower(data.power);
      setTotalNormal(data.totalNormal);
      setTotalCommercial(data.totalCommercial);
      setTotalVehicles(data.totalVehicles);

      setHistoricalPower((prev) => {
        const newData = [...prev, { time: now, power: data.power }];
        const cutoff = now - WINDOW_MINUTES * 60 * 1000;
        return newData.filter((d) => d.time >= cutoff);
      });

      setHistoricalCars((prev) => {
        const newData = [
          ...prev,
          {
            time: now,
            normal: data.totalNormal,
            commercial: data.totalCommercial,
          },
        ];
        const cutoff = now - WINDOW_MINUTES * 60 * 1000;
        return newData.filter((d) => d.time >= cutoff);
      });
    };

    ws.onopen = () => console.log("WebSocket connected");
    ws.onclose = () => console.log("WebSocket disconnected");
    ws.onerror = (error) => console.error("WebSocket error:", error);

    return () => ws.close();
  }, []);

  const formatTime = (timestamp) => {
    const d = new Date(timestamp);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const now = Date.now();
  const start = now - WINDOW_MINUTES * 60 * 1000;
  const tickInterval = (WINDOW_MINUTES * 60 * 1000) / (TICK_COUNT - 1);
  const ticks = Array.from({ length: TICK_COUNT }, (_, i) => start + i * tickInterval);

  return (
    <div className="min-h-screen bg-black text-white p-6 rounded-3xl flex flex-col gap-6 w-full max-w-7xl mx-auto">
      <div className="text-3xl font-bold text-center">
        Ultra-low-power acoustic trafficMonitoring
      </div>

      <div className="flex flex-col md:flex-row gap-6 justify-between">
        {/* Car status */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-1/5 flex flex-col items-center">
          <h2 className="text-4xl mb-4">Last Car</h2>
          <div
            className={`w-40 py-3 mb-4 rounded-xl text-center font-semibold ${
              lastCar === "Commercial"
                ? "bg-yellow-300 text-black"
                : "bg-zinc-700 text-gray-400"
            }`}
          >
            Commercial
          </div>
          <div
            className={`w-40 py-3 rounded-xl text-center font-semibold ${
              lastCar === "Normal"
                ? "bg-yellow-300 text-black"
                : "bg-zinc-700 text-gray-400"
            }`}
          >
            Normal
          </div>
        </div>

        {/* Power chart */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-3/5">
          <h2 className="text-4xl mb-4">Power Consumption Over Time</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={historicalPower}>
              <XAxis
                dataKey="time"
                stroke="#aaa"
                type="number"
                domain={[start, now]}
                tickFormatter={formatTime}
                ticks={ticks}
                scale="time"
                tick={{ fontFamily: "monospace", fontSize: 12 }}
              />
              <YAxis
                stroke="#aaa"
                domain={[0, "auto"]}
                tickFormatter={(v) => `${v}mAh`}
              />
              <Tooltip labelFormatter={formatTime} />
              <Line
                type="monotone"
                dataKey="power"
                stroke="#f6ad55"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Battery */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-1/5 flex flex-col items-center justify-center">
          <h2 className="text-2xl mb-4">Current Power Consumption</h2>
          <div className="w-32 h-32">
            <CircularProgressbar
              value={power}
              maxValue={100}
              text={`${power}mAh`}
              styles={buildStyles({
                pathColor: "red",
                textColor: "white",
                trailColor: "#555",
              })}
            />
          </div>
        </div>
      </div>

      {/* Car chart */}
      <div className="flex flex-col md:flex-row gap-6">
        <div className="bg-zinc-800 rounded-2xl p-6 flex-1 text-base space-y-20">
          <div>Total Normal: {String(totalNormal).padStart(3, "0")}</div>
          <div>Total Commercial: {String(totalCommercial).padStart(3, "0")}</div>
          <div>Total Vehicles: {String(totalVehicles).padStart(3, "0")}</div>
        </div>

        <div className="bg-zinc-800 rounded-2xl p-6 flex-[5]">
          <h2 className="text-4xl mb-4">Cars Passed Over Time</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={historicalCars}>
              <XAxis
                dataKey="time"
                stroke="#aaa"
                type="number"
                domain={[start, now]}
                tickFormatter={formatTime}
                ticks={ticks}
                scale="time"
                tick={{ fontFamily: "monospace", fontSize: 12 }}
              />
              <YAxis stroke="#aaa" domain={[0, "auto"]} />
              <Tooltip labelFormatter={formatTime} />
              <Legend />
              <Line
                type="monotone"
                dataKey="normal"
                stroke="#f6ad55"
                name="Normal"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="commercial"
                stroke="#a78bfa"
                name="Commercial"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default TrafficMonitorUI;

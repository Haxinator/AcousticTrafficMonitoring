
import React, { useState } from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const TrafficMonitorUI = () => {
  const [lastCar, setLastCar] = useState('Normal'); // Simulate the type of the last car {Normal, Commercial, None}
  const [power, setPower] = useState(40); // Simulate the Power Consumption

  // Simulate the number of vehicles
  const totalCommercial = 12;
  const totalNormal = 25;
  const totalVehicles = totalCommercial + totalNormal;

  return (
    <div className="min-h-screen bg-black text-white p-6 rounded-3xl flex flex-col gap-6 w-full max-w-5xl mx-auto">
      <div className="text-2xl font-bold text-center">Ultra-low-power acoustic trafficMonitoring</div>

        <div className="flex flex-col md:flex-row gap-6 justify-between">

        {/* Car Type Indicator */}
    <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-1/3 flex flex-col items-center">
        <h2 className="text-lg mb-4">Last Car</h2>

        {/* Commercial State */}
    <div className={`w-40 py-3 mb-4 rounded-xl text-center font-semibold 
        ${lastCar === 'Commercial' ? 'bg-yellow-300 text-black' : 'bg-zinc-700 text-gray-400'}`}>
        Commercial
    </div>

        {/* Normal State */}
    <div className={`w-40 py-3 rounded-xl text-center font-semibold 
        ${lastCar === 'Normal' ? 'bg-yellow-300 text-black' : 'bg-zinc-700 text-gray-400'}`}>
        Normal
    </div>
    </div>


        {/* Battery Ring Indicator */}
        <div className="bg-zinc-800 rounded-2xl p-6 w-full md:w-2/3 flex flex-col items-center justify-center">
          <h2 className="text-lg mb-4">Current Power Consumption</h2>
          <div className="w-40 h-40">
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

      {/* Vehicle number indicators */}
      <div className="bg-zinc-800 rounded-2xl p-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-center text-lg">
        <div>Total Commercial: {String(totalCommercial).padStart(3, '0')}</div>
        <div>Total Normal: {String(totalNormal).padStart(3, '0')}</div>
        <div>Total Vehicles: {String(totalVehicles).padStart(3, '0')}</div>
      </div>
    </div>
  );
};

export default TrafficMonitorUI;

import { useState, useCallback } from "react";
import { CityGrid } from "./components/CityGrid";
import { HeightFunction } from "./components/HeightFunction";
import { SimulationControls } from "./components/SimulationControls";
import { useReceiveSimulationData } from "./hooks/useReceiveSimulationData";
import styles from "./App.module.css";

function App() {
  const [clickedLines, setClickedLines] = useState<Set<string>>(new Set());
  const [isStart, setIsStart] = useState(false);
  const [isClear, setIsClear] = useState(true);
  const [simulationMode, setSimulationMode] = useState("Paralelo");
  const [size, setSize] = useState(25);
  const [hasStarted, setHasStarted] = useState(false);
  const [showHeightFunction, setShowHeightFunction] = useState(false);
  const [step, setStep] = useState(5);
  const [velocity, setVelocity] = useState(0.5);
  const [particles, setParticles] = useState(100);
  const [densityInit, setDensityInit] = useState(-1);

  const { DiccionarioFuncionAltura } = useReceiveSimulationData(
    isStart,
    velocity
  );

  const generateDataPlot3D = useCallback(() => {
    if (!DiccionarioFuncionAltura) return Array(size).fill(Array(size).fill(0));

    const data = Array(size)
      .fill(null)
      .map(() => Array(size).fill(0));

    Object.entries(DiccionarioFuncionAltura).forEach(([key, value]) => {
      const [x, y] = key.split(",").map(Number);
      if (x < size && y < size) {
        data[y][x] = value || 0;
      }
    });

    return data;
  }, [DiccionarioFuncionAltura, size]);

  const handleStartToggle = () => {
    setIsStart((prev) => !prev);
    if (!hasStarted) {
      setHasStarted(true);
    }
  };

  return (
    <div className={styles.app}>
      <h1 className={styles.title}>City Tasep</h1>
      <div className={styles.gridContainer}>
        {showHeightFunction ? (
          <div className={styles.heightFunctionContainer}>
            <HeightFunction
              data={generateDataPlot3D()}
              size={size}
              clickedLines={clickedLines}
              onClickedLinesUpdate={setClickedLines}
              isStart={isStart}
              isClear={isClear}
              setIsClear={setIsClear}
              simulationMode={simulationMode}
              step={step}
              cantidad_inicial={particles}
              velocidad={velocity}
              densityInit={densityInit}
            />
          </div>
        ) : (
          <CityGrid
            size={size}
            clickedLines={clickedLines}
            onClickedLinesUpdate={setClickedLines}
            isStart={isStart}
            isClear={isClear}
            setIsClear={setIsClear}
            simulationMode={simulationMode}
            step={step}
            cantidad_inicial={particles}
            velocidad={velocity}
            densityInit={densityInit}
          />
        )}
        <SimulationControls
          isStart={isStart}
          onStartToggle={handleStartToggle}
          onModeChange={setSimulationMode}
          size={size}
          onSizeChange={(e) => setSize(Number(e.target.value))}
          hasStarted={hasStarted}
          showHeightFunction={showHeightFunction}
          onToggleView={() => setShowHeightFunction((prev) => !prev)}
          step={step}
          onStepChange={(e) => setStep(Number(e.target.value))}
          velocity={velocity}
          onVelocityChange={(e) => setVelocity(Number(e.target.value))}
          particles={particles}
          onParticlesChange={(e) => setParticles(Number(e.target.value))}
          densityInit={densityInit}
          onDensityInitChange={(e) => setDensityInit(Number(e.target.value))}
        />
      </div>
    </div>
  );
}

export default App;

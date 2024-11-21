import { useState } from "react";
import { Grid } from "../grid";
import { CreateCar } from "../car";
import { useSendSimulationData } from "../../hooks/useSendSimulationData";
import { useReceiveSimulationData } from "../../hooks/useReceiveSimulationData";
import type { CityGridProps } from "./types";
import styles from "./styles.module.css";

export default function CityGrid({
  clickedLines,
  onClickedLinesUpdate,
  isStart,
  isClear,
  setIsClear,
  simulationMode,
  size,
  step,
  cantidad_inicial,
  velocidad,
  densityInit,
}: CityGridProps) {
  const [hasStarted, setHasStarted] = useState(false);
  const { cars } = useReceiveSimulationData(isStart, velocidad);

  useSendSimulationData({
    clickedLines,
    size,
    isStart,
    isClear,
    setIsClear,
    mode: simulationMode,
    step,
    cantidad_inicial,
    velocidad,
    densityInit,
  });

  // Update hasStarted when simulation starts for the first time
  if (isStart && !hasStarted) {
    setHasStarted(true);
  }

  return (
    <div className={styles.carContainer}>
      <Grid
        size={size}
        clickedLines={clickedLines}
        onClickedLinesUpdate={onClickedLinesUpdate}
        hasStarted={hasStarted}
      />
      {cars.map((car) => (
        <CreateCar
          key={car.id}
          position={{ col: car.col, row: car.row }}
          color={car.color}
          id={car.id}
          gridSize={size}
          velocidad={velocidad}
        />
      ))}
    </div>
  );
}

import { useEffect } from "react";
import { calculateRowAndCols } from "../utils/gridCalculations";

interface UseSendSimulationDataProps {
  clickedLines: Set<string>;
  size: number;
  isStart: boolean;
  isClear: boolean;
  setIsClear: (value: boolean) => void;
  mode: string;
  step: number;
  cantidad_inicial: number;
  velocidad: number;
  densityInit: number;
}

export function useSendSimulationData({
  clickedLines,
  size,
  isStart,
  isClear,
  setIsClear,
  mode,
  step,
  cantidad_inicial,
  velocidad,
  densityInit,
}: UseSendSimulationDataProps) {
  useEffect(() => {
    const sendData = async () => {
      try {
        const { calles } = calculateRowAndCols(clickedLines);
        await fetch("http://localhost:5000/update_data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            calles: Array.from(calles),
            size,
            isStart,
            isClear: isClear,
            mode: mode.toLowerCase(),
            step,
            cantidad_inicial,
            velocidad,
            densityInit,
          }),
        });
        if (isClear) {
          setIsClear(false);
        }
      } catch (error) {
        console.error("Error sending data to backend:", error);
      }
    };

    sendData();
  }, [isStart, mode, velocidad]);
}

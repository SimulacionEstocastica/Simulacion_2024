import { useState, useEffect } from "react";
import type { Car } from "../components/CityGrid/types";

export function useReceiveSimulationData(isStart: boolean, velocidad: number) {
  const [cars, setCars] = useState<Car[]>([]);
  const [DiccionarioFuncionAltura, setDiccionarioFuncionAltura] = useState<any>(
    []
  );

  useEffect(() => {
    if (!isStart) return () => {}; // Don't clear cars when stopping

    const fetchState = async () => {
      try {
        const response = await fetch("http://localhost:5000/state");
        const data = await response.json();
        setDiccionarioFuncionAltura(data.diccionario_funcion_altura);
        // Handle new cars
        Object.values(data.particulas_agregadas).forEach((particula: any) => {
          const { col, color, id, row } = particula;
          setCars((prev) => {
            if (!prev.some((car) => car.id === id)) {
              return [...prev, { row: col, col: row, color, id }];
            }
            return prev;
          });
        });

        // Update existing cars
        Object.values(data.particulas).forEach((particula: any) => {
          const { id, new_col, new_row } = particula;
          setCars(
            (prev) =>
              prev
                .map((car) =>
                  car.id === id ? { ...car, row: new_col, col: new_row } : car
                )
                .filter((car) => car.row <= 50 && car.col <= 50) // Remove cars that go out of bounds
          );
        });
      } catch (error) {
        console.error("Error fetching simulation state:", error);
      }
    };

    const intervalId = setInterval(fetchState, (1.3 - velocidad) * 1000);
    return () => clearInterval(intervalId);
  }, [isStart]);

  return { cars, DiccionarioFuncionAltura };
}

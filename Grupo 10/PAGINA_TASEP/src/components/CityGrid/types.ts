export interface Car {
  row: number;
  col: number;
  color: string;
  id: string;
}

export interface CityGridProps {
  clickedLines: Set<string>;
  onClickedLinesUpdate: (clickedLines: Set<string>) => void;
  isStart: boolean;
  isClear: boolean;
  setIsClear: (isClear: boolean) => void;
  simulationMode: string;
  size: number;
  step: number;
  cantidad_inicial: number;
  velocidad: number;
  densityInit: number;
}

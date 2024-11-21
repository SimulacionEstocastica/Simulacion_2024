import { CityGridProps } from "../CityGrid/types";

export interface SceneElementProps {
  position?: [number, number, number];
  scale?: number | [number, number, number];
  rotation?: [number, number, number];
}

export interface HeightFunctionProps extends CityGridProps {
  data: number[][];
  size: number;
}

export interface GridOverlayProps extends CityGridProps {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: number;
}

export interface SceneConfig {
  cameraPosition: [number, number, number];
  axesHelperScale: number;
  axesHelperPosition: [number, number, number];
  surfaceScale: number;
  dataPointsScale: number;
  cityGridScale: number;
  heightScale: number;
  cityGridPosition: [number, number, number];
  cityGridRotation: [number, number, number];
  dataPointsPosition: [number, number, number];
  surfacePosition: [number, number, number];
}

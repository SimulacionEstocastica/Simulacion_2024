import { useMemo } from "react";
import { SceneConfig } from "../types";

export const useSceneConfig = (size: number): SceneConfig => {
  return useMemo(
    () => ({
      cameraPosition: [-1, 0.0421053 * (size - 24) + 1, -1] as [
        number,
        number,
        number
      ],
      axesHelperScale: 0.042 * (size - 24) + 1,
      axesHelperPosition: [-0.5, 0, -0.5] as [number, number, number],
      surfaceScale: 0.0416 * (size - 25) + 1,
      dataPointsScale: 0.0416 * (size - 25) + 1,
      cityGridScale: 0.11,
      heightScale: -0.00266 * (size - 24) + 0.6,
      cityGridPosition: [0.0205 * (size - 24), 0, 2.3245] as [
        number,
        number,
        number
      ],
      cityGridRotation: [-Math.PI / 2, 0, 0] as [number, number, number],
      dataPointsPosition: [0.0208 * (size - 24), 0, 0.0208 * (size - 24)] as [
        number,
        number,
        number
      ],
      surfacePosition: [0.0208 * (size - 24), 0, 0.0208 * (size - 24)] as [
        number,
        number,
        number
      ],
    }),
    [size]
  );
};

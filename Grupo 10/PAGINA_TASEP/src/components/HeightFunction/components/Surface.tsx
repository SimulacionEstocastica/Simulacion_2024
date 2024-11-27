import React, { useMemo } from "react";
import * as THREE from "three";
import { SceneElementProps } from "../types";

interface SurfaceProps extends SceneElementProps {
  data: number[][];
  size: number;
  heightScale: number;
}

export const Surface: React.FC<SurfaceProps> = ({
  data,
  size,
  heightScale,
  position = [0, 0, 0],
  scale = 1,
}) => {
  const { points, lines } = useMemo(() => {
    const pointsArray = [];
    const linesArray = [];
    const validValues = data
      .flat()
      .filter((val) => val !== undefined && val !== null && val !== 0);
    const maxValue = Math.max(...validValues);
    const minValue = Math.min(...validValues);
    const range = maxValue - minValue || 1;

    const getPosition = (x: number, y: number) => {
      const value = data[y][x];
      if (value !== undefined && value !== null && value !== 0) {
        const normalizedValue = ((value - minValue) / range) * heightScale;
        return new THREE.Vector3(
          x / (size - 1) - 0.5,
          normalizedValue,
          y / (size - 1) - 0.5
        );
      }
      return null;
    };

    // Generate points
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pos = getPosition(x, y);
        if (pos) pointsArray.push(pos);
      }
    }

    // Generate connecting lines
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size - 1; x++) {
        const pos1 = getPosition(x, y);
        const pos2 = getPosition(x + 1, y);
        if (pos1 && pos2) {
          linesArray.push(pos1, pos2);
        }
      }
    }

    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size - 1; y++) {
        const pos1 = getPosition(x, y);
        const pos2 = getPosition(x, y + 1);
        if (pos1 && pos2) {
          linesArray.push(pos1, pos2);
        }
      }
    }

    return { points: pointsArray, lines: linesArray };
  }, [data, size, heightScale]);

  const pointsGeometry = useMemo(
    () => new THREE.BufferGeometry().setFromPoints(points),
    [points]
  );

  const linesGeometry = useMemo(
    () => new THREE.BufferGeometry().setFromPoints(lines),
    [lines]
  );

  return (
    <group position={position} scale={scale}>
      <points geometry={pointsGeometry}>
        <pointsMaterial color="#4caf50" size={0.025} sizeAttenuation={true} />
      </points>
      <lineSegments geometry={linesGeometry}>
        <lineBasicMaterial color="#4caf50" opacity={0.3} transparent />
      </lineSegments>
    </group>
  );
};

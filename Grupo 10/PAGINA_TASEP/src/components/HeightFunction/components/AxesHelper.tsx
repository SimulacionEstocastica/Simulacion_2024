import React, { useEffect } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { SceneElementProps } from "../types";

export const AxesHelper: React.FC<SceneElementProps> = ({
  scale = 1,
  position = [0, 0, 0],
}) => {
  const { scene } = useThree();

  useEffect(() => {
    const axesHelper = new THREE.AxesHelper(
      typeof scale === "number" ? scale : 1
    );
    axesHelper.position.set(...position);
    scene.add(axesHelper);

    return () => {
      scene.remove(axesHelper);
    };
  }, [scene, scale, position]);

  return null;
};

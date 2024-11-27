import React, { useRef, useEffect } from "react";
import { useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

interface CameraControllerProps {
  position: [number, number, number];
}

export const CameraController: React.FC<CameraControllerProps> = ({
  position,
}) => {
  const { camera, gl } = useThree();
  const controlsRef = useRef<any>();

  useEffect(() => {
    camera.position.set(...position);
    camera.lookAt(0, 0, 0);
  }, [camera, position]);

  return (
    <OrbitControls
      ref={controlsRef}
      args={[camera, gl.domElement]}
      enableRotate={true}
      enablePan={true}
      enableZoom={true} // Habilitado el zoom
      minDistance={2} // Distancia mínima de zoom
      maxDistance={20} // Distancia máxima de zoom
      minPolarAngle={0} // Ángulo mínimo vertical
      maxPolarAngle={Math.PI / 2} // Ángulo máximo vertical (90 grados)
      zoomSpeed={1} // Velocidad de zoom
    />
  );
};

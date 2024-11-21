import React from "react";
import { Canvas } from "@react-three/fiber";
import { Surface } from "./components/Surface";
import { AxesHelper } from "./components/AxesHelper";
import { CameraController } from "./components/CameraController";
import { GridOverlay } from "./components/GridOverlay";
import { useSceneConfig } from "./hooks/useSceneConfig";
import type { HeightFunctionProps } from "./types";
import styles from "./styles.module.css";

const HeightFunction: React.FC<HeightFunctionProps> = ({
  data,
  size,
  clickedLines,
  onClickedLinesUpdate,
  isStart,
  isClear,
  setIsClear,
  simulationMode,
  step,
  cantidad_inicial,
  velocidad,
  densityInit,
}) => {
  const sceneConfig = useSceneConfig(size);

  return (
    <div className={styles.container}>
      <Canvas
        camera={{
          position: sceneConfig.cameraPosition,
          fov: 50,
          near: 0.1,
          far: 1000,
        }}
        style={{ width: "100%", height: "100%" }}
      >
        <CameraController position={sceneConfig.cameraPosition} />
        <ambientLight intensity={0.5} />
        <pointLight position={[5, 5, 5]} />

        <Surface
          data={data}
          size={size}
          scale={sceneConfig.surfaceScale}
          heightScale={sceneConfig.heightScale}
          position={sceneConfig.surfacePosition}
        />

        <AxesHelper
          scale={sceneConfig.axesHelperScale}
          position={sceneConfig.axesHelperPosition}
        />

        <GridOverlay
          position={sceneConfig.cityGridPosition}
          rotation={sceneConfig.cityGridRotation}
          scale={sceneConfig.cityGridScale}
          size={size}
          clickedLines={clickedLines}
          onClickedLinesUpdate={onClickedLinesUpdate}
          isStart={isStart}
          isClear={isClear}
          setIsClear={setIsClear}
          simulationMode={simulationMode}
          step={step}
          cantidad_inicial={cantidad_inicial}
          velocidad={velocidad}
          densityInit={densityInit}
        />
      </Canvas>
    </div>
  );
};

export default HeightFunction;

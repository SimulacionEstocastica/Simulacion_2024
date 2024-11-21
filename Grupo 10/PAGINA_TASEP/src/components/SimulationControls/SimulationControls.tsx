import { StartButton, ViewToggleButton } from "../buttons";
import { TripleSwitch } from "../switch";
import { Slider } from "../slider";
import type { SimulationControlsProps } from "./types";
import styles from "./styles.module.css";

export default function SimulationControls({
  isStart,
  onStartToggle,
  onModeChange,
  size,
  onSizeChange,
  hasStarted,
  showHeightFunction,
  onToggleView,
  step,
  onStepChange,
  velocity,
  onVelocityChange,
  particles,
  onParticlesChange,
  densityInit,
  onDensityInitChange,
}: SimulationControlsProps) {
  return (
    <div className={styles.controls}>
      <StartButton onClick={onStartToggle} isStart={isStart} />

      <div className={styles.controlsRow}>
        <ViewToggleButton
          onClick={onToggleView}
          showHeightFunction={showHeightFunction}
        />
        <TripleSwitch onChange={onModeChange} />
      </div>
      <div className={styles.slidersContainer}>
        <Slider
          title="Velocity"
          value={velocity}
          onChange={onVelocityChange}
          min={0.1}
          max={1}
          step={0.01}
        />
        {!hasStarted && (
          <>
            <Slider
              title="Step"
              value={step}
              onChange={onStepChange}
              min={1}
              max={50}
              step={1}
            />
            <Slider
              title="Particles"
              value={particles}
              onChange={onParticlesChange}
              min={0}
              max={400}
              step={10}
            />
            <Slider
              title="Grid Size"
              value={size}
              onChange={onSizeChange}
              min={1}
              max={50}
              step={1}
            />
            <Slider
              title="Density"
              value={densityInit}
              onChange={onDensityInitChange}
              min={-1}
              max={size + 1}
              step={1}
            />
          </>
        )}
      </div>
    </div>
  );
}

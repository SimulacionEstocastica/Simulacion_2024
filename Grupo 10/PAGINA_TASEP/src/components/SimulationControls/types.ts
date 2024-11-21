export interface SimulationControlsProps {
  isStart: boolean;
  onStartToggle: () => void;
  onModeChange: (mode: string) => void;
  size: number;
  onSizeChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  hasStarted: boolean;
  showHeightFunction: boolean;
  onToggleView: () => void;
  step: number;
  onStepChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  velocity: number;
  onVelocityChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  particles: number;
  onParticlesChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  densityInit: number;
  onDensityInitChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

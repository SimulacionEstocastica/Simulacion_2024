export interface ButtonProps {
  onClick: () => void;
  isActive?: boolean;
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "tertiary";
}

export interface StartButtonProps {
  onClick: () => void;
  isStart: boolean;
}

export interface ViewToggleButtonProps {
  onClick: () => void;
  showHeightFunction: boolean;
}

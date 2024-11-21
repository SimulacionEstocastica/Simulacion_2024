import Button from "./Button";
import type { ViewToggleButtonProps } from "./types";

const ViewToggleButton = ({
  onClick,
  showHeightFunction,
}: ViewToggleButtonProps) => {
  return (
    <Button onClick={onClick} isActive={showHeightFunction} variant="tertiary">
      {showHeightFunction ? "Show Grid" : "Height Function"}
    </Button>
  );
};

export default ViewToggleButton;

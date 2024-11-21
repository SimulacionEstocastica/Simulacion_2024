import React from "react";
import Button from "../components/Button";

interface ControlPanelProps {
  showIntersections: boolean;
  showRowAndCols: boolean;
  onToggleIntersections: () => void;
  onToggleRowAndCols: () => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  showIntersections,
  showRowAndCols,
  onToggleIntersections,
  onToggleRowAndCols,
}) => {
  return (
    <div>
      <Button onClick={onToggleIntersections}>
        {showIntersections ? "Hide Intersections" : "Show Intersections"}
      </Button>
      <Button onClick={onToggleRowAndCols}>
        {showRowAndCols ? "Hide Columns and Rows" : "Show Columns and Rows"}
      </Button>
    </div>
  );
};

export default ControlPanel;

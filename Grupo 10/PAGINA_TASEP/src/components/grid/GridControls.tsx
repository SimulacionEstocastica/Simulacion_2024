import { useEffect, useCallback } from "react";
import type { GridControlsProps } from "./types";

export default function GridControls({
  onToggleHoverOrientation,
  isGridSelected,
  hasStarted,
}: GridControlsProps) {
  const handleKeyPress = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === "r" && !hasStarted) {
        onToggleHoverOrientation();
      }
    },
    [onToggleHoverOrientation, hasStarted]
  );

  const handleWheel = useCallback(
    (event: WheelEvent) => {
      if (!hasStarted) {
        event.preventDefault();
        onToggleHoverOrientation();
      }
    },
    [onToggleHoverOrientation, hasStarted]
  );

  useEffect(() => {
    if (isGridSelected && !hasStarted) {
      window.addEventListener("keydown", handleKeyPress);
      window.addEventListener("wheel", handleWheel, { passive: false });
    }

    return () => {
      window.removeEventListener("keydown", handleKeyPress);
      window.removeEventListener("wheel", handleWheel);
    };
  }, [isGridSelected, handleKeyPress, handleWheel, hasStarted]);

  return null;
}

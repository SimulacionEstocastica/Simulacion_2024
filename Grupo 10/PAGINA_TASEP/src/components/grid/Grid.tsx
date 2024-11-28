import React, { useState, useCallback } from "react";
import GridCell from "./GridCell";
import styles from "./styles.module.css";
import GridControls from "./GridControls";
import type { GridProps } from "./types";

const Grid: React.FC<GridProps> = ({
  size,
  clickedLines,
  onClickedLinesUpdate,
  hasStarted,
}) => {
  const [hoverCell, setHoverCell] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const [isVerticalHover, setIsVerticalHover] = useState(false);
  const [isGridSelected, setIsGridSelected] = useState(false);

  const handleCellClick = useCallback(
    (row: number, col: number) => {
      if (hoverCell && !hasStarted) {
        const newClickedLines = new Set(clickedLines);
        const lineKey = isVerticalHover ? `col-${col}` : `row-${row}`;
        if (newClickedLines.has(lineKey)) {
          newClickedLines.delete(lineKey);
        } else {
          newClickedLines.add(lineKey);
        }
        onClickedLinesUpdate(newClickedLines);
      }
    },
    [hoverCell, isVerticalHover, onClickedLinesUpdate, hasStarted, clickedLines]
  );

  const toggleHoverOrientation = useCallback(() => {
    if (!hasStarted) {
      setIsVerticalHover((prev) => !prev);
    }
  }, [hasStarted]);

  const handleGridMouseEnter = () => setIsGridSelected(true);
  const handleGridMouseLeave = () => setIsGridSelected(false);

  return (
    <div
      onMouseEnter={handleGridMouseEnter}
      onMouseLeave={handleGridMouseLeave}
      className={styles.gridWrapper}
    >
      <GridControls
        onToggleHoverOrientation={toggleHoverOrientation}
        isGridSelected={isGridSelected}
        hasStarted={hasStarted}
      />
      <div className={styles.grid}>
        {Array.from({ length: size }, (_, index) => {
          const row = index;
          return (
            <div key={row} className={styles.row}>
              {Array.from({ length: size }, (_, col) => (
                <GridCell
                  key={`${row}-${col}`}
                  row={row}
                  col={col}
                  isClicked={
                    clickedLines.has(`row-${row}`) ||
                    clickedLines.has(`col-${col}`)
                  }
                  isHovered={
                    !hasStarted &&
                    (isVerticalHover
                      ? col === hoverCell?.col
                      : row === hoverCell?.row)
                  }
                  isVerticalHover={isVerticalHover}
                  onClick={handleCellClick}
                  onMouseEnter={(cell) =>
                    !hasStarted && setHoverCell({ ...cell, row: row })
                  }
                  onMouseLeave={() => !hasStarted && setHoverCell(null)}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Grid;

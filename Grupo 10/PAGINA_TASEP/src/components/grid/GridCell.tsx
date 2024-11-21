import React from "react";
import styles from "./styles.module.css";
import type { GridCellProps } from "./types";

const GridCell: React.FC<GridCellProps> = ({
  row,
  col,
  isClicked,
  isHovered,
  isVerticalHover,
  onClick,
  onMouseEnter,
  onMouseLeave,
}) => {
  return (
    <div
      className={`${styles.cell} 
        ${isClicked ? styles.clicked : ""}
        ${
          isHovered
            ? isVerticalHover
              ? styles.hoverVertical
              : styles.hoverHorizontal
            : ""
        }`}
      onClick={() => onClick(row, col)}
      onMouseEnter={() => onMouseEnter({ row, col })}
      onMouseLeave={onMouseLeave}
    />
  );
};

export default React.memo(GridCell);

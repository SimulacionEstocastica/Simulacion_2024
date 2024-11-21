export interface GridProps {
  size: number;
  clickedLines: Set<string>;
  onClickedLinesUpdate: (clickedLines: Set<string>) => void;
  hasStarted: boolean;
}

export interface GridCellProps {
  row: number;
  col: number;
  isClicked: boolean;
  isHovered: boolean;
  isVerticalHover: boolean;
  onClick: (row: number, col: number) => void;
  onMouseEnter: (cell: { row: number; col: number }) => void;
  onMouseLeave: () => void;
}

export interface GridControlsProps {
  onToggleHoverOrientation: () => void;
  isGridSelected: boolean;
  hasStarted: boolean;
}

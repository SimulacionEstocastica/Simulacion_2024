import React from "react";

interface DataDisplayProps {
  showIntersections: boolean;
  showRowAndCols: boolean;
  intersections: string[];
  rowAndCols: string[];
}

const DataDisplay: React.FC<DataDisplayProps> = ({
  showIntersections,
  showRowAndCols,
  intersections,
  rowAndCols,
}) => {
  return (
    <>
      {showIntersections && intersections.length > 0 && (
        <div>
          <h2>Intersections:</h2>
          <p>{intersections.join(", ")}</p>
        </div>
      )}
      {showRowAndCols && rowAndCols.length > 0 && (
        <div>
          <h2>Columns and Rows:</h2>
          <p>{rowAndCols.join(", ")}</p>
        </div>
      )}
    </>
  );
};

export default DataDisplay;

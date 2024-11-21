const calculateIntersections = (clickedLines: Set<string>): string[] => {
  const result: string[] = [];
  const rows = new Set();
  const cols = new Set();

  clickedLines.forEach((line) => {
    const [type, index] = line.split("-");
    if (type === "row") {
      rows.add(index);
    } else {
      cols.add(index);
    }
  });

  rows.forEach((row) => {
    cols.forEach((col) => {
      result.push(`(${col}, ${row})`);
    });
  });

  return result;
};

export default calculateIntersections;

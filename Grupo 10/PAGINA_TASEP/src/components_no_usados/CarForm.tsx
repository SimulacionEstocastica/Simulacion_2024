import React, { useState } from "react";
import styles from "./CarForm.module.css";

interface CarFormProps {
  onAddCar: (row: number, col: number, color: string, id: string) => void;
  gridSize: number;
  getNextCarId: () => string;
}

const CarForm: React.FC<CarFormProps> = ({
  onAddCar,
  gridSize,
  getNextCarId,
}) => {
  const [row, setRow] = useState("0");
  const [col, setCol] = useState("0");
  const [color, setColor] = useState("blue");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newId = getNextCarId();
    onAddCar(Number(row), Number(col), color, newId);
    console.log(
      `Car created - ID: ${newId}, Row: ${row}, Column: ${col}, Color: ${color}`
    );
    setRow("0");
    setCol("0");
    setColor("blue");
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <div className={styles.formGroup}>
        <label htmlFor="col">Column</label>
        <input
          type="number"
          id="col"
          value={col}
          onChange={(e) => setCol(e.target.value)}
          required
          min="0"
          max={gridSize - 1}
        />
      </div>
      <div className={styles.formGroup}>
        <label htmlFor="row">Row</label>
        <input
          type="number"
          id="row"
          value={row}
          onChange={(e) => setRow(e.target.value)}
          required
          min="0"
          max={gridSize - 1}
        />
      </div>
      <div className={styles.formGroup}>
        <label htmlFor="color">Color</label>
        <select
          id="color"
          value={color}
          onChange={(e) => setColor(e.target.value)}
        >
          <option value="blue">Blue</option>
          <option value="green">Green</option>
          <option value="red">Red</option>
          <option value="black">Black</option>
          <option value="yellow">Yellow</option>
        </select>
      </div>
      <button type="submit" className={styles.button}>
        Add Car
      </button>
    </form>
  );
};

export default CarForm;

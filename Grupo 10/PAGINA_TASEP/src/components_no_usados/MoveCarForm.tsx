import React, { useState } from "react";
import styles from "./MoveCarForm.module.css";

interface MoveCarFormProps {
  onMoveCar: (id: string, newRow: number, newCol: number) => void;
  gridSize: number;
  cars: Array<{ id: string; row: number; col: number; color: string }>;
}

const MoveCarForm: React.FC<MoveCarFormProps> = ({
  onMoveCar,
  gridSize,
  cars,
}) => {
  const [selectedCarId, setSelectedCarId] = useState("");
  const [newRow, setNewRow] = useState("0");
  const [newCol, setNewCol] = useState("0");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedCarId) {
      onMoveCar(selectedCarId, Number(newRow), Number(newCol));
      console.log(
        `Car moved - ID: ${selectedCarId}, New Row: ${newRow}, New Column: ${newCol}`
      );
      setSelectedCarId("");
      setNewRow("0");
      setNewCol("0");
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <div className={styles.formGroup}>
        <label htmlFor="newCol">New Column</label>
        <input
          type="number"
          id="newCol"
          value={newCol}
          onChange={(e) => setNewCol(e.target.value)}
          required
          min="0"
          max={gridSize - 1}
        />
      </div>
      <div className={styles.formGroup}>
        <label htmlFor="newRow">New Row</label>
        <input
          type="number"
          id="newRow"
          value={newRow}
          onChange={(e) => setNewRow(e.target.value)}
          required
          min="0"
          max={gridSize - 1}
        />
      </div>
      <div className={styles.formGroup}>
        <label htmlFor="carId">Select Car</label>
        <select
          id="carId"
          value={selectedCarId}
          onChange={(e) => setSelectedCarId(e.target.value)}
          required
        >
          <option value="">Select a car</option>
          {cars.map((car) => (
            <option key={car.id} value={car.id}>
              {`${car.id} (${car.color})`}
            </option>
          ))}
        </select>
      </div>
      <button type="submit" className={styles.button}>
        Move Car
      </button>
    </form>
  );
};

export default MoveCarForm;

import React from "react";
import styles from "./styles.module.css";
import type { SliderProps } from "./types";

const Slider: React.FC<SliderProps> = ({
  title,
  value,
  onChange,
  min,
  max,
  step = 1,
}) => {
  return (
    <div className={styles.sliderContainer}>
      <span className={styles.sliderLabel}>{title}:</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        className={styles.sliderInput}
      />
      <span className={styles.sliderValue}>{value}</span>
    </div>
  );
};

export default Slider;

"use client";

import { useEffect, useRef, useState } from "react";
import { carImages } from "./cars";
import type { CreateCarProps } from "./types";
import styles from "./styles.module.css";

export default function CreateCar({
  position,
  color,
  id,
  gridSize,
  velocidad,
}: CreateCarProps) {
  // Don't render if position is negative
  if (position.row < 0 || position.col < 0) {
    return null;
  }

  const carRef = useRef<HTMLDivElement>(null);
  const [prevPosition, setPrevPosition] = useState(position);
  const [isVisible, setIsVisible] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);

  // Determine initial rotation based on movement direction
  const [rotation, setRotation] = useState(() => {
    // If initial position is at col 0, it's moving horizontally (0 degrees)
    // If initial position is at row 0, it's moving vertically (-90 degrees)
    return color === "red" ? 90 : 0;
  });

  useEffect(() => {
    // Fade in when component mounts
    setTimeout(() => setIsVisible(true), 50);
  }, []);

  useEffect(() => {
    if (carRef.current) {
      // Check if car is reaching the grid boundary
      if (position.row >= gridSize || position.col >= gridSize) {
        setIsLeaving(true);
        // Remove car after fade out animation completes
        setTimeout(() => {
          if (carRef.current) {
            carRef.current.style.display = "none";
          }
        }, (1.2 - velocidad) * 1000);
        return;
      }

      const transitionDuration = `${1.3 - velocidad}s`;
      carRef.current.style.transition = `left ${transitionDuration} ease-in-out, bottom ${transitionDuration} ease-in-out, transform ${transitionDuration} ease-in-out, opacity ${transitionDuration} ease-in-out`;
      carRef.current.style.left = `${position.col * 15}px`;
      carRef.current.style.bottom = `${(gridSize - 1 - position.row) * 15}px`;

      const dx = position.col - prevPosition.col;
      const dy = position.row - prevPosition.row;
      let newRotation = rotation;

      if (dx === 1 && dy === 0) newRotation = 0;
      else if (dx === -1 && dy === 0) newRotation = -180;
      else if (dx === 0 && dy === 1) newRotation = -270;
      else if (dx === 0 && dy === -1) newRotation = -90;

      let rotationDiff = newRotation - rotation;
      if (rotationDiff > 180) rotationDiff -= 360;
      if (rotationDiff < -180) rotationDiff += 360;

      setRotation(rotation + rotationDiff);
      carRef.current.style.transform = `rotate(${rotation + rotationDiff}deg)`;

      setPrevPosition(position);
    }
  }, [position, prevPosition, rotation, gridSize, velocidad]);

  return (
    <div
      ref={carRef}
      className={`${styles.car} ${isVisible ? styles.visible : ""} ${
        isLeaving ? styles.leaving : ""
      }`}
      id={id}
      style={{ transform: `rotate(${rotation}deg)` }}
    >
      <img
        src={carImages[color as keyof typeof carImages]}
        alt={`${color} Car`}
        className={styles.carImage}
      />
    </div>
  );
}

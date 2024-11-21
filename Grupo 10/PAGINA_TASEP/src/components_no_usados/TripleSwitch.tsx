// import { useState } from "react";
// import type { Position, TripleSwitchProps } from "./types";
// import styles from "./styles.module.css";

// const TripleSwitch = ({ onChange }: TripleSwitchProps) => {
//   const [position, setPosition] = useState<Position>("left");

//   const handleChange = (newPosition: Position) => {
//     setPosition(newPosition);
//     onChange?.(
//       newPosition === "left"
//         ? "Paralelo"
//         : newPosition === "center"
//         ? "Secuencial"
//         : "Continuo"
//     );
//   };

//   return (
//     <div className={styles.switchContainer}>
//       <div className={`${styles.switch} ${styles[position]}`} />
//       <div className={styles.labels}>
//         <input
//           type="radio"
//           id="left"
//           name="switch"
//           className={styles.input}
//           checked={position === "left"}
//           onChange={() => handleChange("left")}
//         />
//         <label
//           htmlFor="left"
//           className={`${styles.label} ${
//             position === "left" ? styles.active : ""
//           }`}
//         >
//           Paralelo
//         </label>

//         <input
//           type="radio"
//           id="center"
//           name="switch"
//           className={styles.input}
//           checked={position === "center"}
//           onChange={() => handleChange("center")}
//         />
//         <label
//           htmlFor="center"
//           className={`${styles.label} ${
//             position === "center" ? styles.active : ""
//           }`}
//         >
//           Secuencial
//         </label>

//         <input
//           type="radio"
//           id="right"
//           name="switch"
//           className={styles.input}
//           checked={position === "right"}
//           onChange={() => handleChange("right")}
//         />
//         <label
//           htmlFor="right"
//           className={`${styles.label} ${
//             position === "right" ? styles.active : ""
//           }`}
//         >
//           Continuo
//         </label>
//       </div>
//     </div>
//   );
// };

// export default TripleSwitch;

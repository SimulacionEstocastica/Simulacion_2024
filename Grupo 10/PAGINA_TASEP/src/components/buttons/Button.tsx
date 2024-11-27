import type { ButtonProps } from "./types";
import styles from "./styles.module.css";

const Button = ({
  onClick,
  isActive,
  children,
  variant = "primary",
}: ButtonProps) => {
  return (
    <button
      className={`${styles.button} ${styles[variant]} ${
        isActive ? styles.active : ""
      }`}
      onClick={onClick}
    >
      {children}
    </button>
  );
};

export default Button;

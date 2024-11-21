import Button from "./Button";
import type { StartButtonProps } from "./types";

const StartButton = ({ onClick, isStart }: StartButtonProps) => {
  return (
    <Button onClick={onClick} isActive={isStart}>
      {isStart ? "Stop" : "Start"}
    </Button>
  );
};

export default StartButton;

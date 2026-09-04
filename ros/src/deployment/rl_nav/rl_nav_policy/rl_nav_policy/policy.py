"""ONNX Runtime wrapper for the recurrent navigation actor."""

import os

import numpy as np

from .core import LSTM_SIZE, OBSERVATION_DIM


class NavigationPolicy:
    """Validate and execute the exported recurrent ONNX policy."""

    def __init__(self, policy_path: str):
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"navigation policy not found: {policy_path}")
        try:
            saved_stderr = os.dup(2)
            try:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    os.dup2(devnull.fileno(), 2)
                    import onnxruntime as ort
            finally:
                os.dup2(saved_stderr, 2)
                os.close(saved_stderr)
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is required: python3 -m pip install onnxruntime"
            ) from error

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            policy_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        input_names = [item.name for item in inputs]
        output_names = [item.name for item in outputs]
        if input_names != ["obs", "h_in", "c_in"]:
            raise RuntimeError(f"unexpected policy inputs: {input_names}")
        if output_names != ["actions", "h_out", "c_out"]:
            raise RuntimeError(f"unexpected policy outputs: {output_names}")
        if inputs[0].shape[-1] != OBSERVATION_DIM:
            raise RuntimeError(
                f"policy observation dimension is {inputs[0].shape[-1]}, "
                f"expected {OBSERVATION_DIM}"
            )
        if inputs[1].shape[-1] != LSTM_SIZE or inputs[2].shape[-1] != LSTM_SIZE:
            raise RuntimeError("policy recurrent state must have size 512")
        if outputs[0].shape[-1] != 3:
            raise RuntimeError("policy must produce three navigation actions")

        self.hidden = np.zeros((1, 1, LSTM_SIZE), dtype=np.float32)
        self.cell = np.zeros((1, 1, LSTM_SIZE), dtype=np.float32)

    def reset(self) -> None:
        """Reset recurrent state."""
        self.hidden.fill(0.0)
        self.cell.fill(0.0)

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run one recurrent policy step."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (1, OBSERVATION_DIM):
            raise ValueError(
                f"expected observation (1, {OBSERVATION_DIM}), "
                f"got {observation.shape}"
            )
        actions, self.hidden, self.cell = self.session.run(
            ("actions", "h_out", "c_out"),
            {
                "obs": observation,
                "h_in": self.hidden,
                "c_in": self.cell,
            },
        )
        action = np.asarray(actions[0], dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise RuntimeError("policy returned an invalid action")
        return action

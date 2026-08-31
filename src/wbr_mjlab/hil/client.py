"""Synchronous, fail-closed USB client exposed as a ``NativeRunner`` policy."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from random import Random
from typing import Protocol

import numpy as np

from .protocol import (
  ACTION_DIM,
  HISTORY_DIM,
  MAX_FRAME_SIZE,
  MAX_PAYLOAD,
  MODEL_SET_ID,
  OBS_DIM,
  POLICY_PERIOD_US,
  REQUIRED_CAPABILITIES,
  REQUIRED_MODE_MASK,
  VERSION,
  ErrorPayload,
  Frame,
  FrameDecoder,
  HelloRequest,
  HelloResponse,
  InferRequest,
  InferResponse,
  MessageType,
  PolicyMode,
  ProtocolError,
  encode_frame,
)


class SerialPort(Protocol):
  def read(self, size: int = 1) -> bytes: ...

  def write(self, data: bytes | memoryview) -> int | None: ...


class HilError(RuntimeError):
  """Base class for host-side HIL failures."""


class HilTimeoutError(HilError, TimeoutError):
  """The board did not provide a valid response before the deadline."""


class HilTransportError(HilError):
  """The serial transport failed."""


class HilProtocolError(HilError):
  """The board response violates the negotiated HIL contract."""


class HilRemoteError(HilError):
  def __init__(self, code: int, detail: int):
    self.code = code
    self.detail = detail
    super().__init__(f"STM32 HIL error code={code} detail={detail}")


@dataclass(frozen=True)
class InferResult:
  action: np.ndarray
  sequence: int
  mode: PolicyMode
  inference_us: int
  round_trip_us: float


@dataclass
class _TimingAccumulator:
  """Exact aggregates plus a bounded uniform reservoir for percentiles."""

  capacity: int = 4096
  seed: int = 0
  count: int = 0
  total: float = 0.0
  minimum: float = float("inf")
  maximum: float = float("-inf")
  _samples: list[float] = field(default_factory=list)
  _random: Random = field(init=False, repr=False)

  def __post_init__(self) -> None:
    if self.capacity <= 0:
      raise ValueError("Timing reservoir capacity must be positive")
    self._random = Random(self.seed)

  def add(self, value: float | int) -> None:
    sample = float(value)
    self.count += 1
    self.total += sample
    self.minimum = min(self.minimum, sample)
    self.maximum = max(self.maximum, sample)
    if len(self._samples) < self.capacity:
      self._samples.append(sample)
      return
    replacement = self._random.randrange(self.count)
    if replacement < self.capacity:
      self._samples[replacement] = sample

  def summary(self) -> dict[str, float | int] | None:
    if not self.count:
      return None
    data = np.asarray(self._samples, dtype=np.float64)
    return {
      "count": self.count,
      "min": self.minimum,
      "mean": self.total / self.count,
      "p50": float(np.percentile(data, 50)),
      "p95": float(np.percentile(data, 95)),
      "max": self.maximum,
    }


class HilClient:
  """One-outstanding-request client with strict response correlation."""

  def __init__(
    self,
    serial_port: SerialPort | None = None,
    *,
    port: str | None = None,
    baudrate: int = 115_200,
    timeout: float = 0.010,
    handshake_timeout: float = 0.500,
    session: int | None = None,
  ) -> None:
    if timeout <= 0 or not np.isfinite(timeout):
      raise ValueError("HIL timeout must be a positive finite number")
    if handshake_timeout <= 0 or not np.isfinite(handshake_timeout):
      raise ValueError("HIL handshake timeout must be a positive finite number")
    self.timeout = float(timeout)
    self.handshake_timeout = float(handshake_timeout)
    self.session = secrets.randbelow(0xFFFFFFFF) + 1 if session is None else int(session)
    if not 0 < self.session <= 0xFFFFFFFF:
      raise ValueError("HIL session must be a non-zero uint32")
    self._owns_serial = serial_port is None
    self.serial = serial_port if serial_port is not None else self._open_serial(port, baudrate)
    self._decoder = FrameDecoder()
    self._pending: list[Frame] = []
    # A new process must not reuse sequence zero: a late USB response from the
    # previous process can arrive after reset_input_buffer() and otherwise look
    # correlated at the frame-header level. The random session remains the
    # primary connection identity; a random starting sequence adds independent
    # protection against that restart collision.
    self._sequence = secrets.randbits(32)
    self._handshaken = False
    self._lock = threading.Lock()
    self.hello_response: HelloResponse | None = None
    self.last_result: InferResult | None = None
    self._round_trip_timing = _TimingAccumulator(seed=0)
    self._inference_timing = _TimingAccumulator(seed=1)

  def _open_serial(self, port: str | None, baudrate: int) -> SerialPort:
    if not port:
      raise ValueError("Provide a USB serial --port or inject serial_port")
    try:
      import serial
    except ImportError as exc:
      raise HilTransportError(
        "pyserial is required; install it with: "
        "python -m pip install 'pyserial>=3.5,<4'"
      ) from exc
    try:
      return serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=self.timeout,
        write_timeout=self.timeout,
        exclusive=True,
      )
    except Exception as exc:
      raise HilTransportError(f"Could not open HIL serial port {port}: {exc}") from exc

  def __enter__(self) -> HilClient:
    return self

  def __exit__(self, *_exc) -> None:
    self.close()

  def close(self) -> None:
    if self._owns_serial and hasattr(self.serial, "close"):
      self.serial.close()

  def _next_sequence(self) -> int:
    sequence = self._sequence
    self._sequence = (self._sequence + 1) & 0xFFFFFFFF
    return sequence

  def handshake(self) -> HelloResponse:
    with self._lock:
      self._handshaken = False
      self.hello_response = None
      self.last_result = None
      self._decoder.reset()
      self._pending.clear()
      try:
        if hasattr(self.serial, "reset_input_buffer"):
          self.serial.reset_input_buffer()
        sequence = self._next_sequence()
        request = HelloRequest(self.session)
        frame = self._exchange(
          MessageType.HELLO_REQ,
          MessageType.HELLO_RSP,
          sequence,
          request.pack(),
          discard_stale=True,
          timeout=self.handshake_timeout,
        )
        response = HelloResponse.unpack(frame.payload)
        self._validate_hello(response)
      except ProtocolError as exc:
        raise HilProtocolError(str(exc)) from exc
      except HilError:
        raise
      except Exception as exc:
        raise HilTransportError(f"HIL handshake transport failed: {exc}") from exc
      self.hello_response = response
      self._handshaken = True
      return response

  def _validate_hello(self, response: HelloResponse) -> None:
    errors = []
    if response.session != self.session:
      errors.append(f"session {response.session:#x} != {self.session:#x}")
    if response.status != 0:
      errors.append(f"status={response.status}")
    if response.capabilities & REQUIRED_CAPABILITIES != REQUIRED_CAPABILITIES:
      errors.append(f"capabilities={response.capabilities:#x}")
    if response.model_set_id != MODEL_SET_ID:
      errors.append(f"model_set_id={response.model_set_id:#x}")
    if response.period_us != POLICY_PERIOD_US:
      errors.append(f"period_us={response.period_us}")
    dimensions = (response.obs_dim, response.history_dim, response.action_dim)
    if dimensions != (OBS_DIM, HISTORY_DIM, ACTION_DIM):
      errors.append(f"dimensions={dimensions}")
    if response.mode_mask & REQUIRED_MODE_MASK != REQUIRED_MODE_MASK:
      errors.append(f"mode_mask={response.mode_mask:#x}")
    if response.max_payload != MAX_PAYLOAD:
      errors.append(f"max_payload={response.max_payload}")
    if response.protocol_version != VERSION:
      errors.append(f"protocol_version={response.protocol_version}")
    if errors:
      raise HilProtocolError("HELLO_RSP contract mismatch: " + ", ".join(errors))

  def infer(
    self,
    mode: PolicyMode | str | int,
    obs: np.ndarray,
    history: np.ndarray,
  ) -> InferResult:
    with self._lock:
      self.last_result = None
      if not self._handshaken:
        raise HilProtocolError("Call handshake() before infer()")
      selected_mode = PolicyMode.parse(mode)
      obs_vector = self._policy_vector(obs, OBS_DIM, "obs")
      history_vector = self._policy_vector(history, HISTORY_DIM, "history")
      sequence = self._next_sequence()
      request = InferRequest(
        self.session,
        selected_mode,
        tuple(float(value) for value in obs_vector),
        tuple(float(value) for value in history_vector),
      )
      start_ns = time.perf_counter_ns()
      try:
        frame = self._exchange(
          MessageType.INFER_REQ,
          MessageType.INFER_RSP,
          sequence,
          request.pack(),
        )
        round_trip_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        response = InferResponse.unpack(frame.payload)
        action = self._validate_infer(response, sequence, selected_mode)
      except ProtocolError as exc:
        self._handshaken = False
        self.last_result = None
        raise HilProtocolError(str(exc)) from exc
      except (HilError, FloatingPointError):
        self._handshaken = False
        self.last_result = None
        raise
      result = InferResult(
        action=action,
        sequence=sequence,
        mode=selected_mode,
        inference_us=response.inference_us,
        round_trip_us=round_trip_us,
      )
      self.last_result = result
      self._round_trip_timing.add(round_trip_us)
      self._inference_timing.add(response.inference_us)
      return result

  @staticmethod
  def _policy_vector(value: np.ndarray, width: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape == (1, width):
      vector = vector[0]
    if vector.shape != (width,):
      raise ValueError(
        f"{name} must have shape ({width},) or (1, {width}); got {vector.shape}"
      )
    if not np.isfinite(vector).all():
      raise FloatingPointError(f"Non-finite HIL {name}")
    return np.ascontiguousarray(vector)

  def _validate_infer(
    self, response: InferResponse, sequence: int, mode: PolicyMode
  ) -> np.ndarray:
    if response.session != self.session:
      raise HilProtocolError(
        f"INFER_RSP session {response.session:#x} != {self.session:#x}"
      )
    if response.input_sequence != sequence:
      raise HilProtocolError(
        f"INFER_RSP input sequence {response.input_sequence} != {sequence}"
      )
    if response.mode != mode:
      raise HilProtocolError(f"INFER_RSP mode {response.mode.name} != {mode.name}")
    if response.status != 0:
      raise HilProtocolError(f"INFER_RSP status={response.status}")
    if response.reserved != 0:
      raise HilProtocolError(f"INFER_RSP reserved field is {response.reserved}")
    action = np.asarray(response.action, dtype=np.float32)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
      raise FloatingPointError("Invalid HIL action")
    return action

  def _exchange(
    self,
    request_type: MessageType,
    response_type: MessageType,
    sequence: int,
    payload: bytes,
    *,
    discard_stale: bool = False,
    timeout: float | None = None,
  ) -> Frame:
    deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
    self._write_all(encode_frame(request_type, sequence, payload), deadline)
    while True:
      frame = self._read_frame(deadline)
      if frame.sequence != sequence:
        # A response from the request that timed out can race reset_input_buffer()
        # and the next HELLO. During that explicit synchronization barrier only,
        # ignore frames that are provably from another sequence.
        if discard_stale:
          continue
        raise HilProtocolError(f"Response sequence {frame.sequence} != request {sequence}")
      if frame.message_type == MessageType.ERROR:
        remote = ErrorPayload.unpack(frame.payload)
        if remote.session != self.session:
          # During HELLO this is provably a response from an older connection,
          # even if a restarted host happened to reuse its header sequence.
          if discard_stale:
            continue
          raise HilProtocolError(
            f"ERROR session {remote.session:#x} != active session {self.session:#x}"
          )
        if remote.offending_sequence != sequence:
          raise HilProtocolError(
            f"ERROR offending sequence {remote.offending_sequence} != request {sequence}"
          )
        raise HilRemoteError(remote.code, remote.detail)
      if frame.message_type != response_type:
        raise HilProtocolError(
          f"Response type {frame.message_type.name} != expected {response_type.name}"
        )
      return frame

  def _write_all(self, data: bytes, deadline: float) -> None:
    view = memoryview(data)
    while view:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise HilTimeoutError("Timed out while writing HIL request")
      self._set_write_timeout(remaining)
      try:
        count = self.serial.write(view)
      except Exception as exc:
        if exc.__class__.__name__ == "SerialTimeoutException" or time.monotonic() >= deadline:
          raise HilTimeoutError("Timed out while writing HIL request") from exc
        raise HilTransportError(f"HIL serial write failed: {exc}") from exc
      if time.monotonic() >= deadline:
        raise HilTimeoutError("Timed out while writing HIL request")
      count = len(view) if count is None else int(count)
      if not 0 <= count <= len(view):
        raise HilTransportError(f"Invalid serial write count: {count}")
      if count == 0:
        time.sleep(min(0.0001, max(0.0, deadline - time.monotonic())))
      view = view[count:]

  def _read_frame(self, deadline: float) -> Frame:
    crc_start = self._decoder.crc_errors
    header_start = self._decoder.header_errors
    discarded_start = self._decoder.discarded_bytes
    bytes_read = 0

    def timeout_error() -> HilTimeoutError:
      detail = (
        f" (bytes_read={bytes_read}, "
        f"crc_errors={self._decoder.crc_errors - crc_start}, "
        f"header_errors={self._decoder.header_errors - header_start}, "
        f"discarded_bytes={self._decoder.discarded_bytes - discarded_start})"
      )
      return HilTimeoutError("Timed out waiting for a valid HIL response" + detail)

    while True:
      if self._pending:
        return self._pending.pop(0)
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise timeout_error()
      self._set_read_timeout(remaining)
      try:
        available = int(getattr(self.serial, "in_waiting", 0))
        read_size = min(max(available, 1), MAX_FRAME_SIZE)
        chunk = self.serial.read(read_size)
      except Exception as exc:
        raise HilTransportError(f"HIL serial read failed: {exc}") from exc
      bytes_read += len(chunk)
      if time.monotonic() >= deadline:
        raise timeout_error()
      if chunk:
        self._pending.extend(self._decoder.feed(chunk))
      else:
        # Injectable test transports may be non-blocking; avoid a hot spin.
        time.sleep(min(0.0001, remaining))

  def _set_read_timeout(self, remaining: float) -> None:
    if not hasattr(self.serial, "timeout"):
      return
    try:
      self.serial.timeout = max(0.0, remaining)
    except Exception:
      pass

  def _set_write_timeout(self, remaining: float) -> None:
    if not hasattr(self.serial, "write_timeout"):
      return
    try:
      self.serial.write_timeout = max(0.0, remaining)
    except Exception:
      pass

  def timing_summary(self) -> dict[str, dict[str, float | int] | None]:
    return {
      "round_trip_us": self._round_trip_timing.summary(),
      "inference_us": self._inference_timing.summary(),
    }


class UsbPolicy:
  """Adapter satisfying ``NativeRunner``'s ``policy(obs, history)`` contract."""

  def __init__(self, client: HilClient, mode: PolicyMode | str | int):
    self.client = client
    self.mode = PolicyMode.parse(mode)
    self.last_result: InferResult | None = None

  def __call__(self, obs: np.ndarray, history: np.ndarray) -> np.ndarray:
    self.last_result = None
    self.last_result = self.client.infer(self.mode, obs, history)
    return self.last_result.action

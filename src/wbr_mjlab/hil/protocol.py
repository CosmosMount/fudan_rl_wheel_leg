"""Binary USB HIL protocol shared by the host client and its tests."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAGIC = 0x314C5257
VERSION = 1
MAX_PAYLOAD = 608

MODEL_SET_ID = 0x1C28E40F
POLICY_PERIOD_US = 10_000
STRICT_FLAG = 0x01
REQUIRED_CAPABILITIES = 0x0F
REQUIRED_MODE_MASK = 0x03

OBS_DIM = 25
HISTORY_DIM = 125
ACTION_DIM = 6

HEADER = struct.Struct("<IBBHI")
FOOTER = struct.Struct("<H")
HELLO_REQ = struct.Struct("<IIII")
HELLO_RSP = struct.Struct("<IIIIIHHHHHH")
INFER_REQ = struct.Struct("<IB3x150f")
INFER_RSP = struct.Struct("<IIBBHI6f")
ERROR = struct.Struct("<IIII")

MAGIC_BYTES = struct.pack("<I", MAGIC)
MAX_FRAME_SIZE = HEADER.size + MAX_PAYLOAD + FOOTER.size


class MessageType(IntEnum):
  HELLO_REQ = 1
  HELLO_RSP = 2
  INFER_REQ = 3
  INFER_RSP = 4
  ERROR = 5


class PolicyMode(IntEnum):
  PLANE = 0
  JUMP = 1

  @classmethod
  def parse(cls, value: PolicyMode | str | int) -> PolicyMode:
    if isinstance(value, cls):
      return value
    if isinstance(value, str):
      try:
        return cls[value.upper()]
      except KeyError as exc:
        raise ValueError(f"Unknown HIL policy mode: {value}") from exc
    return cls(value)


class ProtocolError(ValueError):
  """A frame or payload does not satisfy the fixed wire contract."""


def _u32(value: int, name: str) -> int:
  value = int(value)
  if not 0 <= value <= 0xFFFFFFFF:
    raise ProtocolError(f"{name} must fit uint32")
  return value


def crc16(data: bytes | bytearray | memoryview) -> int:
  """Reflected CRC-16 with poly 0x8408, seed 0xffff and no final xor."""
  crc = 0xFFFF
  for byte in data:
    crc ^= byte
    for _ in range(8):
      crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
  return crc & 0xFFFF


@dataclass(frozen=True)
class Frame:
  message_type: MessageType
  sequence: int
  payload: bytes


def encode_frame(message_type: MessageType | int, sequence: int, payload: bytes = b"") -> bytes:
  try:
    kind = MessageType(message_type)
  except ValueError as exc:
    raise ProtocolError(f"Unknown message type: {message_type}") from exc
  payload = bytes(payload)
  if len(payload) > MAX_PAYLOAD:
    raise ProtocolError(f"Payload is {len(payload)} bytes; maximum is {MAX_PAYLOAD}")
  header = HEADER.pack(MAGIC, VERSION, int(kind), len(payload), _u32(sequence, "sequence"))
  body = header + payload
  return body + FOOTER.pack(crc16(body))


class FrameDecoder:
  """Incremental decoder tolerant of short reads, glued frames and garbage."""

  def __init__(self) -> None:
    self.buffer = bytearray()
    self.discarded_bytes = 0
    self.header_errors = 0
    self.crc_errors = 0

  def reset(self) -> None:
    self.buffer.clear()

  def feed(self, data: bytes | bytearray | memoryview) -> list[Frame]:
    self.buffer.extend(data)
    frames: list[Frame] = []
    while True:
      if len(self.buffer) < len(MAGIC_BYTES):
        break
      offset = self.buffer.find(MAGIC_BYTES)
      if offset < 0:
        keep = self._magic_prefix_suffix_length()
        dropped = len(self.buffer) - keep
        del self.buffer[:dropped]
        self.discarded_bytes += dropped
        break
      if offset:
        del self.buffer[:offset]
        self.discarded_bytes += offset
      if len(self.buffer) < HEADER.size:
        break
      magic, version, raw_type, payload_len, sequence = HEADER.unpack_from(self.buffer)
      try:
        kind = MessageType(raw_type)
      except ValueError:
        kind = None
      if magic != MAGIC or version != VERSION or kind is None or payload_len > MAX_PAYLOAD:
        del self.buffer[0]
        self.header_errors += 1
        continue
      frame_size = HEADER.size + payload_len + FOOTER.size
      if len(self.buffer) < frame_size:
        break
      body_end = HEADER.size + payload_len
      expected_crc = FOOTER.unpack_from(self.buffer, body_end)[0]
      actual_crc = crc16(memoryview(self.buffer)[:body_end])
      if actual_crc != expected_crc:
        # Drop only one byte. A valid frame may already follow a damaged length field.
        del self.buffer[0]
        self.crc_errors += 1
        continue
      payload = bytes(self.buffer[HEADER.size:body_end])
      del self.buffer[:frame_size]
      frames.append(Frame(kind, sequence, payload))
    return frames

  def _magic_prefix_suffix_length(self) -> int:
    limit = min(len(self.buffer), len(MAGIC_BYTES) - 1)
    for length in range(limit, 0, -1):
      if self.buffer[-length:] == MAGIC_BYTES[:length]:
        return length
    return 0


def _unpack_exact(codec: struct.Struct, payload: bytes, name: str) -> tuple:
  if len(payload) != codec.size:
    raise ProtocolError(f"{name} payload is {len(payload)} bytes; expected {codec.size}")
  return codec.unpack(payload)


@dataclass(frozen=True)
class HelloRequest:
  session: int
  period_us: int = POLICY_PERIOD_US
  model_set_id: int = MODEL_SET_ID
  flags: int = STRICT_FLAG

  def pack(self) -> bytes:
    return HELLO_REQ.pack(
      _u32(self.session, "session"),
      _u32(self.period_us, "period_us"),
      _u32(self.model_set_id, "model_set_id"),
      _u32(self.flags, "flags"),
    )

  @classmethod
  def unpack(cls, payload: bytes) -> HelloRequest:
    return cls(*_unpack_exact(HELLO_REQ, payload, "HELLO_REQ"))


@dataclass(frozen=True)
class HelloResponse:
  session: int
  capabilities: int
  model_set_id: int
  period_us: int
  status: int
  obs_dim: int
  history_dim: int
  action_dim: int
  mode_mask: int
  max_payload: int
  protocol_version: int

  def pack(self) -> bytes:
    return HELLO_RSP.pack(
      _u32(self.session, "session"),
      _u32(self.capabilities, "capabilities"),
      _u32(self.model_set_id, "model_set_id"),
      _u32(self.period_us, "period_us"),
      _u32(self.status, "status"),
      self.obs_dim,
      self.history_dim,
      self.action_dim,
      self.mode_mask,
      self.max_payload,
      self.protocol_version,
    )

  @classmethod
  def unpack(cls, payload: bytes) -> HelloResponse:
    return cls(*_unpack_exact(HELLO_RSP, payload, "HELLO_RSP"))


@dataclass(frozen=True)
class InferRequest:
  session: int
  mode: PolicyMode
  obs: tuple[float, ...]
  history: tuple[float, ...]

  def pack(self) -> bytes:
    if len(self.obs) != OBS_DIM or len(self.history) != HISTORY_DIM:
      raise ProtocolError("INFER_REQ requires 25 observation and 125 history floats")
    return INFER_REQ.pack(
      _u32(self.session, "session"),
      int(PolicyMode.parse(self.mode)),
      *self.obs,
      *self.history,
    )

  @classmethod
  def unpack(cls, payload: bytes) -> InferRequest:
    values = _unpack_exact(INFER_REQ, payload, "INFER_REQ")
    try:
      mode = PolicyMode(values[1])
    except ValueError as exc:
      raise ProtocolError(f"INFER_REQ has invalid mode {values[1]}") from exc
    return cls(values[0], mode, values[2:27], values[27:])


@dataclass(frozen=True)
class InferResponse:
  session: int
  input_sequence: int
  mode: PolicyMode
  status: int
  reserved: int
  inference_us: int
  action: tuple[float, ...]

  def pack(self) -> bytes:
    if len(self.action) != ACTION_DIM:
      raise ProtocolError("INFER_RSP requires 6 action floats")
    return INFER_RSP.pack(
      _u32(self.session, "session"),
      _u32(self.input_sequence, "input_sequence"),
      int(PolicyMode.parse(self.mode)),
      self.status,
      self.reserved,
      _u32(self.inference_us, "inference_us"),
      *self.action,
    )

  @classmethod
  def unpack(cls, payload: bytes) -> InferResponse:
    values = _unpack_exact(INFER_RSP, payload, "INFER_RSP")
    try:
      mode = PolicyMode(values[2])
    except ValueError as exc:
      raise ProtocolError(f"INFER_RSP has invalid mode {values[2]}") from exc
    return cls(values[0], values[1], mode, *values[3:6], values[6:])


@dataclass(frozen=True)
class ErrorPayload:
  session: int
  code: int
  detail: int
  offending_sequence: int

  def pack(self) -> bytes:
    return ERROR.pack(
      _u32(self.session, "session"),
      _u32(self.code, "code"),
      _u32(self.detail, "detail"),
      _u32(self.offending_sequence, "offending_sequence"),
    )

  @classmethod
  def unpack(cls, payload: bytes) -> ErrorPayload:
    return cls(*_unpack_exact(ERROR, payload, "ERROR"))


assert INFER_REQ.size == MAX_PAYLOAD

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from wbr_mjlab.hil.client import (
  HilClient,
  HilProtocolError,
  HilRemoteError,
  HilTimeoutError,
  UsbPolicy,
  _TimingAccumulator,
)
from wbr_mjlab.hil.protocol import (
  ACTION_DIM,
  ERROR,
  HEADER,
  HELLO_REQ,
  HELLO_RSP,
  HISTORY_DIM,
  INFER_REQ,
  INFER_RSP,
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
  crc16,
  encode_frame,
)


def valid_hello(session: int) -> HelloResponse:
  return HelloResponse(
    session=session,
    capabilities=REQUIRED_CAPABILITIES,
    model_set_id=MODEL_SET_ID,
    period_us=POLICY_PERIOD_US,
    status=0,
    obs_dim=OBS_DIM,
    history_dim=HISTORY_DIM,
    action_dim=ACTION_DIM,
    mode_mask=REQUIRED_MODE_MASK,
    max_payload=MAX_PAYLOAD,
    protocol_version=VERSION,
  )


class MemorySerial:
  """Incremental device double supporting short writes and short reads."""

  def __init__(self, handler, *, write_size=19, read_sizes=(1, 2, 7, 31)):
    self.handler = handler
    self.write_size = write_size
    self.read_sizes = tuple(read_sizes)
    self.read_index = 0
    self.timeout = None
    self.requests: list[Frame] = []
    self._request_decoder = FrameDecoder()
    self._rx = bytearray()

  @property
  def in_waiting(self):
    return len(self._rx)

  def reset_input_buffer(self):
    self._rx.clear()

  def write(self, data):
    size = min(len(data), self.write_size)
    for frame in self._request_decoder.feed(bytes(data[:size])):
      self.requests.append(frame)
      response = self.handler(frame)
      if response:
        self._rx.extend(response)
    return size

  def read(self, size=1):
    if not self._rx:
      return b""
    chunk_size = self.read_sizes[self.read_index % len(self.read_sizes)]
    self.read_index += 1
    count = min(size, chunk_size, len(self._rx))
    result = bytes(self._rx[:count])
    del self._rx[:count]
    return result


def device_handler(*, hello=None, infer=None, response_prefix=b""):
  def handle(frame):
    if frame.message_type == MessageType.HELLO_REQ:
      request = HelloRequest.unpack(frame.payload)
      response = valid_hello(request.session) if hello is None else hello(request, frame)
      return response_prefix + encode_frame(MessageType.HELLO_RSP, frame.sequence, response.pack())
    if frame.message_type == MessageType.INFER_REQ:
      request = InferRequest.unpack(frame.payload)
      response = (
        InferResponse(
          session=request.session,
          input_sequence=frame.sequence,
          mode=request.mode,
          status=0,
          reserved=0,
          inference_us=731,
          action=(0.1, -0.2, 0.3, -0.4, 0.5, -0.6),
        )
        if infer is None
        else infer(request, frame)
      )
      if isinstance(response, bytes):
        return response
      return response_prefix + encode_frame(MessageType.INFER_RSP, frame.sequence, response.pack())
    raise AssertionError(frame.message_type)

  return handle


def test_wire_sizes_crc_and_payload_round_trips():
  assert HEADER.size == 12
  assert HELLO_REQ.size == 16
  assert HELLO_RSP.size == 32
  assert INFER_REQ.size == MAX_PAYLOAD == 608
  assert INFER_RSP.size == 40
  assert ERROR.size == 16
  assert MAX_FRAME_SIZE == 622
  assert crc16(b"123456789") == 0x6F91

  hello = valid_hello(0x12345678)
  assert HelloResponse.unpack(hello.pack()) == hello
  request = InferRequest(
    0x12345678,
    PolicyMode.JUMP,
    tuple(float(i) for i in range(OBS_DIM)),
    tuple(float(i) / 10 for i in range(HISTORY_DIM)),
  )
  unpacked = InferRequest.unpack(request.pack())
  assert unpacked.session == request.session and unpacked.mode == request.mode
  np.testing.assert_allclose(unpacked.obs, request.obs)
  np.testing.assert_allclose(unpacked.history, request.history)
  error = ErrorPayload(0x12345678, 7, 11, 99)
  assert ErrorPayload.unpack(error.pack()) == error


def test_hello_request_golden_frame():
  frame = encode_frame(MessageType.HELLO_REQ, 1, HelloRequest(0x11223344).pack())
  assert frame.hex() == "57524c31010110000100000044332211102700000fe4281c010000004167"


def test_incremental_decoder_handles_short_reads_glued_frames_and_crc_resync():
  first = encode_frame(MessageType.HELLO_REQ, 3, HelloRequest(9).pack())
  second = encode_frame(MessageType.ERROR, 4, ErrorPayload(9, 2, 5, 4).pack())
  damaged = bytearray(first)
  damaged[-1] ^= 0x80
  stream = b"noise" + bytes(damaged) + b"garbage" + first + second
  decoder = FrameDecoder()
  frames = []
  for byte in stream:
    frames.extend(decoder.feed(bytes((byte,))))
  assert [(frame.message_type, frame.sequence) for frame in frames] == [
    (MessageType.HELLO_REQ, 3),
    (MessageType.ERROR, 4),
  ]
  assert decoder.crc_errors == 1
  assert decoder.discarded_bytes >= len("noisegarbage")


def test_client_handshake_and_inference_survive_transport_fragmentation_and_garbage():
  captured = {}

  def infer(request, frame):
    captured["request"] = request
    response = InferResponse(
      request.session,
      frame.sequence,
      request.mode,
      0,
      0,
      731,
      (0.1, -0.2, 0.3, -0.4, 0.5, -0.6),
    )
    valid = encode_frame(MessageType.INFER_RSP, frame.sequence, response.pack())
    damaged = bytearray(valid)
    damaged[-2] ^= 1
    return b"misaligned" + bytes(damaged) + valid

  serial = MemorySerial(device_handler(infer=infer), write_size=11, read_sizes=(1, 3, 5, 13))
  client = HilClient(serial, timeout=0.05, session=0x10203040)
  hello = client.handshake()
  assert hello.session == client.session and hello.mode_mask == REQUIRED_MODE_MASK
  obs = np.linspace(-1, 1, OBS_DIM, dtype=np.float32)[None]
  history = np.linspace(-2, 2, HISTORY_DIM, dtype=np.float32)[None]
  result = client.infer("jump", obs, history)
  np.testing.assert_array_equal(captured["request"].obs, obs[0])
  np.testing.assert_array_equal(captured["request"].history, history[0])
  np.testing.assert_allclose(result.action, [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
  assert result.mode == PolicyMode.JUMP and result.inference_us == 731
  assert len(serial.requests) == 2
  assert serial.requests[1].sequence == result.sequence
  assert client.timing_summary()["round_trip_us"]["count"] == 1


def test_timing_accumulator_has_bounded_percentile_storage_and_exact_aggregates():
  timing = _TimingAccumulator(capacity=8, seed=7)
  for value in range(100):
    timing.add(value)
  summary = timing.summary()
  assert len(timing._samples) == 8
  assert summary["count"] == 100
  assert summary["min"] == 0
  assert summary["mean"] == 49.5
  assert summary["max"] == 99


def test_handshake_discards_late_response_from_an_old_sequence():
  stale = InferResponse(
    session=7,
    input_sequence=99,
    mode=PolicyMode.PLANE,
    status=0,
    reserved=0,
    inference_us=100,
    action=(0.0,) * ACTION_DIM,
  )
  prefix = encode_frame(MessageType.INFER_RSP, 99, stale.pack())
  serial = MemorySerial(device_handler(response_prefix=prefix))
  client = HilClient(serial, timeout=0.02, session=7)
  assert client.handshake().session == 7


def test_handshake_discards_same_sequence_error_from_an_old_session():
  def handler(frame):
    request = HelloRequest.unpack(frame.payload)
    stale = ErrorPayload(request.session + 1, 5, 99, frame.sequence)
    return encode_frame(MessageType.ERROR, frame.sequence, stale.pack()) + encode_frame(
      MessageType.HELLO_RSP,
      frame.sequence,
      valid_hello(request.session).pack(),
    )

  client = HilClient(MemorySerial(handler), timeout=0.02, session=7)
  assert client.handshake().session == 7


def test_write_is_part_of_the_whole_request_deadline():
  class SlowSerial:
    timeout = None
    write_timeout = None

    def reset_input_buffer(self):
      pass

    def write(self, data):
      time.sleep(0.015)
      return len(data)

    def read(self, _size=1):
      return b""

  serial = SlowSerial()
  client = HilClient(serial, timeout=0.003, handshake_timeout=0.003, session=7)
  started = time.monotonic()
  with pytest.raises(HilTimeoutError, match="writing"):
    client.handshake()
  assert time.monotonic() - started < 0.1
  assert 0 < serial.write_timeout <= 0.003


def test_read_is_part_of_the_whole_request_deadline():
  class SlowReadSerial(MemorySerial):
    def read(self, size=1):
      time.sleep(0.015)
      return super().read(size)

  client = HilClient(
    SlowReadSerial(device_handler()),
    timeout=0.003,
    handshake_timeout=0.003,
    session=7,
  )
  with pytest.raises(HilTimeoutError, match=r"waiting.*bytes_read="):
    client.handshake()


def test_client_generates_nonzero_session_and_rejects_zero(monkeypatch):
  import wbr_mjlab.hil.client as client_module

  monkeypatch.setattr(client_module.secrets, "randbelow", lambda _limit: 0)
  client = HilClient(MemorySerial(device_handler()), timeout=0.02)
  assert client.session == 1
  with pytest.raises(ValueError, match="non-zero uint32"):
    HilClient(MemorySerial(device_handler()), timeout=0.02, session=0)


@pytest.mark.parametrize(
  "change,needle",
  (
    ({"session": 8}, "session"),
    ({"capabilities": 0x07}, "capabilities"),
    ({"model_set_id": 4}, "model_set_id"),
    ({"period_us": 20_000}, "period_us"),
    ({"obs_dim": 24}, "dimensions"),
    ({"mode_mask": 1}, "mode_mask"),
    ({"max_payload": 607}, "max_payload"),
    ({"protocol_version": 2}, "protocol_version"),
  ),
)
def test_handshake_rejects_contract_mismatch(change, needle):
  def hello(request, _frame):
    return replace(valid_hello(request.session), **change)

  client = HilClient(MemorySerial(device_handler(hello=hello)), timeout=0.02, session=7)
  with pytest.raises(HilProtocolError, match=needle):
    client.handshake()


@pytest.mark.parametrize(
  "fault", ("header_sequence", "input_sequence", "session", "mode", "invalid_mode")
)
def test_inference_correlation_fault_invalidates_session(fault):
  def infer(request, frame):
    if fault == "invalid_mode":
      payload = INFER_RSP.pack(
        request.session,
        frame.sequence,
        2,
        0,
        0,
        100,
        *((0.0,) * ACTION_DIM),
      )
      return encode_frame(MessageType.INFER_RSP, frame.sequence, payload)
    response = InferResponse(
      request.session + (fault == "session"),
      frame.sequence + (fault == "input_sequence"),
      PolicyMode.JUMP if fault == "mode" else request.mode,
      0,
      0,
      100,
      (0.0,) * ACTION_DIM,
    )
    sequence = frame.sequence + (fault == "header_sequence")
    return encode_frame(MessageType.INFER_RSP, sequence, response.pack())

  serial = MemorySerial(device_handler(infer=infer))
  client = HilClient(serial, timeout=0.02, session=7)
  client.handshake()
  with pytest.raises(HilProtocolError):
    client.infer("plane", np.zeros(OBS_DIM), np.zeros(HISTORY_DIM))
  request_count = len(serial.requests)
  with pytest.raises(HilProtocolError, match=r"handshake\(\)"):
    client.infer("plane", np.zeros(OBS_DIM), np.zeros(HISTORY_DIM))
  assert len(serial.requests) == request_count


def test_timeout_and_remote_error_are_fail_closed_until_new_handshake():
  infer_calls = 0

  def infer(request, frame):
    nonlocal infer_calls
    infer_calls += 1
    if infer_calls == 1:
      return b""
    error = ErrorPayload(request.session, 12, 34, frame.sequence)
    return encode_frame(MessageType.ERROR, frame.sequence, error.pack())

  client = HilClient(MemorySerial(device_handler(infer=infer)), timeout=0.002, session=7)
  client.handshake()
  with pytest.raises(HilTimeoutError):
    client.infer("plane", np.zeros(OBS_DIM), np.zeros(HISTORY_DIM))
  with pytest.raises(HilProtocolError, match=r"handshake\(\)"):
    client.infer("plane", np.zeros(OBS_DIM), np.zeros(HISTORY_DIM))
  client.handshake()
  with pytest.raises(HilRemoteError, match="code=12"):
    client.infer("plane", np.zeros(OBS_DIM), np.zeros(HISTORY_DIM))


def test_client_validates_shapes_finite_values_and_usb_policy_adapter():
  client = HilClient(MemorySerial(device_handler()), timeout=0.02, session=7)
  client.handshake()
  policy = UsbPolicy(client, "plane")
  action = policy(np.zeros((1, OBS_DIM), np.float32), np.zeros((1, HISTORY_DIM), np.float32))
  assert action.shape == (ACTION_DIM,)
  with pytest.raises(ValueError, match="obs must have shape"):
    client.infer("plane", np.zeros(24), np.zeros(HISTORY_DIM))
  assert client.last_result is None
  with pytest.raises(FloatingPointError, match="history"):
    client.infer("plane", np.zeros(OBS_DIM), np.full(HISTORY_DIM, np.nan))

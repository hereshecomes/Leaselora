from __future__ import annotations

import time
import logging
from dataclasses import dataclass

logger = logging.getLogger("fl_testbed.network.measure")

@dataclass
class UploadMeasurement:
    payload_bytes: int
    upload_time_sec: float
    success: bool
    attempt: int
    error: str | None = None
    measured_throughput_mbps: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self):
        if self.success and self.upload_time_sec > 0:
            self.measured_throughput_mbps = (
                self.payload_bytes * 8 / (self.upload_time_sec * 1e6)
            )

def measure_upload_with_retry(
    upload_fn,
    max_retries: int = 3,
    retry_backoff_sec: list[float] | None = None,
    upload_timeout_sec: float = 60.0,
) -> tuple[any, list[UploadMeasurement]]:
    if retry_backoff_sec is None:
        retry_backoff_sec = [1.0, 2.0, 4.0]

    measurements = []
    last_result = None

    for attempt in range(max_retries):
        t_start = time.time()
        try:
            result, upload_time, payload_bytes = upload_fn()
            t_end = time.time()
            measurement = UploadMeasurement(
                payload_bytes=payload_bytes,
                upload_time_sec=upload_time,
                success=True,
                attempt=attempt + 1,
                start_time=t_start,
                end_time=t_end,
            )
            measurements.append(measurement)
            last_result = result
            return last_result, measurements

        except Exception as e:
            t_end = time.time()
            measurement = UploadMeasurement(
                payload_bytes=0,
                upload_time_sec=t_end - t_start,
                success=False,
                attempt=attempt + 1,
                error=str(e)[:200],
                start_time=t_start,
                end_time=t_end,
            )
            measurements.append(measurement)
            logger.warning(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")

            if attempt < max_retries - 1:
                backoff = retry_backoff_sec[min(attempt, len(retry_backoff_sec) - 1)]
                time.sleep(backoff)

    return None, measurements

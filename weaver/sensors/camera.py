# camera.py — Pi Camera module for Weaver
#
# This module handles all computer vision for the hexapod. It captures frames
# from the Pi Camera (CSI ribbon cable), runs lightweight OpenCV detection
# (color tracking + Haar cascade face detection for human safety), and publishes
# results to the event bus. It also serves a JPEG video stream for the web
# dashboard.
#
# Hardware: Raspberry Pi Camera Module v2 (or v3) connected via CSI.
# Software: picamera2 (libcamera-based, the modern Pi camera stack),
#           opencv-python-headless for CV processing.
#
# Learning notes:
#   - picamera2 replaced the legacy picamera library; it uses libcamera under
#     the hood and works on Pi 5's new camera stack.
#   - We run CV detection on a *downscaled* frame to save CPU — the Pi 5 is
#     fast but doing full-res Haar cascades at 30fps is wasteful.
#   - Face detection uses Haar cascades (built into OpenCV). It's not as
#     accurate as deep learning models but needs zero GPU/NPU and runs at
#     10-15fps on a Pi 5, which is plenty for a slow-moving hexapod.
#   - The web stream is MJPEG over HTTP — the simplest possible streaming
#     protocol. A multipart/x-mixed-replace response serves frames indefinitely.
#
# ──────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

import numpy as np
from loguru import logger

from weaver.config import CameraConfig, HardwareMode, get_config
from weaver.event_bus import Event, EventType, get_event_bus

# ─── Optional hardware/library imports ───────────────────────────────────
# We try to import these at module load. If they're not installed (e.g. on a
# dev laptop without a Pi Camera), we gracefully fall back to mock mode.
# This pattern lets the same code run everywhere — no try/except needed in
# the main logic.

try:
    from picamera2 import Picamera2  # type: ignore[import-untyped]
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

try:
    import cv2  # type: ignore[import-untyped]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class CameraSensor:
    """Pi Camera sensor with computer vision capabilities.

    This class manages the Pi Camera and runs two forms of lightweight
    object detection:

    1. **Color tracking** — converts frames to HSV and thresholds for a
       target color range. Returns the largest contour's centroid and area.
       Useful for following a colored ball or marker.

    2. **Face detection** — uses OpenCV's Haar cascade classifier to find
       human faces in the frame. This is the primary input to the human-safety
       subsystem: if a face is detected, the safety governor slows the robot
       and maintains distance.

    Events published:
        - CAMERA_FRAME:           every captured frame (raw ndarray + JPEG)
        - CAMERA_OBJECT_DETECTED: when color tracking finds an object
        - CAMERA_FACE_DETECTED:   when a face is found (safety-critical!)

    The web dashboard consumes the JPEG bytes from the latest frame for its
    live video stream via :meth:`get_jpeg_frame`.

    Mock mode:
        When hardware is unavailable, a synthetic gradient frame is generated
        so the rest of the pipeline (web dashboard, event subscribers) can be
        developed and tested without a camera.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        """Initialize the camera sensor.

        Args:
            config: Camera configuration. If None, loads from global config.
        """
        cfg = get_config()
        self.config: CameraConfig = config or cfg.camera
        self.hardware_mode: HardwareMode = cfg.hardware_mode

        self.bus = get_event_bus()

        # picamera2 instance (real mode only)
        self._picam2: Any = None

        # OpenCV Haar cascade classifier for face detection
        self._face_cascade: Any = None

        # Latest JPEG-encoded frame for web streaming (thread-safe-ish: we
        # only write from the capture loop and read from the web handler).
        self._latest_jpeg: bytes | None = None

        # Running tasks
        self._capture_task: asyncio.Task | None = None
        self._detection_task: asyncio.Task | None = None

        # Track whether we're in mock mode (may be forced if libs missing)
        self._mock: bool = self._should_use_mock()

        # Color tracking target (HSV lower/upper bounds).
        # Default: track a bright orange object (good contrast on most floors).
        # HSV = Hue [0,179], Saturation [0,255], Value [0,255] in OpenCV.
        self._color_lower = np.array([10, 100, 100])
        self._color_upper = np.array([25, 255, 255])

        # Frame counter for logging cadence
        self._frame_count: int = 0

    def _should_use_mock(self) -> bool:
        """Determine whether to use mock mode.

        Mock mode is active when:
        - The global hardware mode is MOCK, OR
        - picamera2 is not installed, OR
        - OpenCV is not installed (needed for detection)
        """
        if self.hardware_mode == HardwareMode.MOCK:
            return True
        if not _PICAMERA2_AVAILABLE:
            logger.warning(
                "📷 picamera2 not installed — camera running in MOCK mode. "
                "Install with: pip install picamera2"
            )
            return True
        if not _CV2_AVAILABLE:
            logger.warning(
                "📷 opencv not installed — camera running in MOCK mode. "
                "Install with: pip install opencv-python-headless"
            )
            return True
        return False

    async def start(self) -> None:
        """Start the camera and detection loops."""
        logger.info("📷 Camera sensor starting (mode: {})", "MOCK" if self._mock else "REAL")

        if not self._mock:
            await self._init_real_hardware()

        # Start the async capture + detection loop
        self._capture_task = asyncio.create_task(self._capture_loop())

        logger.info("✅ Camera sensor started")

    async def stop(self) -> None:
        """Stop the camera and release hardware resources."""
        logger.info("📷 Camera sensor stopping...")

        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass

        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception as e:
                logger.error(f"Error stopping picamera2: {e}")
            self._picam2 = None

        logger.info("✅ Camera sensor stopped")

    # ─── Public API ───────────────────────────────────────────────────

    def get_jpeg_frame(self) -> bytes | None:
        """Get the latest JPEG-encoded frame for web streaming.

        Returns:
            JPEG bytes, or None if no frame has been captured yet.
        """
        return self._latest_jpeg

    def set_color_target(self, hsv_lower: list[int], hsv_upper: list[int]) -> None:
        """Set the HSV color range for color tracking.

        Args:
            hsv_lower: Lower HSV bound [H, S, V] (each 0-179/255).
            hsv_upper: Upper HSV bound [H, S, V].
        """
        self._color_lower = np.array(hsv_lower)
        self._color_upper = np.array(hsv_upper)
        logger.debug(f"Color target set: {hsv_lower} → {hsv_upper}")

    # ─── Initialization ───────────────────────────────────────────────

    async def _init_real_hardware(self) -> None:
        """Initialize the real Pi Camera and OpenCV cascade."""
        try:
            # picamera2 uses a configuration dict to set resolution/framerate.
            # We create a "preview" stream configuration optimized for our use
            # case: moderate resolution for CV, decent framerate.
            self._picam2 = Picamera2()

            # Configure the camera with our target resolution and framerate.
            # picamera2 uses "StreamConfiguration" objects.
            video_config = self._picam2.create_video_configuration(
                main={
                    "size": self.config.resolution,
                    "format": "RGB888",
                },
                controls={
                    "FrameRate": self.config.framerate,
                },
            )
            self._picam2.configure(video_config)

            # Load Haar cascade for face detection.
            # This XML file ships with OpenCV and contains pre-trained
            # features for detecting frontal human faces.
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)

            if self._face_cascade.empty():
                logger.warning("⚠️  Haar cascade file not found — face detection disabled")
                self._face_cascade = None

            # Start the camera
            self._picam2.start()
            logger.info(f"📷 Pi Camera initialized: {self.config.resolution} @ {self.config.framerate}fps")

        except Exception as e:
            logger.error(f"Failed to initialize Pi Camera: {e} — falling back to MOCK mode")
            self._mock = True
            self._picam2 = None

    # ─── Capture Loop ─────────────────────────────────────────────────

    async def _capture_loop(self) -> None:
        """Main capture and processing loop.

        This loop runs at the configured stream framerate. For each iteration:
        1. Capture a frame (real camera or mock)
        2. Encode to JPEG for the web stream
        3. Run color tracking (if enabled)
        4. Run face detection (if enabled)
        5. Publish events to the bus

        We interleave small ``await asyncio.sleep(0)`` calls around the
        blocking CV work to keep the event loop responsive.
        """
        interval = 1.0 / max(1, self.config.stream_framerate)

        while True:
            try:
                frame = await self._capture_frame()
                if frame is None:
                    await asyncio.sleep(interval)
                    continue

                # Encode JPEG for web stream (always, so dashboard has video)
                jpeg_bytes = await self._encode_jpeg(frame)
                self._latest_jpeg = jpeg_bytes

                # Publish CAMERA_FRAME event
                await self.bus.publish(Event(
                    type=EventType.CAMERA_FRAME,
                    data={
                        "frame_shape": list(frame.shape),
                        "jpeg_size": len(jpeg_bytes),
                        "timestamp": time.time(),
                    },
                    source="camera",
                ))

                # Run detection (if enabled and OpenCV available)
                if _CV2_AVAILABLE:
                    if self.config.color_tracking_enabled:
                        await self._detect_color(frame)

                    if self.config.face_detection_enabled:
                        await self._detect_faces(frame)

                self._frame_count += 1
                # Log every 100 frames to avoid spam
                if self._frame_count % 100 == 0:
                    logger.debug(f"📷 Captured {self._frame_count} frames")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Camera capture loop error: {e}")

            await asyncio.sleep(interval)

    async def _capture_frame(self) -> np.ndarray | None:
        """Capture a single frame from the camera.

        Returns:
            numpy ndarray of shape (H, W, 3) in RGB format, or None on error.

        In mock mode, generates a synthetic gradient frame with a moving
        "object" so color tracking and detection code paths can be tested.
        """
        if self._mock:
            return self._generate_mock_frame()

        try:
            # picamera2.capture_array() returns the latest frame as an ndarray.
            # The format is set by our configuration (RGB888).
            frame = self._picam2.capture_array()
            # Yield to the event loop briefly (capture_array can block ~10-30ms)
            await asyncio.sleep(0)
            return frame
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def _generate_mock_frame(self) -> np.ndarray:
        """Generate a synthetic frame for mock mode.

        Creates a gradient background with a colored "blob" that moves in a
        circle. This lets the color tracking and face detection code paths
        execute (color tracking will find the blob; face detection will find
        nothing, which is correct for a synthetic frame).
        """
        w, h = self.config.resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Gradient background (looks like a camera feed, not just noise)
        for i in range(h):
            frame[i, :, 0] = int(30 + 20 * (i / h))  # Blue-ish gradient
            frame[i, :, 1] = int(20 + 10 * (i / h))
            frame[i, :, 2] = int(40 + 15 * (i / h))

        # Draw a moving orange blob (circles around the center).
        # This exercises the color tracking code path.
        t = self._frame_count * 0.05
        cx = int(w / 2 + 100 * np.cos(t))
        cy = int(h / 2 + 80 * np.sin(t))
        cv2_circle_mock(frame, cx, cy, 30, (0, 165, 255), -1)  # Orange (BGR-ish)

        return frame

    async def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Encode a frame to JPEG bytes.

        Args:
            frame: RGB ndarray to encode.

        Returns:
            JPEG-encoded bytes. Returns empty bytes if encoding fails.
        """
        if _CV2_AVAILABLE:
            try:
                # cv2.imencode returns (success, buffer)
                # The quality parameter controls compression vs size.
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.stream_quality]
                result = cv2.imencode(".jpg", frame, encode_params)
                # Handle both 2-tuple (newer cv2) and 3-tuple (older cv2) returns
                if isinstance(result, tuple) and len(result) >= 2:
                    success, buffer = result[0], result[1]
                    if success:
                        return buffer.tobytes()
                logger.warning("JPEG encoding returned unexpected result")
            except Exception as e:
                logger.debug(f"JPEG encoding error (cv2 may not be functional): {e}")

        # Fallback: can't encode without functional OpenCV, return empty bytes
        await asyncio.sleep(0)
        return b""

    # ─── Detection ────────────────────────────────────────────────────

    async def _detect_color(self, frame: np.ndarray) -> None:
        """Run color tracking on the frame.

        Converts the frame to HSV, applies a color threshold, finds contours,
        and publishes CAMERA_OBJECT_DETECTED if a sufficiently large blob is
        found.

        Learning note: HSV (Hue-Saturation-Value) is better than RGB for
        color detection because it separates color information (hue) from
        lighting (value). This makes detection robust to varying brightness.
        """
        try:
            # Convert RGB → BGR (OpenCV expects BGR) → HSV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image for our target color range
            mask = cv2.inRange(hsv, self._color_lower, self._color_upper)

            # Find contours (connected components of the mask).
            # OpenCV 3 returns 3 values, 4+ returns 2 values.
            # Using unpacking with * to handle both.
            contour_result = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # In cv2 4.x: (contours, hierarchy); in cv2 3.x: (image, contours, hierarchy)
            contours = contour_result[-2] if isinstance(contour_result, tuple) else []

            if not contours:
                return

            # Find the largest contour (most likely our target object)
            largest = max(contours, key=cv2.contourArea)

            # Ignore tiny detections (noise)
            min_area = 100
            area = cv2.contourArea(largest)
            if area < min_area:
                return

            # Compute centroid using image moments.
            # The moment M00 is the area; M10/M00 is the x-center, M01/M00 is y.
            m = cv2.moments(largest)
            if m["m00"] == 0:
                return

            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            area = float(m["m00"])

            await self.bus.publish(Event(
                type=EventType.CAMERA_OBJECT_DETECTED,
                data={
                    "type": "color",
                    "centroid": (cx, cy),
                    "area": area,
                    "color_range": (
                        self._color_lower.tolist(),
                        self._color_upper.tolist(),
                    ),
                },
                source="camera",
            ))

        except Exception as e:
            logger.error(f"Color detection error: {e}")

    async def _detect_faces(self, frame: np.ndarray) -> None:
        """Run face detection using Haar cascades.

        This is the human-safety input. When a face is detected, the safety
        governor will reduce the robot's speed and maintain distance.

        Learning note: Haar cascades work by sliding a window across the image
        and applying a series of simple "feature" tests (edge, line, four-
        rectangle patterns). A trained cascade has thousands of features
        organized in stages — if a window fails an early stage, it's rejected
        quickly. This makes them very fast compared to CNNs.

        We downscale to 320px wide for detection — this halves the pixels to
        scan and gives us ~2x speedup with minimal accuracy loss for faces
        within a few meters.
        """
        if self._face_cascade is None:
            return

        try:
            # Convert to grayscale (Haar cascades only need luminance)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Downscale for speed (face detection is O(n) in image area)
            h, w = gray.shape
            scale = 320 / w if w > 320 else 1.0
            if scale < 1.0:
                small = cv2.resize(gray, (320, int(h * scale)))
            else:
                small = gray

            # detectMultiScale params:
            #   scaleFactor: how much image shrinks between cascade stages
            #   minNeighbors: how many detections needed to confirm a face
            #   minSize: minimum face size in pixels
            faces = self._face_cascade.detectMultiScale(
                small,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) == 0:
                return

            # Convert face coordinates back to original image scale
            for i, (x, y, fw, fh) in enumerate(faces):
                if scale < 1.0:
                    x, y, fw, fh = int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)

                # Estimate distance using face size heuristic.
                # A typical face is ~15cm wide. The Pi Camera v2 has a 62° FOV.
                # Using the pinhole camera model: distance = (real_width * focal_length) / pixel_width
                # For a rough estimate at 640px width: focal ≈ 360px
                # distance_cm ≈ (15 * 360) / fw  (very approximate!)
                focal_length_px = 360  # Approximate for Pi Camera v2 at 640px
                estimated_distance = (15.0 * focal_length_px) / max(fw, 1)

                await self.bus.publish(Event(
                    type=EventType.CAMERA_FACE_DETECTED,
                    data={
                        "face_index": i,
                        "bbox": (x, y, fw, fh),
                        "area": fw * fh,
                        "distance_cm": round(estimated_distance, 1),
                        "num_faces": len(faces),
                    },
                    source="camera",
                ))

                logger.info(
                    f"👤 Face detected at ({x},{y}) {fw}x{fh}px — "
                    f"est. {estimated_distance:.0f}cm away"
                )

        except Exception as e:
            logger.error(f"Face detection error: {e}")


# ─── Helper: draw a filled circle without requiring OpenCV ────────────────
# This is used by the mock frame generator so that mock frames can show a
# "blob" for color tracking even if OpenCV is not installed.

def cv2_circle_mock(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a filled circle on an ndarray frame (numpy-only fallback).

    A minimal reimplementation of cv2.circle for mock mode when OpenCV
    isn't available. Draws a filled circle by testing pixel distance.
    """
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    frame[mask] = color
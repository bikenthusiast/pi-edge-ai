"""Gesture recognition from 8×8 time-of-flight depth frames.

Frames come from a `FrameSource` (the SEN0628 sensor, a recording, or a
synthetic generator), a `SwipeClassifier` turns them into at most one gesture
per hand movement, and `edge.gesture.run` stores and publishes the result on
`edge/<device>/gesture` — see docs/mqtt.md and ADR 0004.

Everything here except the SEN0628 adapter is pure Python and runs on the Mac.
"""

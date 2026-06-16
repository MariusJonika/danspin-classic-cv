#!/usr/bin/env python3
"""
detect_breaks_v2.py  —  Yarn break detector for diagonal warping-machine yarns.
                         Hough-line edition.

GEOMETRY
────────
End-on camera view of a warping machine.  Yarns are diagonal straight lines
spanning the frame: left-side yarns lean lower-left, right-side yarns lean
lower-right, centre yarns are nearly vertical.  Each yarn originates from its
own individual bobbin, so origins are spread along the roller — not a single
convergence point.  Angles at the frame edges can reach 60–70° from vertical.

ALGORITHM
─────────
  BGR → grey → CLAHE → Gaussian blur
  → ROI crop (--roi-top / --roi-bottom)
  → proportional downscale (--subsample-width)
  → Frangi vesselness filter (enhances ridges at ALL orientations)
  → threshold → binary
  → Probabilistic Hough Line Transform
  → cluster nearby segments → one canonical x per yarn
  → Hungarian-optimal assignment to frame-to-frame tracks
  → N-frame miss counter → BREAK alert

WHY HOUGH INSTEAD OF STRIP-BASED DETECTION
───────────────────────────────────────────
Strip-based detection samples each yarn's x-position at one (or a few)
horizontal heights and tries to match those samples across heights.  This
fails for steep yarns: a yarn at 70° from vertical shifts ~850 full-res px in
x between the top and bottom of a 300 px ROI.  No fixed x-tolerance can match
such a shift.

The Hough transform works in (rho, theta) line-parameter space.  Each yarn —
regardless of angle — produces one cluster of votes.  We then compute a
*canonical x*: where the detected line crosses the ROI reference y.  This is a
stable, angle-independent identifier for each yarn.  The tracker is unchanged;
it tracks canonical-x values exactly as before.

FALLEN BROKEN-END DETECTION
────────────────────────────
A complete yarn spans the full ROI height → long Hough segment → counted.
A broken yarn end that has drooped onto an adjacent yarn is:
  (a) kinked/curved → not a straight line → Hough ignores it, or
  (b) short → below --hough-min-length → filtered out.
Either way the track loses the detection and fires a BREAK after --break-frames
consecutive misses.

USAGE
─────
  # Tune on one frame, save 3-panel diagnostic PNG:
  python3 detect_breaks_v2.py --video recording.mkv --tune-frame 1 --save-diag ./diag

  # Run on a file:
  python3 detect_breaks_v2.py --video recording.mkv --roi-top 0.57 --roi-bottom 0.93

  # Process directory of kurokesu_*.mkv in order (state carries across files):
  python3 detect_breaks_v2.py --dir ~/recordings/

  # Live camera:
  python3 detect_breaks_v2.py --camera 0

TUNING GUIDE
────────────
  Run --tune-frame N --save-diag ./diag and open the PNG (3 panels).

  Panel A  Original frame + ROI (cyan) + detected yarn segments (green)
  Panel B  Frangi ridge map (INFERNO) + binary threshold (white regions)
           + detected segments (yellow)
  Panel C  Segment length histogram — helps set --hough-min-length

  1. ROI box (cyan, Panel A): must cover only the target yarn layer.
     Adjust --roi-top / --roi-bottom until the box excludes rollers and
     machine frame.

  2. Panel B brightness: Frangi should glow on yarns, dark elsewhere.
     Uniformly dim → try smaller --sigmas (e.g. 1 2).
     Uniformly bright → try larger (e.g. 2 4 8).
     Yarns darker than BG → add --black-ridges.

  3. Binary threshold (Panel B white regions): white blobs should sit on
     yarns.  If too much background noise is white, increase --hough-ridge-thr
     (try 0.10).  If yarns are only partially white, decrease it (try 0.03).

  4. Segment length (Panel C): the tall bars should be the full-yarn segments
     (long).  Short bars at the left are noise/fluff.  Set --hough-min-length
     so it sits between the two groups.

  5. Panel A green lines: one per yarn, following the actual yarn angle.
     Too many → increase --hough-ridge-thr or --hough-min-length.
     Too few → decrease them, or check --sigmas.

  6. --min-dist: set to ~70–80% of the inter-yarn pixel spacing (full-res).

  7. --break-frames: at 1 fps, default 4 = 4-second persistence before alert.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.filters import frangi
from skimage.morphology import skeletonize

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("detector")


# ── Layer presets ───────────────────────────────────────────────────────────────
# Each yarn layer needs its own parameter bundle.  Selecting --layer loads one of
# these; any individual CLI flag the user passes still overrides the preset value.
#
# WHY THIS EXISTS
# ───────────────
# The bottom and top layers are physically different detection problems:
#   • Bottom layer  — bright, high-contrast, continuous yarns.  No skeletonisation
#                     needed; a lenient y-span filter (0.15) is fine.
#   • Top layer     — yarns cross a bright specular roller band, so they go dim in
#                     the middle and split into pieces.  Needs skeletonisation
#                     (collapse thick/duplicate ridges to one centreline), an
#                     aggressive y-span filter (0.55) to kill short noise, and a
#                     large max-gap (0.60) to bridge each yarn's two halves back
#                     into one full-height segment that survives the span filter.
# Mixing these — e.g. the top layer's 0.55 y-span leaking onto the bottom layer —
# is what silently broke bottom-layer detection.  Keeping them in named presets
# stops that.
#
# NOTE ON ROI: these ROI fractions are camera-framing dependent.  Adjust per rig.
LAYER_PRESETS: dict[str, dict] = {
    "bottom": dict(
        roi_top               = 0.65,   # current bottom-layer framing — verify per camera
        roi_bottom            = 0.98,
        frangi_sigmas         = (1,),
        hough_ridge_threshold = 0.2,
        hough_min_length_frac = 0.4,
        hough_max_gap_frac    = 0.20,
        ref_y_frac            = 0.05,
        peak_min_dist         = 50,
        min_y_span_frac       = 0.15,
        skeletonize_mask      = False,
        skeleton_close        = (3, 3),
    ),
    "top": dict(
        roi_top               = 0.25,
        roi_bottom            = 0.40,
        frangi_sigmas         = (1, 2, 3),
        hough_ridge_threshold = 0.04,
        hough_min_length_frac = 0.30,
        hough_max_gap_frac    = 0.60,
        ref_y_frac            = 0.55,
        peak_min_dist         = 40,
        min_y_span_frac       = 0.55,
        skeletonize_mask      = True,
        skeleton_close        = (3, 3),   # try (1, 3) vertical-only if neighbours weld together
    ),
}


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── ROI (fraction of frame height: 0.0 = top, 1.0 = bottom)
    roi_top:    float = 0.57
    roi_bottom: float = 0.93

    # ── Frangi vesselness
    # sigma ≈ half the yarn pixel-width in the DOWNSCALED image.
    # 800 TEX yarn ≈ 3–4 px downscaled → start with (1, 2, 3).
    frangi_sigmas: tuple = (1, 2, 3)
    # False = bright ridges on dark BG (default).
    # True  = dark ridges on bright BG.
    black_ridges: bool = False

    # ── Mask thinning (top layer)
    # skeletonize_mask: collapse thick / duplicate ridges to a 1-px centreline
    #   before Hough.  Eliminates the parallel-duplicate segments a thick ridge
    #   produces (one yarn → many near-collinear lines → false doubles).
    # skeleton_close: morphology-close kernel (w, h) applied before skeletonising,
    #   to bridge sub-threshold gaps along a ridge.  (0, 0) = no close.
    #   Use a vertical-only (1, h) kernel if a general (w, h) kernel welds
    #   closely-spaced neighbouring yarns into one blob.
    skeletonize_mask: bool  = False
    skeleton_close:   tuple = (3, 3)

    # ── Hough line detection
    # Frangi output is normalised to [0,1]; pixels above this threshold are
    # treated as ridge pixels for the Hough transform.
    hough_ridge_threshold: float = 0.05

    # Minimum Hough accumulator votes to accept a line.
    # Lower → more sensitive, more false lines.  Start at 15.
    hough_threshold: int = 15

    # Minimum segment length as a fraction of ROI height (downscaled).
    # A complete yarn spans most of the ROI; a fallen end is much shorter.
    # With ROI height 350 px (downscaled), 0.25 → min 87 px.
    hough_min_length_frac: float = 0.25

    # Maximum gap within a yarn ridge that the Hough algorithm will bridge
    # (fraction of ROI height).  Covers fluff occlusions.
    hough_max_gap_frac: float = 0.04

    # Minimum vertical span a segment must cover (fraction of ROI height) to be
    # kept.  Rejects short noise/fluff stubs.  A complete yarn — even a steep
    # 70° edge yarn — spans most of the ROI vertically, so it survives.  Keep
    # LOW (0.15) for the bottom layer; raise (0.55) for the top layer where it
    # is the main noise-rejection lever (paired with a large max-gap so real
    # gapped yarns are bridged back to full span before this test).
    min_y_span_frac: float = 0.15

    # ── Canonical-x reference y
    # Yarns are identified by where their Hough line crosses this y-level
    # (expressed as a fraction of ROI height, 0=top 1=bottom).
    # Put it roughly in the middle of the visible yarn region.
    ref_y_frac: float = 0.55

    # ── Peak clustering (full-resolution pixels)
    # Two Hough segments within this x-distance at ref-y are merged into one
    # yarn detection.  Set to ~70–80% of inter-yarn spacing.
    peak_min_dist: int = 60

    # ── Reference count
    # ref_count_override > 0  →  skip auto-detect, use this value from frame 1.
    # ref_stabilise_frames    →  when auto-detecting, collect counts over this many
    #   frames and lock to the MAXIMUM seen.  Protects against a bad first frame
    #   (one occluded yarn) setting a permanently wrong reference.
    ref_count_override:   int = 0    # 0 = auto-detect
    ref_stabilise_frames: int = 10   # frames to observe before locking

    # ── Temporal tracking
    drift_gate_px:  int = 80  # max lateral shift per frame to still match a track
                               # steep yarns (70°) can shift 100+ px between frames
    break_frames:   int = 8    # consecutive missed frames → BREAK alert
                               # at 1 fps: 8 = 8-second persistence before alert
    min_track_hits: int = 3    # frames a track must be seen before becoming break-eligible
    smooth_x_window: int = 5   # frames averaged when reporting a confirmed yarn's x
                               # (removes residual canonical-x jitter; 1 = off)

    # Suppress a break event if any currently-detected yarn is within this
    # distance of the lost track's last known x.  Prevents false positives when
    # canonical-x drift causes a track to drop and re-spawn.  0 = disabled.
    suppress_radius_px: int = 80   # must be < half inter-yarn spacing

    # count_drop_frames is always equal to break_frames — see _check_count_drop()

    # ── Edge margins (full-res px) — suppress partially-visible edge yarns
    margin_left_px:  int = 0
    margin_right_px: int = 0

    # ── Speed
    subsample_width: int = 1920   # downscale ROI to this width before Frangi

    # ── Output
    annotate:  bool = False
    show:      bool = False
    log_csv:   bool = True
    save_diag: str  = ""
    skip:      int  = 0


# ── Binary mask preparation ─────────────────────────────────────────────────────
def prepare_binary(ridge: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Threshold the Frangi ridge map into the binary mask fed to HoughLinesP.

    If cfg.skeletonize_mask is set, optionally close small along-ridge gaps and
    then thin every ridge to a 1-px centreline.  This is the SINGLE source of
    truth for binarisation — detection (_hough_detect) and the diagnostic panels
    both call it, so the diagnostic always shows exactly what detection sees.
    """
    binary = ((ridge > cfg.hough_ridge_threshold) * 255).astype(np.uint8)
    if cfg.skeletonize_mask:
        kw, kh = int(cfg.skeleton_close[0]), int(cfg.skeleton_close[1])
        if kw > 0 and kh > 0:
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
        binary = (skeletonize(binary > 0).astype(np.uint8)) * 255
    return binary


# ── Yarn track ────────────────────────────────────────────────────────────────
class YarnTrack:
    _next_id: int = 1
    smooth_window: int = 5   # frames averaged for smooth_x; set by Detector from cfg

    def __init__(self, x: float) -> None:
        self.id      = YarnTrack._next_id
        YarnTrack._next_id += 1
        self.x       = x
        self.missing = 0
        self.broken  = False
        self._hist:  list[float] = [x]
        self._hits:  int         = 1

    def hit(self, x: float) -> None:
        self.x = x
        self.missing = 0
        self._hist.append(x)
        self._hits += 1

    def miss(self) -> None:
        self.missing += 1

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def smooth_x(self) -> int:
        # Windowed MEAN of recent canonical-x.  Yarns are physically static, so
        # this position should be constant; the per-frame variation is angle-fit
        # noise amplified by the long extrapolation to ref_y (worst for steep
        # right-edge yarns).  A windowed mean removes it.  Mean (not median) is
        # used because the residual after the joint line fit is symmetric jitter,
        # not outliers — mean suppresses symmetric noise better.  Raw self.x is
        # kept for break-timing/suppression; only reporting & association use this.
        w = max(1, YarnTrack.smooth_window)
        return int(np.mean(self._hist[-w:]))


# ── Detector ──────────────────────────────────────────────────────────────────
class Detector:
    """
    Stateful yarn-break detector.  Call process(bgr_frame) once per frame.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg        = cfg
        YarnTrack.smooth_window = max(1, cfg.smooth_x_window)
        self.tracks:    list[YarnTrack] = []
        self.ref_count: Optional[int]   = (
            cfg.ref_count_override if cfg.ref_count_override > 0 else None
        )
        self._fidx:       int       = 0
        self._frame_w:    int       = 0
        self._count_hist:      list[int] = []   # counts during stabilisation window
        self._below_ref_streak: int       = 0    # consecutive frames below ref_count
        self._last_full_pos:    list[float] = []  # positions from last full-count frame

    def reset(self) -> None:
        self.tracks.clear()
        self.ref_count          = None
        self._fidx              = 0
        self._frame_w           = 0
        self._count_hist        = []
        self._below_ref_streak  = 0
        self._last_full_pos     = []
        YarnTrack._next_id      = 1

    # ─────────────────────────────────────────────────────────────────────────
    def process(self, bgr: np.ndarray) -> dict:
        cfg = self.cfg
        self._fidx += 1
        H, W = bgr.shape[:2]
        if self._frame_w == 0:
            self._frame_w = W

        # 1 — greyscale + CLAHE + blur
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

        # 2 — ROI
        y0  = int(H * cfg.roi_top)
        y1  = int(H * cfg.roi_bottom)
        roi = gray[y0:y1]

        # 3 — downscale
        rH, rW = roi.shape
        scale  = min(1.0, cfg.subsample_width / rW)
        small  = cv2.resize(roi, (int(rW * scale), int(rH * scale))) if scale < 1.0 else roi
        sH, sW = small.shape

        # 4 — Frangi
        f32   = small.astype(np.float32) / 255.0
        ridge = frangi(f32, sigmas=cfg.frangi_sigmas, black_ridges=cfg.black_ridges)
        rmax  = ridge.max()
        if rmax > 0:
            ridge = ridge / rmax

        # 5 — Hough detection
        positions, segments = self._hough_detect(ridge, sW, sH, scale)

        # 6 — track
        breaks = self._track(positions)

        # Confirmed yarns: tracks seen >= min_track_hits frames AND present this
        # frame.  The raw per-frame cluster list (`positions`) flickers — a yarn
        # momentarily splitting into two clusters, or a one-frame spurious ridge,
        # appears and vanishes frame to frame, inflating the count and making the
        # bubbles jump even when the yarns are physically still.  A flickering
        # detection never survives min_track_hits consecutive-ish frames, so it is
        # excluded here.  This mirrors the persistence the break logic already
        # requires (break_frames) — we simply apply the same standard to what
        # counts as a present yarn.  Raw `positions`/`segments` are still returned
        # below for the diagnostic panels, so you can see raw-vs-confirmed.
        confirmed = [
            t for t in self.tracks
            if t.hits >= cfg.min_track_hits and t.missing == 0
        ]
        confirmed_x = sorted(t.smooth_x for t in confirmed)   # smoothed positions

        # 7 — stabilise reference count over first ref_stabilise_frames frames.
        #     Use the CONFIRMED count (flicker excluded) so the reference can't
        #     latch onto a one-frame phantom-inflated count.
        if self.ref_count is None and cfg.ref_count_override == 0:
            if len(self.tracks) >= 2:
                self._count_hist.append(len(confirmed_x))
            if len(self._count_hist) >= cfg.ref_stabilise_frames:
                self.ref_count = max(self._count_hist)
                log.info(
                    f"  Reference yarn count locked: {self.ref_count} "
                    f"(max over {cfg.ref_stabilise_frames} frames, "
                    f"counts seen: {sorted(set(self._count_hist))})"
                )

        # Sustained count-drop detection
        if cfg.break_frames > 0 and self.ref_count is not None:
            if len(positions) >= self.ref_count:
                self._below_ref_streak = 0
                self._last_full_pos = list(positions)  # snapshot of a good frame
            else:
                self._below_ref_streak += 1
            if self._below_ref_streak == cfg.break_frames:
                # Find which position from the last good frame is absent now.
                # This is far more accurate than "most misses" — it directly
                # compares what we had vs what we have now.
                suspect_x   = -1
                # tolerance = 40% of estimated inter-yarn spacing
                # — large enough to cover canonical-x drift, small enough
                #   not to accidentally match a neighbouring yarn
                est_spacing = (self._frame_w / self.ref_count) if self.ref_count else cfg.peak_min_dist * 2
                tol         = est_spacing * 0.40
                # Diff all positions — midpoints are always within frame
                valid_ref = list(self._last_full_pos)
                valid_cur = list(positions)
                if valid_ref and valid_cur:
                    for ref_x in valid_ref:
                        if not any(abs(ref_x - p) < tol for p in valid_cur):
                            suspect_x = int(ref_x)
                            break
                elif valid_ref:
                    suspect_x = int(valid_ref[0])

                # Suppress count-drop if any track already has enough misses to
                # trigger its own per-track break — they fire on different frames
                # so checking same-frame breaks is insufficient.
                per_track_fired  = any("type" not in b for b in breaks)
                track_will_break = any(t.missing >= cfg.break_frames
                                       for t in self.tracks)
                if per_track_fired or track_will_break:
                    log.warning(
                        f"  !!  COUNT DROP  "
                        f"{len(positions)}/{self.ref_count} yarns  "
                        f"for {self._below_ref_streak} consecutive frames  "
                        f"(suppressed — per-track break firing)"
                    )
                else:
                    # Genuine fallback: no per-track break active — fire count-drop
                    breaks.append({
                        "track_id":    -1,
                        "position_px": suspect_x,
                        "type":        "count_drop",
                    })
                    log.warning(
                        f"  !!  COUNT DROP  "
                        f"{len(positions)}/{self.ref_count} yarns  "
                        f"for {self._below_ref_streak} consecutive frames  "
                        f"missing x={suspect_x} px (vs last good frame)"
                    )

        return {
            "frame":        self._fidx,
            "yarn_count":   len(confirmed_x),          # confirmed tracks (flicker-free)
            "ref_count":    self.ref_count,
            "positions_px": [int(p) for p in confirmed_x],
            "breaks":       breaks,
            # diagnostic helpers
            "_raw_positions_px": [int(p) for p in positions],  # raw clusters this frame
            "_y0":          y0,
            "_y1":          y1,
            "_ref_y":       y0 + int(sH * cfg.ref_y_frac / scale),
            "_frame_w":     W,
            "_ridge":       ridge,
            "_segments":    segments,   # Nx4 int array: x0,y0,x1,y1 (downscaled ROI coords)
            "_scale":       scale,
            "_sH":          sH,
            "_sW":          sW,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _hough_detect(
        self,
        ridge:  np.ndarray,
        sW:     int,
        sH:     int,
        scale:  float,
    ) -> tuple[list[float], np.ndarray]:
        """
        Detect yarns as Hough line segments in the Frangi ridge map.

        Returns
        -------
        positions : list[float]
            Canonical x-positions (full-res px) — where each detected yarn
            crosses the reference y level.
        segments : np.ndarray, shape (N, 4)
            Raw Hough segments (x0, y0, x1, y1) in downscaled ROI coordinates,
            used for diagnostics.
        """
        cfg = self.cfg

        # Binarise (+ optional skeletonisation, per layer config)
        binary = prepare_binary(ridge, cfg)

        min_len = max(5, int(sH * cfg.hough_min_length_frac))
        max_gap = max(2, int(sH * cfg.hough_max_gap_frac))

        raw = cv2.HoughLinesP(
            binary,
            rho=1, theta=np.pi / 180,
            threshold=cfg.hough_threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        empty = np.empty((0, 4), dtype=int)
        if raw is None:
            return [], empty

        segments = raw.reshape(-1, 4)   # (N, 4): x0 y0 x1 y1  (downscaled coords)

        # Filter out short / near-horizontal segments by vertical span.
        # Threshold is layer-configurable (cfg.min_y_span_frac): low for the
        # bottom layer, high for the top layer.  Even a 70°-from-vertical edge
        # yarn spans most of the ROI vertically, so real yarns survive.
        min_y_span = max(2, int(sH * cfg.min_y_span_frac))
        segments   = segments[np.abs(segments[:, 3] - segments[:, 1]) >= min_y_span]
        if len(segments) == 0:
            return [], empty

        # Canonical x: where each segment's line crosses ref_y.
        # Used here only to GROUP segments belonging to the same yarn; the final
        # per-yarn x comes from a joint line fit below, not from these values.
        ref_y  = sH * cfg.ref_y_frac
        ref_xs: list[float] = []
        for x0, y0, x1, y1 in segments:
            dy = float(y1 - y0)
            if abs(dy) < 1e-3:
                ref_xs.append((float(x0) + float(x1)) / 2.0)
            else:
                t = (ref_y - float(y0)) / dy
                ref_xs.append(float(x0) + t * float(x1 - x0))

        # Cluster nearby ref_xs — multiple segments from the same yarn.
        # Gap-based rule: start a new cluster only when the gap to the IMMEDIATELY
        # PRECEDING segment exceeds min_gap.  (The old rule compared each segment
        # to the running cluster *mean*, so a yarn that emits several segments
        # with accumulating x-spread — common for steep right-edge yarns — would
        # cross the mean-distance threshold partway along and split into two
        # phantom yarns a small distance apart.  Comparing consecutive gaps
        # instead keeps such a yarn as one cluster, while still separating
        # genuinely distinct yarns at their true spacing.)
        min_gap_scaled = max(1, int(cfg.peak_min_dist * scale))
        order          = np.argsort(ref_xs)
        clusters: list[list[int]] = []   # each cluster = list of segment indices
        prev_x: Optional[float] = None
        for i in order:
            x = ref_xs[i]
            if clusters and prev_x is not None and (x - prev_x) < min_gap_scaled:
                clusters[-1].append(int(i))
            else:
                clusters.append([int(i)])
            prev_x = x

        # Per-yarn canonical x via a JOINT line fit, not median-of-crossings.
        #
        # WHY: each Hough segment's angle is quantised/noisy (~1°).  Extrapolating
        # each segment INDIVIDUALLY to ref_y (then taking the median) amplifies
        # that per-segment angle noise by the extrapolation arm BEFORE aggregating
        # — and at ref_y_frac≈0.05 the arm is nearly the full ROI height, so the
        # crossing-x of a STEEP (right-edge) yarn swings tens of px per frame for a
        # sub-degree wobble.  Instead we pool ALL endpoints of all segments in the
        # cluster and fit ONE line x = m·y + b by least squares, then evaluate it at
        # ref_y.  Averaging happens in slope-space over many points first, so the
        # single extrapolation rests on a stable angle.  This shrinks the jitter
        # most where it is worst (steep yarns), exactly matching the observed
        # left→right jitter gradient.
        #
        # x = m·y + b (regress x on y) handles near-vertical AND steep yarns
        # uniformly; a y = f(x) fit would blow up for vertical centre yarns.
        positions: list[float] = []
        for idxs in clusters:
            ys: list[float] = []
            xs: list[float] = []
            for k in idxs:
                x0, y0, x1, y1 = segments[k]
                ys.extend((float(y0), float(y1)))
                xs.extend((float(x0), float(x1)))
            ys_a = np.asarray(ys); xs_a = np.asarray(xs)
            if len(ys_a) >= 2 and (ys_a.max() - ys_a.min()) > 1e-3:
                # slope m and intercept b for x = m·y + b
                m, b = np.polyfit(ys_a, xs_a, 1)
                cross = m * ref_y + b
            else:
                # degenerate (all endpoints at same y) — fall back to mean x
                cross = float(xs_a.mean())
            positions.append(cross / scale)
        positions = sorted(positions)
        return positions, segments

    # ─────────────────────────────────────────────────────────────────────────
    def _track(self, positions: list[float]) -> list[dict]:
        cfg = self.cfg

        if not self.tracks:
            for x in positions:
                self.tracks.append(YarnTrack(x))
            return []

        if not positions:
            for t in self.tracks:
                t.miss()
            return self._check_breaks([])

        track_xs = np.array([t.x for t in self.tracks], dtype=float)
        det_xs   = np.array(positions,                  dtype=float)
        cost     = np.abs(track_xs[:, None] - det_xs[None, :])
        cost[cost > cfg.drift_gate_px] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks:    set[int] = set()
        matched_positions: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e8:
                self.tracks[r].hit(det_xs[c])
                matched_tracks.add(r)
                matched_positions.add(c)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.miss()

        for j, x in enumerate(positions):
            if j not in matched_positions:
                self.tracks.append(YarnTrack(x))

        return self._check_breaks(positions)

    # ─────────────────────────────────────────────────────────────────────────
    def _check_breaks(self, current_positions: list[float]) -> list[dict]:
        """
        Emit break events for tracks that have been missing for break_frames
        consecutive frames.

        Suppression: if any currently-detected yarn is within suppress_radius_px
        of the lost track's last known x, the break is suppressed.  This handles
        the common false-positive where Hough canonical-x drift causes the tracker
        to drop and re-spawn a yarn (new track = new green line) while the old
        track fires a break — both pointing at the same physical yarn.
        """
        cfg         = self.cfg
        right_limit = (self._frame_w - cfg.margin_right_px) if self._frame_w > 0 else 99_999
        breaks:     list[dict] = []

        for t in self.tracks:
            if t.missing == cfg.break_frames and not t.broken:
                in_zone     = cfg.margin_left_px <= t.x <= right_limit
                established = t.hits >= cfg.min_track_hits

                # Suppress if a live detection is nearby — same yarn, drifted ID
                nearby_xs = [
                    px for px in current_positions
                    if cfg.suppress_radius_px > 0 and abs(px - t.x) <= cfg.suppress_radius_px
                ]
                suppressed_nearby = len(nearby_xs) > 0

                if in_zone and established and not suppressed_nearby:
                    t.broken = True
                    breaks.append({"track_id": t.id, "position_px": int(t.x)})
                else:
                    reasons = []
                    if not established:
                        reasons.append(f"only {t.hits} hit(s)")
                    if not in_zone:
                        reasons.append("margin zone")
                    if suppressed_nearby:
                        reasons.append(f"active yarn nearby @ {[int(p) for p in nearby_xs]}")
                    log.debug(f"  track #{t.id} x={int(t.x)} suppressed ({', '.join(reasons)})")
        return breaks


# ── Frame annotation ──────────────────────────────────────────────────────────
def annotate_frame(frame: np.ndarray, result: dict) -> np.ndarray:
    out   = frame.copy()
    H, W  = out.shape[:2]
    y0    = result["_y0"]
    y1    = result["_y1"]
    ref_y = result["_ref_y"]
    scale = result["_scale"]
    segs  = result["_segments"]   # downscaled ROI coords
    ml    = result["_margin_left_px"] if "_margin_left_px" in result else 0
    mr    = result["_margin_right_px"] if "_margin_right_px" in result else 0

    # ROI box (cyan)
    cv2.rectangle(out, (0, y0), (W - 1, y1), (0, 220, 220), 2)
    # Reference y line (dim yellow)
    cv2.line(out, (0, ref_y), (W - 1, ref_y), (0, 160, 200), 1)

    # Draw Hough segments as actual angled lines (full-res coords)
    for x0s, y0s, x1s, y1s in segs:
        # convert downscaled ROI coords → full-res frame coords
        fx0 = int(x0s / scale)
        fy0 = y0 + int(y0s / scale)
        fx1 = int(x1s / scale)
        fy1 = y0 + int(y1s / scale)
        cv2.line(out, (fx0, fy0), (fx1, fy1), (0, 220, 0), 2)

    # Canonical-x markers on reference y:
    #   • small faint dots  = RAW clusters this frame (includes flicker)
    #   • bright bubbles     = CONFIRMED yarns (seen ≥min_track_hits frames)
    # Where a faint dot has no bubble, that detection was filtered as flicker.
    for x in result["_raw_positions_px"]:
        cv2.circle(out, (x, ref_y), 3, (0, 120, 60), -1)
    for x in result["positions_px"]:
        cv2.circle(out, (x, ref_y), 6, (0, 255, 100), -1)

    # Break markers — draw persistent breaks (stay for rest of video)
    # then highlight current-frame breaks on top
    all_to_draw = result.get("_persistent_breaks", result["breaks"])
    for brk in all_to_draw:
        x    = brk["position_px"]
        kind = brk.get("type", "track")
        col  = (0, 140, 255) if kind == "count_drop" else (0, 0, 255)
        lbl  = "COUNT DROP" if kind == "count_drop" else f"BREAK #{brk['track_id']}"
        if 0 <= x < W:
            cv2.line(out, (x, 0), (x, H), col, 3)
            cv2.putText(out, lbl, (max(0, x - 90), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        else:
            cv2.rectangle(out, (0, 0), (W - 1, 70), (0, 0, 100), -1)
            cv2.putText(out, f"{lbl}  (position outside frame)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 140, 255), 3)

    hud = (
        f"frame {result['frame']}  "
        f"yarns {result['yarn_count']}  "
        f"ref {result['ref_count'] or '?'}"
    )
    cv2.putText(out, hud, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return out


# ── Diagnostic PNG ────────────────────────────────────────────────────────────
def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


def build_diagnostic(frame: np.ndarray, result: dict, cfg: Config) -> np.ndarray:
    """
    Three-panel diagnostic image (1280 px wide, stacked vertically).

    Panel A  Downscaled frame  + ROI (cyan) + ref-y (dim yellow)
             + detected Hough segments drawn as angled lines (green)
             + canonical-x ticks (bright dots on ref-y line)
    Panel B  Frangi ridge map (INFERNO colourmap)
             + binary threshold overlay (white)
             + Hough segments (yellow)
    Panel C  Segment-length histogram  — use this to tune --hough-min-length
    """
    DW      = 1280
    H, W    = frame.shape[:2]
    ds      = DW / W
    ridge   = result["_ridge"]
    segs    = result["_segments"]
    scale   = result["_scale"]
    y0, y1  = result["_y0"], result["_y1"]
    ref_y   = result["_ref_y"]
    sH      = result["_sH"]
    sW      = result["_sW"]

    # ── Panel A ───────────────────────────────────────────────────────────────
    panel_a = cv2.resize(frame, (DW, max(1, int(H * ds))))
    aH      = panel_a.shape[0]
    ay0     = int(y0 * ds);  ay1 = int(y1 * ds)
    aref    = int(ref_y * ds)

    cv2.rectangle(panel_a, (0, ay0), (DW - 1, ay1), (0, 220, 220), 2)
    cv2.line(panel_a, (0, aref), (DW - 1, aref), (0, 140, 180), 1)

    for x0s, y0s, x1s, y1s in segs:
        fx0 = int(x0s / scale * ds)
        fy0 = ay0 + int(y0s / scale * ds)
        fx1 = int(x1s / scale * ds)
        fy1 = ay0 + int(y1s / scale * ds)
        cv2.line(panel_a, (fx0, fy0), (fx1, fy1), (0, 210, 0), 2)

    for x in result["_raw_positions_px"]:
        xd = int(x * ds)
        cv2.circle(panel_a, (xd, aref), 4, (0, 255, 100), -1)

    for brk in result["breaks"]:
        xd = int(brk["position_px"] * ds)
        cv2.line(panel_a, (xd, 0), (xd, aH), (0, 0, 255), 3)

    _label(panel_a,
           f"A  ROI (cyan) + ref-y (yellow) + {len(segs)} Hough segments (green) "
           f"+ {len(result['_raw_positions_px'])} raw clusters (dots), "
           f"{result['yarn_count']} confirmed")

    # ── Panel B: Frangi + binary + segments ───────────────────────────────────
    ridge_u8  = (ridge * 255).astype(np.uint8)
    ridge_col = cv2.applyColorMap(ridge_u8, cv2.COLORMAP_INFERNO)

    # White overlay where the ACTUAL detection mask fires (skeleton if enabled,
    # so a working skeleton shows as hairline-thin white — the quick visual check)
    bin_mask = prepare_binary(ridge, cfg) > 0
    ridge_col[bin_mask] = (255, 255, 255)

    # Reference y line
    ref_y_s = int(sH * cfg.ref_y_frac)
    cv2.line(ridge_col, (0, ref_y_s), (sW - 1, ref_y_s), (0, 140, 180), 1)

    # Hough segments (yellow)
    for x0s, y0s, x1s, y1s in segs:
        cv2.line(ridge_col, (int(x0s), int(y0s)), (int(x1s), int(y1s)), (0, 220, 255), 1)

    bH      = max(1, int(sH * DW / max(sW, 1)))
    panel_b = cv2.resize(ridge_col, (DW, bH))
    _label(panel_b,
           f"B  Frangi (INFERNO) + binary thr={cfg.hough_ridge_threshold:.2f} (white) "
           f"+ Hough segments (yellow), ref-y (dim)")

    # ── Panel C: segment length histogram ────────────────────────────────────
    # Run a SEPARATE Hough pass with minLineLength=1 so we see ALL potential
    # segments — including short fluff ones that get filtered in detection.
    # This makes the threshold line meaningful: bars to the left are filtered
    # out; bars to the right are counted as yarns.
    CHART_H   = 150
    panel_c   = np.zeros((CHART_H, DW, 3), dtype=np.uint8)
    min_len_px = max(5, int(sH * cfg.hough_min_length_frac))

    binary_diag = prepare_binary(ridge, cfg)   # same mask detection uses
    raw_all = cv2.HoughLinesP(
        binary_diag,
        rho=1, theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=1,          # no filter — show everything
        maxLineGap=max(2, int(sH * cfg.hough_max_gap_frac)),
    )

    if raw_all is not None:
        segs_all = raw_all.reshape(-1, 4)
        lengths  = np.sqrt(
            (segs_all[:, 2] - segs_all[:, 0]).astype(float) ** 2 +
            (segs_all[:, 3] - segs_all[:, 1]).astype(float) ** 2
        )
        max_len = max(lengths.max(), float(min_len_px) * 2, 1.0)
        n_bins  = 80
        hist, _ = np.histogram(lengths, bins=n_bins, range=(0, max_len))
        hist_max = max(hist.max(), 1)

        for i, count in enumerate(hist):
            bin_lo = i       * max_len / n_bins
            bin_hi = (i + 1) * max_len / n_bins
            x_left  = int(i       * DW / n_bins)
            x_right = int((i + 1) * DW / n_bins)
            bar_h   = int(count / hist_max * (CHART_H - 28))
            if bar_h > 0:
                # grey = filtered out (below min-length), green = counted as yarn
                colour = (60, 180, 60) if bin_lo >= min_len_px else (90, 90, 90)
                cv2.rectangle(panel_c,
                               (x_left, CHART_H - 1 - bar_h),
                               (max(x_left + 1, x_right - 1), CHART_H - 1),
                               colour, -1)

        # Min-length threshold line (blue)
        thr_x = int(min_len_px / max_len * DW)
        cv2.line(panel_c, (thr_x, 0), (thr_x, CHART_H - 1), (220, 80, 0), 2)
        cv2.putText(panel_c, "min-length", (max(0, thr_x - 2), 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 140, 20), 1)

        n_short = int((lengths < min_len_px).sum())
        n_long  = int((lengths >= min_len_px).sum())
        cv2.putText(panel_c,
                    f"grey={n_short} filtered  green={n_long} counted  "
                    f"(all segments before length filter)",
                    (8, CHART_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)

    _label(panel_c,
           f"C  ALL Hough segments before filtering  |  "
           f"grey=filtered  green=counted as yarn  |  "
           f"blue line = min-length ({cfg.hough_min_length_frac:.2f} = {min_len_px} px downscaled)")

    return np.vstack([panel_a, panel_b, panel_c])


def save_diagnostic(frame: np.ndarray, result: dict, cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, build_diagnostic(frame, result, cfg))


# ── Reference count detection ─────────────────────────────────────────────────
def detect_ref_count(video_path: str, cfg: Config, max_frames: int = 50) -> int:
    """
    Scan up to max_frames of a known-good reference video and return the MODE
    (most common yarn count) across all sampled frames.

    Why mode, not max:
      max returns 22 if Hough finds one spurious extra segment — a count nobody
      reaches in normal operation, so the break detector fires on every frame.
      Mode returns the count present in most frames: robust to both single-frame
      dropouts (too low) and one-off false extras (too high).
    """
    from collections import Counter
    quiet_cfg = Config(**{**cfg.__dict__,
                          "ref_count_override": 0,
                          "ref_stabilise_frames": 9999})
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open reference video: {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det    = Detector(quiet_cfg)
    counts: list[int] = []
    for _ in range(min(max_frames, max(total, 1))):
        ret, frame = cap.read()
        if not ret:
            break
        r = det.process(frame)
        if r["yarn_count"] > 0:
            counts.append(r["yarn_count"])
    cap.release()
    if not counts:
        raise RuntimeError(
            f"No yarns detected in {video_path} — check --roi-top/bottom and --sigmas"
        )
    from collections import Counter
    freq = Counter(counts)
    ref  = freq.most_common(1)[0][0]
    log.info(f"  Reference video : {os.path.basename(video_path)}")
    log.info(f"  Counts seen     : {dict(sorted(freq.items()))}  ->  ref_count = {ref} (mode)")
    return ref

# ── Tune mode ─────────────────────────────────────────────────────────────────
def tune_frame(path: str, frame_n: int, cfg: Config) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error(f"Cannot open: {path}"); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_n - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        log.error(f"Could not read frame {frame_n}"); return

    det    = Detector(cfg)
    result = det.process(frame)
    H, W   = frame.shape[:2]
    ridge  = result["_ridge"]
    segs   = result["_segments"]
    sH     = result["_sH"]

    print()
    print(f"═══ Tune: frame {frame_n}  ·  {os.path.basename(path)} ═══")
    print(f"  Frame          : {W} × {H} px")
    print(f"  ROI            : y = {result['_y0']}–{result['_y1']}  ({result['_y1'] - result['_y0']} px)")
    print(f"  Frangi max     : {ridge.max():.5f}" +
          ("  ✓" if ridge.max() > 1e-5 else "  ← LOW — try smaller --sigmas or --black-ridges"))
    print(f"  Sigmas         : {cfg.frangi_sigmas}")
    print(f"  Ridge thr      : {cfg.hough_ridge_threshold}  →  "
          f"{int((ridge > cfg.hough_ridge_threshold).sum())} px above threshold")
    print(f"  Skeletonize    : {cfg.skeletonize_mask}" +
          (f"  (close kernel {cfg.skeleton_close})" if cfg.skeletonize_mask else ""))
    print(f"  Min y-span     : {cfg.min_y_span_frac:.2f} × {sH} px = "
          f"{max(2, int(sH * cfg.min_y_span_frac))} px (downscaled)")
    raw_count = len(result["_raw_positions_px"])
    print(f"  Hough segments : {len(segs)} raw  →  {raw_count} clustered yarns")
    print(f"  (single frame: shows RAW clusters. In video runs the reported count is "
          f"CONFIRMED tracks — those seen ≥{cfg.min_track_hits} frames — which excludes flicker.)")
    min_len_px = max(5, int(sH * cfg.hough_min_length_frac))
    print(f"  Min length     : {cfg.hough_min_length_frac:.2f} × {sH} px = {min_len_px} px (downscaled)")
    print(f"  Ref y          : {cfg.ref_y_frac:.0%} of ROI  (full-res y = {result['_ref_y']})")
    if result["_raw_positions_px"]:
        print(f"  Yarn positions : {result['_raw_positions_px']}")
    else:
        print(f"  Yarn positions : none — check ROI, sigmas, ridge-thr, min-length")
    print()
    print("  Checklist:")
    print("    □  Panel A: green lines follow actual yarn angles (not vertical)")
    print("    □  Panel A: one green line per yarn, none duplicated")
    print("    □  Panel B: white glow sits on yarns, not on background")
    print("    □  Panel C: blue threshold line sits between short-noise bars and long-yarn bars")
    print("    □  Yarn count matches what you count by eye")
    print()

    if not cfg.save_diag and not cfg.show:
        print("  Tip: add  --save-diag ./diag  to write the PNG.")
        return

    diag = build_diagnostic(frame, result, cfg)

    if cfg.save_diag:
        os.makedirs(cfg.save_diag, exist_ok=True)
        out_path = os.path.join(cfg.save_diag, f"tune_frame{frame_n:04d}.png")
        cv2.imwrite(out_path, diag)
        print(f"  Diagnostic → {out_path}")

    if cfg.show:
        cv2.imshow("Tune (any key=close)", diag)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows(); sys.exit(0)
        cv2.destroyAllWindows()


# ── Per-source processing ─────────────────────────────────────────────────────
def process_source(source, cfg: Config, det: Detector, csv_writer=None) -> list[dict]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open: {source!r}"); return []

    is_file  = isinstance(source, str)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else -1
    fps      = cap.get(cv2.CAP_PROP_FPS) or 1.0
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_name = os.path.basename(str(source)) if is_file else f"camera:{source}"

    log.info(f"→ {src_name}  {W}×{H}  " +
             (f"{total} frames @ {fps:.1f} fps" if total > 0 else "live camera"))

    ann_writer = None
    ann_path   = None
    if cfg.annotate and is_file:
        import platform
        ann_path = str(source).rsplit(".", 1)[0] + "_annotated.mp4"
        if platform.system() == "Darwin":
            # macOS: try avc1 (H.264) first — most reliable at 4K
            ann_writer = cv2.VideoWriter(
                ann_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
            if not ann_writer.isOpened():
                # fallback to mp4v
                ann_writer = cv2.VideoWriter(
                    ann_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        else:
            ann_writer = cv2.VideoWriter(
                ann_path, cv2.VideoWriter_fourcc(*"X264"), fps, (W, H))
        if not ann_writer.isOpened():
            log.error(f"VideoWriter failed to open: {ann_path}")
            ann_writer = None
        else:
            log.info(f"Annotated video → {ann_path}")

    # Diag folder created on demand when first break fires (save_diagnostic handles makedirs)

    file_breaks:       list[dict] = []
    persistent_breaks: list[dict] = []   # all breaks fired so far — drawn on every frame
    frame_num = 0
    t0        = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if cfg.skip > 0 and (frame_num - 1) % (cfg.skip + 1) != 0:
            continue

        result = det.process(frame)
        # Accumulate fired breaks so annotation persists for rest of video
        persistent_breaks.extend(result["breaks"])
        result["_persistent_breaks"] = list(persistent_breaks)

        for brk in result["breaks"]:
            ts_sec = frame_num / fps
            log.warning(f"  ⚠  BREAK  track #{brk['track_id']}"
                        f"  x={brk['position_px']} px"
                        f"  t={ts_sec:.1f} s  frame {frame_num}")
            event = {
                "source":      src_name,
                "frame":       frame_num,
                "time_sec":    round(ts_sec, 2),
                "track_id":    brk["track_id"],
                "position_px": brk["position_px"],
            }
            file_breaks.append(event)
            if csv_writer:
                csv_writer.writerow(event)
            if cfg.save_diag:
                png = os.path.join(cfg.save_diag,
                                   f"break_f{frame_num:05d}_t{brk['track_id']}.png")
                save_diagnostic(frame, result, cfg, png)
                log.info(f"  Diagnostic → {png}")

        if cfg.annotate or cfg.show:
            annotated = annotate_frame(frame, result)
            if ann_writer:
                ann_writer.write(annotated)
            if cfg.show:
                cv2.imshow("Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if frame_num % 20 == 0 or (total > 0 and frame_num == total):
            pct = f"{100 * frame_num / total:5.1f}%  " if total > 0 else ""
            log.info(f"  {pct}frame {frame_num}" +
                     (f"/{total}" if total > 0 else "") +
                     f"  yarns {result['yarn_count']}" +
                     (f"  breaks {len(file_breaks)}" if file_breaks else ""))

    cap.release()
    if ann_writer:
        ann_writer.release()

    elapsed = time.monotonic() - t0
    log.info(f"  Done — {frame_num} frames  {len(file_breaks)} break event(s)  {elapsed:.1f} s")
    return file_breaks


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Yarn break detector v2 — Hough line edition, warping machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --video rec.mkv --tune-frame 1 --save-diag ./diag\n"
            "  %(prog)s --video rec.mkv --roi-top 0.57 --roi-bottom 0.93\n"
            "  %(prog)s --dir ~/recordings/\n"
            "  %(prog)s --camera 0\n"
        ),
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  metavar="PATH")
    src.add_argument("--dir",    metavar="DIR")
    src.add_argument("--camera", metavar="N", type=int)

    ap.add_argument("--tune-frame", metavar="N", type=int)

    # Layer preset — bundles the layer-dependent defaults below.  Any flag passed
    # explicitly overrides the preset.  See LAYER_PRESETS at the top of this file.
    ap.add_argument("--layer", choices=sorted(LAYER_PRESETS), default="bottom",
                    help="Yarn-layer preset (default: bottom). Sets ROI, ridge-thr, "
                         "min-length, max-gap, ref-y, min-dist, y-span, skeletonize.")

    # ROI  (default None → taken from --layer preset)
    ap.add_argument("--roi-top",    type=float, default=None, metavar="F")
    ap.add_argument("--roi-bottom", type=float, default=None, metavar="F")

    # Frangi
    ap.add_argument("--sigmas",      type=float, nargs="+", default=None, metavar="S",
                    help="Frangi scales in px (preset default: 1 2 3)")
    ap.add_argument("--black-ridges", action="store_true",
                    help="Detect dark yarns on bright background")

    # Mask thinning
    ap.add_argument("--skeletonize", action=argparse.BooleanOptionalAction, default=None,
                    help="Thin ridges to 1-px centrelines before Hough "
                         "(preset: off=bottom, on=top). Use --no-skeletonize to force off.")
    ap.add_argument("--skeleton-close", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Morphology-close kernel before skeletonising (preset: 3 3). "
                         "Try '1 3' if neighbouring yarns weld together.")

    # Hough  (default None → taken from --layer preset)
    ap.add_argument("--hough-ridge-thr", type=float, default=None, metavar="F",
                    help="Frangi threshold for binarisation (preset: bottom 0.05, top 0.04)")
    ap.add_argument("--hough-threshold", type=int,   default=15,   metavar="N",
                    help="Min Hough accumulator votes (default 15)")
    ap.add_argument("--hough-min-length", type=float, default=None, metavar="F",
                    help="Min segment length as fraction of ROI height (preset: bottom 0.25, top 0.30)")
    ap.add_argument("--hough-max-gap",    type=float, default=None, metavar="F",
                    help="Max gap in a ridge as fraction of ROI height (preset: bottom 0.10, top 0.60)")
    ap.add_argument("--ref-y",            type=float, default=None, metavar="F",
                    help="Reference y for canonical x (fraction of ROI, preset 0.55)")
    ap.add_argument("--min-y-span",       type=float, default=None, metavar="F",
                    help="Min vertical span to keep a segment (preset: bottom 0.15, top 0.55)")

    # Detection
    ap.add_argument("--min-dist", type=int,   default=None,   metavar="PX",
                    help="Min inter-yarn x spacing for clustering, full-res px (preset: bottom 60, top 40)")

    # Tracking
    ap.add_argument("--ref-video",     type=str, default=None, metavar="PATH",
                    help="Known-good video — sets ref yarn count before processing")
    ap.add_argument("--ref-count",     type=int, default=0,    metavar="N",
                    help="Hardcode expected yarn count (0 = auto-detect)")
    ap.add_argument("--ref-stabilise", type=int, default=10,   metavar="N",
                    help="Frames to observe before locking ref count (default 10)")
    ap.add_argument("--drift",            type=int, default=150, metavar="PX",
                    help="Max lateral drift per frame, full-res px (default 150)")
    ap.add_argument("--break-frames",     type=int, default=21,   metavar="N",
                    help="Consecutive missed frames → BREAK (default 8)")
    ap.add_argument("--min-hits",         type=int, default=3,   metavar="N",
                    help="Min track detections before break-eligible (default 3)")
    ap.add_argument("--smooth-x-window",  type=int, default=7,   metavar="N",
                    help="Frames averaged when reporting a yarn's x position; "
                         "removes canonical-x jitter (1 = off, default 5)")
    ap.add_argument("--suppress-radius",  type=int, default=80, metavar="PX",
                    help="Suppress break if live yarn within this distance (default 80; 0=off)")

    # Margins
    ap.add_argument("--margin-left",  type=int, default=0, metavar="PX")
    ap.add_argument("--margin-right", type=int, default=0, metavar="PX")

    # Speed / output
    ap.add_argument("--subsample-width", type=int,  default=1920, metavar="PX")
    ap.add_argument("--skip",     type=int,  default=0,  metavar="N",
                    help="Process every (N+1)th frame (default 0 = every frame)")
    ap.add_argument("--annotate",  action="store_true")
    ap.add_argument("--show",      action="store_true")
    ap.add_argument("--no-csv",    action="store_true")
    ap.add_argument("--save-diag", metavar="DIR", default="",
                    help="Save 3-panel diagnostic PNG on every break event")

    args = ap.parse_args()

    # ── Resolve layer preset ────────────────────────────────────────────────────
    # For each layer-dependent parameter: use the explicit CLI value if the user
    # passed one (not None), otherwise fall back to the selected layer's preset.
    preset = LAYER_PRESETS[args.layer]
    def pick(val, key):
        return preset[key] if val is None else val

    cfg = Config(
        roi_top               = pick(args.roi_top,           "roi_top"),
        roi_bottom            = pick(args.roi_bottom,        "roi_bottom"),
        frangi_sigmas         = tuple(pick(args.sigmas,      "frangi_sigmas")),
        black_ridges          = args.black_ridges,
        skeletonize_mask      = pick(args.skeletonize,       "skeletonize_mask"),
        skeleton_close        = tuple(pick(args.skeleton_close, "skeleton_close")),
        hough_ridge_threshold = pick(args.hough_ridge_thr,   "hough_ridge_threshold"),
        hough_threshold       = args.hough_threshold,
        hough_min_length_frac = pick(args.hough_min_length,  "hough_min_length_frac"),
        hough_max_gap_frac    = pick(args.hough_max_gap,     "hough_max_gap_frac"),
        min_y_span_frac       = pick(args.min_y_span,        "min_y_span_frac"),
        ref_y_frac            = pick(args.ref_y,             "ref_y_frac"),
        peak_min_dist         = pick(args.min_dist,          "peak_min_dist"),
        ref_count_override    = args.ref_count,
        ref_stabilise_frames  = args.ref_stabilise,
        drift_gate_px         = args.drift,
        break_frames          = args.break_frames,
        min_track_hits        = args.min_hits,
        smooth_x_window       = args.smooth_x_window,
        suppress_radius_px    = args.suppress_radius,
        margin_left_px        = args.margin_left,
        margin_right_px       = args.margin_right,
        subsample_width       = max(args.subsample_width, 100),
        annotate              = args.annotate,
        show                  = args.show,
        log_csv               = not args.no_csv,
        save_diag             = args.save_diag,
        skip                  = args.skip,
    )
    log.info(f"Layer preset: {args.layer}  "
             f"(ROI {cfg.roi_top}-{cfg.roi_bottom}, ridge-thr {cfg.hough_ridge_threshold}, "
             f"max-gap {cfg.hough_max_gap_frac}, min-len {cfg.hough_min_length_frac}, "
             f"y-span {cfg.min_y_span_frac}, skeletonize {cfg.skeletonize_mask})")

    # ── Tune mode ──────────────────────────────────────────────────────────────
    if args.tune_frame is not None:
        if args.camera is not None:
            log.error("--tune-frame requires --video or --dir"); sys.exit(1)
        path = args.video
        if args.dir:
            files = sorted(Path(args.dir).glob("kurokesu_*.mkv"))
            if not files:
                log.error(f"No kurokesu_*.mkv in {args.dir}"); sys.exit(1)
            path = str(files[0])
        tune_frame(path, args.tune_frame, cfg)
        return

    # ── Source list ────────────────────────────────────────────────────────────
    if args.video:
        sources: list = [args.video]
    elif args.dir:
        sources = sorted(str(p) for p in Path(args.dir).glob("kurokesu_*.mkv"))
        if not sources:
            log.error(f"No kurokesu_*.mkv in {args.dir}"); sys.exit(1)
        log.info(f"Found {len(sources)} file(s)")
    else:
        sources = [args.camera]

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_file = csv_writer = None
    if cfg.log_csv and args.camera is None:
        log_dir    = Path(str(sources[0])).parent
        csv_path   = log_dir / "breaks.csv"
        csv_file   = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["source", "frame", "time_sec", "track_id", "position_px"],
        )
        csv_writer.writeheader()
        log.info(f"Break log → {csv_path}")

    # ── Reference video ───────────────────────────────────────────────────────
    if args.ref_video and args.ref_count == 0:
        log.info(f"Learning reference count from: {args.ref_video}")
        ref_n = detect_ref_count(args.ref_video, cfg)
        cfg   = Config(**{**cfg.__dict__, "ref_count_override": ref_n})

    # ── Run ────────────────────────────────────────────────────────────────────
    det        = Detector(cfg)
    all_breaks: list[dict] = []

    try:
        for s in sources:
            all_breaks.extend(process_source(s, cfg, det, csv_writer))
    finally:
        if csv_file:
            csv_file.close()
        if cfg.show:
            cv2.destroyAllWindows()

    print()
    print("═" * 60)
    print(f"  Total break events: {len(all_breaks)}")
    for b in all_breaks:
        print(f"    [{b['source']}]  frame {b['frame']}  t={b['time_sec']} s  "
              f"x={b['position_px']} px  track #{b['track_id']}")
    if cfg.log_csv and csv_file:
        print(f"  Log → {csv_file.name}")
    print("═" * 60)


if __name__ == "__main__":
    main()

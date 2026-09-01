"""Multi-Modal Defensive Avionics AI Simulator — Full-Screen Engineering HUD Dashboard.

Streamlit tactical decision support interface featuring dark navy/cyan glassmorphism,
real-time multi-sensor awareness, AI classification, PPO policy, collaborative sensing,
and a high-performance live camera WebRTC HUD with asynchronous background inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import psutil
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from defensive_avionics.integration.orchestrator import (
    ComprehensiveSnapshot,
    IntegratedOrchestrator,
)
from defensive_avionics.signal.pipeline import SignalPipeline
from defensive_avionics.vision.live_camera import LiveCameraProcessor
from defensive_avionics.vision.pipeline import VisionPipeline

# Configure page layout and HUD theme
st.set_page_config(
    page_title="Defensive Avionics AI Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom HUD Styling (Full-Screen Dark Navy/Cyan Tactical Aesthetics)
CUSTOM_CSS = """
<style>
    /* Safely hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}

    .stApp {
        background-color: #030912;
        color: #E8F6FF;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 1.2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 100%;
    }

    /* Fixed HUD Status Strip */
    .hud-status-strip {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        background: #071625;
        border: 1px solid #0B526D;
        border-top: 2px solid #00E5FF;
        border-radius: 4px;
        padding: 8px 16px;
        margin-bottom: 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.78rem;
        box-shadow: 0 4px 16px rgba(0,229,255,0.06);
    }

    .hud-badge {
        font-family: 'Consolas', monospace;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 0.5px;
    }
    .badge-active {
        background: rgba(46,230,166,0.15);
        color: #2EE6A6;
        border: 1px solid rgba(46,230,166,0.4);
    }
    .badge-caution {
        background: rgba(246,183,60,0.15);
        color: #F6B73C;
        border: 1px solid rgba(246,183,60,0.4);
    }
    .badge-critical {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border: 1px solid rgba(239,68,68,0.4);
    }
    .badge-mode {
        background: rgba(0,229,255,0.12);
        color: #00E5FF;
        border: 1px solid rgba(0,229,255,0.35);
    }

    /* Standardized HUD Panels */
    .hud-panel {
        background: #071625;
        border: 1px solid #0B526D;
        border-radius: 4px;
        padding: 12px 14px;
        height: 100%;
        box-shadow: inset 0 0 16px rgba(0,229,255,0.02);
    }
    .panel-title {
        font-family: 'Consolas', 'Segoe UI', monospace;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #00E5FF;
        border-bottom: 1px solid #0B526D;
        padding-bottom: 6px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .panel-subtitle {
        font-family: 'Consolas', monospace;
        font-size: 0.70rem;
        color: #91A8B8;
        font-weight: normal;
    }

    .hud-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px dashed rgba(11,82,109,0.4);
        font-family: 'Consolas', monospace;
        font-size: 0.80rem;
    }
    .hud-metric-label {
        color: #91A8B8;
    }
    .hud-metric-value {
        font-weight: 600;
        color: #00E5FF;
    }

    /* Live Camera Frame Container */
    .camera-viewport-card {
        background: #040E1A;
        border: 1px solid #0B526D;
        border-radius: 4px;
        padding: 6px;
        position: relative;
    }

    .hud-footer {
        text-align: center;
        font-family: 'Consolas', monospace;
        font-size: 0.80rem;
        letter-spacing: 1.5px;
        color: #F6B73C;
        background: rgba(246,183,60,0.06);
        border: 1px solid rgba(246,183,60,0.25);
        border-radius: 4px;
        padding: 8px 14px;
        margin-top: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_orchestrator(seed: int = 42) -> IntegratedOrchestrator:
    """Instantiate and cache the heavy ML models and orchestrator."""
    return IntegratedOrchestrator(
        seed=seed,
        difficulty="medium",
        device="cpu",
    )


@st.cache_resource
def get_camera_processor() -> LiveCameraProcessor:
    """Instantiate and cache the thread-safe live camera processor once."""
    return LiveCameraProcessor(
        model_mode="classroom",
        confidence=0.20,
        image_size=320,
        device="cpu",
    )


def initialize_session_state() -> None:
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "medium"
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "vision_input_mode" not in st.session_state:
        st.session_state.vision_input_mode = "Live Camera"
    if "camera_model_mode" not in st.session_state:
        st.session_state.camera_model_mode = "Classroom Objects"
    if "cam_confidence_threshold" not in st.session_state:
        st.session_state.cam_confidence_threshold = 0.20
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = get_orchestrator(st.session_state.seed)
    if "latest_snapshot" not in st.session_state:
        st.session_state.latest_snapshot = st.session_state.orchestrator.step()
    if "area_history" not in st.session_state:
        st.session_state.area_history = [0.02]


initialize_session_state()
orchestrator: IntegratedOrchestrator = st.session_state.orchestrator
camera_processor: LiveCameraProcessor = get_camera_processor()


# --- TOP CONTROL BAR ---
c_col1, c_col2, c_col3, c_col4, c_col5, c_col6 = st.columns(
    [2.2, 1.2, 0.9, 0.9, 0.9, 0.9]
)

with c_col1:
    header_html = """
    <div style="padding:4px 0;">
        <div style="font-family:'Consolas','Segoe UI',monospace; font-size:1.15rem;
        font-weight:800; color:#00E5FF; letter-spacing:1.5px;">
            MULTI-MODAL DEFENSIVE AVIONICS AI SIMULATOR
        </div>
        <div style="font-size:0.72rem; color:#91A8B8; font-family:'Consolas',monospace;">
            INTEGRATED TACTICAL DECISION SUPPORT & MULTI-SENSOR AWARENESS
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

with c_col2:
    mode_choice = st.radio(
        "Vision Input Source",
        ["Live Camera", "Synthetic Demo"],
        index=0 if st.session_state.vision_input_mode == "Live Camera" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode_choice != st.session_state.vision_input_mode:
        st.session_state.vision_input_mode = mode_choice
        st.rerun()

with c_col3:
    button_label = "▶ STEP (+1s)" if not st.session_state.sim_running else "⏸ PAUSE"
    if st.button(button_label, use_container_width=True):
        live_obs = (
            camera_processor.to_observation()
            if st.session_state.vision_input_mode == "Live Camera"
            else None
        )
        st.session_state.latest_snapshot = orchestrator.step(live_observation=live_obs)
        st.session_state.step_count += 1
        st.session_state.area_history.append(
            0.02 + 0.015 * (st.session_state.step_count % 8)
        )
        if len(st.session_state.area_history) > 20:
            st.session_state.area_history.pop(0)
        st.rerun()

with c_col4:
    if st.button("🔄 RESET", use_container_width=True):
        camera_processor.reset()
        st.session_state.latest_snapshot = orchestrator.reset(
            seed=st.session_state.seed,
            difficulty=st.session_state.difficulty,
        )
        st.session_state.step_count = 0
        st.session_state.area_history = [0.02]
        st.rerun()

with c_col5:
    diff_choice = st.selectbox(
        "Difficulty Preset",
        ["low", "medium", "high"],
        index=["low", "medium", "high"].index(st.session_state.difficulty),
        label_visibility="collapsed",
    )
    if diff_choice != st.session_state.difficulty:
        st.session_state.difficulty = diff_choice
        orchestrator.scenario_engine.difficulty = diff_choice
        st.session_state.latest_snapshot = orchestrator.reset(difficulty=diff_choice)
        st.rerun()

with c_col6:
    seed_input = st.number_input(
        "Simulation Seed",
        min_value=1,
        max_value=9999,
        value=st.session_state.seed,
        step=1,
        label_visibility="collapsed",
    )
    if seed_input != st.session_state.seed:
        st.session_state.seed = seed_input
        st.session_state.latest_snapshot = orchestrator.reset(seed=seed_input)
        st.rerun()

snapshot: ComprehensiveSnapshot = st.session_state.latest_snapshot

# Header status badges
status_badge_class = (
    "badge-critical"
    if snapshot.status == "critical"
    else ("badge-caution" if snapshot.status == "caution" else "badge-active")
)

threat_count = sum(
    1 for o in snapshot.scenario.objects if o.urgency in {"approaching", "critical"}
)

mode_display_text = (
    "LIVE CAMERA PROTOTYPE"
    if st.session_state.vision_input_mode == "Live Camera"
    else "OFFLINE SIMULATION"
)

# Render Fixed Status Strip
cam_snap = camera_processor.get_state()
status_strip_html = f"""
<div class="hud-status-strip">
    <div><strong>MODE:</strong> <span class="hud-badge badge-mode">{mode_display_text}</span></div>
    <div><strong>STATUS:</strong>
        <span class="hud-badge {status_badge_class}">{snapshot.status.upper()}</span>
    </div>
    <div><strong>MODEL:</strong> <span style="color:#26A7FF;">GENERIC YOLO NANO</span></div>
    <div><strong>SIM TIME:</strong>
        <span style="color:#00E5FF;">T+{snapshot.sim_time_sec:05.1f}s</span>
    </div>
    <div><strong>FRAME:</strong> <span style="color:#00E5FF;">#{snapshot.frame_id:04d}</span></div>
    <div><strong>LATENCY:</strong>
        <span style="color:#E8F6FF;">{cam_snap.inference_ms:02.0f} ms</span>
    </div>
    <div><strong>THREATS:</strong> <span style="color:#F6B73C;">{threat_count}</span></div>
</div>
"""
st.markdown(status_strip_html, unsafe_allow_html=True)


# ==========================================
# ROW 1: SIGNAL ANALYSIS | SCENARIO RADAR | VISION ANALYSIS
# ==========================================
row1_col1, row1_col2, row1_col3 = st.columns(3)

# 1. TOP-LEFT: SIGNAL ANALYSIS
with row1_col1:
    panel_signal_html = f"""
    <div class="hud-panel">
        <div class="panel-title">
            <span>📡 SIGNAL ANALYSIS</span>
            <span class="panel-subtitle">{snapshot.signal_source_mode}</span>
        </div>
    </div>
    """
    st.markdown(panel_signal_html, unsafe_allow_html=True)

    mod = snapshot.signal.label
    iq_data = orchestrator.signal_pipeline.generate_synthetic_iq(
        modulation=mod if mod in ["QPSK", "BPSK", "8PSK"] else "QPSK",
        snr_db=snapshot.signal.snr_db or 10.0,
        seed=st.session_state.seed + snapshot.frame_id,
    )

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(4.8, 3.4))
    fig.patch.set_facecolor("#071625")

    ax_wave.set_facecolor("#030912")
    ax_wave.plot(iq_data[0][:64], color="#00E5FF", label="I (In-Phase)", linewidth=1.2)
    ax_wave.plot(
        iq_data[1][:64], color="#26A7FF", label="Q (Quad)", linewidth=1.0, linestyle="--"
    )
    ax_wave.set_title("I/Q Baseband Waveform", color="#91A8B8", fontsize=8)
    ax_wave.tick_params(colors="#91A8B8", labelsize=7)
    ax_wave.grid(alpha=0.15, color="#0B526D")
    ax_wave.legend(
        loc="upper right",
        fontsize=6,
        facecolor="#071625",
        edgecolor="#0B526D",
        labelcolor="#E8F6FF",
    )

    ax_spec.set_facecolor("#030912")
    freqs, times, spec = SignalPipeline.compute_spectrogram(iq_data, nperseg=32)
    ax_spec.pcolormesh(times, freqs, spec, shading="gouraud", cmap="plasma")
    ax_spec.set_title("STFT Power Spectrogram (dB)", color="#91A8B8", fontsize=8)
    ax_spec.tick_params(colors="#91A8B8", labelsize=7)
    ax_spec.set_ylabel("Freq", color="#91A8B8", fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    snr_val = snapshot.signal.snr_db or 8.0
    sig_info_html = f"""
    <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
    padding:8px 12px; margin-top:8px;">
        <div class="hud-metric-row">
            <span class="hud-metric-label">MODULATION CLASS:</span>
            <span class="hud-metric-value" style="font-size:0.95rem; color:#00E5FF;">
                {snapshot.signal.label} ({snapshot.signal.confidence:.0%})
            </span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">ESTIMATED SNR:</span>
            <span class="hud-metric-value">{snr_val:0.1f} dB</span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">RECEIVER STATUS:</span>
            <span class="hud-metric-value" style="color:#2EE6A6;">LOCK ACTIVE</span>
        </div>
    </div>
    """
    st.markdown(sig_info_html, unsafe_allow_html=True)


# 2. TOP-CENTER: SYNTHETIC SCENARIO (RADAR-STYLE POLAR)
with row1_col2:
    panel_scenario_html = """
    <div class="hud-panel">
        <div class="panel-title">
            <span>🌐 SYNTHETIC SCENARIO</span>
            <span class="panel-subtitle">RADAR TELEMETRY</span>
        </div>
    </div>
    """
    st.markdown(panel_scenario_html, unsafe_allow_html=True)

    fig_radar = plt.figure(figsize=(4.8, 3.4))
    fig_radar.patch.set_facecolor("#071625")
    ax_radar = fig_radar.add_subplot(111, polar=True)
    ax_radar.set_facecolor("#030912")

    ax_radar.set_theta_zero_location("N")
    ax_radar.set_theta_direction(-1)
    ax_radar.set_ylim(0, 0.6)
    ax_radar.set_yticks([0.15, 0.30, 0.45, 0.60])
    ax_radar.set_yticklabels(
        ["0.25", "0.50", "0.75", "1.00"], color="#00E5FF66", fontsize=6
    )
    ax_radar.tick_params(colors="#00E5FF66", labelsize=7)
    ax_radar.grid(color="#0B526D", alpha=0.35, linestyle=":")

    ax_radar.scatter(
        [0], [0], color="#00E5FF", s=140, marker="^", zorder=5, label="Ownship"
    )

    cat_colors = {
        "friendly": "#2EE6A6",
        "neutral": "#00E5FF",
        "unknown": "#F6B73C",
        "resource": "#A855F7",
    }
    cat_markers = {
        "friendly": "o",
        "neutral": "s",
        "unknown": "D",
        "resource": "P",
    }

    for obj in snapshot.scenario.objects:
        theta = math.radians(obj.bearing_deg)
        dist = obj.distance_from_ownship
        color = (
            "#EF4444"
            if obj.urgency == "critical"
            else cat_colors.get(obj.category, "#00E5FF")
        )
        marker = cat_markers.get(obj.category, "o")

        ax_radar.scatter(
            [theta],
            [dist],
            color=color,
            s=70,
            marker=marker,
            edgecolors="#ffffff",
            linewidths=0.7,
            zorder=4,
        )
        ax_radar.text(
            theta,
            dist + 0.04,
            obj.id,
            color=color,
            fontsize=7,
            fontweight="bold",
            ha="center",
        )

    plt.tight_layout()
    st.pyplot(fig_radar)
    plt.close()

    contact_count = len(snapshot.scenario.objects)
    radar_info_html = f"""
    <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
    padding:8px 12px; margin-top:8px;">
        <div style="display:flex; justify-content:space-between; font-size:0.75rem;
        font-family:'Consolas',monospace;">
            <span style="color:#2EE6A6;">● FRIENDLY</span>
            <span style="color:#00E5FF;">■ NEUTRAL</span>
            <span style="color:#F6B73C;">◆ UNKNOWN</span>
            <span style="color:#A855F7;">✚ RESOURCE</span>
        </div>
        <div class="hud-metric-row" style="margin-top:4px;">
            <span class="hud-metric-label">TRACKED TARGETS:</span>
            <span class="hud-metric-value">{contact_count} CONTACTS</span>
        </div>
    </div>
    """
    st.markdown(radar_info_html, unsafe_allow_html=True)


# 3. TOP-RIGHT: VISION ANALYSIS (SYNTHETIC OR LIVE CAMERA)
with row1_col3:
    is_live_mode = st.session_state.vision_input_mode == "Live Camera"
    vis_subtitle = "LIVE WEBRTC WEBCAM" if is_live_mode else snapshot.vision_source_mode

    panel_vision_html = f"""
    <div class="hud-panel">
        <div class="panel-title">
            <span>📷 VISION ANALYSIS</span>
            <span class="panel-subtitle">{vis_subtitle}</span>
        </div>
    </div>
    """
    st.markdown(panel_vision_html, unsafe_allow_html=True)

    if not is_live_mode:
        shape_types = ["triangle", "diamond", "circle"]
        target_shape = shape_types[snapshot.frame_id % len(shape_types)]
        scale = 0.8 + 0.2 * (snapshot.frame_id % 4)

        raw_frame, box = VisionPipeline.generate_synthetic_sky_frame(
            image_size=320,
            shape_type=target_shape,
            scale=scale,
            seed=st.session_state.seed + snapshot.frame_id,
        )

        annotated_frame, detections, estimate = (
            orchestrator.vision_pipeline.process_frame(raw_frame)
        )
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        vis_c1, vis_c2 = st.columns([1.2, 1])
        with vis_c1:
            st.image(
                frame_rgb,
                use_container_width=True,
                caption=f"Synthetic Sky: {target_shape.upper()}",
            )

        with vis_c2:
            fig_area, ax_area = plt.subplots(figsize=(2.2, 2.0))
            fig_area.patch.set_facecolor("#071625")
            ax_area.set_facecolor("#030912")
            ax_area.plot(
                st.session_state.area_history,
                color="#00E5FF",
                marker="o",
                markersize=3,
                linewidth=1.5,
            )
            ax_area.set_title("Box Expansion Trend", color="#91A8B8", fontsize=7)
            ax_area.tick_params(colors="#91A8B8", labelsize=6)
            ax_area.grid(alpha=0.15, color="#0B526D")
            plt.tight_layout()
            st.pyplot(fig_area)
            plt.close()

        trend_color = (
            "#EF4444"
            if estimate.trend == "rapid_growth"
            else ("#F6B73C" if estimate.trend == "growing" else "#00E5FF")
        )
        trend_text = estimate.trend.replace("_", " ").upper()
        vision_info_html = f"""
        <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
        padding:8px 12px; margin-top:8px;">
            <div class="hud-metric-row">
                <span class="hud-metric-label">APPROACH TREND:</span>
                <span class="hud-metric-value" style="color:{trend_color}; font-weight:bold;">
                    {trend_text}
                </span>
            </div>
            <div class="hud-metric-row">
                <span class="hud-metric-label">GROWTH RATE:</span>
                <span class="hud-metric-value">{estimate.relative_growth:+.2%}/step</span>
            </div>
            <div style="font-size:0.68rem; color:#91A8B8; margin-top:4px;">
                * Normalized expansion trend — not a physical distance estimate.
            </div>
        </div>
        """
        st.markdown(vision_info_html, unsafe_allow_html=True)

    else:
        # Live WebRTC Camera Mode
        cam_c1, cam_c2 = st.columns([1.1, 1.0])
        with cam_c1:
            cam_model_choice = st.selectbox(
                "Detection Target Class",
                ["Classroom Objects", "Synthetic Aerial Object"],
                index=(
                    0
                    if st.session_state.camera_model_mode == "Classroom Objects"
                    else 1
                ),
                key="cam_model_select",
                label_visibility="collapsed",
            )
            if cam_model_choice != st.session_state.camera_model_mode:
                st.session_state.camera_model_mode = cam_model_choice
                camera_processor.set_model_mode(
                    "classroom"
                    if cam_model_choice == "Classroom Objects"
                    else "synthetic"
                )

        with cam_c2:
            conf_slider = st.slider(
                "Confidence Threshold",
                min_value=0.15,
                max_value=0.60,
                value=float(st.session_state.cam_confidence_threshold),
                step=0.05,
                label_visibility="collapsed",
            )
            if conf_slider != st.session_state.cam_confidence_threshold:
                st.session_state.cam_confidence_threshold = conf_slider
                camera_processor.set_confidence(conf_slider)

        webrtc_ctx = webrtc_streamer(
            key="defensive-avionics-live-camera",
            video_frame_callback=camera_processor.video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            camera_processor.reset()

        # Telemetry panel rendered as auto-refreshing fragment
        @st.fragment(run_every=0.8)
        def render_live_telemetry() -> None:
            cam_state = camera_processor.get_state()
            cam_status_text = (
                "STREAMING" if webrtc_ctx.state.playing else "STANDBY / STOPPED"
            )
            cam_status_color = "#2EE6A6" if webrtc_ctx.state.playing else "#91A8B8"

            trend_color = (
                "#EF4444"
                if cam_state.trend == "rapid_growth"
                else (
                    "#F6B73C"
                    if cam_state.trend == "growing"
                    else ("#91A8B8" if cam_state.is_stale else "#00E5FF")
                )
            )
            trend_text = (
                "STALE RESULT"
                if cam_state.is_stale
                else cam_state.trend.replace("_", " ").upper()
            )

            conf_display = (
                f"{cam_state.confidence * 100:.1f}% (raw: {cam_state.raw_confidence * 100:.1f}%)"
                if cam_state.confidence > 0
                else "--"
            )

            live_telemetry_html = f"""
            <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
            padding:8px 12px; margin-top:8px;">
                <div class="hud-metric-row">
                    <span class="hud-metric-label">CAMERA STATUS:</span>
                    <span class="hud-metric-value"
                          style="color:{cam_status_color}; font-weight:bold;">
                        ● {cam_status_text}
                    </span>
                </div>
                <div class="hud-metric-row">
                    <span class="hud-metric-label">PRIMARY TARGET:</span>
                    <span class="hud-metric-value" style="color:#00E5FF; font-weight:bold;">
                        {cam_state.label.upper()}
                    </span>
                </div>
                <div class="hud-metric-row">
                    <span class="hud-metric-label">CONFIDENCE (EMA):</span>
                    <span class="hud-metric-value">{conf_display}</span>
                </div>
                <div class="hud-metric-row">
                    <span class="hud-metric-label">APPROACH TREND:</span>
                    <span class="hud-metric-value" style="color:{trend_color}; font-weight:bold;">
                        {trend_text}
                    </span>
                </div>
                <div class="hud-metric-row">
                    <span class="hud-metric-label">GROWTH RATE / AREA:</span>
                    <span class="hud-metric-value">
                        {cam_state.relative_growth:+.2%}/s | {cam_state.area_ratio:.3f}
                    </span>
                </div>
                <div class="hud-metric-row">
                    <span class="hud-metric-label">FPS / LATENCY / AGE:</span>
                    <span class="hud-metric-value">
                        {cam_state.fps:04.1f} FPS | {cam_state.inference_ms:02.0f}ms
                        | {cam_state.result_age_ms:02.0f}ms
                    </span>
                </div>
                <div style="font-size:0.68rem; color:#91A8B8; margin-top:6px; font-style:italic;">
                    🔒 Camera frames are processed locally and are not recorded or uploaded.
                </div>
            </div>
            """
            st.markdown(live_telemetry_html, unsafe_allow_html=True)

        render_live_telemetry()

        with st.expander("ℹ️ Live Camera Test Guide", expanded=False):
            st.markdown(
                """
                **How to test bounding-box growth & sensor fusion:**
                1. Click **START** above and allow browser camera permission.
                2. Hold a **bottle, book, cell phone, cup, or backpack** in view.
                3. Move it closer to test bounding-box growth (`GROWING` / `RAPID GROWTH`).
                4. Move it away to demonstrate a `RECEDING` trend.
                5. Click **STEP (+1s)** to inject live detection into Sensor Fusion and PPO.
                6. Click **STOP** when finished to release the camera.
                """
            )


# ==========================================
# ROW 2: PPO POLICY | SENSOR FUSION | PERFORMANCE & NODES
# ==========================================
row2_col1, row2_col2, row2_col3 = st.columns(3)

# 4. BOTTOM-LEFT: PPO POLICY
with row2_col1:
    panel_policy_html = f"""
    <div class="hud-panel">
        <div class="panel-title">
            <span>🧠 PPO DECISION POLICY</span>
            <span class="panel-subtitle">{snapshot.policy_source_mode}</span>
        </div>
    </div>
    """
    st.markdown(panel_policy_html, unsafe_allow_html=True)

    action_color = (
        "#2EE6A6"
        if "observe" in snapshot.policy_action.lower()
        else (
            "#F6B73C"
            if "resource" in snapshot.policy_action.lower()
            else "#00E5FF"
        )
    )

    action_text = snapshot.policy_action.replace("_", " ").upper()
    action_banner_html = f"""
    <div style="background:rgba(0,229,255,0.06); border:1px solid #0B526D;
    border-radius:4px; padding:8px; text-align:center; margin-bottom:8px;">
        <div style="font-size:0.72rem; color:#91A8B8; font-family:'Consolas',monospace;">
            RECOMMENDED ACTION:
        </div>
        <div style="font-family:'Consolas','Segoe UI',monospace; font-size:1.05rem; font-weight:800;
        color:{action_color}; letter-spacing:1px; margin-top:2px;">
            {action_text}
        </div>
    </div>
    """
    st.markdown(action_banner_html, unsafe_allow_html=True)

    obs_names = [
        "Intensity",
        "Uncertainty",
        "Change Rate",
        "Signal Conf",
        "Visual Urg",
        "Resource A",
        "Resource B",
    ]
    st.markdown(
        "<div style='font-size:0.72rem; color:#91A8B8; margin-bottom:4px;'>"
        "ABSTRACT STATE (NORMALIZED [0-1]):</div>",
        unsafe_allow_html=True,
    )
    for name, val in zip(obs_names, snapshot.policy_observation, strict=True):
        col_m1, col_m2 = st.columns([1, 2.5])
        with col_m1:
            st.markdown(
                f"<span style='font-size:0.72rem; font-family:Consolas;'>{name}:</span>",
                unsafe_allow_html=True,
            )
        with col_m2:
            st.progress(float(min(1.0, max(0.0, val))))


# 5. BOTTOM-CENTER: SENSOR FUSION
with row2_col2:
    panel_fusion_html = """
    <div class="hud-panel">
        <div class="panel-title">
            <span>🧬 SENSOR FUSION ENGINE</span>
            <span class="panel-subtitle">EXPLAINABLE CONSENSUS</span>
        </div>
    </div>
    """
    st.markdown(panel_fusion_html, unsafe_allow_html=True)

    fused = snapshot.fused
    fused_urg_color = (
        "#EF4444"
        if fused.relative_urgency == "critical"
        else ("#F6B73C" if fused.relative_urgency == "approaching" else "#2EE6A6")
    )

    fusion_info_html = f"""
    <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
    padding:8px 12px; margin-bottom:8px;">
        <div class="hud-metric-row">
            <span class="hud-metric-label">FUSED CONSENSUS:</span>
            <span class="hud-metric-value" style="font-size:0.95rem; color:#00E5FF;">
                {fused.fused_label}
            </span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">FUSED CONFIDENCE:</span>
            <span class="hud-metric-value">{fused.fused_confidence:.1%}</span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">FUSED UNCERTAINTY:</span>
            <span class="hud-metric-value" style="color:#F6B73C;">
                {fused.fused_uncertainty:.1%}
            </span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">RELATIVE URGENCY:</span>
            <span class="hud-metric-value" style="color:{fused_urg_color};">
                {fused.relative_urgency.upper()}
            </span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">CONTRIBUTING SOURCES:</span>
            <span class="hud-metric-value">{fused.contributing_sources_count} NODES</span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">INFORMATION FRESHNESS:</span>
            <span class="hud-metric-value">{fused.information_freshness:.1%}</span>
        </div>
    </div>
    """
    st.markdown(fusion_info_html, unsafe_allow_html=True)

    explanation_html = f"""
    <div style="background:rgba(0,229,255,0.04); border-left:3px solid #00E5FF;
    padding:6px 10px; font-size:0.72rem; font-family:'Consolas',monospace; color:#E8F6FF;">
        <strong>FUSION RATIONALE:</strong><br/>{fused.explanation}
    </div>
    """
    st.markdown(explanation_html, unsafe_allow_html=True)


# 6. BOTTOM-RIGHT: PERFORMANCE & COLLABORATIVE NODES
with row2_col3:
    panel_nodes_html = """
    <div class="hud-panel">
        <div class="panel-title">
            <span>🌐 COLLABORATIVE NODES</span>
            <span class="panel-subtitle">SYSTEM TELEMETRY</span>
        </div>
    </div>
    """
    st.markdown(panel_nodes_html, unsafe_allow_html=True)

    for node in snapshot.scenario.nodes:
        status_color = "#2EE6A6" if node.status == "connected" else "#F6B73C"
        node_html = f"""
        <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
        padding:5px 8px; margin-bottom:5px;">
            <div style="display:flex; justify-content:space-between;
            font-family:'Consolas',monospace; font-size:0.75rem;">
                <span style="font-weight:600; color:#E8F6FF;">{node.label}</span>
                <span style="color:{status_color}; font-size:0.68rem; font-weight:bold;">
                    ● {node.status.upper()}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between; color:#91A8B8;
            font-size:0.68rem; margin-top:2px; font-family:'Consolas',monospace;">
                <span>Link: {node.link_quality:.0%}</span>
                <span>Lat: {node.latency_ms:.1f}ms</span>
                <span>Pkts: {node.packets_exchanged}</span>
            </div>
        </div>
        """
        st.markdown(node_html, unsafe_allow_html=True)

    cpu_pct = psutil.cpu_percent()
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    perf_html = f"""
    <div style="background:#040E1A; border:1px solid #0B526D; border-radius:4px;
    padding:6px 10px; margin-top:6px;">
        <div class="hud-metric-row">
            <span class="hud-metric-label">HOST CPU USAGE:</span>
            <span class="hud-metric-value">{cpu_pct:.1f}%</span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">PROCESS MEMORY:</span>
            <span class="hud-metric-value">{mem_mb:.1f} MB</span>
        </div>
        <div class="hud-metric-row">
            <span class="hud-metric-label">INFERENCE BACKEND:</span>
            <span class="hud-metric-value">ASYNC WORKER (CPU)</span>
        </div>
    </div>
    """
    st.markdown(perf_html, unsafe_allow_html=True)


# --- BENCHMARK & EVALUATION EXPANDER ---
with st.expander("📊 Offline Model Evaluation Reports & Benchmarks", expanded=False):
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.markdown("**Module 1: RadioML Signal Classifier**")
        sig_rep_path = Path("outputs/reports/signal_metrics.json")
        if sig_rep_path.is_file():
            with sig_rep_path.open("r", encoding="utf-8") as f:
                sig_data = json.load(f)
            st.json(sig_data)
        else:
            st.info("Run `python -m defensive_avionics.signal.evaluate` to generate report.")

    with exp_col2:
        st.markdown("**Module 2: PPO vs Rule-Based Benchmark**")
        pol_rep_path = Path("outputs/reports/policy_evaluation.json")
        if pol_rep_path.is_file():
            with pol_rep_path.open("r", encoding="utf-8") as f:
                pol_data = json.load(f)
            st.json(pol_data)
        else:
            st.info("Run `python -m defensive_avionics.policy.evaluate` to generate report.")

    with exp_col3:
        st.markdown("**Module 3: Synthetic YOLO Detector**")
        vis_rep_path = Path("outputs/reports/vision_evaluation.json")
        if vis_rep_path.is_file():
            with vis_rep_path.open("r", encoding="utf-8") as f:
                vis_data = json.load(f)
            st.json(vis_data)
        else:
            st.info("Run `python -m defensive_avionics.vision.evaluate` to generate report.")


# --- FOOTER (STRICT COMPLIANCE REQUIREMENT) ---
footer_html = """
<div class="hud-footer">
    SYNTHETIC CLASSROOM SIMULATION — NO REAL-WORLD TARGETING
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

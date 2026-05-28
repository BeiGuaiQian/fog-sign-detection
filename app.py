"""Streamlit app for foggy traffic sign dehazing and detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from src.dehaze_dcp import dehaze_dcp_bgr
from src.detect_yolo import TrafficSignDetector
from src.metrics import compare_results, detections_to_dataframe
from src.utils import bgr_to_rgb, pil_to_bgr


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "weights" / "best.pt"


@st.cache_resource
def load_detector(weight_path: str, device: str | None) -> TrafficSignDetector:
    """Load and cache the YOLOv8 traffic sign detector."""
    return TrafficSignDetector(weight_path=weight_path, device=device)


def encode_jpg(bgr) -> bytes:
    """Encode a BGR image as JPG bytes for Streamlit download buttons."""
    success, buffer = cv2.imencode(".jpg", bgr)
    if not success:
        raise OSError("Failed to encode image as JPG.")
    return buffer.tobytes()


def gray_to_rgb(gray):
    """Convert a grayscale image to RGB for Streamlit display."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def resolve_weight_path(weight_text: str) -> Path:
    """Resolve a sidebar weight path relative to the project root."""
    weight_path = Path(weight_text)
    if not weight_path.is_absolute():
        weight_path = PROJECT_ROOT / weight_path
    return weight_path


def render_sidebar() -> dict:
    """Render sidebar controls and return selected parameters."""
    st.sidebar.header("参数设置")

    weight_text = st.sidebar.text_input(
        "模型权重路径",
        value=str(Path("weights") / "best.pt"),
    )
    device_choice = st.sidebar.selectbox("device", options=["auto", "cpu", "0"], index=0)
    conf = st.sidebar.slider(
        "YOLO 置信度阈值",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    iou = st.sidebar.slider(
        "YOLO IoU 阈值",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )
    window_size = st.sidebar.slider(
        "DCP window_size",
        min_value=3,
        max_value=31,
        value=15,
        step=2,
    )
    omega = st.sidebar.slider(
        "omega",
        min_value=0.50,
        max_value=1.00,
        value=0.95,
        step=0.01,
    )
    t0 = st.sidebar.slider(
        "t0",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
    )
    guided_radius = st.sidebar.slider(
        "guided_radius",
        min_value=5,
        max_value=80,
        value=40,
        step=1,
    )

    device = None if device_choice == "auto" else device_choice
    return {
        "weight_text": weight_text,
        "weight_path": resolve_weight_path(weight_text),
        "device": device,
        "conf": conf,
        "iou": iou,
        "window_size": window_size,
        "omega": omega,
        "t0": t0,
        "guided_radius": guided_radius,
    }


def show_image_pair(left_title: str, left_bgr, right_title: str, right_bgr) -> None:
    """Display two BGR images in a two-column Streamlit layout."""
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader(left_title)
        st.image(bgr_to_rgb(left_bgr), use_container_width=True)
    with right_col:
        st.subheader(right_title)
        st.image(bgr_to_rgb(right_bgr), use_container_width=True)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="雾天路标检测系统", layout="wide")

    params = render_sidebar()

    st.title("基于图像去雾与目标定位的雾天路标检测系统")
    st.markdown("处理流程：雾天图像 → DCP 去雾 → YOLO 路标检测 → 指标对比。")

    if not params["weight_path"].exists():
        st.warning(
            "未找到自定义权重，当前将使用 yolov8n.pt 临时模型。"
            "建议后续替换为交通路标训练权重 weights/best.pt。"
        )

    uploaded_file = st.file_uploader(
        "上传一张雾天路标图片",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is None:
        st.info("请上传 jpg、jpeg 或 png 格式图片开始处理。")
        return

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        original_bgr = pil_to_bgr(pil_image)
    except Exception as exc:
        st.error("上传图片读取失败，请检查图片文件是否损坏。")
        st.exception(exc)
        return

    try:
        with st.spinner("正在进行 DCP 暗通道先验去雾..."):
            dehazed_bgr, dark_channel_img, transmission_img, debug_info = dehaze_dcp_bgr(
                original_bgr,
                window_size=params["window_size"],
                omega=params["omega"],
                t0=params["t0"],
                guided_radius=params["guided_radius"],
            )
    except Exception as exc:
        st.error("DCP 去雾处理失败。")
        st.exception(exc)
        return

    show_image_pair("原始雾天图像", original_bgr, "DCP 去雾结果", dehazed_bgr)

    st.subheader("DCP 中间结果")
    dark_col, trans_col = st.columns(2)
    with dark_col:
        st.image(gray_to_rgb(dark_channel_img), caption="暗通道图", use_container_width=True)
    with trans_col:
        st.image(gray_to_rgb(transmission_img), caption="透射率图", use_container_width=True)

    with st.expander("DCP 调试参数"):
        st.json(debug_info)

    try:
        with st.spinner("正在加载 YOLOv8 模型并执行检测..."):
            detector = load_detector(str(params["weight_path"]), params["device"])
            original_detected_bgr, original_dets = detector.detect(
                original_bgr,
                conf=params["conf"],
                iou=params["iou"],
            )
            dehazed_detected_bgr, dehazed_dets = detector.detect(
                dehazed_bgr,
                conf=params["conf"],
                iou=params["iou"],
            )
    except Exception as exc:
        st.error("YOLO 推理失败。")
        st.exception(exc)
        return

    if detector.used_fallback:
        st.warning(
            "当前使用 yolov8n.pt 临时模型，检测类别为通用 COCO 类别。"
            "课程最终展示建议使用交通路标数据集训练得到的 weights/best.pt。"
        )

    show_image_pair(
        "原图 YOLO 检测结果",
        original_detected_bgr,
        "去雾后 YOLO 检测结果",
        dehazed_detected_bgr,
    )

    st.subheader("检测框信息表")
    original_df = detections_to_dataframe(original_dets)
    dehazed_df = detections_to_dataframe(dehazed_dets)
    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.markdown("**原图检测框**")
        st.dataframe(original_df, use_container_width=True)
    with table_col2:
        st.markdown("**去雾后检测框**")
        st.dataframe(dehazed_df, use_container_width=True)

    st.subheader("图像质量指标和检测指标对比")
    comparison = compare_results(original_bgr, dehazed_bgr, original_dets, dehazed_dets)
    comparison_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in comparison.items()]
    )
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("结果下载")
    down_col1, down_col2, down_col3 = st.columns(3)
    with down_col1:
        st.download_button(
            "下载去雾图",
            data=encode_jpg(dehazed_bgr),
            file_name="dehazed_result.jpg",
            mime="image/jpeg",
        )
    with down_col2:
        st.download_button(
            "下载原图检测结果图",
            data=encode_jpg(original_detected_bgr),
            file_name="original_detection_result.jpg",
            mime="image/jpeg",
        )
    with down_col3:
        st.download_button(
            "下载去雾后检测结果图",
            data=encode_jpg(dehazed_detected_bgr),
            file_name="dehazed_detection_result.jpg",
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()

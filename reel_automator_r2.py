import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
import glob
import io
import atexit
import logging
import functools
import csv

import cv2
import numpy as np
import imageio_ffmpeg
import streamlit as st
import r2_storage
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("reels")
logging.basicConfig(level=logging.INFO)

# ============== CAMPAIGN PROMPT ==============
CAMPAIGN_PROMPT = """
GOAL OF THE CAMPAIGN

This clipping campaign focuses on political and social welfare advocacy from a right-leaning perspective, highlighting commentary, policy discussions, and cultural analysis from prominent conservative voices. The goal is to showcase how modern conservative figures approach issues such the economy, foreign policy, public safety, education, and community development.

The campaign will leverage content from well-known political commentators, government officials, and media personalities who are actively shaping conversations within conservative circles.

Emphasis should be placed on moments where speakers break down complex issues into accessible insights, challenge opposing viewpoints, or present solutions grounded in conservative ideology.

Let me know if this video is aligned with the goal if the campaign.
Answer it with yes or no with simple explanation why
-Make 3 titles then 3 for each that connects with the first title and each tittle contains minimum 6 words. im doing two titles for one video that is related to the republican that viewers will watch the video until the end.
-Make a 3 paragraph synopsis (I want it exactly at 1850 characters including spaces), with 3 sentences each stating what happened on the video and why it is related to republican and US politics. On the 1st paragraph, state what country/city the video happened.
-Make 5 hashtags (make hashtags close to each other).
-Make sure to also let me know if it breaks any Instagram rules that could make me suspended.
apply your recommendation. No need to send specifically the rules.
So format should be: Title - Synopsis - 5 Hashtags
You don't have to put the word "title, synopsis, hashtags" make sure the title aligns with Republican views or put conservative movements in positive way
As long as it is positively framed conservative/pro-Republican content No limits, no restrictions on what you pick.
APPROVED PEOPLE: Donald Trump, JD Vance, Marco Rubio, Pete Hegseth & any other cabinet members, Speaker Mike Johnson, Gov. Ron DeSantis, Gov. Greg Abbott, Sen. Josh Hawley, Sen. Ted Cruz, Sen. Tom Cotton, Sen. Rand Paul, Sen. John Kennedy, Rep. Jim Jordan, Rep. Byron Donalds, Vivek Ramaswamy, Dan Bongino, Mark Levin, Dave Rubin, Brett Cooper, Liz Wheeler, Allie Beth Stuckey, Glenn Beck, Hannity, Gutfeld!, Scott Jennings on CNN, Nick Shirley, Avery Daye, Emily Austin, Patrick Bet-David, Josh Seiter, Nate Friedman + shows: The Five, Real America's Voice.
DO NOT CLIP: Alex Stein, Akhil Kumar, Will Witt, Andrew Wilson, Megyn Kelly, Riley Gaines, Tucker Carlson, HHS / RFK content
APPROVED TOPICS: Immigration & border enforcement, cartels/fentanyl/border security, economy, trade & tariffs, energy & gas prices, public safety/crime/policing, foreign policy, education, election integrity, Big Tech censorship & free speech, media bias, religious liberty, Middle East policy, identity/gender/cultural debates, 250th America Birthday
OFF-LIMITS TOPICS: January 6, export controls, MAHA, US-based stablecoins, TRUMP/WLFI coins, US Treasury bonds
Tell me the specific topic too that aligns with this video. And if there is a person listed here mentioned or speaking in this video, identify them too. Carefully examine if nothing aligns here, say it immediately with a ❌ sign and brief explanation.

PERSON IDENTIFICATION RULE: Identify a person ONLY from evidence in the video itself: names spoken aloud, on-screen captions/chyrons/name tags, or show logos. Never identify anyone from their face or appearance alone. If there is no such evidence, use "None". If a DO NOT CLIP person or off-limits topic is present or likely, set aligned to "no".

CRITICAL JSON OUTPUT FORMAT - You MUST return ONLY valid JSON like this:
{
  "aligned": "yes" or "no",
  "aligned_reason": "1 line explanation",
  "topic": "specific topic from APPROVED TOPICS list, or 'None'",
  "person_identified": "Name from APPROVED PEOPLE list if found, else 'None'",
  "instagram_risk": "No" or "Yes - reason",
  "titles": [
    ["Title 1A - Hook", "Title 1B - Continuation that connects with 1A"],
    ["Title 2A - Hook", "Title 2B - Continuation that connects with 2A"],
    ["Title 3A - Hook", "Title 3B - Continuation that connects with 3A"]
  ],
  "synopsis": "Exactly 1850 characters including spaces, 3 paragraphs, 3 sentences each. Para 1 must state country/city.",
  "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5"]
}
"""
BANNED_NAMES = ["Alex Stein", "Akhil Kumar", "Will Witt", "Andrew Wilson", "Megyn Kelly",
                "Riley Gaines", "Tucker Carlson", "RFK", "Kennedy Jr"]

# Google retires models often; resolve_model() checks the name at run time and falls back.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
SYNOPSIS_TARGET, SYNOPSIS_TOL = 1850, 40

st.set_page_config(page_title="Batch Reel Automator V5 (R2)", layout="wide")
st.title("⚡ V5 - R2 Batch: Analyze + Render + Captions")

_pw = os.environ.get("APP_PASSWORD")             # optional password gate (set APP_PASSWORD in Render)
if _pw and not st.session_state.get("authed"):
    if st.text_input("Password", type="password", key="pw_in") == _pw:
        st.session_state["authed"] = True
        st.rerun()
    st.stop()

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
TARGET_W, TARGET_H = 1080, 1920
CPU = os.cpu_count() or 4

if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix="reels_")
    atexit.register(functools.partial(shutil.rmtree, st.session_state.workdir, ignore_errors=True))
for k, dflt in [("video_queue", []), ("uid_counter", 0), ("rendered_files", []),
                ("caption_files", []), ("zip_path", None)]:
    if k not in st.session_state:
        st.session_state[k] = dflt

WORKDIR = st.session_state.workdir


def tmp(suffix, d=None):
    fd, p = tempfile.mkstemp(suffix=suffix, dir=d or WORKDIR)
    os.close(fd)
    return p


def in_workdir(path):
    try:
        wd = os.path.realpath(WORKDIR)
        return os.path.commonpath([os.path.realpath(path), wd]) == wd
    except ValueError:          # e.g. different drives on Windows
        return False


def safe_remove(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError as e:
            log.warning("remove failed %s: %s", p, e)


# ---------------- Sidebar
st.sidebar.header("🔑 API Keys")
gemini_keys_input = st.sidebar.text_area("Gemini API Keys:", placeholder="AIzaSy...Key1\nAIzaSy...Key2", height=80)
model_name = st.sidebar.text_input("Gemini Model", value=DEFAULT_MODEL)
use_gpu = st.sidebar.checkbox("NVIDIA GPU (h264_nvenc)", value=False)
max_workers_ui = max(2, min(6, CPU))
render_workers = st.sidebar.slider("Parallel render workers", 1, max_workers_ui, 1)

st.sidebar.markdown("---")
header_logo_file = st.sidebar.file_uploader("Top Watermark Banner", type=["png", "jpg", "jpeg"])
@st.cache_data(show_spinner=False)
def prep_logo(file_id, _data):
    # _data is excluded from hashing; file_id is the cache key. Result is a small PNG.
    im = Image.open(io.BytesIO(_data)).convert("RGB")
    w = TARGET_W - 80
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


header_logo_bytes = None
if header_logo_file:
    try:
        _lid = getattr(header_logo_file, "file_id", None) or f"{header_logo_file.name}_{header_logo_file.size}"
        header_logo_bytes = prep_logo(_lid, header_logo_file.getvalue())
    except Exception as e:
        st.sidebar.error(f"Logo could not be read: {e}")
enable_text_watermark = st.sidebar.checkbox("Text Watermark", value=True)
manual_channel_name = st.sidebar.text_input("Channel Handle", value="@MyPoliticalChannel")
watermark_position = st.sidebar.selectbox("Watermark Position", ["Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left", "Center"])
watermark_opacity = st.sidebar.slider("Watermark Opacity", 0.1, 1.0, 0.8)


def parse_keys(text):
    return [k.strip() for k in re.split(r"[,\n]+", text or "") if k.strip()]


# ---------------- Video helpers
def get_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frames / fps) if fps else 0


def detect_crop_fast(frames, orig_w, orig_h):
    """Find the MAIN VIDEO rectangle (left, top, right, bottom) on ANY background (white/black/colored).

    - Background colour is learned from the frame border.
    - Thin things (old title text, logos, handles) are removed with a morphological opening,
      so only the big solid video area survives.
    - Several frames are OR-ed together so a dark/faded first frame does not break detection.
    """
    if isinstance(frames, np.ndarray):
        frames = [frames]
    default = (0, int(orig_h * 0.28), orig_w, int(orig_h * 0.72))
    full = (0, 0, orig_w, orig_h)
    sw = 320
    sh = max(8, int(sw * orig_h / orig_w))
    mask = np.zeros((sh, sw), dtype=bool)
    bars_found = False

    for fr in frames:
        small = cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA)
        mx, my = max(2, int(sw * 0.02)), max(2, int(sh * 0.01))
        strips = [small[:, :mx], small[:, -mx:], small[:my], small[-my:]]
        border = np.concatenate([x.reshape(-1, 3) for x in strips])

        # 1) background colour = most common colour on the border
        q = (border // 32).astype(np.int32)
        keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
        vals, counts = np.unique(keys, return_counts=True)
        bg = np.median(border[keys == vals[counts.argmax()]], axis=0)

        # 2) if no side of the frame is (almost) one flat colour, there are no bars -> plain video
        for x in strips:
            side_diff = np.abs(x.reshape(-1, 3).astype(np.int16) - bg.astype(np.int16)).max(axis=1)
            if (side_diff <= 28).mean() > 0.9:
                bars_found = True
                break

        diff = np.abs(small.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
        mask |= diff > 28

    if not bars_found:
        return full

    # 3) remove thin strokes (title text, logos, handles); the solid video block stays
    k = max(5, int(sw * 0.022)) | 1
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0

    def tallest_run(flags, bridge):
        f = cv2.morphologyEx(flags.astype(np.uint8).reshape(-1, 1) * 255, cv2.MORPH_CLOSE,
                             np.ones((2 * bridge + 1, 1), np.uint8)).ravel() > 0
        d = np.diff(np.concatenate([[0], f.astype(np.int8), [0]]))
        st_, en_ = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(st_) == 0:
            return None
        i = int((en_ - st_).argmax())
        return int(st_[i]), int(en_[i])

    # 4) rows: the video band
    rows = tallest_run(m.mean(axis=1) > 0.35, max(2, int(sh * 0.03)))
    if not rows or (rows[1] - rows[0]) < sh * 0.15:
        return default
    s, e = rows

    # 5) columns: left/right edges of the video inside that band
    cols = tallest_run(m[s:e].mean(axis=0) > 0.5, max(2, int(sw * 0.02)))
    if not cols or (cols[1] - cols[0]) < sw * 0.15:
        l, r = 0, sw
    else:
        l, r = cols

    # 6) trim 1 px (small scale) of edge blur / rounded-corner fringe where the edge is not the frame edge
    if s > 0: s += 1
    if e < sh: e -= 1
    if l > 0: l += 1
    if r < sw: r -= 1

    sx, sy = orig_w / sw, orig_h / sh
    left, right = max(0, int(l * sx)), min(orig_w, int(r * sx))
    top, bottom = max(0, int(s * sy)), min(orig_h, int(e * sy))
    if (r - l) >= sw * 0.97 and (e - s) >= sh * 0.97:
        return full
    if top >= bottom - 20 or left >= right - 20:
        return default
    return left, top, right, bottom


def read_sample_frames(path):
    """First frame + two later frames (a first frame is often black / a fade-in)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = [0] + ([int(total * 0.35), int(total * 0.7)] if total > 30 else [])
    frames = []
    for pos in positions:
        if pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, fr = cap.read()
        if ret and fr is not None:
            frames.append(fr)
    cap.release()
    return fps, frames


def crop_box(v):
    """Clamped, even-sized crop rectangle -> (x, y, w, h) in source pixels."""
    W, H = int(v["orig_w"]), int(v["orig_h"])
    l = min(max(0, int(v.get("crop_left", 0))), W - 2)
    r = min(W, max(l + 2, int(v.get("crop_right", W))))
    t = min(max(0, int(v["crop_top"])), H - 2)
    b = min(H, max(t + 2, int(v["crop_bottom"])))
    l -= l % 2
    t -= t % 2
    w = max(2, (r - l) - ((r - l) % 2))
    h = max(2, (b - t) - ((b - t) % 2))
    return l, t, w, h


def ingest_video(path):
    fps, frames = read_sample_frames(path)
    if not frames:
        return None
    frame = frames[0]
    orig_h, orig_w = frame.shape[:2]
    if orig_w <= 0 or orig_h <= 0:
        return None
    left, top, right, bottom = detect_crop_fast(frames, orig_w, orig_h)
    sample = frames[1] if len(frames) > 1 else frames[0]      # a later frame makes a better preview
    _, buf = cv2.imencode(".jpg", sample, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"path": path, "orig_w": orig_w, "orig_h": orig_h, "fps": fps,
            "sample_jpg": buf.tobytes(), "crop_left": left, "crop_right": right,
            "crop_top": top, "crop_bottom": bottom,
            "title": "Republican Leader Confronts The Bias Directly",
            "font_size": 66, "gap": 35, "ai_data": None}


def safe_ingest(path):
    try:
        return ingest_video(path)
    except Exception as e:
        log.warning("ingest failed for %s: %s", path, e)
        return None


def delete_video(uid):
    for v in st.session_state.video_queue:
        if v["uid"] == uid and in_workdir(v["path"]):     # never touch originals from a user folder
            safe_remove(v["path"])
            try:
                os.rmdir(os.path.dirname(v["path"]))      # drop the empty uploads/<file_id> folder
            except OSError:
                pass
    st.session_state.video_queue = [v for v in st.session_state.video_queue if v["uid"] != uid]


@st.cache_resource
def get_bold_font(size):
    for p in ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "assets/arialbd.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception as e:
                log.warning("font load failed %s: %s", p, e)
    return ImageFont.load_default()


@st.cache_data(show_spinner=False, max_entries=2)
def create_static_canvas(title, font_size, gap, crop_h, crop_w, logo_bytes):
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = 40
    if logo_bytes:
        try:
            logo = Image.open(io.BytesIO(logo_bytes)).convert("RGB")
            lw, lh = logo.size
            tw = TARGET_W - 80
            th = int(lh * (tw / lw))
            canvas.paste(logo.resize((tw, th), Image.Resampling.LANCZOS), ((TARGET_W - tw) // 2, y))
            y += th + 30
        except Exception as e:
            log.warning("logo failed: %s", e)
    font = get_bold_font(font_size)
    lines, curr = [], []
    for word in str(title or "").split():
        curr.append(word)
        b = draw.textbbox((0, 0), " ".join(curr), font=font)
        if (b[2] - b[0]) > TARGET_W - 90 and len(curr) > 1:
            curr.pop()
            lines.append(" ".join(curr))
            curr = [word]
    if curr:
        lines.append(" ".join(curr))
    for line in lines:
        b = draw.textbbox((0, 0), line, font=font)
        draw.text(((TARGET_W - (b[2] - b[0])) // 2, y), line, fill=(255, 255, 255), font=font)
        y += font_size + 12
    scale = TARGET_W / crop_w
    v_w, v_h = TARGET_W, int(crop_h * scale)
    if v_h > int(TARGET_H * 0.65):
        scale = (TARGET_H * 0.65) / crop_h
        v_w, v_h = int(crop_w * scale), int(crop_h * scale)
    v_w -= v_w % 2
    v_h -= v_h % 2
    pos_x = (TARGET_W - v_w) // 2
    pos_y = y + gap
    pos_y -= pos_y % 2
    return canvas, pos_x, pos_y, v_w, v_h


@st.cache_data(show_spinner=False, max_entries=2)
def make_watermark_layer(text, position, opacity, px, py, vw, vh):
    layer = Image.new("RGBA", (TARGET_W, TARGET_H), (255, 255, 255, 0))
    if not (text or "").strip():
        return layer
    draw = ImageDraw.Draw(layer)
    font = get_bold_font(32)
    b = draw.textbbox((0, 0), text, font=font)
    tw, th, m = b[2] - b[0], b[3] - b[1], 30
    pos_map = {"Top-Right": (px + vw - tw - m, py + m), "Top-Left": (px + m, py + m),
               "Bottom-Right": (px + vw - tw - m, py + vh - th - m),
               "Bottom-Left": (px + m, py + vh - th - m),
               "Center": (px + (vw - tw) // 2, py + (vh - th) // 2)}
    pos = pos_map.get(position, pos_map["Center"])
    draw.text(pos, text, fill=(255, 255, 255, int(255 * opacity)), font=font)
    return layer


# ---------------- Synopsis length handling
def trim_to_sentence(text, target=SYNOPSIS_TARGET):
    if len(text) <= target:
        return text
    cut = text[:target]
    idx = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[:idx + 1] if idx > target * 0.8 else cut


def fit_synopsis(client, model, text, target=SYNOPSIS_TARGET, tol=SYNOPSIS_TOL, tries=2):
    """Ask the model to rewrite to length; fall back to trimming at a sentence end. Never pads."""
    text = (text or "").replace("\r", "").strip()
    for _ in range(tries):
        if abs(len(text) - target) <= tol:
            break
        diff = target - len(text)
        prompt = (f"This text is {len(text)} characters including spaces. Rewrite it to be {target} characters "
                  f"(you need to {'add' if diff > 0 else 'cut'} about {abs(diff)}). Keep exactly 3 paragraphs "
                  f"of 3 sentences each, same facts, paragraphs separated by a blank line. "
                  f"Return only the text.\n\n{text}")
        try:
            r = client.models.generate_content(model=model, contents=[prompt])
            if r.text and r.text.strip():
                text = r.text.strip()
        except Exception as e:
            log.warning("synopsis rewrite failed: %s", e)
            break
    return trim_to_sentence(text, target)


def flag_banned(ad):
    blob = " ".join([str(ad.get("person_identified", "")), str(ad.get("synopsis", "")),
                     " ".join(str(t) for t in ad.get("titles", []))]).lower()
    return [n for n in BANNED_NAMES if n.lower() in blob]


def build_caption_txt_content(ai_data):
    """caption.txt = the 3-paragraph synopsis + the 5 hashtags. Nothing else."""
    synopsis = (ai_data.get("synopsis") or "").replace("\r", "").strip()
    hashtags = " ".join(ai_data.get("hashtags", [])[:5])
    return f"{synopsis}\n\n{hashtags}\n"


# ---------------- Rendering
def fps_expr(fps):
    """Exact rational fps so 29.97 stays 29.97 (no 30fps resample -> no dup/drop judder)."""
    for num, den in ((24000, 1001), (30000, 1001), (60000, 1001)):
        if abs(fps - num / den) < 0.02:
            return f"{num}/{den}"
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def render_reel(v, canvas_png, wm_png, px, py, vw, vh, out_path, gpu):
    cx, cy, cw, ch = crop_box(v)
    fps = fps_expr(v["fps"] or 30.0)
    fc = f"[1:v]crop={cw}:{ch}:{cx}:{cy},scale={vw}:{vh}[vid];[0:v][vid]overlay={px}:{py}[bg];"
    fc += "[bg][2:v]overlay=0:0:format=auto,format=yuv420p[out]" if wm_png else "[bg]format=yuv420p[out]"
    cmd = [FFMPEG_BIN, "-y", "-framerate", str(fps), "-loop", "1", "-i", canvas_png, "-i", v["path"]]
    if wm_png:
        cmd += ["-framerate", str(fps), "-loop", "1", "-i", wm_png]
    cmd += ["-filter_complex", fc, "-map", "[out]", "-map", "1:a:0?"]
    cmd += (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "24"] if gpu
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-threads", "0"])
    cmd += ["-r", str(fps), "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="ignore")[-800:])
    return out_path


def render_job(v, gpu, title, font_size, gap, wm_args, logo_bytes):
    _, _, cw, ch = crop_box(v)
    canvas, px, py, vw, vh = create_static_canvas(title, font_size, gap, ch, cw, logo_bytes)
    # Matching names: <uid>_<source>_reel.mp4 <-> <uid>_<source>_caption.txt
    base = f"{v['uid']}_{os.path.splitext(os.path.basename(v['path']))[0]}"
    out = os.path.join(WORKDIR, base + "_reel.mp4")
    caption_path = os.path.join(WORKDIR, base + "_caption.txt")
    canvas_png, wm_png = tmp(".png"), None
    canvas.save(canvas_png, compress_level=1, optimize=False)
    try:
        if wm_args:
            wm_png = tmp(".png")
            make_watermark_layer(*wm_args, px, py, vw, vh).save(wm_png, compress_level=1, optimize=False)
        render_reel(v, canvas_png, wm_png, px, py, vw, vh, out, gpu)
        ai_data = v.get("ai_data")
        if ai_data:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(build_caption_txt_content(ai_data))
        else:
            caption_path = None          # no AI data -> no caption file (no filler text)
        return v["uid"], out, caption_path, None
    except Exception as e:
        log.exception("render failed")
        return v["uid"], None, None, str(e)
    finally:
        safe_remove(canvas_png, wm_png)


# ---------------- Gemini
_clients, _lock = {}, threading.Lock()


def get_client(key):
    with _lock:
        if key not in _clients:
            _clients[key] = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120_000))
        return _clients[key]


def make_smart_proxy(path, d):
    """Every temp file goes into job dir `d`; the caller deletes the whole dir, so nothing can leak."""
    try:
        dur = get_duration(path)
        if dur < 7:
            small = tmp(".mp4", d)
            cmd = [FFMPEG_BIN, "-y", "-i", path, "-vf", "scale=-2:240:flags=fast_bilinear,fps=4", "-c:v", "libx264",
                   "-preset", "ultrafast", "-crf", "32", "-c:a", "aac", "-b:a", "32k", "-ac", "1", small]
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            return small if r.returncode == 0 and os.path.getsize(small) > 5000 else path
        mid = dur / 2
        video_only, smart_final = tmp(".mp4", d), tmp(".mp4", d)
        sc = "scale=-2:240:flags=fast_bilinear,fps=4,setpts=PTS-STARTPTS"
        cmd_v = [FFMPEG_BIN, "-y", "-ss", "0", "-t", "2", "-i", path, "-ss", str(max(0, mid - 1)), "-t", "2", "-i", path,
                 "-ss", str(max(0, dur - 2)), "-t", "2", "-i", path, "-filter_complex",
                 f"[0:v]{sc}[v0];[1:v]{sc}[v1];[2:v]{sc}[v2];[v0][v1][v2]concat=n=3:v=1:a=0[v]",
                 "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32", "-an", video_only]
        if subprocess.run(cmd_v, capture_output=True, timeout=20).returncode != 0:
            raise RuntimeError("concat fail")
        cmd_a = [FFMPEG_BIN, "-y", "-i", video_only, "-i", path, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "32k", "-ac", "1", "-shortest", smart_final]
        r2 = subprocess.run(cmd_a, capture_output=True, timeout=15)
        ok = r2.returncode == 0 and os.path.getsize(smart_final) > 5000
        return smart_final if ok else video_only
    except Exception as e:                      # includes subprocess.TimeoutExpired
        log.warning("proxy failed, using original: %s", e)
        return path


def extract_key_frames_fast(path, d):
    dur = get_duration(path)
    times = [0.5, dur / 2, max(0, dur - 0.8)] if dur > 2 else [0.5]
    frames = []
    for t in times:
        out = tmp(".jpg", d)
        try:
            subprocess.run([FFMPEG_BIN, "-y", "-ss", str(t), "-i", path, "-vframes", "1", "-q:v", "2", out],
                           capture_output=True, timeout=10)
            if os.path.getsize(out) > 0:
                with Image.open(out) as im:
                    frames.append(im.convert("RGB"))
        except Exception as e:
            log.warning("frame extract failed: %s", e)
    return frames


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            raise ValueError("No JSON in model response")
        return json.loads(m.group(0))


def analyze_with_key_smart(key, video_path, model):
    client = get_client(key)
    job_dir = tempfile.mkdtemp(prefix="job_", dir=WORKDIR)
    frames, f = [], None
    try:
        proxy = make_smart_proxy(video_path, job_dir)
        frames = extract_key_frames_fast(video_path, job_dir)
        f = client.files.upload(file=proxy)
        deadline = time.time() + 60
        while True:
            f = client.files.get(name=f.name)
            state = getattr(f.state, "name", str(f.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini file processing FAILED")
            if time.time() > deadline:
                raise TimeoutError("Gemini file not ACTIVE within 60s")
            time.sleep(1.2)
        res = client.models.generate_content(
            model=model, contents=[f] + frames + [CAMPAIGN_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        data = parse_json(res.text)
        if data.get("synopsis"):
            data["synopsis"] = fit_synopsis(client, model, data["synopsis"])
        return data
    finally:
        try:
            if f:
                client.files.delete(name=f.name)
        except Exception as e:
            log.warning("remote delete failed: %s", e)
        for fr in frames:
            fr.close()
        shutil.rmtree(job_dir, ignore_errors=True)   # removes proxy + every temp file at once


def resolve_model(key, wanted):
    """Return the first model that exists for this key: the requested one, then known fallbacks."""
    client = get_client(key)
    for m in [wanted] + [x for x in MODEL_FALLBACKS if x != wanted]:
        try:
            client.models.get(model=m)
            return m
        except Exception as e:
            log.info("model %s unavailable: %s", m, e)
    raise RuntimeError("No usable Gemini model found. Enter a current model name in the sidebar.")


BAD_KEY_HINTS = ("api key", "api_key_invalid", "permission_denied", "401", "403")
BLOCKED_HINTS = ("blocked", "prohibited", "safety")
NOT_FOUND_HINTS = ("404", "not_found", "no longer available", "not found")


def process_single_ai(uid, path, keys, start_idx, model):
    last_err = "Unknown"
    for attempt in range(max(3, len(keys))):
        key = keys[(start_idx + attempt) % len(keys)]
        try:
            return uid, analyze_with_key_smart(key, path, model), None
        except Exception as e:
            last_err = str(e)
            low = last_err.lower()
            if any(h in low for h in BLOCKED_HINTS):
                return uid, None, f"Blocked by model: {last_err}"   # retrying won't help
            if any(h in low for h in NOT_FOUND_HINTS):
                return uid, None, f"Model unavailable: {last_err}"    # same for every key, don't retry
            if any(h in low for h in BAD_KEY_HINTS):
                continue                                           # try next key immediately
            if "429" in low or "quota" in low:
                time.sleep(2)
                continue
            time.sleep(min(2 ** attempt, 4))
    return uid, None, last_err


# ======================================================================
# Cloudflare R2 pipeline: base bucket (read-only) -> edit -> new bucket named after the watermark
# ======================================================================
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
MANIFEST_KEY = "manifest.json"
_mlock = threading.Lock()

st.sidebar.markdown("---")
st.sidebar.header("☁️ Cloudflare R2")
base_bucket = st.sidebar.text_input("Base videos bucket (read-only)", value=os.environ.get("R2_BASE_BUCKET", "basevideos")).strip()
base_prefix = st.sidebar.text_input("Folder inside it (optional)", value=os.environ.get("R2_BASE_PREFIX", "")).strip().lstrip("/")
max_per_run = st.sidebar.number_input("Videos this run (0 = all)", min_value=0, value=0, step=1)
r2_storage.protect(base_bucket)          # from here on, any write to the base bucket raises an error


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def bucket_slug(name):
    """'@My Political Channel!' -> 'my-political-channel' (R2 rules: 3-63 chars, a-z 0-9 -)."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    if len(s) < 3:
        s = f"{s}-reels" if s else "reels-output"
    return s[:63].strip("-")


def load_manifest(bucket):
    raw = r2_storage.get_bytes(bucket, MANIFEST_KEY)
    if raw:
        try:
            m = json.loads(raw)
            m.setdefault("next", 1)
            m.setdefault("numbers", {})
            m.setdefault("items", {})
            return m
        except ValueError:
            log.warning("manifest unreadable, starting fresh")
    return {"next": 1, "numbers": {}, "items": {}}


def save_manifest(bucket, m):
    r2_storage.put_bytes(bucket, MANIFEST_KEY, json.dumps(m, indent=1).encode("utf-8"), "application/json")


def process_item(src_key, n, base_b, out_b, manifest, keys, model, wm_args, logo_bytes, gpu):
    """download -> crop detect -> Gemini -> render -> upload videoN.mp4 + captionN.txt -> cleanup.
    Returns (n, src_key, entry_or_None, error_or_None). Never writes to the base bucket."""
    job = tempfile.mkdtemp(prefix="item_", dir=WORKDIR)
    try:
        ext = os.path.splitext(src_key)[1].lower() or ".mp4"
        local = os.path.join(job, "source" + ext)
        r2_storage.download_file(base_b, src_key, local)

        info = ingest_video(local)
        if not info:
            return n, src_key, None, "Video could not be read"
        info["uid"] = n

        _, ai, err = process_single_ai(n, local, keys, n, model)
        if not ai:
            return n, src_key, None, f"AI failed: {err}"
        titles = ai.get("titles") or []
        first = titles[0] if titles else ""
        title = first[0] if isinstance(first, list) and first else first
        if not str(title or "").strip():
            return n, src_key, None, "AI returned no title"

        cx, cy, cw, ch = crop_box(info)
        canvas, px, py, vw, vh = create_static_canvas(title, 66, 35, ch, cw, logo_bytes)
        canvas_png, wm_png = tmp(".png", job), None
        canvas.save(canvas_png, compress_level=1, optimize=False)
        if wm_args:
            wm_png = tmp(".png", job)
            make_watermark_layer(*wm_args, px, py, vw, vh).save(wm_png, compress_level=1, optimize=False)
        out_mp4 = os.path.join(job, f"video{n}.mp4")
        render_reel(info, canvas_png, wm_png, px, py, vw, vh, out_mp4, gpu)

        r2_storage.upload_file(out_b, out_mp4, f"video{n}.mp4", "video/mp4")
        r2_storage.put_bytes(out_b, f"caption{n}.txt", build_caption_txt_content(ai).encode("utf-8"),
                             "text/plain; charset=utf-8")

        entry = {"n": n, "title": str(title), "aligned": ai.get("aligned"), "topic": ai.get("topic"),
                 "person": ai.get("person_identified"), "ig_risk": ai.get("instagram_risk"),
                 "flags": ", ".join(flag_banned(ai)), "reason": ai.get("aligned_reason")}
        with _mlock:
            manifest["items"][src_key] = entry
            save_manifest(out_b, manifest)
        return n, src_key, entry, None
    except Exception as e:
        log.exception("item failed: %s", src_key)
        return n, src_key, None, str(e)
    finally:
        shutil.rmtree(job, ignore_errors=True)       # source, canvas, mp4: everything local is removed


def build_report_csv(manifest):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["number", "source_video", "title", "aligned", "topic", "person", "instagram_risk",
                "do_not_clip_flags", "reason"])
    for src, e in sorted(manifest["items"].items(), key=lambda kv: kv[1]["n"]):
        w.writerow([e["n"], src, e.get("title"), e.get("aligned"), e.get("topic"), e.get("person"),
                    e.get("ig_risk"), e.get("flags"), e.get("reason")])
    return buf.getvalue().encode("utf-8")


# ---------------- UI
st.header("🚀 Analyze and Render")
out_bucket = bucket_slug(manual_channel_name)
keys = parse_keys(gemini_keys_input) or parse_keys(os.environ.get("GEMINI_KEYS"))
st.markdown(f"**Source:** `{base_bucket}/{base_prefix}` (read-only)  →  **Output bucket:** `{out_bucket}`")
st.caption("Output files: video1.mp4 + caption1.txt, video2.mp4 + caption2.txt ... "
           "(plus manifest.json and report.csv). Re-running continues where it stopped.")

if st.button("🚀 Analyze and Render", type="primary"):
    problems = []
    if not r2_storage.enabled():
        problems.append("R2 credentials missing (R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY).")
    if not keys:
        problems.append("Add at least one Gemini API key in the sidebar.")
    if not (manual_channel_name or "").strip():
        problems.append("Enter the channel name / watermark in the sidebar.")
    if out_bucket == base_bucket:
        problems.append("Output bucket would be the same as the base bucket. Change the channel name.")
    if not header_logo_bytes:
        st.warning("No banner uploaded: videos will be rendered without the top banner.")
    if problems:
        for p_ in problems:
            st.error(p_)
        st.stop()

    try:
        active_model = resolve_model(keys[0], model_name.strip())
        sources = sorted((o["key"] for o in r2_storage.list_objects(base_bucket, base_prefix)
                          if o["key"].lower().endswith(VIDEO_EXTS) and o["size"] > 0), key=natural_key)
        if not sources:
            st.error(f"No videos found in '{base_bucket}/{base_prefix}'.")
            st.stop()
        state = r2_storage.ensure_bucket(out_bucket)
        manifest = load_manifest(out_bucket)
    except Exception as e:
        st.error(f"Setup failed: {e}")
        st.stop()

    st.info(f"Output bucket '{out_bucket}' {state}. Found {len(sources)} base videos.")
    pending = [k for k in sources if k not in manifest["items"]]
    for k in pending:                                   # stable numbers, kept across re-runs
        if k not in manifest["numbers"]:
            manifest["numbers"][k] = manifest["next"]
            manifest["next"] += 1
    todo = pending[:int(max_per_run)] if max_per_run else pending
    save_manifest(out_bucket, manifest)
    st.write(f"Already done: {len(sources) - len(pending)} | To process now: {len(todo)}")

    wm_args = (manual_channel_name, watermark_position, watermark_opacity) \
        if enable_text_watermark and manual_channel_name else None
    bar, logbox = st.progress(0), st.container()
    ok = bad = 0
    if todo:
        with cf.ThreadPoolExecutor(max_workers=render_workers) as ex:
            futs = [ex.submit(process_item, k, manifest["numbers"][k], base_bucket, out_bucket, manifest,
                              keys, active_model, wm_args, header_logo_bytes, use_gpu) for k in todo]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                n, src, entry, err = fut.result()
                if entry:
                    ok += 1
                    note = ""
                    if entry["flags"]:
                        note += f" ⚠️ DO-NOT-CLIP: {entry['flags']}"
                    if str(entry.get("aligned", "")).lower() != "yes":
                        note += " ❌ not aligned"
                    logbox.success(f"video{n}.mp4 / caption{n}.txt  <-  {src}{note}")
                else:
                    bad += 1
                    logbox.error(f"#{n} {src} failed: {err}")
                bar.progress(i / len(futs))

    try:
        with _mlock:
            report = build_report_csv(manifest)
        r2_storage.put_bytes(out_bucket, "report.csv", report, "text/csv")
        st.session_state["report_url"] = r2_storage.presigned_url(out_bucket, "report.csv", 86400)
    except Exception as e:
        st.warning(f"Could not write report.csv: {e}")
    st.success(f"Finished: {ok} done, {bad} failed. Failed videos are retried automatically on the next run.")
    if st.session_state.get("report_url"):
        st.link_button("⬇️ Open report.csv (alignment + DO-NOT-CLIP flags)", st.session_state["report_url"])

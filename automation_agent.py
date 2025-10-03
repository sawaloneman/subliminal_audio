"""Automation agent for creating and publishing subliminal audio videos.

This module orchestrates the end-to-end workflow required to transform
affirmations into a YouTube-ready subliminal video.  It integrates with an
existing Streamlit generator (expected to expose a
``generate_subliminal_audio`` function) and automates the following steps:

1. Gather run configuration either interactively (manual mode) or on a
   schedule (auto mode).
2. Use the OpenAI API to generate affirmations, metadata, and an image prompt
   for the thumbnail artwork.
3. Persist affirmations to disk, call the Streamlit audio generator, and keep
   track of the produced audio artifact.
4. Convert the generated audio into an MP4 video by controlling
   https://www.onlineconverter.com/audio-to-video with Selenium.
5. Upload the resulting video and thumbnail to YouTube using the Data API.

Environment variables (or `.env` management) should supply the required
credentials:

- ``OPENAI_API_KEY`` – API key for OpenAI.
- ``GOOGLE_CLIENT_SECRETS`` – Path to the OAuth client secrets JSON used for
  YouTube uploads.
- ``YOUTUBE_TOKEN`` – Optional path for storing the OAuth token; defaults to
  ``token.json`` in the working directory.
- ``CHROMEDRIVER_PATH`` – Optional path to the ChromeDriver binary. If not
  provided the Selenium manager will attempt to locate a driver on the PATH.
- ``STREAMLIT_APP_MODULE`` – Optional module path for the Streamlit app that
  defines ``generate_subliminal_audio`` (defaults to ``streamlit_app``).

The script can be executed directly; when run as a module it prompts the user
to select manual or automatic mode.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


_GOOGLE_CLIENT_CACHE: Optional[Tuple] = None


@dataclass
class DiagnosticResult:
    """Result of an environment diagnostic check."""

    name: str
    status: Literal["pass", "fail", "warn"]
    details: str = ""
    duration: Optional[float] = None


def _load_google_client_dependencies():
    """Return Google client helpers, ensuring dependencies are installed."""

    global _GOOGLE_CLIENT_CACHE
    if _GOOGLE_CLIENT_CACHE is not None:
        return _GOOGLE_CLIENT_CACHE

    required_packages = {
        "google.auth": "google-auth",
        "google.oauth2": "google-auth",
        "googleapiclient": "google-api-python-client",
        "google_auth_oauthlib": "google-auth-oauthlib",
    }

    missing = [
        package_name
        for module_name, package_name in required_packages.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing:
        install_hint = "pip install " + " ".join(sorted(set(missing)))
        raise ModuleNotFoundError(
            "YouTube uploads require Google API client libraries. "
            f"Install them with `{install_hint}` before running the agent."
        )

    from google.auth.transport.requests import Request as _Request
    from google.oauth2.credentials import Credentials as _Credentials
    from googleapiclient.discovery import build as _build
    from googleapiclient.http import MediaFileUpload as _MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow as _InstalledAppFlow

    _GOOGLE_CLIENT_CACHE = (
        _Request,
        _Credentials,
        _build,
        _MediaFileUpload,
        _InstalledAppFlow,
    )
    return _GOOGLE_CLIENT_CACHE


def _load_openai_client() -> "OpenAIClient":
    """Dynamically load the OpenAI client, guiding users to install it."""

    if importlib.util.find_spec("openai") is None:
        raise ModuleNotFoundError(
            "OpenAI API support requires the `openai` package. "
            "Install it with `pip install openai` before running the agent."
        )

    from openai import OpenAI as _OpenAI

    return _OpenAI

# The Streamlit generator is expected to live in the same repository.  Update
# the environment variables below to match the actual module/function names if
# they differ from the defaults provided here.
STREAMLIT_APP_MODULE = os.environ.get("STREAMLIT_APP_MODULE")
STREAMLIT_GENERATOR_NAME = os.environ.get(
    "STREAMLIT_GENERATOR_NAME", "generate_subliminal_audio"
)


def _resolve_streamlit_generator() -> Callable[..., Path]:
    """Locate the Streamlit audio generator function.

    The automation agent was originally designed to import a module named
    ``streamlit_app`` that exposes a ``generate_subliminal_audio`` callable.  In
    practice, projects frequently organize their Streamlit apps differently. To
    minimise friction we search a handful of likely module names and honour the
    optional ``STREAMLIT_APP_MODULE`` and ``STREAMLIT_GENERATOR_NAME``
    environment overrides.  A descriptive error message is raised if we exhaust
    the options without finding a usable function.
    """

    attempted: List[str] = []

    def candidate_modules() -> Iterable[str]:
        if STREAMLIT_APP_MODULE:
            yield STREAMLIT_APP_MODULE
        for default_name in ("streamlit_app", "app", "main", "subliminal"):
            if default_name != STREAMLIT_APP_MODULE:
                yield default_name

    for module_name in dict.fromkeys(candidate_modules()):
        attempted.append(module_name)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        generator = getattr(module, STREAMLIT_GENERATOR_NAME, None)
        if callable(generator):
            return generator

    attempted_list = ", ".join(attempted) or "<none>"
    raise AttributeError(
        "Unable to locate a Streamlit generator function named "
        f"`{STREAMLIT_GENERATOR_NAME}`. Set the STREAMLIT_APP_MODULE and/"
        "or STREAMLIT_GENERATOR_NAME environment variables so the automation "
        "agent can import the correct function. Attempted modules: "
        f"{attempted_list}."
    )


generate_subliminal_audio = _resolve_streamlit_generator()


# --- Configuration -----------------------------------------------------------------

DEFAULT_NOISE_TYPES: Sequence[str] = (
    "rainbow",
    "white",
    "gray",
    "black",
    "pink",
    "brown",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "purple",
    "center",
    "center-pulse",
    "subconscious-hack",
    "conscious-hack",
    "mind-hijack",
    "binaural-delta",
    "binaural-theta",
    "binaural-alpha",
    "binaural-beta",
    "binaural-gamma",
    "binaural-epsilon",
    "binaural-lambda",
    "binaural-mu",
    "morphic-field",
    "metaliminal",
    "supraliminal",
    "phantomliminal",
    "binaural",
    "none",
)

AFFIRMATION_COUNT_OPTIONS: Sequence[int] = (5, 8, 10, 12, 15, 20)

PLAYBACK_SPEED_OPTIONS: Sequence[float] = (0.75, 1.0, 1.15, 1.25, 1.5)

AFFIRMATION_THEME_OPTIONS: Sequence[str] = (
    "Calm confidence",
    "Deep sleep",
    "Wealth mindset",
    "Self-love",
    "Productivity focus",
    "Motivation boost",
)

LAYER_COUNT_OPTIONS: Sequence[int] = (1, 2, 3, 4, 5)
LAYER_VARIATION_OPTIONS: Sequence[float] = (0.25, 0.35, 0.45, 0.55, 0.65)
CENTER_FOCUS_OPTIONS: Sequence[float] = (0.3, 0.45, 0.55, 0.65, 0.75)
SUBCONSCIOUS_INTENSITY_OPTIONS: Sequence[float] = (0.2, 0.35, 0.45, 0.55, 0.65)
CONSCIOUS_INTENSITY_OPTIONS: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5)
OVERRIDE_INTENSITY_OPTIONS: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


@dataclass
class GeneratorSettings:
    """Container for Streamlit generator parameters."""

    noise_type: str
    affirmation_count: int
    playback_speed: float
    theme: str = "Calm focus"
    auto_layer: bool = True
    layer_count: int = 3
    layer_variation: float = 0.4
    layer_seed: Optional[int] = None
    center_focus: float = 0.55
    subconscious_intensity: float = 0.4
    conscious_intensity: float = 0.25
    override_intensity: float = 0.35


@dataclass
class GeneratedAssets:
    """Paths to artifacts created during a single automation run."""

    text_path: Path
    audio_path: Path
    video_path: Path
    thumbnail_path: Path


# --- Helper utilities --------------------------------------------------------------


def build_openai_client() -> "OpenAIClient":
    """Instantiate an OpenAI client using the environment or prompted API key."""

    OpenAI = _load_openai_client()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.")
        api_key = getpass("Enter your OpenAI API key (input hidden): ").strip()
        if not api_key:
            raise RuntimeError(
                "An OpenAI API key is required. Set the OPENAI_API_KEY environment "
                "variable or provide it interactively when prompted."
            )
        os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)


def load_noise_options() -> Sequence[str]:
    """Return the noise options exposed by the Streamlit generator.

    The Streamlit app can expose an ``AVAILABLE_NOISE_TYPES`` attribute. If it
    does not, the fallback defaults declared in this module are returned.
    """

    noise_options = getattr(sys.modules[generate_subliminal_audio.__module__], "AVAILABLE_NOISE_TYPES", None)
    if noise_options:
        return tuple(noise_options)
    return DEFAULT_NOISE_TYPES


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def prompt_user(prompt: str, valid: Optional[Iterable[str]] = None) -> str:
    while True:
        response = input(prompt).strip()
        if not valid or response.lower() in {v.lower() for v in valid}:
            return response
        print(f"Invalid response. Expected one of: {', '.join(valid)}")


# --- OpenAI content generation -----------------------------------------------------


def parse_openai_text(response) -> str:
    if hasattr(response, "output_text"):
        return response.output_text.strip()
    if hasattr(response, "choices"):
        return response.choices[0].message["content"].strip()
    raise ValueError("Unexpected OpenAI response format.")


def generate_affirmations(client: "OpenAIClient", theme: str, count: int) -> List[str]:
    prompt = (
        "You are crafting concise subliminal affirmations. "
        f"Write {count} first-person present-tense statements tailored for a "
        f"{theme} themed relaxation track. Provide them as a numbered list."
    )
    response = client.responses.create(model="gpt-4o-mini", input=prompt)
    text = parse_openai_text(response)
    affirmations: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "." in line[:4]:
            line = line.split(".", 1)[1].strip()
        affirmations.append(line)
    return affirmations[:count]


def generate_metadata(
    client: "OpenAIClient", affirmations: Sequence[str], noise_type: str, theme: str
) -> Tuple[str, str, str]:
    joined_affirmations = "\n".join(f"- {a}" for a in affirmations)
    prompt = (
        "Create a compelling YouTube title and description for the following "
        "subliminal affirmations track. Also suggest a vivid thumbnail prompt "
        "suitable for text-to-image generation. Return a JSON object with "
        "keys: title, description, thumbnail_prompt.\n\n"
        f"Noise type: {noise_type}\n"
        f"Theme: {theme}\n"
        f"Affirmations:\n{joined_affirmations}"
    )
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        response_format={"type": "json_object"},
    )
    content = parse_openai_text(response)
    payload: Dict[str, str] = json.loads(content)
    return payload["title"].strip(), payload["description"].strip(), payload["thumbnail_prompt"].strip()


def generate_thumbnail(client: "OpenAIClient", prompt: str, output_path: Path) -> Path:
    response = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
    image_b64 = response.data[0].b64_json
    output_path.write_bytes(base64.b64decode(image_b64))
    return output_path


# --- Affirmation persistence -------------------------------------------------------


def save_affirmations(affirmations: Sequence[str], output_dir: Path, stem: str) -> Path:
    ensure_directory(output_dir)
    path = output_dir / f"{stem}_affirmations.txt"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(affirmations))
    return path


# --- Streamlit generator integration ----------------------------------------------


def render_audio(
    generator: Callable[..., Path],
    settings: GeneratorSettings,
    affirmations: Sequence[str],
    output_dir: Path,
) -> Path:
    ensure_directory(output_dir)
    generator_kwargs = dict(
        noise_type=settings.noise_type,
        affirmations=list(affirmations),
        playback_speed=settings.playback_speed,
        output_dir=str(output_dir),
        auto_layer=settings.auto_layer,
        layer_count=settings.layer_count,
        layer_variation=settings.layer_variation,
        center_focus=settings.center_focus,
        subconscious_intensity=settings.subconscious_intensity,
        conscious_intensity=settings.conscious_intensity,
        override_intensity=settings.override_intensity,
    )
    if settings.layer_seed is not None:
        generator_kwargs["layer_seed"] = settings.layer_seed

    result = generator(**generator_kwargs)
    if isinstance(result, (list, tuple)):
        result = result[0]
    if isinstance(result, dict):
        result = result.get("audio_path")
    if not result:
        raise ValueError("generate_subliminal_audio did not return an audio path.")
    return Path(result)


# --- Selenium automation -----------------------------------------------------------


def configure_webdriver(download_dir: Path) -> webdriver.Chrome:
    ensure_directory(download_dir)
    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)
    driver_path = os.environ.get("CHROMEDRIVER_PATH")
    if driver_path:
        return webdriver.Chrome(driver_path, options=options)
    return webdriver.Chrome(options=options)


def wait_for_file(directory: Path, suffix: str, timeout: int = 600, existing: Optional[Iterable[Path]] = None) -> Path:
    deadline = time.time() + timeout
    known = {p.resolve() for p in existing or []}
    while time.time() < deadline:
        for candidate in directory.glob(f"*{suffix}"):
            if candidate.stat().st_size > 0 and candidate.resolve() not in known:
                return candidate
        time.sleep(2)
    raise TimeoutError(f"No file with suffix {suffix} found in {directory} within {timeout} seconds")


def convert_audio_to_video(audio_path: Path, download_dir: Path) -> Path:
    driver = configure_webdriver(download_dir)
    try:
        driver.get("https://www.onlineconverter.com/audio-to-video")

        file_input = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        file_input.send_keys(str(audio_path.resolve()))

        convert_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit'][value='Convert']"))
        )
        convert_button.click()

        download_link = WebDriverWait(driver, 300).until(
            EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Download"))
        )
        existing = list(download_dir.glob("*.mp4"))
        download_link.click()

        video_file = wait_for_file(download_dir, ".mp4", existing=existing)
    finally:
        driver.quit()
    return video_file


# --- YouTube upload ----------------------------------------------------------------


def build_youtube_client(token_path: Path, client_secrets: Path):
    Request, Credentials, build, _, InstalledAppFlow = _load_google_client_dependencies()

    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            creds = flow.run_local_server(port=0)
        with token_path.open("w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload_video_to_youtube(
    youtube,
    video_path: Path,
    title: str,
    description: str,
    thumbnail_path: Path,
    tags: Optional[Sequence[str]] = None,
    privacy_status: str = "public",
) -> str:
    _, _, _, MediaFileUpload, _ = _load_google_client_dependencies()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": list(tags or []),
            "categoryId": "22",  # People & Blogs
        },
        "status": {"privacyStatus": privacy_status},
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    video_id = response["id"]

    thumbnail_media = MediaFileUpload(str(thumbnail_path), mimetype="image/png")
    youtube.thumbnails().set(videoId=video_id, media_body=thumbnail_media).execute()
    return video_id


# --- Run orchestration -------------------------------------------------------------


def choose_random_settings() -> GeneratorSettings:
    return GeneratorSettings(
        noise_type=random.choice(load_noise_options()),
        affirmation_count=random.choice(AFFIRMATION_COUNT_OPTIONS),
        playback_speed=random.choice(PLAYBACK_SPEED_OPTIONS),
        theme=random.choice(AFFIRMATION_THEME_OPTIONS),
        auto_layer=True,
        layer_count=random.choice(LAYER_COUNT_OPTIONS[1:]),
        layer_variation=random.choice(LAYER_VARIATION_OPTIONS),
        layer_seed=random.randint(0, 9999),
        center_focus=random.choice(CENTER_FOCUS_OPTIONS),
        subconscious_intensity=random.choice(SUBCONSCIOUS_INTENSITY_OPTIONS),
        conscious_intensity=random.choice(CONSCIOUS_INTENSITY_OPTIONS),
        override_intensity=random.choice(OVERRIDE_INTENSITY_OPTIONS),
    )


def collect_manual_settings(*, require_theme: bool = False) -> Optional[GeneratorSettings]:
    response = prompt_user("Do you want to generate a new video? (Y/N): ", {"y", "n"}).lower()
    if response != "y":
        return None

    noise_options = list(load_noise_options())
    print("Available noise types:")
    for idx, name in enumerate(noise_options, start=1):
        print(f"  {idx}. {name}")

    while True:
        selection = prompt_user("Select noise type by number: ")
        if selection.isdigit() and 1 <= int(selection) <= len(noise_options):
            noise_type = noise_options[int(selection) - 1]
            break
        print("Invalid selection. Please try again.")

    theme = "Calm focus"
    if require_theme:
        theme_input = prompt_user(
            "Affirmation theme (e.g. Calm confidence): "
        ).strip()
        if theme_input:
            theme = theme_input

    while True:
        affirmation_count_input = prompt_user("Number of affirmations: ")
        if affirmation_count_input.isdigit() and int(affirmation_count_input) > 0:
            affirmation_count = int(affirmation_count_input)
            break
        print("Please enter a positive integer.")

    while True:
        speed_input = prompt_user("Playback speed (e.g. 1.0): ")
        try:
            playback_speed = float(speed_input)
            if playback_speed > 0:
                break
        except ValueError:
            pass
        print("Playback speed must be a positive number.")

    auto_layer_choice = prompt_user("Enable auto-layered ambience? (Y/N): ", {"y", "n"}).lower()
    auto_layer = auto_layer_choice == "y"
    layer_count = 1
    layer_variation = 0.0
    layer_seed: Optional[int] = None

    if auto_layer:
        while True:
            layer_input = prompt_user("Number of noise layers: ")
            if layer_input.isdigit() and int(layer_input) > 0:
                layer_count = int(layer_input)
                break
            print("Please enter a positive integer for the layer count.")

        while True:
            variation_input = prompt_user("Layer variation (0.0 - 1.0, e.g. 0.45): ")
            try:
                layer_variation = float(variation_input)
                if 0.0 <= layer_variation <= 1.0:
                    break
            except ValueError:
                pass
            print("Layer variation must be a number between 0.0 and 1.0.")

        seed_input = prompt_user("Layer random seed (optional, press enter to skip): ")
        if seed_input:
            try:
                layer_seed = int(seed_input)
            except ValueError:
                print("Seed must be numeric. Leaving it unset.")

    def ask_ratio(label: str, default: float) -> float:
        while True:
            raw = input(f"{label} (0.0 - 1.0, default {default:.2f}): ").strip()
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError:
                print("Value must be a number between 0.0 and 1.0.")
                continue
            if 0.0 <= value <= 1.0:
                return value
            print("Value must be between 0.0 and 1.0.")

    center_focus = ask_ratio("Center technique focus", 0.55)
    subconscious_intensity = ask_ratio("Subconscious hack intensity", 0.4)
    conscious_intensity = ask_ratio("Conscious hack intensity", 0.25)
    override_intensity = ask_ratio("Mind override layering", 0.35)

    settings = GeneratorSettings(
        noise_type=noise_type,
        affirmation_count=affirmation_count,
        playback_speed=playback_speed,
        theme=theme,
        auto_layer=auto_layer,
        layer_count=layer_count,
        layer_variation=layer_variation,
        layer_seed=layer_seed,
        center_focus=center_focus,
        subconscious_intensity=subconscious_intensity,
        conscious_intensity=conscious_intensity,
        override_intensity=override_intensity,
    )

    print("\nSelected configuration:")
    print(settings)
    confirm = prompt_user("Proceed with these settings? (Y/N): ", {"y", "n"}).lower()
    if confirm != "y":
        return collect_manual_settings()
    return settings


def collect_affirmations_interactively(expected_count: int) -> List[str]:
    print(
        "Enter your affirmations one per line. Submit an empty line to finish. "
        f"You indicated a target of {expected_count} affirmations, but you can "
        "provide more or fewer if desired."
    )
    affirmations: List[str] = []
    while True:
        entry = input(f"Affirmation {len(affirmations) + 1}: ").strip()
        if not entry:
            if affirmations:
                break
            print("At least one affirmation is required.")
            continue
        affirmations.append(entry)
        if len(affirmations) >= expected_count:
            more = prompt_user("Add another affirmation? (Y/N): ", {"y", "n"}).lower()
            if more != "y":
                break
    return affirmations


def prompt_multiline(prompt: str) -> str:
    print(prompt)
    print("Enter a blank line to finish.")
    lines: List[str] = []
    while True:
        line = input().rstrip()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def build_output_stem(settings: GeneratorSettings) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return sanitize_filename(f"{settings.noise_type}_{timestamp}")


def run_basic_generator(
    generator: Callable[..., Path] = generate_subliminal_audio,
    client_secrets_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> None:
    """Interactive workflow that skips the AI assistant."""

    youtube_client = None
    while True:
        settings = collect_manual_settings()
        if settings is None:
            print("Exiting manual generator.")
            return

        stem = build_output_stem(settings)
        session_dir = ensure_directory(Path("manual_artifacts") / stem)

        affirmations = collect_affirmations_interactively(settings.affirmation_count)
        text_path = save_affirmations(affirmations, session_dir, stem)
        print(f"Saved affirmations to {text_path}")

        audio_path = render_audio(generator, settings, affirmations, session_dir)
        print(f"Generated audio: {audio_path}")

        video_path: Optional[Path] = None
        convert_choice = prompt_user("Convert audio to video now? (Y/N): ", {"y", "n"}).lower()
        if convert_choice == "y":
            video_path = convert_audio_to_video(audio_path, session_dir)
            print(f"Video downloaded to {video_path}")

        if not video_path:
            continue

        if not client_secrets_path:
            print(
                "GOOGLE_CLIENT_SECRETS is not set. Skipping YouTube upload."
            )
            continue

        upload_choice = prompt_user("Upload the video to YouTube? (Y/N): ", {"y", "n"}).lower()
        if upload_choice != "y":
            continue

        if youtube_client is None:
            youtube_client = build_youtube_client(
                token_path or Path("token.json"), client_secrets_path
            )

        title = input("Video title: ").strip()
        if not title:
            title = stem.replace("_", " ").title()

        description = prompt_multiline("Enter the video description")
        if not description:
            description = "Generated with the manual subliminal audio workflow."

        thumbnail_path: Optional[Path] = None
        if prompt_user("Provide a thumbnail image path? (Y/N): ", {"y", "n"}).lower() == "y":
            candidate = Path(input("Thumbnail image path: ").strip()).expanduser()
            if candidate.exists():
                thumbnail_path = candidate
            else:
                print(f"Thumbnail not found at {candidate}. Skipping upload.")

        if not thumbnail_path:
            print("A thumbnail is required for upload. Skipping YouTube publishing.")
            continue

        video_id = upload_video_to_youtube(
            youtube_client,
            video_path=video_path,
            title=title,
            description=description,
            thumbnail_path=thumbnail_path,
            tags=[settings.noise_type, "subliminal", "affirmations"],
        )
        print(f"Video uploaded successfully: https://youtube.com/watch?v={video_id}")


def run_single_workflow(
    settings: GeneratorSettings,
    openai_client: "OpenAIClient",
    generator: Callable[..., Path] = generate_subliminal_audio,
    base_output_dir: Path = Path("artifacts"),
    youtube_client=None,
    interactive: bool = False,
) -> Optional[GeneratedAssets]:
    stem = build_output_stem(settings)
    session_dir = ensure_directory(base_output_dir / stem)

    affirmations = generate_affirmations(openai_client, settings.theme, settings.affirmation_count)
    if not affirmations:
        raise ValueError("OpenAI did not return any affirmations.")
    text_path = save_affirmations(affirmations, session_dir, stem)

    title, description, thumbnail_prompt = generate_metadata(
        openai_client, affirmations, settings.noise_type, settings.theme
    )

    audio_path = render_audio(generator, settings, affirmations, session_dir)

    if interactive:
        confirm = prompt_user("Audio generated. Proceed to video creation? (Y/N): ", {"y", "n"}).lower()
        if confirm != "y":
            print("Aborting before video creation per user request.")
            return None

    video_path = convert_audio_to_video(audio_path, session_dir)

    thumbnail_path = session_dir / f"{stem}_thumbnail.png"
    generate_thumbnail(openai_client, thumbnail_prompt, thumbnail_path)

    if youtube_client:
        video_id = upload_video_to_youtube(
            youtube_client,
            video_path=video_path,
            title=title,
            description=description,
            thumbnail_path=thumbnail_path,
            tags=[settings.noise_type, "subliminal", "affirmations"],
        )
        print(f"Video uploaded successfully. Video ID: {video_id}")

    return GeneratedAssets(
        text_path=text_path,
        audio_path=audio_path,
        video_path=video_path,
        thumbnail_path=thumbnail_path,
    )


# --- Diagnostics & benchmarking -----------------------------------------------------


def _human_readable_bytes(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    value = float(size)
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _summarize_exception(exc: BaseException) -> str:
    tail = "\n".join(traceback.format_exc().strip().splitlines()[-3:])
    return f"{exc.__class__.__name__}: {exc}\n{tail}".strip()


def run_diagnostics(include_benchmark: bool = True) -> List[DiagnosticResult]:
    """Run environment diagnostics and optional audio generator benchmark."""

    results: List[DiagnosticResult] = []

    def record(
        name: str,
        func: Callable[[], Tuple[str, str] | Tuple[str] | str | None],
    ) -> None:
        start = time.perf_counter()
        status: Literal["pass", "fail", "warn"] = "pass"
        details = ""
        try:
            outcome = func()
            if isinstance(outcome, tuple):
                if len(outcome) == 2:
                    status, details = outcome  # type: ignore[misc]
                elif len(outcome) == 1:
                    (details,) = outcome
            elif isinstance(outcome, str):
                details = outcome
        except Exception as exc:  # noqa: BLE001
            status = "fail"
            details = _summarize_exception(exc)
        duration = time.perf_counter() - start
        results.append(DiagnosticResult(name=name, status=status, details=details.strip(), duration=duration))

    def check_python_version() -> Tuple[str, str]:
        version = ".".join(str(part) for part in sys.version_info[:3])
        if sys.version_info < (3, 9):
            return "warn", f"Python {version} detected. Python 3.9+ is recommended."
        return "pass", f"Python {version} detected."

    def check_ffmpeg() -> Tuple[str, str]:
        from pydub import AudioSegment

        converter = AudioSegment.converter
        if not converter:
            return "warn", "pydub could not determine the ffmpeg binary location."
        converter_path = Path(converter)
        if not converter_path.exists():
            return "warn", f"Expected ffmpeg at {converter_path}, but it was not found."
        return "pass", f"ffmpeg located at {converter_path}"

    def check_pyttsx3() -> Tuple[str, str]:
        if importlib.util.find_spec("pyttsx3") is None:
            return "warn", "pyttsx3 is not installed. Install it to enable speech synthesis."
        import pyttsx3

        engine = pyttsx3.init()
        try:
            voices = engine.getProperty("voices") or []
        finally:
            engine.stop()
        if voices:
            return "pass", f"pyttsx3 initialised with {len(voices)} available voices."
        return "warn", "pyttsx3 initialised but no voices were reported."

    def check_openai_package() -> Tuple[str, str]:
        if importlib.util.find_spec("openai") is None:
            return "warn", "openai package missing. Install it to enable AI workflows."
        return "pass", "openai package available."

    def check_google_clients() -> Tuple[str, str]:
        try:
            _load_google_client_dependencies()
        except ModuleNotFoundError as exc:
            return "warn", str(exc)
        return "pass", "Google API client libraries available."

    def check_selenium() -> Tuple[str, str]:
        if shutil.which("chromedriver") is None:
            return "warn", "ChromeDriver not found on PATH. Provide CHROMEDRIVER_PATH or install it."
        return "pass", "ChromeDriver detected on PATH."

    def check_streamlit_generator() -> Tuple[str, str]:
        module_name = generate_subliminal_audio.__module__
        return "pass", f"Using {module_name}.{generate_subliminal_audio.__name__}"

    def benchmark_generator() -> Tuple[str, str]:
        sample_affirmations = [
            "I command my subconscious to absorb aligned truths.",
            "My conscious mind welcomes empowered focus.",
            "Every breath layers confidence through my aura.",
        ]
        with tempfile.TemporaryDirectory(prefix="subliminal_diag_") as tmp_dir:
            output_path = generate_subliminal_audio(
                noise_type="center-pulse",
                affirmations=sample_affirmations,
                playback_speed=1.0,
                output_dir=tmp_dir,
                voice_rate=150,
                voice_volume=0.85,
                base_filename="diagnostic_mix",
                auto_layer=True,
                layer_count=2,
                layer_variation=0.35,
                layer_seed=1234,
                center_focus=0.6,
                subconscious_intensity=0.4,
                conscious_intensity=0.25,
                override_intensity=0.35,
            )
            size = output_path.stat().st_size
        return "pass", f"Generated diagnostic mix ({_human_readable_bytes(size)})."

    record("Python version", check_python_version)
    record("ffmpeg availability", check_ffmpeg)
    record("pyttsx3 speech engine", check_pyttsx3)
    record("openai package", check_openai_package)
    record("Google client libraries", check_google_clients)
    record("ChromeDriver", check_selenium)
    record("Streamlit generator", check_streamlit_generator)
    if include_benchmark:
        record("Audio generator benchmark", benchmark_generator)

    return results


def run_diagnostics_cli(include_benchmark: bool = True) -> None:
    """Pretty-print diagnostics to the console."""

    results = run_diagnostics(include_benchmark=include_benchmark)
    if not results:
        print("No diagnostics were executed.")
        return

    name_width = max(len(result.name) for result in results) + 2
    status_icons = {"pass": "✔", "warn": "⚠", "fail": "✖"}

    print("\nDiagnostic report")
    print("=" * (name_width + 32))

    for result in results:
        icon = status_icons.get(result.status, "•")
        duration = f"{result.duration:.2f}s" if result.duration is not None else "-"
        details = result.details.replace("\n", " ").strip()
        print(f"{icon} {result.name.ljust(name_width)} {result.status.upper():<5} {duration:>8}  {details}")

    failures = [r for r in results if r.status == "fail"]
    warnings = [r for r in results if r.status == "warn"]

    if failures:
        print("\nOne or more diagnostics failed. Review the details above before running automation workflows.")
    elif warnings:
        print("\nDiagnostics completed with warnings. Some features may be unavailable until they are addressed.")
    else:
        print("\nAll diagnostics passed. The automation agent is ready to run.")


def run_manual_mode(openai_client: "OpenAIClient", youtube_client) -> None:
    while True:
        settings = collect_manual_settings(require_theme=True)
        if settings is None:
            print("Exiting manual mode.")
            return
        assets = run_single_workflow(settings, openai_client, youtube_client=youtube_client, interactive=True)
        if assets:
            print(f"Workflow complete. Video saved to {assets.video_path}")


def run_auto_mode(openai_client: "OpenAIClient", youtube_client, interval_minutes: int = 5) -> None:
    print("Starting auto mode. Press Ctrl+C to stop.")
    while True:
        settings = choose_random_settings()
        print(f"Running automated session with settings: {settings}")
        assets = run_single_workflow(settings, openai_client, youtube_client=youtube_client)
        if assets:
            print(f"Automated workflow complete. Video saved to {assets.video_path}")
        print(f"Waiting {interval_minutes} minutes before next run...")
        time.sleep(interval_minutes * 60)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Subliminal automation agent")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run environment diagnostics and exit.",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the audio rendering benchmark during diagnostics.",
    )

    args = parser.parse_args(argv)

    if args.diagnostics:
        run_diagnostics_cli(include_benchmark=not args.no_benchmark)
        return

    client_secrets_env = os.environ.get("GOOGLE_CLIENT_SECRETS")
    client_secrets_path = Path(client_secrets_env).expanduser() if client_secrets_env else None
    token_path = Path(os.environ.get("YOUTUBE_TOKEN", "token.json")).expanduser()

    print("Select workflow mode:")
    print("  [1] Manual generator (no AI agent)")
    print("  [2] AI agent – manual confirmation")
    print("  [3] AI agent – automated schedule")
    print("  [4] Diagnostics & benchmark")
    selection = prompt_user("Choose an option (1/2/3/4): ", {"1", "2", "3", "4"})

    if selection == "1":
        run_basic_generator(
            client_secrets_path=client_secrets_path,
            token_path=token_path,
        )
        return

    if selection == "4":
        include_benchmark = (
            prompt_user("Run the audio benchmark? (Y/N): ", {"y", "n"}).lower() == "y"
        )
        run_diagnostics_cli(include_benchmark=include_benchmark)
        return

    openai_client = build_openai_client()

    youtube_client = None
    if client_secrets_path:
        youtube_client = build_youtube_client(token_path, client_secrets_path)
    else:
        print("GOOGLE_CLIENT_SECRETS not set. YouTube upload will be skipped.")

    if selection == "2":
        run_manual_mode(openai_client, youtube_client)
    else:
        run_auto_mode(openai_client, youtube_client)


if __name__ == "__main__":
    main(sys.argv[1:])


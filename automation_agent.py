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

import base64
import importlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def _load_google_client_dependencies():
    """Return Google client helpers, ensuring dependencies are installed."""

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

    return _Request, _Credentials, _build, _MediaFileUpload, _InstalledAppFlow


Request, Credentials, build, MediaFileUpload, InstalledAppFlow = _load_google_client_dependencies()

# The Streamlit generator is expected to live in the same repository.  Update
# the import below to match the actual module name if it differs.
STREAMLIT_APP_MODULE = os.environ.get("STREAMLIT_APP_MODULE", "streamlit_app")
_streamlit_module = importlib.import_module(STREAMLIT_APP_MODULE)
generate_subliminal_audio = getattr(_streamlit_module, "generate_subliminal_audio")


# --- Configuration -----------------------------------------------------------------

DEFAULT_NOISE_TYPES: Sequence[str] = (
    "purple",
    "brown",
    "pink",
    "center",
    "binaural",
    "white",
)

AFFIRMATION_COUNT_OPTIONS: Sequence[int] = (5, 8, 10, 12, 15, 20)

PLAYBACK_SPEED_OPTIONS: Sequence[float] = (0.75, 1.0, 1.15, 1.25, 1.5)

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


@dataclass
class GeneratedAssets:
    """Paths to artifacts created during a single automation run."""

    text_path: Path
    audio_path: Path
    video_path: Path
    thumbnail_path: Path


# --- Helper utilities --------------------------------------------------------------


def build_openai_client() -> OpenAI:
    """Instantiate an OpenAI client using the environment API key."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
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


def generate_affirmations(client: OpenAI, theme: str, count: int) -> List[str]:
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


def generate_metadata(client: OpenAI, affirmations: Sequence[str], noise_type: str) -> Tuple[str, str, str]:
    joined_affirmations = "\n".join(f"- {a}" for a in affirmations)
    prompt = (
        "Create a compelling YouTube title and description for the following "
        "subliminal affirmations track. Also suggest a vivid thumbnail prompt "
        "suitable for text-to-image generation. Return a JSON object with "
        "keys: title, description, thumbnail_prompt.\n\n"
        f"Noise type: {noise_type}\n"
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


def generate_thumbnail(client: OpenAI, prompt: str, output_path: Path) -> Path:
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
    result = generator(
        noise_type=settings.noise_type,
        affirmations=list(affirmations),
        playback_speed=settings.playback_speed,
        output_dir=str(output_dir),
    )
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
    )


def collect_manual_settings() -> Optional[GeneratorSettings]:
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

    settings = GeneratorSettings(
        noise_type=noise_type,
        affirmation_count=affirmation_count,
        playback_speed=playback_speed,
    )

    print("\nSelected configuration:")
    print(settings)
    confirm = prompt_user("Proceed with these settings? (Y/N): ", {"y", "n"}).lower()
    if confirm != "y":
        return collect_manual_settings()
    return settings


def build_output_stem(settings: GeneratorSettings) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return sanitize_filename(f"{settings.noise_type}_{timestamp}")


def run_single_workflow(
    settings: GeneratorSettings,
    openai_client: OpenAI,
    generator: Callable[..., Path] = generate_subliminal_audio,
    base_output_dir: Path = Path("artifacts"),
    youtube_client=None,
    interactive: bool = False,
) -> Optional[GeneratedAssets]:
    stem = build_output_stem(settings)
    session_dir = ensure_directory(base_output_dir / stem)

    affirmations = generate_affirmations(openai_client, settings.noise_type, settings.affirmation_count)
    if not affirmations:
        raise ValueError("OpenAI did not return any affirmations.")
    text_path = save_affirmations(affirmations, session_dir, stem)

    title, description, thumbnail_prompt = generate_metadata(openai_client, affirmations, settings.noise_type)

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


def run_manual_mode(openai_client: OpenAI, youtube_client) -> None:
    while True:
        settings = collect_manual_settings()
        if settings is None:
            print("Exiting manual mode.")
            return
        assets = run_single_workflow(settings, openai_client, youtube_client=youtube_client, interactive=True)
        if assets:
            print(f"Workflow complete. Video saved to {assets.video_path}")


def run_auto_mode(openai_client: OpenAI, youtube_client, interval_minutes: int = 5) -> None:
    print("Starting auto mode. Press Ctrl+C to stop.")
    while True:
        settings = choose_random_settings()
        print(f"Running automated session with settings: {settings}")
        assets = run_single_workflow(settings, openai_client, youtube_client=youtube_client)
        if assets:
            print(f"Automated workflow complete. Video saved to {assets.video_path}")
        print(f"Waiting {interval_minutes} minutes before next run...")
        time.sleep(interval_minutes * 60)


def main() -> None:
    openai_client = build_openai_client()

    client_secrets_path = os.environ.get("GOOGLE_CLIENT_SECRETS")
    youtube_client = None
    if client_secrets_path:
        token_path = Path(os.environ.get("YOUTUBE_TOKEN", "token.json"))
        youtube_client = build_youtube_client(token_path, Path(client_secrets_path))
    else:
        print("GOOGLE_CLIENT_SECRETS not set. YouTube upload will be skipped.")

    mode = prompt_user("Select mode: [1] Manual  [2] Auto : ", {"1", "2"})
    if mode == "1":
        run_manual_mode(openai_client, youtube_client)
    else:
        run_auto_mode(openai_client, youtube_client)


if __name__ == "__main__":
    main()


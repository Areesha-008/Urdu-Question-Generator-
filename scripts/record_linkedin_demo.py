"""Record a LinkedIn-ready operability demo of the live Urdu Question Studio.

Requires the app at http://127.0.0.1:8011 and:
    pip install playwright
    python -m playwright install chromium
    ffmpeg on PATH (Homebrew or imageio-ffmpeg)
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from playwright.async_api import Page, Route, async_playwright

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'results' / 'demo'
APP = 'http://127.0.0.1:8011'
SIZE = {'width': 1920, 'height': 1080}

DEMO_CSS = """
html, body { overflow: hidden; }
main { max-width: 1500px; padding: 28px 72px 118px; }
header { padding: 28px 0 20px; max-width: 820px; }
h1 { font-size: 46px; margin: 12px 0 14px; }
textarea { height: 118px; }
article { min-height: 150px; }
footer { display: none; }
#demo-caption {
  position: fixed; left: 80px; right: 80px; bottom: 32px;
  text-align: center; z-index: 20;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 28px; line-height: 1.35; color: #101828;
}
"""


async def caption(page: Page, text: str) -> None:
    await page.evaluate(
        """text => {
          let el = document.getElementById('demo-caption');
          if (!el) {
            el = document.createElement('div');
            el.id = 'demo-caption';
            document.body.appendChild(el);
          }
          el.textContent = text;
        }""",
        text,
    )


async def prepare_app(page: Page) -> None:
    await page.add_style_tag(content=DEMO_CSS)
    await page.wait_for_function(
        "document.getElementById('health')?.textContent === 'Model ready'",
        timeout=15000,
    )


async def delay_generate(route: Route) -> None:
    response = await route.fetch()
    await asyncio.sleep(1.7)
    await route.fulfill(response=response)


async def type_field(page: Page, selector: str, text: str, delay: int) -> None:
    await page.click(selector, timeout=10000)
    await page.fill(selector, '')
    await page.type(selector, text, delay=delay)


async def generate(page: Page) -> None:
    await page.click('#submit')
    await page.wait_for_function(
        "document.querySelector('#message')?.textContent.includes('Questions ready')",
        timeout=30000,
    )


async def hold(page: Page, seconds: float) -> None:
    await page.wait_for_timeout(int(seconds * 1000))


def find_ffmpeg() -> str:
    for candidate in (
        shutil.which('ffmpeg'),
        '/opt/homebrew/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
    ):
        if candidate and Path(candidate).exists():
            return candidate
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(source: Path, dest: Path) -> None:
    ffmpeg = find_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, '-y',
        '-i', str(source),
        '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
        '-crf', '18', '-preset', 'medium', '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        str(dest),
    ]
    subprocess.run(command, check=True)


async def record() -> Path:
    DEMO.mkdir(parents=True, exist_ok=True)
    raw_dir = DEMO / 'raw'
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir()
    title = (DEMO / 'title.html').resolve().as_uri()
    end = (DEMO / 'end.html').resolve().as_uri()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=SIZE,
            device_scale_factor=1,
            record_video_dir=str(raw_dir),
            record_video_size=SIZE,
        )
        page = await context.new_page()
        await page.route('**/generate', delay_generate)

        await page.goto(title)
        await hold(page, 5.2)

        await page.goto(APP, wait_until='networkidle')
        await prepare_app(page)
        await caption(page, 'The trained model is live in the browser.')
        await hold(page, 2.6)

        await caption(page, 'Type a sentence, then mark the answer to ask about.')
        await type_field(page, '#sentence', 'پاکستان کا دارالحکومت اسلام آباد ہے۔', 48)
        await hold(page, 0.7)
        await type_field(page, '#answer', 'اسلام آباد', 70)
        await page.locator('h1').click()
        await hold(page, 1.0)

        await caption(page, 'Generate a question with greedy search and beam search.')
        await generate(page)
        await caption(page, 'Both decoders ask: what is the capital of Pakistan?')
        await hold(page, 5.4)

        await caption(page, 'Same model. Now ask about a quantity.')
        await page.click('button[data-example="2"]')
        await hold(page, 2.0)
        await generate(page)
        await caption(page, 'Beam search tightens the wording: how many books?')
        await hold(page, 5.6)

        await page.goto(end)
        await hold(page, 6.0)

        video = page.video
        await context.close()
        await browser.close()
        assert video is not None
        path = Path(await video.path())
        return path


def poster(video: Path, dest: Path) -> None:
    ffmpeg = find_ffmpeg()
    subprocess.run(
        [ffmpeg, '-y', '-ss', '12', '-i', str(video), '-frames:v', '1', '-q:v', '2', str(dest)],
        check=True,
    )


def main() -> None:
    raw = asyncio.run(record())
    mp4 = DEMO / 'urdu-qg-linkedin-demo.mp4'
    encode(raw, mp4)
    poster(mp4, DEMO / 'poster.jpg')
    print(mp4)
    print(f'{mp4.stat().st_size / 1_000_000:.1f} MB')


if __name__ == '__main__':
    main()

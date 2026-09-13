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
main { max-width: 1500px; padding: 28px 72px 110px; }
header { padding: 28px 0 20px; max-width: 820px; }
h1 { font-size: 46px; margin: 12px 0 14px; }
textarea { height: 118px; }
article { min-height: 150px; }
footer { display: none; }
#demo-caption {
  position: fixed; left: 72px; right: 72px; bottom: 28px;
  text-align: center; font-size: 22px; line-height: 1.4;
  letter-spacing: 0.01em; color: #344e6c; z-index: 20;
}
#demo-mouse {
  position: fixed; width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid #101828; background: #c7a463;
  transform: translate(-50%, -50%); pointer-events: none; z-index: 30;
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
    await page.evaluate(
        """css => {
          const style = document.createElement('style');
          style.textContent = css;
          document.head.appendChild(style);
          const mouse = document.createElement('div');
          mouse.id = 'demo-mouse';
          document.body.appendChild(mouse);
          document.addEventListener('mousemove', event => {
            mouse.style.left = event.clientX + 'px';
            mouse.style.top = event.clientY + 'px';
          });
        }""",
        DEMO_CSS,
    )
    await page.wait_for_function(
        "document.getElementById('health')?.textContent === 'Model ready'",
        timeout=15000,
    )


async def delay_generate(route: Route) -> None:
    response = await route.fetch()
    await asyncio.sleep(1.15)
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
        await hold(page, 4.6)

        await page.goto(APP, wait_until='networkidle')
        await prepare_app(page)
        await caption(page, 'A live model. A sentence in. A question out.')
        await hold(page, 2.4)

        await caption(page, 'Write an Urdu sentence, then mark the answer to ask about.')
        await type_field(page, '#sentence', 'پاکستان کا دارالحکومت اسلام آباد ہے۔', 42)
        await hold(page, 0.6)
        await type_field(page, '#answer', 'اسلام آباد', 55)
        await hold(page, 0.8)

        await caption(page, 'Generate with greedy decoding and beam search.')
        await generate(page)
        await caption(page, 'Both decoders ask: what is the capital of Pakistan?')
        await hold(page, 4.2)

        await caption(page, 'Same model. Now a how-many question.')
        await page.click('button[data-example="2"]')
        await hold(page, 1.6)
        await generate(page)
        await caption(page, 'Beam search repairs the agreement: کتنی, not کتنے.')
        await hold(page, 4.6)

        await page.goto(end)
        await hold(page, 5.4)

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

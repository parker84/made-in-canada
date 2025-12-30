#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import colorlog

# Setup colored logging
def setup_logger() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    ))
    logger = logging.getLogger("run_scrapes")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG") else logging.INFO)
    return logger

log = setup_logger()

# Track job states for status updates
queued_jobs: set[str] = set()  # Jobs waiting for semaphore
running_jobs: dict[str, float] = {}  # job_name -> start_time (actually executing)
completed_jobs: list[dict] = []
failed_jobs: list[dict] = []


@dataclass
class Job:
    name: str
    base: str
    store_type: Optional[str] = None  # e.g. "shopify"
    use_browser: bool = True
    max_categories: int = 1000
    url_regex: Optional[str] = None
    download_images: bool = False
    use_postgres: bool = True
    out: Optional[str] = None  # Only needed if downloading images or not using postgres


def build_cmd(job: Job) -> List[str]:
    cmd = ["python", "scrape_products.py", "--base", job.base]

    if job.out:
        cmd += ["--out", job.out]
    if job.use_browser:
        cmd.append("--use-browser")
    if job.store_type:
        cmd += ["--store-type", job.store_type]
    if job.max_categories is not None:
        cmd += ["--max-categories", str(job.max_categories)]
    if job.url_regex:
        cmd += [f"--url-regex={job.url_regex}"]
    if job.download_images:
        cmd.append("--download-images")
    if job.use_postgres:
        cmd.append("--use-postgres")

    return cmd


def looks_like_block(stderr_text: str, stdout_text: str) -> bool:
    combined = (stderr_text + "\n" + stdout_text).lower()
    return "403" in combined and "forbidden" in combined


def extract_stats_from_logs(stderr_text: str, stdout_text: str) -> dict:
    """Extract useful stats from the scraper output logs."""
    combined = stderr_text + "\n" + stdout_text
    stats = {
        "products_found": None,
        "products_saved": None,
        "categories_found": None,
        "images_extracted": None,
    }
    
    # Look for common patterns in the scraper output
    import re
    
    # "Found X product URLs"
    match = re.search(r'Found (\d+) product URLs?', combined)
    if match:
        stats["products_found"] = int(match.group(1))
    
    # "Saved X products to PostgreSQL"
    match = re.search(r'Saved (\d+) products? to PostgreSQL', combined)
    if match:
        stats["products_saved"] = int(match.group(1))
    
    # "Found X category pages"
    match = re.search(r'Found (\d+) category pages?', combined)
    if match:
        stats["categories_found"] = int(match.group(1))
    
    # Count image extractions
    image_matches = re.findall(r'🖼️ Extracted (\d+) image', combined)
    if image_matches:
        stats["images_extracted"] = sum(int(m) for m in image_matches)
    
    return stats


async def run_job(job: Job, sem: asyncio.Semaphore, logs_dir: Path, cooldown_s: float = 0.0, job_index: int = 0, total_jobs: int = 0) -> dict:
    # optional spacing between store jobs to smooth bursts
    if cooldown_s > 0:
        log.debug(f"⏳ [{job.name}] Waiting {cooldown_s}s cooldown before starting...")
        await asyncio.sleep(cooldown_s)

    cmd = build_cmd(job)
    stdout_path = logs_dir / f"{job.name}.out.log"
    stderr_path = logs_dir / f"{job.name}.err.log"

    # Mark as queued (waiting for semaphore)
    queued_jobs.add(job.name)
    queued_count = len(queued_jobs)
    running_count = len(running_jobs)
    log.info(f"📋 [{job.name}] Queued (waiting for slot) - {running_count} running, {queued_count} queued")

    async with sem:
        # Move from queued to running
        queued_jobs.discard(job.name)
        started_at = time.time()
        running_jobs[job.name] = started_at
        
        running_count = len(running_jobs)
        queued_count = len(queued_jobs)
        log.info(f"🚀 [{job.name}] Starting scrape of {job.base} ({running_count} running, {queued_count} queued)")
        log.debug(f"   Command: {' '.join(cmd)}")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Read output with periodic status updates
        stdout_chunks = []
        stderr_chunks = []
        last_status_time = time.time()
        status_interval = 30  # Log status every 30 seconds
        
        async def read_stream(stream, chunks, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                chunks.append(decoded)
                # Log interesting lines in real-time at debug level
                if any(keyword in decoded.lower() for keyword in ["error", "found", "saved", "✅", "❌", "⚠️", "🖼️", "⭐"]):
                    log.debug(f"   [{job.name}] {decoded.strip()}")
        
        # Read both streams concurrently
        await asyncio.gather(
            read_stream(proc.stdout, stdout_chunks, "stdout"),
            read_stream(proc.stderr, stderr_chunks, "stderr"),
        )
        
        await proc.wait()

    finished_at = time.time()
    duration = finished_at - started_at
    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    stdout_path.write_text(stdout_text)
    stderr_path.write_text(stderr_text)
    
    # Remove from running jobs
    running_jobs.pop(job.name, None)
    
    # Extract stats from logs
    stats = extract_stats_from_logs(stderr_text, stdout_text)
    
    is_blocked = looks_like_block(stderr_text, stdout_text)
    
    # Log completion with details
    running_count = len(running_jobs)
    queued_count = len(queued_jobs)
    completed_count = len(completed_jobs) + 1  # +1 for this job
    
    if proc.returncode == 0:
        products_info = f"products: {stats['products_saved'] or stats['products_found'] or '?'}"
        log.info(f"✅ [{job.name}] Completed in {duration:.1f}s ({duration/60:.1f}min) - {products_info} ({running_count} running, {queued_count} queued)")
        completed_jobs.append({"name": job.name, "duration": duration, "stats": stats})
    elif is_blocked:
        log.warning(f"🚫 [{job.name}] BLOCKED (403) after {duration:.1f}s - site may be blocking scrapers ({running_count} running, {queued_count} queued)")
        failed_jobs.append({"name": job.name, "reason": "403_blocked", "duration": duration})
    else:
        log.error(f"❌ [{job.name}] FAILED (exit code {proc.returncode}) after {duration:.1f}s ({running_count} running, {queued_count} queued)")
        # Log last few lines of stderr for debugging
        stderr_lines = stderr_text.strip().split('\n')[-5:]
        for line in stderr_lines:
            if line.strip():
                log.error(f"   {line.strip()}")
        failed_jobs.append({"name": job.name, "reason": f"exit_{proc.returncode}", "duration": duration})

    result = {
        "name": job.name,
        "base": job.base,
        "out": job.out,
        "cmd": cmd,
        "exit_code": proc.returncode,
        "duration_s": round(duration, 2),
        "duration_min": round(duration / 60, 2),
        "blocked_403": is_blocked,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "ts_start": started_at,
        "ts_end": finished_at,
        "stats": stats,
    }
    return result


async def status_reporter(stop: asyncio.Event, total_jobs: int, max_parallel: int):
    """Periodically report status of running jobs."""
    while not stop.is_set():
        await asyncio.sleep(30)  # Report every 30 seconds
        if stop.is_set():
            break
        
        now = time.time()
        running_count = len(running_jobs)
        queued_count = len(queued_jobs)
        completed_count = len(completed_jobs)
        failed_count = len(failed_jobs)
        
        log.info(f"📊 Status: ✅ {completed_count}/{total_jobs} done | 🏃 {running_count}/{max_parallel} running | 📋 {queued_count} queued | ❌ {failed_count} failed")
        
        if running_jobs:
            for name, start_time in running_jobs.items():
                elapsed = now - start_time
                log.info(f"   ⏳ {name}: running for {elapsed:.0f}s ({elapsed/60:.1f}min)")


async def main():
    # ✅ Keep this low if you use --use-browser (Playwright/Chromium is heavy).
    # Start with 2. If stable, try 3–4.
    max_parallel = int(os.getenv("SCRAPE_MAX_PARALLEL", "2"))

    # Optional: space out store starts a bit to avoid bursty behavior
    cooldown_s = float(os.getenv("SCRAPE_COOLDOWN_S", "1.0"))

    if not os.path.isdir("./logs"):
        os.makedirs("./logs", exist_ok=True)

    logs_dir = Path("./logs/scrape_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path("./logs/scrape_report.jsonl")

    jobs: List[Job] = [
        Job("roots", "https://www.roots.com", store_type="roots", url_regex=r"\.html"),
        Job("province_of_canada", "https://provinceofcanada.com", store_type="shopify"),
        Job("manmade", "https://manmadebrand.com/", store_type="shopify"),
        Job("tilley", "https://ca.tilley.com/", store_type="shopify"),
        Job("tentree", "https://www.tentree.com/", store_type="shopify"),
        Job("kamik", "https://www.kamik.com/", store_type="shopify"),
        Job("sheertex", "https://sheertex.com/", store_type="shopify"),
        Job("baffin", "https://www.baffin.com/", store_type="shopify"),
        Job("bushbalm", "https://bushbalm.com/", store_type="shopify"),
        Job("soma", "https://www.somachocolate.com/", store_type="shopify"),
        Job("stanfields", "https://www.stanfields.com/", store_type="shopify"),
        Job("balzacs", "https://balzacs.com/", store_type="shopify"),
        Job("muttonhead", "https://muttonheadstore.com/", store_type="shopify"),
        Job("naked_and_famous", "https://nakedandfamousdenim.com/", store_type="shopify"),
        Job("regimenlab", "https://regimenlab.ca/", store_type="shopify"),
        Job("craigs_cookies", "https://craigscookies.com/", store_type="shopify"),
        Job("jenny_bird", "https://jenny-bird.ca/", store_type="shopify"),
        Job("green_beaver", "https://greenbeaver.com/", store_type="shopify"),
        Job("manitobah", "https://www.manitobah.ca/", store_type="shopify"),
        Job("moose_knuckles", "https://www.mooseknucklescanada.com/", store_type="shopify"),
        Job("rheo_thompson", "https://rheothompson.com/", store_type="shopify"),
        Job("davids_tea", "https://davidstea.com/", store_type="shopify"),
        Job("rocky_mountain_soap", "https://www.rockymountainsoap.com/", store_type="shopify"),
        Job("kicking_horse", "https://kickinghorsecoffee.ca/", store_type="shopify"),
        Job("st_viateur", "https://stviateurbagel.com/", store_type="shopify"),
    ]

    total_jobs = len(jobs)
    start_time = time.time()
    
    log.info("=" * 60)
    log.info(f"🍁 Made in Canada Scraper - Starting {total_jobs} jobs")
    log.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"⚙️  Max parallel: {max_parallel}, Cooldown: {cooldown_s}s")
    log.info(f"📁 Logs: {logs_dir.resolve()}")
    log.info("=" * 60)
    
    # List all jobs
    log.info("📋 Jobs to run:")
    for i, j in enumerate(jobs, 1):
        log.info(f"   {i:2}. {j.name:<20} → {j.base}")

    sem = asyncio.Semaphore(max_parallel)

    # graceful ctrl+c
    stop = asyncio.Event()

    def handle_sigint(*_):
        log.warning("\n⚠️  Received interrupt signal, stopping gracefully...")
        stop.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # Start status reporter
    status_task = asyncio.create_task(status_reporter(stop, total_jobs, max_parallel))

    tasks = []
    for i, j in enumerate(jobs):
        if stop.is_set():
            break
        tasks.append(asyncio.create_task(run_job(j, sem, logs_dir, cooldown_s=cooldown_s, job_index=i, total_jobs=total_jobs)))

    results = []
    for t in asyncio.as_completed(tasks):
        if stop.is_set():
            break
        res = await t
        results.append(res)

        # append JSONL row as each job completes (so you keep progress if interrupted)
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        
        # Progress update
        completed = len(results)
        remaining = total_jobs - completed
        log.info(f"📈 Progress: {completed}/{total_jobs} complete ({remaining} remaining)")

    # Stop status reporter
    stop.set()
    status_task.cancel()
    try:
        await status_task
    except asyncio.CancelledError:
        pass

    # Summary
    total_duration = time.time() - start_time
    ok = sum(1 for r in results if r["exit_code"] == 0)
    fail = len(results) - ok
    blocked = sum(1 for r in results if r["blocked_403"])
    total_products = sum(r.get("stats", {}).get("products_saved") or r.get("stats", {}).get("products_found") or 0 for r in results)
    
    log.info("")
    log.info("=" * 60)
    log.info("🏁 SCRAPE RUN COMPLETE")
    log.info("=" * 60)
    log.info(f"⏱️  Total time: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    log.info(f"✅ Successful: {ok}")
    log.info(f"❌ Failed: {fail}")
    log.info(f"🚫 Blocked (403): {blocked}")
    log.info(f"📦 Total products scraped: ~{total_products}")
    log.info("")
    
    if completed_jobs:
        log.info("✅ Completed jobs:")
        for job in completed_jobs:
            stats = job.get("stats", {})
            products = stats.get("products_saved") or stats.get("products_found") or "?"
            log.info(f"   • {job['name']:<20} {job['duration']:.1f}s - {products} products")
    
    if failed_jobs:
        log.warning("❌ Failed jobs:")
        for job in failed_jobs:
            log.warning(f"   • {job['name']:<20} {job['duration']:.1f}s - {job['reason']}")
    
    log.info("")
    log.info(f"📄 Report: {report_path}")
    log.info(f"📁 Logs:   {logs_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())

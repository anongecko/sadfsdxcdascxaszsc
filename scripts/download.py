import requests
import os
import time
import sys
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import hashlib
from tqdm import tqdm
import signal
import tempfile
import shutil


class ChunkDownloader:
    def __init__(self, url, output_path, chunk_size=1024 * 1024, num_threads=8, expected_sha256=None):
        self.url = url
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.num_threads = num_threads
        self.expected_sha256 = expected_sha256
        self.temp_dir = tempfile.mkdtemp()
        self.chunks_queue = Queue()
        self.downloaded_chunks = set()
        self.lock = threading.Lock()
        self.failed_chunks = Queue()
        self.stop_event = threading.Event()

        # Initialize progress tracking
        self.total_size = 0
        self.current_size = 0
        self.start_time = time.time()
        self.speeds = []

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        print("\nGracefully stopping download. Progress will be saved...")
        self.stop_event.set()

    def get_file_info(self):
        response = requests.head(self.url, allow_redirects=True)
        self.total_size = int(response.headers.get("content-length", 0))
        self.num_chunks = (self.total_size + self.chunk_size - 1) // self.chunk_size

    def load_progress(self):
        if os.path.exists(self.output_path):
            # Load existing chunks
            size = os.path.getsize(self.output_path)
            self.current_size = size
            complete_chunks = size // self.chunk_size
            for i in range(complete_chunks):
                self.downloaded_chunks.add(i)

    def calculate_sha256(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def download_chunk(self, chunk_id):
        if self.stop_event.is_set():
            return

        start_byte = chunk_id * self.chunk_size
        end_byte = min(start_byte + self.chunk_size - 1, self.total_size - 1)

        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(self.url, headers=headers, timeout=30)
                if response.status_code in [206, 200]:
                    chunk_file = os.path.join(self.temp_dir, f"chunk_{chunk_id}")
                    with open(chunk_file, "wb") as f:
                        f.write(response.content)

                    with self.lock:
                        self.downloaded_chunks.add(chunk_id)
                        self.current_size += len(response.content)
                    return True
            except (requests.exceptions.RequestException, IOError):
                retry_count += 1
                if retry_count == max_retries:
                    self.failed_chunks.put(chunk_id)
                time.sleep(1)
        return False

    def merge_chunks(self):
        print("\nMerging chunks...")
        with open(self.output_path, "wb") as outfile:
            for i in range(self.num_chunks):
                chunk_file = os.path.join(self.temp_dir, f"chunk_{i}")
                if os.path.exists(chunk_file):
                    with open(chunk_file, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def download(self):
        self.get_file_info()
        self.load_progress()

        # Initialize progress bar
        progress = tqdm(
            total=self.total_size,
            initial=self.current_size,
            unit="iB",
            unit_scale=True,
            desc="Downloading",
            bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        # Create download tasks
        remaining_chunks = [i for i in range(self.num_chunks) if i not in self.downloaded_chunks]

        def update_progress():
            last_size = self.current_size
            while not self.stop_event.is_set() and self.current_size < self.total_size:
                time.sleep(0.1)
                with self.lock:
                    delta = self.current_size - last_size
                    if delta > 0:
                        progress.update(delta)
                        last_size = self.current_size

                    # Calculate speed
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        speed = self.current_size / elapsed
                        self.speeds.append(speed)
                        if len(self.speeds) > 50:  # Keep last 50 speed measurements
                            self.speeds.pop(0)

                        # Update progress bar description with current speed
                        avg_speed = sum(self.speeds) / len(self.speeds)
                        progress.set_description(f"Downloading [{self.format_speed(avg_speed)}]")

        # Start progress monitoring thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()

        # Download chunks using thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.download_chunk, chunk_id) for chunk_id in remaining_chunks]

            # Wait for all downloads to complete or stop event
            while not all(future.done() for future in futures):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

        progress.close()

        if not self.stop_event.is_set():
            self.merge_chunks()

            if self.expected_sha256:
                print("Verifying SHA256...")
                actual_sha256 = self.calculate_sha256(self.output_path)
                if actual_sha256 != self.expected_sha256:
                    print(f"SHA256 verification failed!")
                    print(f"Expected: {self.expected_sha256}")
                    print(f"Got: {actual_sha256}")
                    return False
                print("SHA256 verification successful!")

            print(f"\nDownload completed! Average speed: {self.format_speed(sum(self.speeds) / len(self.speeds))}")
        else:
            print("\nDownload interrupted. Progress saved.")

        self.cleanup()
        return not self.stop_event.is_set()

    @staticmethod
    def format_speed(speed):
        if speed >= 1024 * 1024:
            return f"{speed / 1024 / 1024:.2f} MB/s"
        elif speed >= 1024:
            return f"{speed / 1024:.2f} KB/s"
        else:
            return f"{speed:.2f} B/s"


def main():
    parser = argparse.ArgumentParser(description="High-performance file downloader with resume capability")
    parser.add_argument("url", help="URL to download from")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024 * 4, help="Chunk size in bytes (default: 4MB)")
    parser.add_argument("--threads", type=int, default=8, help="Number of download threads (default: 8)")
    parser.add_argument("--sha256", help="Expected SHA256 hash for verification")

    args = parser.parse_args()

    downloader = ChunkDownloader(args.url, args.output, chunk_size=args.chunk_size, num_threads=args.threads, expected_sha256=args.sha256)

    downloader.download()


if __name__ == "__main__":
    main()

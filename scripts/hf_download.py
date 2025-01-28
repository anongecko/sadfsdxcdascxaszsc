import requests
import os
import time
import sys
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from tqdm import tqdm
import signal
import tempfile
import mmap
import getpass
from urllib.parse import urlparse


class HFChunkDownloader:
    def __init__(self, url, output_path, token=None, chunk_size=8 * 1024 * 1024, num_threads=32, expected_sha256=None):
        self.url = url
        self.output_path = output_path
        self.token = token
        self.chunk_size = chunk_size
        self.num_threads = num_threads
        self.expected_sha256 = expected_sha256
        self.temp_dir = tempfile.mkdtemp(dir="/dev/shm" if os.path.exists("/dev/shm") else None)
        self.downloaded_chunks = set()
        self.verified_chunks = set()
        self.lock = threading.Lock()
        self.failed_chunks = Queue()
        self.stop_event = threading.Event()
        self.session = self._create_session()

        # Progress tracking
        self.total_size = 0
        self.downloaded_bytes = 0
        self.start_time = time.time()
        self.speeds = []
        self.last_progress_update = time.time()

    def _create_session(self):
        session = requests.Session()
        if self.token:
            session.headers.update({"Authorization": f"Bearer {self.token}"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=self.num_threads, pool_maxsize=self.num_threads * 2, max_retries=3, pool_block=False)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def handle_interrupt(self, signum, frame):
        print("\nGracefully stopping download. Progress will be saved...")
        self.stop_event.set()

    def get_file_info(self):
        try:
            response = self.session.head(self.url, allow_redirects=True, timeout=30)

            # Check for authentication issues
            if response.status_code == 401:
                raise ValueError("Authentication failed. Please check your token.")
            elif response.status_code == 403:
                raise ValueError("Access forbidden. You may not have access to this resource.")
            elif response.status_code != 200:
                raise ValueError(f"Failed to access URL. Status code: {response.status_code}")

            self.total_size = int(response.headers.get("content-length", 0))
            if self.total_size == 0:
                raise ValueError("Could not determine file size")
            self.num_chunks = (self.total_size + self.chunk_size - 1) // self.chunk_size

            # Pre-allocate the output file
            if not os.path.exists(self.output_path):
                with open(self.output_path, "wb") as f:
                    f.seek(self.total_size - 1)
                    f.write(b"\0")
                print(f"Pre-allocated file: {self.total_size / (1024 * 1024 * 1024):.2f} GB")

        except requests.exceptions.RequestException as e:
            print(f"Network error getting file info: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error getting file info: {e}")
            sys.exit(1)

    def verify_chunk(self, chunk_id):
        start_byte = chunk_id * self.chunk_size
        chunk_size = min(self.chunk_size, self.total_size - start_byte)

        try:
            with open(self.output_path, "rb") as f:
                f.seek(start_byte)
                chunk_data = f.read(chunk_size)
                return any(chunk_data)
        except Exception:
            return False

    def download_chunk(self, chunk_id):
        if self.stop_event.is_set():
            return False

        start_byte = chunk_id * self.chunk_size
        end_byte = min(start_byte + self.chunk_size - 1, self.total_size - 1)
        chunk_size = end_byte - start_byte + 1

        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries and not self.stop_event.is_set():
            try:
                response = self.session.get(self.url, headers=headers, timeout=30, stream=True)

                if response.status_code in [206, 200]:
                    chunk_buffer = bytearray(chunk_size)
                    bytes_read = 0

                    for data in response.iter_content(chunk_size=1024 * 1024):
                        if self.stop_event.is_set():
                            return False
                        chunk_buffer[bytes_read : bytes_read + len(data)] = data
                        bytes_read += len(data)

                        with self.lock:
                            self.downloaded_bytes += len(data)
                            self.last_progress_update = time.time()

                    if bytes_read != chunk_size:
                        raise IOError(f"Incomplete chunk download: got {bytes_read} bytes, expected {chunk_size}")

                    with open(self.output_path, "r+b") as f:
                        mm = mmap.mmap(f.fileno(), 0)
                        try:
                            mm.seek(start_byte)
                            mm.write(chunk_buffer)
                        finally:
                            mm.flush()
                            mm.close()

                    if self.verify_chunk(chunk_id):
                        with self.lock:
                            self.downloaded_chunks.add(chunk_id)
                            self.verified_chunks.add(chunk_id)
                        return True
                    else:
                        raise IOError("Chunk verification failed")

            except Exception as e:
                print(f"\nError downloading chunk {chunk_id}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)

        if retry_count == max_retries:
            self.failed_chunks.put(chunk_id)
        return False

    def download(self):
        try:
            self.get_file_info()

            progress = tqdm(
                total=self.total_size,
                initial=0,
                unit="iB",
                unit_scale=True,
                desc="Downloading",
                bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            chunks_to_download = list(range(self.num_chunks))

            def update_progress():
                last_size = 0
                while not self.stop_event.is_set() and len(self.verified_chunks) < self.num_chunks:
                    time.sleep(0.1)
                    current_time = time.time()

                    if current_time - self.last_progress_update > 60:
                        print("\nDownload appears to be stuck. Initiating graceful shutdown...")
                        self.stop_event.set()
                        break

                    with self.lock:
                        delta = self.downloaded_bytes - last_size
                        if delta > 0:
                            progress.update(delta)
                            last_size = self.downloaded_bytes

                            elapsed = current_time - self.start_time
                            if elapsed > 0:
                                speed = self.downloaded_bytes / elapsed
                                self.speeds.append(speed)
                                if len(self.speeds) > 50:
                                    self.speeds.pop(0)
                                if self.speeds:
                                    avg_speed = sum(self.speeds) / len(self.speeds)
                                    progress.set_description(f"Downloading [{self.format_speed(avg_speed)}] ({len(self.verified_chunks)}/{self.num_chunks} chunks)")

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                future_to_chunk = {executor.submit(self.download_chunk, chunk_id): chunk_id for chunk_id in chunks_to_download}

                try:
                    for future in as_completed(future_to_chunk.keys()):
                        chunk_id = future_to_chunk[future]
                        try:
                            success = future.result()
                            if not success and not self.stop_event.is_set():
                                print(f"\nFailed to download chunk {chunk_id}")
                        except Exception as e:
                            print(f"\nError processing chunk {chunk_id}: {e}")
                except Exception as e:
                    print(f"\nError during download: {e}")
                    self.stop_event.set()

            progress.close()

            if not self.stop_event.is_set() and not self.failed_chunks.qsize():
                print("\nVerifying download integrity...")
                if self.verify_download():
                    if self.expected_sha256:
                        print("Verifying SHA256...")
                        actual_sha256 = self.calculate_sha256(self.output_path)
                        if actual_sha256 != self.expected_sha256:
                            print(f"SHA256 verification failed!")
                            print(f"Expected: {self.expected_sha256}")
                            print(f"Got: {actual_sha256}")
                            return False
                        print("SHA256 verification successful!")

                    if self.speeds:
                        avg_speed = sum(self.speeds) / len(self.speeds)
                        print(f"\nDownload completed! Average speed: {self.format_speed(avg_speed)}")
                    else:
                        print("\nDownload completed!")
                    return True
                else:
                    print("Download verification failed! File may be corrupted.")
                    return False
            else:
                failed_count = self.failed_chunks.qsize()
                if failed_count > 0:
                    print(f"\nDownload incomplete. {failed_count} chunks failed to download.")
                else:
                    print("\nDownload interrupted.")
                return False

        except Exception as e:
            print(f"\nUnexpected error during download: {e}")
            return False

    def verify_download(self):
        try:
            if not os.path.exists(self.output_path):
                print("Error: Output file does not exist!")
                return False

            actual_size = os.path.getsize(self.output_path)
            if actual_size != self.total_size:
                print(f"\nFile size mismatch!")
                print(f"Expected: {self.total_size:,} bytes")
                print(f"Got: {actual_size:,} bytes")
                return False

            print("Verifying file contents...")
            for chunk_id in range(self.num_chunks):
                if not self.verify_chunk(chunk_id):
                    print(f"Error: Chunk {chunk_id} appears to be empty or corrupted")
                    return False

            return True

        except Exception as e:
            print(f"Error during verification: {e}")
            return False

    @staticmethod
    def format_speed(speed):
        if speed >= 1024 * 1024:
            return f"{speed / 1024 / 1024:.2f} MB/s"
        elif speed >= 1024:
            return f"{speed / 1024:.2f} KB/s"
        else:
            return f"{speed:.2f} B/s"

    def calculate_sha256(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(1024 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def is_huggingface_url(url):
    """Check if the URL is from Hugging Face"""
    parsed = urlparse(url)
    return any(domain in parsed.netloc for domain in ["huggingface.co", "hf.co"])


def main():
    parser = argparse.ArgumentParser(description="High-performance Hugging Face file downloader with authentication")
    parser.add_argument("url", help="URL to download from")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--token", help="Hugging Face API token")
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024, help="Chunk size in bytes (default: 8MB)")
    parser.add_argument("--threads", type=int, default=32, help="Number of download threads (default: 32)")
    parser.add_argument("--sha256", help="Expected SHA256 hash for verification")

    args = parser.parse_args()

    # Check if URL is from Hugging Face
    if is_huggingface_url(args.url):
        # If token not provided as argument, try to get it from environment or prompt
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("Hugging Face URL detected. Please enter your authentication token.")
            print("You can find your token at https://huggingface.co/settings/tokens")
            token = getpass.getpass("Enter your Hugging Face token: ")

        downloader = HFChunkDownloader(args.url, args.output, token=token, chunk_size=args.chunk_size, num_threads=args.threads, expected_sha256=args.sha256)
    else:
        # For non-Hugging Face URLs, use without token
        downloader = HFChunkDownloader(args.url, args.output, chunk_size=args.chunk_size, num_threads=args.threads, expected_sha256=args.sha256)

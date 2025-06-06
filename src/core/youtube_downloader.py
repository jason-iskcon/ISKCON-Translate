"""YouTube video downloader for ISKCON-Translate."""
import os
import re
import yt_dlp
from typing import Optional
from logging_utils import get_logger

logger = get_logger(__name__)

class YouTubeDownloader:
    """Downloads YouTube videos to a cache directory."""
    
    def __init__(self, cache_dir: str = None):
        """Initialize the downloader.
        
        Args:
            cache_dir: Directory to store downloaded videos (defaults to ~/.video_cache)
        """
        if cache_dir is None:
            # Use user's home directory with .video_cache folder
            cache_dir = os.path.expanduser("~/.video_cache")
        
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"YouTube cache directory: {self.cache_dir}")
    
    def is_youtube_url(self, url: str) -> bool:
        """Check if the given string is a YouTube URL.
        
        Args:
            url: String to check
            
        Returns:
            bool: True if it's a YouTube URL
        """
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True
        return False
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            str: Video ID or None if not found
        """
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_cached_video_path(self, video_id: str) -> Optional[str]:
        """Check if video is already cached.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            str: Path to cached video or None if not found
        """
        cache_path = os.path.join(self.cache_dir, f"{video_id}.mp4")
        if os.path.exists(cache_path):
            logger.info(f"Found cached video: {cache_path}")
            return cache_path
        return None
    
    def download_video(self, url: str) -> str:
        """Download a YouTube video and return the path to the downloaded file.
        
        Args:
            url: YouTube URL
            
        Returns:
            str: Path to the downloaded MP4 file
            
        Raises:
            Exception: If download fails
        """
        if not self.is_youtube_url(url):
            raise ValueError(f"Not a valid YouTube URL: {url}")
        
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        # Check if already cached
        cached_path = self.get_cached_video_path(video_id)
        if cached_path:
            return cached_path
        
        # Configure yt-dlp options
        output_path = os.path.join(self.cache_dir, f"{video_id}.%(ext)s")
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer MP4 format
            'outtmpl': output_path,
            'no_warnings': False,
            'extractaudio': False,
            'audioformat': 'mp3',
            'embed_subs': False,
            'writesubtitles': False,
        }
        
        try:
            logger.info(f"Downloading YouTube video: {url}")
            logger.info(f"Video ID: {video_id}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                logger.info(f"Video title: {title}")
                logger.info(f"Duration: {duration} seconds")
                
                # Download the video
                ydl.download([url])
            
            # Find the downloaded file
            final_path = os.path.join(self.cache_dir, f"{video_id}.mp4")
            if not os.path.exists(final_path):
                # Check for other extensions
                for ext in ['webm', 'mkv', 'avi']:
                    alt_path = os.path.join(self.cache_dir, f"{video_id}.{ext}")
                    if os.path.exists(alt_path):
                        final_path = alt_path
                        break
            
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"Downloaded file not found: {final_path}")
            
            logger.info(f"Successfully downloaded video to: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to download YouTube video: {e}")
            raise

def download_youtube_video(url: str, cache_dir: str = None) -> str:
    """Convenience function to download a YouTube video.
    
    Args:
        url: YouTube URL
        cache_dir: Directory to store downloaded videos (defaults to ~/.video_cache)
        
    Returns:
        str: Path to the downloaded MP4 file
    """
    downloader = YouTubeDownloader(cache_dir)
    return downloader.download_video(url) 
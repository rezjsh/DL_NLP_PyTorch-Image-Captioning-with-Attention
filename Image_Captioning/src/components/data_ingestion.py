import os
import requests
import zipfile
import shutil

from Image_Captioning.src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    A utility class to download and extract the dataset.
    This includes both the image files and the captions file.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the downloader.
        Args:
            download_dir (str): The directory where the dataset will be downloaded and extracted.
        """
        self.config = config

    def _download_file(self, url, filename):
        """
        Downloads a file from a given URL.
        Args:
            url (str): The URL of the file to download.
            filename (str): The local filename to save the downloaded content.
        Returns:
            bool: True if download is successful, False otherwise.
        """
        print(f"Downloading {filename} from {url}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded {filename}.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return False

    def _extract_zip(self, zip_path, extract_to):
        """
        Extracts a zip file to a specified directory.
        Args:
            zip_path (str): Path to the zip file.
            extract_to (str): Directory where contents will be extracted.
        Returns:
            bool: True if extraction is successful, False otherwise.
        """
        print(f"Extracting {zip_path} to {extract_to}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Successfully extracted {zip_path}.")
            return True
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file.")
            return False
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False

    def download_flickr8k(self):
        """
        Downloads and extracts the Flickr8k dataset (images and captions)
        into the specified download directory.
        """
        # Create download directory if it doesn't exist
        if not os.path.exists(self.config.download_dir):
            os.makedirs(self.config.download_dir)
            print(f"Created directory: {self.config.download_dir}")

        # --- Download Images ---
        images_zip_path = os.path.join(self.config.download_dir, self.config.images_zip_name)
        if not os.path.exists(images_zip_path):
            if not self._download_file(self.config.images_zip_url, images_zip_path):
                print("Failed to download image dataset. Exiting.")
                return
        else:
            print(f"Image zip file already exists: {images_zip_path}")

        # --- Download Captions ---
        captions_zip_path = os.path.join(self.config.download_dir, self.config.captions_zip_name)
        if not os.path.exists(captions_zip_path):
            if not self._download_file(self.config.captions_url, captions_zip_path):
                print("Failed to download captions dataset. Exiting.")
                return
        else:
            print(f"Captions zip file already exists: {captions_zip_path}")

        # --- Extract Images ---
        images_extract_path = os.path.join(self.config.download_dir, "images")
        if not os.path.exists(images_extract_path):
            if not self._extract_zip(images_zip_path, images_extract_path):
                print("Failed to extract image dataset. Exiting.")
                return
        else:
            print(f"Image directory already exists: {images_extract_path}")

        # --- Extract and Process Captions ---
        captions_temp_extract_path = os.path.join(self.config.download_dir, "temp_captions")
        if not os.path.exists(captions_temp_extract_path):
            if not self._extract_zip(captions_zip_path, captions_temp_extract_path):
                print("Failed to extract captions dataset. Exiting.")
                return
            
            # The Flickr8k.token.txt is usually inside a subdirectory like 'Flickr8k_text'
            # Find the actual captions file
            found_caption_file = None
            for root, _, files in os.walk(captions_temp_extract_path):
                if self.config.captions_file_name in files:
                    found_caption_file = os.path.join(root, self.config.captions_file_name)
                    break
            
            if found_caption_file:
                # Move the captions file to the main download directory
                target_captions_path = os.path.join(self.config.download_dir, "captions.txt")
                shutil.move(found_caption_file, target_captions_path)
                print(f"Moved captions file to: {target_captions_path}")
            else:
                print(f"Warning: Could not find '{self.config.captions_file_name}' inside the captions zip.")

            # Clean up temporary captions extraction directory
            shutil.rmtree(captions_temp_extract_path)
            print(f"Cleaned up temporary captions directory: {captions_temp_extract_path}")
        else:
            print(f"Captions directory already exists (or captions.txt already moved).")


        print("\nFlickr8k dataset download and extraction complete!")
        print(f"Images are in: {os.path.join(self.config.download_dir, 'images')}")
        print(f"Captions are in: {os.path.join(self.config.download_dir, 'captions.txt')}")

# --- Main execution ---
if __name__ == "__main__":
    downloader = DataIngestion()
    downloader.download_flickr8k()

    # After running this script, you can update your Config in the image captioning model:
    # from your_script_name import Config # assuming your main script is image_captioning_attn.py
    # Config.root_dir = "flickr8k_dataset/images"
    # Config.caption_path = "flickr8k_dataset/captions.txt"
    # Then run your image captioning model.

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union, Any
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from src.file import read_json

def get_all_data(
    main_data_folder: Union[str, Path] = "data/",
    metadata_to_keep: List[str] = ["name", "keywords"]
) -> Dict[str, List[Any]]:
    """
    Collects image paths and selected metadata from a structured dataset directory.

    The expected structure of each subfolder inside `main_data_folder` is:
        <data_source>/
            images/
                image1.jpg
                image2.jpg
                ...
            source_info.json

    Parameters
    ----------
    main_data_folder : str or Path, optional
        Path to the main directory containing multiple data source folders.
        Defaults to "data/".

    metadata_to_keep : list of str, optional
        Keys to extract from each `source_info.json` file.
        Defaults to ["name", "keywords"].

    Returns
    -------
    dict
        A dictionary where:
        - "img_path" contains a list of image paths (Path objects)
        - For each metadata key `k` in `metadata_to_keep`, there is a corresponding
          list stored under the key `"source_{k}"`.

    """
    
    # Validate folder type
    if not isinstance(main_data_folder, (str, Path)):
        raise TypeError(
            f"`main_data_folder` must be a string or Path object, got {type(main_data_folder)}."
        )

    main_data_folder = Path(main_data_folder)

    # Validate folder existence and type
    if not main_data_folder.exists():
        raise ValueError(f"The folder '{main_data_folder}' does not exist.")
    if not main_data_folder.is_dir():
        raise ValueError(f"'{main_data_folder}' is not a directory.")

    combined_data_dict = defaultdict(list)

    # Iterate through all subdirectories
    for data_folder in main_data_folder.iterdir():
        if not data_folder.is_dir():
            continue

        image_folder = data_folder / "images"
        source_info_file = data_folder / "source_info.json"

        # Skip invalid folders
        if not (image_folder.is_dir() and source_info_file.is_file()):
            print(f"Skipping {data_folder}: missing 'images/' or 'source_info.json'.")
            continue

        # Read metadata JSON
        data_source_dict = read_json(source_info_file)

        # Collect image paths and metadata
        for image_path in image_folder.glob("*"):
            combined_data_dict["img_path"].append(image_path)

            for key in metadata_to_keep:
                combined_data_dict[f"source_{key}"].append(
                    data_source_dict.get(key, None)
                )

    return combined_data_dict

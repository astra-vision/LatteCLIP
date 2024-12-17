import os
import tarfile
import click


@click.command()
@click.option("--input_directory")
@click.option("--output_directory")
@click.option("--items_per_tar", default=1024)
def create_tarfiles(input_directory, output_directory, items_per_tar=1024):
    """
    Create tarfiles from a directory of JSON, TXT, and JPG files.
    The function is called with the following command:
    python create_tarfiles.py --input_directory /path/to/input/directory --output_directory /path/to/output/directory --items_per_tar 1024
    Args:
        input_directory (str): The directory path containing the JSON, TXT, and JPG files.
        output_directory (str): The directory path to save the tarfiles.
        items_per_tar (int): The number of items per tarfile. Default is 1024.
    """
    os.makedirs(output_directory, exist_ok=True)
    # Assumption: For each item, there exists one .json, one .txt, and one .jpg file.
    # Generate a list of unique basenames without extensions.
    all_files = os.listdir(input_directory)
    unique_basenames = set(os.path.splitext(file)[0] for file in all_files)
    
    basename_chunks = [list(unique_basenames)[i:i + items_per_tar] 
                       for i in range(0, len(unique_basenames), items_per_tar)]
    total_item = 0
    for index, chunk in enumerate(basename_chunks):
        tar_filename = os.path.join(output_directory, f"{index:05d}.tar")
        with tarfile.open(tar_filename, "w") as tar:
            for basename in chunk:
                total_item += 1
                for extension in ['.json', '.txt', '.jpg']:
                    file_name = f"{basename}{extension}"
                    file_path = os.path.join(input_directory, file_name)
                    # Check if the file exists to avoid errors.
                    if os.path.isfile(file_path):
                        tar.add(file_path, arcname=file_name)
            print(f"Created {tar_filename}, total items: {total_item}/{len(unique_basenames)}")


if __name__ == "__main__":
    create_tarfiles()
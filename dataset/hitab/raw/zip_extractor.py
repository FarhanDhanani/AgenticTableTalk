import zipfile
import os

zip_path = os.path.join(os.getcwd(), "raw-tables.zip")
output_path = os.path.join(os.getcwd())

# Confirm the zip file exists
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Zip file not found at: {zip_path}")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        filename = os.path.basename(member)  # strip folders
        if not filename:
            continue  # skip directories
        source = zip_ref.open(member)
        target_path = os.path.join(output_path, filename)

        # If duplicate filenames exist, handle or overwrite — here we overwrite
        with open(target_path, "wb") as target:
            target.write(source.read())

print(f"✅ All files extracted (flat) to: {output_path}")

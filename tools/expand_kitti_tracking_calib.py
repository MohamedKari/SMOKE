from pathlib import Path
import shutil

root_dir = "."

root_image_dir = Path(Path(root_dir), "image_02")
root_calib_dir = Path(Path(root_dir), "calib")
root_imagesets_dir = Path(Path(root_dir), "ImageSets")

calib_txt_paths = sorted(root_calib_dir.iterdir())

for calib_txt_path in calib_txt_paths:
    if not calib_txt_path.is_file():
        continue

    track_id = calib_txt_path.stem
    track_image_dir = Path(root_image_dir, f"{int(track_id):04d}")
    track_calib_dir = Path(root_calib_dir, track_id)

    track_calib_dir.mkdir(exist_ok=True, parents=True)

    image_ids = list()

    for track_image_png in sorted(track_image_dir.iterdir()):        
        image_id = track_image_png.stem
        image_ids.append(image_id)

        target = Path(track_calib_dir, track_image_png.stem + ".txt")

        if target.exists():
            target.unlink()

        print(f"moving {str(calib_txt_path)} to {str(target)}")
        shutil.copyfile(str(calib_txt_path), str(target))

    track_imagesets_dir = Path(root_imagesets_dir, track_id)
    track_imagesets_dir.mkdir(exist_ok=True, parents=True)

    Path(track_imagesets_dir, "test.txt").write_text("\n".join(image_ids))
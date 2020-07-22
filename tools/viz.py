from argparse import ArgumentParser
import sys
from pathlib import Path
import pprint
import random

from PIL import Image, ImageDraw
from io import BytesIO

class MatchingError(Exception): pass

def get_viz(detection_dict: dict):
    image = detection_dict["image"]
    
    draw = ImageDraw.Draw(image)

    for instance in detection_dict["instances"]:
        draw.rectangle((
            instance["box_x1"],
            instance["box_y1"],
            instance["box_x2"],
            instance["box_y2"],
        ))

    image.show()


def get_input_data(image_input_dir: str, prediction_dir: str):
    image_input_paths = sorted(Path(image_input_dir).iterdir())
    prediction_paths = sorted(Path(prediction_dir).iterdir())

    if len(image_input_paths) != len(prediction_paths):
        raise MatchingError(
            f"image_input_dir and prediction_dir have differing numbers of child elements"
            f"{len(image_input_paths)} vs {len(prediction_paths)}"
            f"Should be equal.")

    for image_path, prediction_path in zip(image_input_paths, prediction_paths):
        if image_path.stem != prediction_path.stem:
            raise MatchingError(f"Couldn't match image and prediction: {image_input_paths} couldn't be mapped to {prediction_path}.")
        
        detection_dict = dict()

        image = Image.open(BytesIO(image_path.read_bytes()))
        detection_dict["image"] = image

        prediction_text = prediction_path.read_text().strip("\n")
        detection_dict["instances"] = list()

        for detection_instance in prediction_text.split("\n"):
            if not detection_instance: 
                break

            (
                instance_class, 
                trash_1, 
                trash_2,
                box_alpha, 
                box_x1,
                box_y1,
                box_x2,
                box_y2,
                h, 
                w, 
                l,
                t1,
                t2,
                t3,
                ry,
                thresh
            ) = detection_instance.split(" ")
            
            detection_dict["instances"].append({
                "instance_class": instance_class,
                "trash_1": trash_1, 
                "trash_2": trash_2,
                "box_alpha": float(box_alpha),
                "box_x1": float(box_x1),
                "box_y1": float(box_y1),
                "box_x2": float(box_x2),
                "box_y2": float(box_y2),
                "h": float(h), 
                "w": float(w), 
                "l": float(l),
                "t1": float(t1),
                "t2": float(t2),
                "t3": float(t3),
                "ry": float(ry),
                "thresh": (thresh)
            })

        yield detection_dict


def run(image_input_dir: str, prediction_dir: str, viz_output_dir: str):
    input_data = get_input_data(image_input_dir, prediction_dir)
    
    input_data_list = list(input_data)
    input_data_item = random.choice(input_data_list)
    pprint.pprint(input_data_item)
    get_viz(input_data_item)

    #for detection_dict in input_data:
    #    pprint.pprint(detection_dict)
    #    get_viz(detection_dict)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image_input_dir")
    parser.add_argument("prediction_dir")
    parser.add_argument("viz_output_dir")

    args = parser.parse_args(sys.argv[1:])

    run(
        args.image_input_dir,
        args.prediction_dir,
        args.viz_output_dir
    )

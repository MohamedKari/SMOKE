from typing import Union, Tuple
from argparse import ArgumentParser
import sys

from PIL import Image
from torchvision.transforms.functional import (
    resize,
    crop,
    pad)

class ReversiblePadding():
    
    def __init__(self, source_image: Image.Image, target_size: Union[Tuple[int], int]):
        self.source_image = source_image
        self.target_size = target_size
        self.source_width, self.source_height = source_image.size
        self.box_xmin, self.box_ymin, self.box_xmax, self.box_ymax = None, None, None, None
        
    def get_padded_image(self):
        padded_image, self.box_xmin, self.box_ymin, self.box_xmax, self.box_ymax = \
            fit(self.source_image, self.target_size, fitting_mode="pad")

        return padded_image

    def revert_on_image(self, image: Image.Image, revert_resizing: bool = True):
        """
        Reverts the padding process on the given image (e. g. the annotated image), by unpadding and upsizing the pad box.
        Note, that resizing is lossy process.
        """

        reverted_image = image

        # crop
        print("Using stored box info for unpadding: xmin, ymin, xmax, ymax, width, height",
              self.box_xmin, 
              self.box_ymin, 
              self.box_xmax,
              self.box_ymax,
              self.box_xmax - self.box_xmin + 1,
              self.box_ymax - self.box_ymin + 1)

        print(reverted_image.size)
        print("(i, j): ", self.box_ymin, self.box_xmin)
        reverted_image = crop(
            reverted_image, 
            self.box_ymin, 
            self.box_xmin, 
            self.box_ymax - self.box_ymin + 1,
            self.box_xmax - self.box_xmin + 1)

        # resize
        if revert_resizing:
            reverted_image = resize(
                reverted_image,
                (self.source_height, self.source_width))
        
        return reverted_image

    # TODO: implement context manager

def fit(source_image: Image.Image, target_size: Union[Tuple[int], int], fitting_mode="crop") -> Image.Image:
    """
    Args:
        source_image: PIL Image
        target_size: Tuple of ints (height, width) or single int for square target
        fitting_mode: Either 'crop' or 'pad'.
    """
    source_width, source_height = source_image.size
    if isinstance(target_size, int):
        target_height, target_width = target_size, target_size
    elif isinstance(target_size, tuple) and len(target_size) == 2:
        target_height, target_width = target_size
    else:
        raise TypeError("invalid type of target_size")

    source_ratio = source_height / source_width
    target_ratio = target_height / target_width

    target_image = None
    box_xmin, box_ymin, box_xmax, box_ymax = None, None, None, None

    if fitting_mode == "crop":
        if source_ratio == target_ratio:
            # simple resize
            target_image = resize(
                source_image,
                (
                    target_height,
                    target_width,
                )
            )
        elif source_ratio > target_ratio:
            # align width, then crop
            overheight = int(source_height * (target_width / source_width))
            
            target_image = resize(
                source_image,
                (overheight, target_width)
            )                
            
            target_image = crop(
                target_image,
                int((overheight - target_height) / 2),
                0,
                target_height,
                target_width
            )

        elif source_ratio < target_ratio:
            # align height, then crop
            overwidth = int(source_width * (target_height / source_height))

            target_image = resize(
                source_image,
                (target_height, overwidth)
            )

            target_image = crop(
                target_image,
                0,
                int((overwidth - target_width) / 2),
                target_height,
                target_width
            )

            # TODO: Implement crop box info if wanted

    elif fitting_mode == "pad":
        if source_ratio == target_ratio:
            # simple resize
            target_image = resize(
                source_image,
                (
                    target_height,
                    target_width,
                )
            )
        
            box_xmin, box_ymin = 0, 0
            box_xmax, box_ymax = target_width - 1, target_height - 1

        elif source_ratio > target_ratio:
            # align height, then pad
            underwidth = int(source_width * (target_height / source_height))

            target_image = resize(
                source_image,
                (target_height, underwidth))

            target_image = pad(
                target_image,

                # add 1 in case (target_height - underheight) is odd, so to ensure putting out the desired size
                padding=( # left, top, right, bottom
                    int((target_width - underwidth) / 2),
                    0,
                    int((target_width - underwidth) / 2) + (target_width - underwidth) % 2, 
                    0))

            box_xmin, box_ymin = int((target_width - underwidth) / 2), 0
            box_xmax, box_ymax = int((target_width - underwidth) / 2) + underwidth - 1, target_height - 1
    
        
        elif source_ratio < target_ratio:
            # align width, then pad
            underheight = int(source_height * (target_width / source_width))

            target_image = resize(
                source_image,
                (underheight, target_width))

            target_image = pad(
                target_image,

                # add 1 in case (target_height - underheight) is odd, so to ensure putting out the desired size
                padding=( # left, top, right, bottom
                    0, 
                    int((target_height - underheight) / 2),
                    0,
                    int((target_height - underheight) / 2) + + (target_width - underwidth) % 2 
                    ))
            
            box_xmin, box_ymin = 0, int((target_height - underheight) / 2)
            box_xmax, box_ymax = target_width - 1, int((target_height - underheight) / 2) + underheight - 1

    assert target_image.size[0] == target_width
    assert target_image.size[1] == target_height
    
    return target_image, box_xmin, box_ymin, box_xmax, box_ymax

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("mode", choices=["crop", "pad"])
    parser.add_argument("height", type=int)
    parser.add_argument("width", type=int)

    args = parser.parse_args(sys.argv[1:])

    with open(args.path, "rb") as image_file:
        input_image = Image.open(image_file)
        
        input_image.show()
        
        reversible_padding = ReversiblePadding(input_image, (args.height, args.width))
        _padded_image = reversible_padding.get_padded_image()
        _padded_image.show()

        _reverted_image = reversible_padding.revert_on_image(_padded_image) 
        _reverted_image.show()





    
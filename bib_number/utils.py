import os
import cv2
from random import randint

COLOR = (150, 230, 210)

def invert_color(r, g, b):
    x = [r, g, b]
    x.sort()
    k = x[0] + x[-1]
    return tuple(min(k, 255) - u for u in (r, g, b))


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=3,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, max(0.5, font_scale), text_color, font_thickness)

    return text_size

def draw_bbox(_img, _bbox, _object_class, _color=COLOR, add_border=False, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1):
    """draw bounding box on image

    Args:
        _img (cv_image): opencv image
        _bbox (list): bbox for the object with x,y,w,h format
        _object_class (str): object class name
        _color (tuple): selected color for the class
        add_border (bool, optional): to add border for the box or not. Defaults to False.
    """
    _bbox = [int(cord) for cord in _bbox]
    pt1 = (max(0, int(round(_bbox[0]))), max(0, int(round(_bbox[1]))))
    pt2 = (int(round(_bbox[0] + _bbox[2])), int(round(_bbox[1] + _bbox[3])))

    # when the area is zero it is either captioning or context
    area = (pt2[1] - pt1[1]) * (pt2[0] - pt1[0])
    pta = (pt1[0], pt1[1] - 15)
    ptb = (pt2[0], pt1[1])

    if not area:
        pt1 = (10, pt1[1])
        pt2 = (10, pt2[1])
        _color_inverted = (255, 255, 255)
        bg_color = (0, 0, 0)
    else:
        _color_inverted = invert_color(*_color)
        cv2.rectangle(_img, pt1, pt2, _color, 2)
        bg_color = _color

    if add_border:
        cv2.rectangle(_img, pta, ptb, _color, 2)
        cv2.rectangle(_img, pta, ptb, _color, -1)
    draw_text(_img, _object_class, font, (pta[0], pta[1]-5), 1, 1, _color_inverted, bg_color)


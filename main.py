import argparse
import os
from nets import nn


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', type=str, default='./demo/inputs/images/10.jpg')
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--input-video', type=str, default='./demo/inputs/videos/9.mp4')
    parser.add_argument('--output-dir', type=str, default='./demo/results')

    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    live_portrait = nn.LPP(args, False, False)
    live_portrait.execute()


if __name__ == "__main__":
    demo()

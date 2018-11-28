import pandas as pd
import cv2
import numpy as np
import os
import expyriment

exp = expyriment.design.Experiment(name="First Experiment")

expyriment.control.initialize(exp)

expyriment.control.start()

DF100 = pd.read_csv('DF100.csv')


def key_responses():
    while not my_video.new_frame_available:  # or not my_video2.new_frame_available:
        key = exp.keyboard.check()
        # if key is None:
        #     show_text()
        if key is not None:
            print
            "key:", key

        wkey = 119
        ekey = 101
        rkey = 114

        if key == wkey:
            answer = 'LEFT'
        if key == ekey:
            answer = 'DON\'T KNOW'
        if key == rkey:
            answer = 'RIGHT'


for pair in range(len(DF100)):

    pair = DF100.iloc[pair]

    video_pair = pair['videoPair']

    my_video = expyriment.stimuli.Video(video_pair)

    expyriment.control.stop_audiosystem()

    my_video.preload()

    my_video.play()

    personality_text = expyriment.stimuli.TextBox('Friendly', (3, 3))

    while my_video.is_playing and my_video.frame < 100:
        key_responses()

        my_video.update()

        personality_text.present(clear=False)

    print
    "my_video1.frame:", my_video.frame

    my_video.stop()

expyriment.control.end()

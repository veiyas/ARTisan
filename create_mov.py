import moviepy.video.io.ImageSequenceClip
import os
import re

def create_mov_from_img_seq(folder, video_file_name, fps):
    seq = [os.path.join(folder, img) for img in os.listdir(folder) if img.startswith("result-at-")]
    seq = sorted(seq, key=lambda name: int(re.findall("\d+", name)[2]))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(seq, fps=fps)
    clip.write_videofile(os.path.join(folder, video_file_name))

def main():
    create_mov_from_img_seq(
        os.path.join("output", "started-at-20211006-190059"),
        "seq.mp4",
        fps=30
    )

if __name__ == "__main__":
    main()
# Live Portraits AI

<p align="center">  
  <h1 align="center">LivePortrait: Bring portraits to life!</h1>
  <img src="demo/results/demo1.gif">
  <img src="demo/results/demo2.gif">
</p>

<p align="center">  
  <h2 align="center">Efficient Portrait Animation with Stitching and Retargeting Control</h2>
</p>

### Transform static portraits into lifelike animations with ** just one click! 🔥🔥🔥** 

### 💪Features

- **One-Click Animation:** Effortlessly generate animated portraits with a single command.
- **Static Image Animation:** Breathe life into your static images with our advanced models.
- **Keypoint Detection:** Accurately detect and transform facial keypoints for realistic movements.
- **Model Integration:** Seamlessly integrate various deep learning models to enhance animations.
- **Video Processing:** Utilize video inputs to drive stunning animations.
- **Template Creation:** Easily generate motion templates for consistent and repeatable animations.

### 🤗 Project Structure

- `main.py`: Entry point for the demo script.
- `nn.py`: Contains the neural network definitions and model loading functions.
- `util.py`: Utility functions for image and video processing.
- `nets/`: Directory containing various model definitions and configurations.
- `demo/`: Directory for demo inputs and outputs.

### 🚀🚀 Pretrained Weights

1. Download pre-trained model weights and place them in the `weights` directory. Ensure the directory structure is as follows:
    ```
    live-portraits-ai/
    ├── weights/
    │   ├── afx.pth
    │   ├── mx.pth
    │   ├── wn.pth
    │   ├── spad.pth
    │   ├── stitch.pth
    │   └── insightface
    ```

### 🚀🚀🚀 Usage: Running the Demo

Experience the magic of animation with our easy-to-use demo:

1. Configure your video path in `main.py` for visualizing the demo.
2. Run the demo script:
    ```bash
    python main.py --input-image ./demo/inputs/images/10.jpg --input-video ./demo/inputs/videos/9.mp4 --output-dir ./demo/results
    ```

With **just one click**, you can transform any portrait into a captivating animated version, perfect for showcasing on social media, websites, or presentations.

#### Reference

- [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)

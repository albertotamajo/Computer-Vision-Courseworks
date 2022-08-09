package uk.ac.soton.ecs.at2n19.ch7;

import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.video.Video;
import org.openimaj.video.VideoDisplay;
import org.openimaj.video.VideoDisplayListener;
import org.openimaj.video.xuggle.XuggleVideo;

import java.net.MalformedURLException;
import java.net.URL;

/*
    Try a different processing operation and see how it affects the frames of your video.
 */
public class Exercise1 {
    // Given that I convolve each frame of the video with a Gaussian kernel then the resulting video is blurred
    public static void main(String[] args) throws MalformedURLException {

        // Load a video from a URL
        Video<MBFImage> video = new XuggleVideo(new URL("http://static.openimaj.org/media/tutorial/keyboardcat.flv"));
        // Create a video display
        VideoDisplay<MBFImage> display = VideoDisplay.createVideoDisplay(video);
        // Add a listener which automatically blurs each frame of the video with a Gaussian Kernel
        display.addVideoListener(
                new VideoDisplayListener<MBFImage>() {
                    public void beforeUpdate(MBFImage frame) {
                        frame.processInplace(new FGaussianConvolve(15));
                    }

                    public void afterUpdate(VideoDisplay<MBFImage> display) {
                    }
                });
    }
}

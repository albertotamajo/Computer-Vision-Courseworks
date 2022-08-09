package uk.ac.soton.ecs.at2n19.ch7;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.video.Video;
import org.openimaj.video.VideoDisplay;
import org.openimaj.video.VideoDisplayListener;
import org.openimaj.video.capture.VideoCapture;
import org.openimaj.video.capture.VideoCaptureException;
import org.openimaj.video.xuggle.XuggleVideo;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;

public class Tutorial {
    public static void main(String[] args) throws MalformedURLException, VideoCaptureException {

        // Create a video instance which could hold coloured frames
        Video<MBFImage> video;
        // Load a video from a file
        video = new XuggleVideo(new File("C:\\Users\\tamaj\\OneDrive\\University of Southampton\\University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\keyboardcat.flv"));
        // Load a video from a URL
        video = new XuggleVideo(new URL("http://static.openimaj.org/media/tutorial/keyboardcat.flv"));
        // Capture a video from a capture device
        video = new VideoCapture(1920, 1080);
        // Display the video
        VideoDisplay.createVideoDisplay(video);

        // Detect the edges of each frame using the Canny Edge Detector algorithm and display it
        for (MBFImage mbfImage : video) {
            DisplayUtilities.displayName(mbfImage.process(new CannyEdgeDetector()), "videoFrames");
        }

        // Create a video display
        VideoDisplay<MBFImage> display = VideoDisplay.createVideoDisplay(video);
        // Add a listener which automatically detects the edges of each frame using the Canny Edge Detector algorithm
        // and then display it
        display.addVideoListener(
                new VideoDisplayListener<MBFImage>() {
                    public void beforeUpdate(MBFImage frame) {
                        frame.processInplace(new CannyEdgeDetector());
                    }

                    public void afterUpdate(VideoDisplay<MBFImage> display) {
                    }
                });
    }
}
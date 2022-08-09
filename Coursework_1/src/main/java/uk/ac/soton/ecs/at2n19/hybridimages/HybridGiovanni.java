package uk.ac.soton.ecs.at2n19.hybridimages;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.Gaussian2D;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class HybridGiovanni {

    /**
     * This method computes a hybrid image by combining low-pass and high-pass filtered images.
     * Note that the input images are expected to have the same size, and the output image will also have the same
     * height & width as the inputs.
     *
     * @param lowImage          The image to which apply the low pass filter
     * @param lowSigma          The standard deviation of the low-pass filter
     * @param highImage         The image to which apply the high pass filter
     * @param highSigma         The standard deviation of the low-pass component of computing the high-pass filtered image
     * @return                  The computed hybrid image
     */
    public static MBFImage makeHybrid(MBFImage lowImage, float lowSigma, MBFImage highImage, float highSigma) {

        // Generating the low pass filtered version of lowImage
        lowImage = generateLowPass(lowImage, lowSigma);

        // Generating the low pass filtered version of highImage
        MBFImage lowPassedComponent = generateLowPass(highImage, highSigma);

        // Generating the high pass filtered version of highImage
        highImage = highImage.subtract(lowPassedComponent);

        // Calculating the hybrid image
        return lowImage.add(highImage);
    }



    /**
     * This method generates a low-pass filtered image from an input image and a sigma value.
     *
     * @param image         The image to filter
     * @param sigma         Cutoff frequency control
     * @return              The resulting low-pass image
     */
    public static MBFImage generateLowPass(MBFImage image, float sigma) {
        int size = (int) (8.0f * sigma + 1.0f);
        if (size % 2 == 0) size++;
        System.out.println("Using Kernel of size " + size);

        // Generating Low-Pass filter kernel
        float[][] lowPassKernel = Gaussian2D.createKernelImage(size, sigma).pixels;

        // Performing Low-Pass filtering convolution on first image
        ConvolutionGiovanni lowPassConvolution = new ConvolutionGiovanni(lowPassKernel);
        image = image.process(lowPassConvolution);
        return image;
    }



    /**
     * This method displays the cascading sized hybrid images to emphasise the hybrid image effect
     *
     * @param hybridImage       The generated hybrid image
     * @param cascadeAmount     The amount of times to cascade (halve) the image by
     */
    public static void displayCascade(MBFImage hybridImage, int cascadeAmount) {
        int requiredWidth = 0;
        int widthBuffer = hybridImage.getWidth();

        for (int i = 0; i <= cascadeAmount; i++) {
            requiredWidth += widthBuffer + 10;
            widthBuffer /= 2;
        }

        // Creating a new larger MBFImage in order to contain the cascade of resized images
        MBFImage canvas = new MBFImage(requiredWidth, hybridImage.getHeight());

        // Drawing the full size hybrid image
        canvas.drawImage(hybridImage, 0, 0);

        // Adding horizontal offset
        int horizontalOffset = hybridImage.getWidth() + 10;
        ResizeProcessor resizeHalf = new ResizeProcessor(ResizeProcessor.Mode.HALF);

        // Creating 4 cascading sized hybrid images
        for (int i = 0; i < cascadeAmount; i++) {
            hybridImage = hybridImage.process(resizeHalf);
            canvas.drawImage(hybridImage, horizontalOffset, canvas.getHeight()-hybridImage.getHeight());
            horizontalOffset += hybridImage.getWidth() + 10;
        }

        // Displaying the cascading hybrid images
        DisplayUtilities.display(canvas);
    }
}

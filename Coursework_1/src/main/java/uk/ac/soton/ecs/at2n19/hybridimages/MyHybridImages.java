package uk.ac.soton.ecs.at2n19.hybridimages;

import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.Gaussian2D;

public class MyHybridImages {
    /**
     * Compute a hybrid image combining low-pass and high-pass filtered images
     *
     * @param lowImage
     *            the image to which apply the low pass filter
     * @param lowSigma
     *            the standard deviation of the low-pass filter
     * @param highImage
     *            the image to which apply the high pass filter
     * @param highSigma
     *            the standard deviation of the low-pass component of computing the
     *            high-pass filtered image
     * @return the computed hybrid image
     */
    public static MBFImage makeHybrid(MBFImage lowImage, float lowSigma, MBFImage highImage, float highSigma) {
        // Remove high-frequencies from the lowImage
        lowImage = gaussianConvolve(lowImage, lowSigma);
        // Remove high-frequencies from the highImage
        MBFImage lowFrequencyHighImage = gaussianConvolve(highImage, highSigma);
        // Subtract the low frequency version of the high image from itself
        highImage = highImage.subtract(lowFrequencyHighImage);
        // Return the hybrid image
        return lowImage.add(highImage);
    }

    // Convolve an image with a gaussian kernel
    public static MBFImage gaussianConvolve(MBFImage image, float sigma){
        // Set the size of the Gaussian kernel
        int size = (int) (8.0f * sigma + 1.0f);
        if (size % 2 == 0) size++;
        // Create Gaussian Kernel
        float[][] kernel = Gaussian2D.createKernelImage(size, sigma).pixels;
        // Return the convolved image
        return image.process(new MyConvolution(kernel));
    }
}

package uk.ac.soton.ecs.at2n19.hybridimages;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.Gaussian2D;

import java.io.File;
import java.io.IOException;

public class HybridAlessandro {
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
        return lowPass(lowImage, lowSigma).add(highPass(highImage, highSigma));
    }

    public static MBFImage lowPass(MBFImage image, float sigma){
        ConvolutionAlessandro convolution = new ConvolutionAlessandro(HybridAlessandro.makeGaussian(sigma));
        FImage[] images = new FImage[3];
        for (int band=0; band<3; band++){
            FImage tempImage = new FImage(image.getBand(band).pixels);
            convolution.processImage(tempImage);
            images[band] = tempImage;
        }
        return new MBFImage(images);
    }

    public static MBFImage highPass(MBFImage image, float sigma){
        MBFImage lowPassImage = lowPass(image, sigma);
        try {
            ImageUtilities.write(image.add(0.5f).subtract(lowPassImage), new File("HighPass.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image.subtract(lowPassImage);
    }

    public static float[][] makeGaussian(float sigma){
        int size = (int) (8.0f * sigma + 1.0f); // (this implies the window is +/- 4 sigmas from the centre of the Gaussian)
        if (size % 2 == 0) size++; // size must be odd
        return new Gaussian2D(size, sigma).kernel.pixels;
    }

    public static void main(String[] args) throws IOException {
        float sigmaLow = 3f;
        float sigmaHigh = 8f;
        MBFImage imageLow = ImageUtilities.readMBF(new File(args[0]));
        MBFImage imageHigh =  ImageUtilities.readMBF(new File(args[1]));
        ImageUtilities.write(MyHybridImages.makeHybrid(imageLow, sigmaLow, imageHigh, sigmaHigh), new File("Result.png"));
    }
}

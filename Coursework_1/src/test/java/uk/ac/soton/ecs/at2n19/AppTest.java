package uk.ac.soton.ecs.at2n19;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.api.Assertions;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import uk.ac.soton.ecs.at2n19.hybridimages.*;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Unit test for simple App.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class AppTest {
    /**
     * Rigourous Test :-)
     */
	@ParameterizedTest
    @MethodSource("argumentsTestConvolution")
    void testConvolution(MBFImage image, float[][] kernel) {
        MBFImage myConv = image.process(new MyConvolution(kernel));
        FImage actualFirstBand = new FImage(Convolution.convolve(image.getBand(0).pixels, kernel));
        FImage actualSecondBand = new FImage(Convolution.convolve(image.getBand(1).pixels, kernel));
        FImage actualThirdBand = new FImage(Convolution.convolve(image.getBand(2).pixels, kernel));
        MBFImage actualConv = new MBFImage(actualFirstBand, actualSecondBand, actualThirdBand);
        // Assert the size of the convoluted image is the same as the original
        Assertions.assertEquals(image.getWidth(), myConv.getWidth());
        Assertions.assertEquals(image.getHeight(), myConv.getHeight());
        // Assert the size of the convoluted image is the same as the actual convolution
        Assertions.assertEquals(actualConv.getWidth(), myConv.getWidth());
        Assertions.assertEquals(actualConv.getHeight(), myConv.getHeight());

        // Rounding off to 5 decimal places
        roundoff(1, actualConv.getBand(0).pixels);
        roundoff(1, actualConv.getBand(1).pixels);
        roundoff(1, actualConv.getBand(2).pixels);
        roundoff(1, myConv.getBand(0).pixels);
        roundoff(1, myConv.getBand(1).pixels);
        roundoff(1, myConv.getBand(2).pixels);

        // Assert the convoluted image is convoluted correctly using the OpenCV convolution as reference
        Assertions.assertArrayEquals(actualConv.getBand(0).pixels, myConv.getBand(0).pixels);
        Assertions.assertArrayEquals(actualConv.getBand(1).pixels, myConv.getBand(1).pixels);
        Assertions.assertArrayEquals(actualConv.getBand(2).pixels, myConv.getBand(2).pixels);
        // Assert the convoluted image is equivalent to the actual convolution
        Assertions.assertEquals(actualConv,myConv);
    }


    @ParameterizedTest
    @MethodSource("argumentsTestConvolution")
    void testConvolutionAlessandro(MBFImage image, float[][] kernel) {
        MBFImage myConv = image.process(new MyConvolution(kernel));
        MBFImage aleConv = image.process(new ConvolutionAlessandro(kernel));
        // Assert the size of the convoluted image is the same as the original
        Assertions.assertEquals(image.getWidth(), myConv.getWidth());
        Assertions.assertEquals(image.getHeight(), myConv.getHeight());
        // Assert the size of the convoluted image is the same as the actual convolution
        Assertions.assertEquals(aleConv.getWidth(), myConv.getWidth());
        Assertions.assertEquals(aleConv.getHeight(), myConv.getHeight());


        // Assert the convoluted image is convoluted correctly using the OpenCV convolution as reference
        Assertions.assertArrayEquals(aleConv.getBand(0).pixels, myConv.getBand(0).pixels);
        Assertions.assertArrayEquals(aleConv.getBand(1).pixels, myConv.getBand(1).pixels);
        Assertions.assertArrayEquals(aleConv.getBand(2).pixels, myConv.getBand(2).pixels);
        // Assert the convoluted image is equivalent to the actual convolution
        Assertions.assertEquals(aleConv,myConv);
    }

    @ParameterizedTest
    @MethodSource("argumentsTestConvolution")
    void testConvolutionGiovanni(MBFImage image, float[][] kernel) {
        MBFImage myConv = image.process(new MyConvolution(kernel));
        MBFImage gioConv = image.process(new ConvolutionGiovanni(kernel));
        // Assert the size of the convoluted image is the same as the original
        Assertions.assertEquals(image.getWidth(), myConv.getWidth());
        Assertions.assertEquals(image.getHeight(), myConv.getHeight());
        // Assert the size of the convoluted image is the same as the actual convolution
        Assertions.assertEquals(gioConv.getWidth(), myConv.getWidth());
        Assertions.assertEquals(gioConv.getHeight(), myConv.getHeight());


        // Assert the convoluted image is convoluted correctly using the OpenCV convolution as reference
        Assertions.assertArrayEquals(gioConv.getBand(0).pixels, myConv.getBand(0).pixels);
        Assertions.assertArrayEquals(gioConv.getBand(1).pixels, myConv.getBand(1).pixels);
        Assertions.assertArrayEquals(gioConv.getBand(2).pixels, myConv.getBand(2).pixels);
        // Assert the convoluted image is equivalent to the actual convolution
        Assertions.assertEquals(gioConv,myConv);
    }

    @ParameterizedTest
    @MethodSource("argumentsTestHybrid")
    void testHybridGiovanni(MBFImage lowImage, float lowSigma, MBFImage highImage, float highSigma) {
	    MBFImage myHybrid = MyHybridImages.makeHybrid(lowImage, lowSigma, highImage, highSigma);
        MBFImage gioHybrid = HybridGiovanni.makeHybrid(lowImage, lowSigma, highImage, highSigma);
        // Assert the size of the hybrid image is the same as the original
        Assertions.assertEquals(myHybrid.getWidth(), lowImage.getWidth());
        Assertions.assertEquals(myHybrid.getHeight(), lowImage.getHeight());
        // Assert the size of the convoluted image is the same as the actual convolution
        Assertions.assertEquals(gioHybrid.getWidth(), myHybrid.getWidth());
        Assertions.assertEquals(gioHybrid.getHeight(), myHybrid.getHeight());


        // Assert the convoluted image is convoluted correctly using the OpenCV convolution as reference
        Assertions.assertArrayEquals(gioHybrid.getBand(0).pixels, myHybrid.getBand(0).pixels);
        Assertions.assertArrayEquals(gioHybrid.getBand(1).pixels, myHybrid.getBand(1).pixels);
        Assertions.assertArrayEquals(gioHybrid.getBand(2).pixels, myHybrid.getBand(2).pixels);
        // Assert the convoluted image is equivalent to the actual convolution
        Assertions.assertEquals(gioHybrid,myHybrid);
    }

    @ParameterizedTest
    @MethodSource("argumentsTestHybrid")
    void testHybridAlessandro(MBFImage lowImage, float lowSigma, MBFImage highImage, float highSigma) {
        MBFImage myHybrid = MyHybridImages.makeHybrid(lowImage, lowSigma, highImage, highSigma);
        MBFImage aleHybrid = HybridAlessandro.makeHybrid(lowImage, lowSigma, highImage, highSigma);
        // Assert the size of the hybrid image is the same as the original
        Assertions.assertEquals(myHybrid.getWidth(), lowImage.getWidth());
        Assertions.assertEquals(myHybrid.getHeight(), lowImage.getHeight());
        // Assert the size of the convoluted image is the same as the actual convolution
        Assertions.assertEquals(aleHybrid.getWidth(), myHybrid.getWidth());
        Assertions.assertEquals(aleHybrid.getHeight(), myHybrid.getHeight());


        // Assert the convoluted image is convoluted correctly using the OpenCV convolution as reference
        Assertions.assertArrayEquals(aleHybrid.getBand(0).pixels, myHybrid.getBand(0).pixels);
        Assertions.assertArrayEquals(aleHybrid.getBand(1).pixels, myHybrid.getBand(1).pixels);
        Assertions.assertArrayEquals(aleHybrid.getBand(2).pixels, myHybrid.getBand(2).pixels);
        // Assert the convoluted image is equivalent to the actual convolution
        Assertions.assertEquals(aleHybrid,myHybrid);
    }


    Stream<Arguments> argumentsTestConvolution() throws IOException {

        String path = "C:\\Users\\tamaj\\OneDrive\\University of Southampton\\" +
                "University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\" +
                "OpenIMAJ-Tutorials\\images\\";

        MBFImage dog = ImageUtilities.readMBF(new File(path + "dog.bmp"));
        MBFImage landscape = ImageUtilities.readMBF(new File(path + "landscape.jpg"));

        return Stream.of(
                // Kernels which have 1x1 size
                Arguments.of(dog, new float[][]{{1f}}),
                Arguments.of(dog, new float[][]{{0.2f}}),
                Arguments.of(dog, new float[][]{{10f}}),
                Arguments.of(dog, new float[][]{{-1f}}),
                Arguments.of(landscape, new float[][]{{1f}}),
                Arguments.of(landscape, new float[][]{{0.2f}}),
                Arguments.of(landscape, new float[][]{{10f}}),
                Arguments.of(landscape, new float[][]{{-1f}}),
                // Kernels which have 3x1 size
                Arguments.of(dog, new float[][]{{1/3f},{1/3f},{1/3f}}),
                Arguments.of(dog, new float[][]{{1f},{1.3f},{1.2f}}),
                Arguments.of(dog, new float[][]{{0.2f}, {0.3f},{0.1f}}),
                Arguments.of(dog, new float[][]{{10f}, {5f}, {4f}}),
                Arguments.of(dog, new float[][]{{-1f}, {-3f}, {-0.4f}}),
                Arguments.of(landscape, new float[][]{{1/3f},{1/3f},{1/3f}}),
                Arguments.of(landscape, new float[][]{{1f},{1.3f},{1.2f}}),
                Arguments.of(landscape, new float[][]{{0.2f}, {0.3f},{0.1f}}),
                Arguments.of(landscape, new float[][]{{10f}, {5f}, {4f}}),
                Arguments.of(landscape, new float[][]{{-1f}, {-3f}, {-0.4f}}),
                // Kernels which have 1x3 size
                Arguments.of(dog, new float[][]{{1/3f,1/3f,1/3f}}),
                Arguments.of(dog, new float[][]{{1f,1.3f,1.2f}}),
                Arguments.of(dog, new float[][]{{0.2f, 0.3f, 0.1f}}),
                Arguments.of(dog, new float[][]{{10f, 5f, 4f}}),
                Arguments.of(dog, new float[][]{{-1f, -3f, -0.4f}}),
                Arguments.of(dog, new float[][]{{1/3f,1/3f,1/3f}}),
                Arguments.of(landscape, new float[][]{{1f,1.3f,1.2f}}),
                Arguments.of(landscape, new float[][]{{0.2f, 0.3f, 0.1f}}),
                Arguments.of(landscape, new float[][]{{10f, 5f, 4f}}),
                Arguments.of(landscape, new float[][]{{-1f, -3f, -0.4f}}),
                // Kernels which have 3x3 size
                Arguments.of(dog, new float[][]{{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f}}),
                Arguments.of(dog, new float[][]{{1f,1.3f,1.2f},{1.2f,1.1f,1.7f},{1.3f,1.9f,1.3f}}),
                Arguments.of(dog, new float[][]{{0.2f, 0.3f, 0.1f},{0.123f, 0.24f, 0.34f},{0.3f,0.3456f, 0.12f}}),
                Arguments.of(dog, new float[][]{{10f, 5f, 4f}, {5.1f, 4.4f, 6.9f}, {6.1f, 5.76f, 8.765f}}),
                Arguments.of(dog, new float[][]{{-1f, -3f, -0.4f}, {-2.3f, -5.4f, -8.21f}, {-0.23f, -3.21f, -6.54f}}),
                Arguments.of(landscape, new float[][]{{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f}}),
                Arguments.of(landscape, new float[][]{{1f,1.3f,1.2f},{1.2f,1.1f,1.7f},{1.3f,1.9f,1.1123f}}),
                Arguments.of(landscape, new float[][]{{0.2f, 0.3f, 0.1f},{0.123f, 0.24f, 0.34f},{0.3f,0.3456f, 0.12f}}),
                Arguments.of(landscape, new float[][]{{10f, 5f, 4f}, {5.1f, 4.4f, 6.9f}, {6.1f, 5.76f, 8.765f}}),
                Arguments.of(landscape, new float[][]{{-1f, -3f, -0.4f}, {-2.3f, -5.4f, -8.21f}, {-0.23f, -3.21f, -6.54f}}),
                // Kernels which have 5x3 size
                Arguments.of(dog, new float[][]{{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f}, {1/3f,1/3f,1/3f}, {1/3f,1/3f,1/3f}}),
                Arguments.of(dog, new float[][]{{1f,1.3f,1.2f},{1.2f,1.1f,1.7f},{1.3f,1.9f,1.3f}, {1.2f,1.1f,1.7f}, {1f,1.3f,1.2f}}),
                Arguments.of(dog, new float[][]{{0.2f, 0.3f, 0.1f},{0.123f, 0.24f, 0.34f},{0.3f,0.3456f, 0.12f}, {0.123f, 0.24f, 0.34f}, {0.3f,0.3456f, 0.12f}}),
                Arguments.of(dog, new float[][]{{10f, 5f, 4f}, {5.1f, 4.4f, 6.9f}, {6.1f, 5.76f, 8.765f}, {10f, 5f, 4f}, {10f, 5f, 4f}}),
                Arguments.of(dog, new float[][]{{-1f, -3f, -0.4f}, {-2.3f, -5.4f, -8.21f}, {-0.23f, -3.21f, -6.54f}, {5.1f, 4.4f, 6.9f}, {10f, 5f, 4f}}),
                Arguments.of(landscape, new float[][]{{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f},{1/3f,1/3f,1/3f}, {1/3f,1/3f,1/3f}, {1/3f,1/3f,1/3f}}),
                Arguments.of(landscape, new float[][]{{1f,1.3f,1.2f},{1.2f,1.1f,1.7f},{1.3f,1.9f,1.3f}, {1.2f,1.1f,1.7f}, {1f,1.3f,1.2f}}),
                Arguments.of(landscape, new float[][]{{0.2f, 0.3f, 0.1f},{0.123f, 0.24f, 0.34f},{0.3f,0.3456f, 0.12f}, {0.123f, 0.24f, 0.34f}, {0.3f,0.3456f, 0.12f}}),
                Arguments.of(landscape, new float[][]{{10f, 5f, 4f}, {5.1f, 4.4f, 6.9f}, {6.1f, 5.76f, 8.765f}, {10f, 5f, 4f}, {10f, 5f, 4f}}),
                Arguments.of(landscape, new float[][]{{-1f, -3f, -0.4f}, {-2.3f, -5.4f, -8.21f}, {-0.23f, -3.21f, -6.54f}, {5.1f, 4.4f, 6.9f}, {10f, 5f, 4f}}),
                // Kernels which have 3x5 size
                Arguments.of(dog, new float[][]{{1/3f, 1/3f,1/3f, 1/3f, 1/3f},{1/3f, 1/3f,1/3f, 1/3f, 1/3f},{1/3f, 1/3f,1/3f, 1/3f, 1/3f}}),
                Arguments.of(dog, new float[][]{{1f,1.3f,1.2f, 1.2f, 1.1f},{1.2f,1.1f,1.7f, 1.3f, 1.9f},{1.3f,1.9f,1.3f, 1.2f, 1.1f}}),
                Arguments.of(dog, new float[][]{{0.2f, 0.3f, 0.1f, 0.123f, 0.24f},{0.123f, 0.24f, 0.34f, 0.3f, 0.3456f},{0.3f,0.3456f, 0.12f, 0.123f, 0.24f}}),
                Arguments.of(dog, new float[][]{{10f, 5f, 4f, 5.1f, 4.4f}, {5.1f, 4.4f, 6.9f, 6.1f, 5.76f}, {6.1f, 5.76f, 8.765f, 10f, 5f}}),
                Arguments.of(dog, new float[][]{{-1f, -3f, -0.4f, -2.3f, -5.4f}, {-2.3f, -5.4f, -8.21f, -0,23f, -3.21f}, {-0.23f, -3.21f, -6.54f, 5.1f, 4.4f}}),
                Arguments.of(landscape, new float[][]{{1/3f, 1/3f,1/3f, 1/3f, 1/3f},{1/3f, 1/3f,1/3f, 1/3f, 1/3f},{1/3f, 1/3f,1/3f, 1/3f, 1/3f}}),
                Arguments.of(landscape, new float[][]{{1f,1.3f,1.2f, 1.2f, 1.1f},{1.2f,1.1f,1.7f, 1.3f, 1.9f},{1.3f,1.9f,1.3f, 1.2f, 1.1f}}),
                Arguments.of(landscape, new float[][]{{0.2f, 0.3f, 0.1f, 0.123f, 0.24f},{0.123f, 0.24f, 0.34f, 0.3f, 0.3456f},{0.3f,0.3456f, 0.12f, 0.123f, 0.24f}}),
                Arguments.of(landscape, new float[][]{{10f, 5f, 4f, 5.1f, 4.4f}, {5.1f, 4.4f, 6.9f, 6.1f, 5.76f}, {6.1f, 5.76f, 8.765f, 10f, 5f}}),
                Arguments.of(landscape, new float[][]{{-1f, -3f, -0.4f, -2.3f, -5.4f}, {-2.3f, -5.4f, -8.21f, -0,23f, -3.21f}, {-0.23f, -3.21f, -6.54f, 5.1f, 4.4f}}),
                // Kernels which have size 1x5
                Arguments.of(dog, createKernel(1,5,0f,1f)),
                Arguments.of(dog, createKernel(1,5,1f,2f)),
                Arguments.of(dog, createKernel(1,5,-1f,3f)),
                Arguments.of(dog, createKernel(1,5,5f,0.2f)),
                Arguments.of(dog, createKernel(1,5,-5f,10f)),
                Arguments.of(landscape, createKernel(1,5,0f,1f)),
                Arguments.of(landscape, createKernel(1,5,1f,2f)),
                Arguments.of(landscape, createKernel(1,5,-1f,3f)),
                Arguments.of(landscape, createKernel(1,5,5f,0.2f)),
                Arguments.of(landscape, createKernel(1,5,-5f,10f)),
                // Kernels which have size 5x1
                Arguments.of(dog, createKernel(5,1,0f,1f)),
                Arguments.of(dog, createKernel(5,1,1f,2f)),
                Arguments.of(dog, createKernel(5,1,-1f,3f)),
                Arguments.of(dog, createKernel(5,1,5f,0.2f)),
                Arguments.of(dog, createKernel(5,1,-5f,10f)),
                Arguments.of(landscape, createKernel(5,1,0f,1f)),
                Arguments.of(landscape, createKernel(5,1,1f,2f)),
                Arguments.of(landscape, createKernel(5,1,-1f,3f)),
                Arguments.of(landscape, createKernel(5,1,5f,0.2f)),
                Arguments.of(landscape, createKernel(5,1,-5f,10f)),
                // Kernels which have size 5x5
                Arguments.of(dog, createKernel(5,5,0f,1f)),
                Arguments.of(dog, createKernel(5,5,1f,2f)),
                Arguments.of(dog, createKernel(5,5,-1f,3f)),
                Arguments.of(dog, createKernel(5,5,5f,0.2f)),
                Arguments.of(dog, createKernel(5,5,-5f,10f)),
                Arguments.of(landscape, createKernel(5,5,0f,1f)),
                Arguments.of(landscape, createKernel(5,5,1f,2f)),
                Arguments.of(landscape, createKernel(5,5,-1f,3f)),
                Arguments.of(landscape, createKernel(5,5,5f,0.2f)),
                Arguments.of(landscape, createKernel(5,5,-5f,10f)),
                // Kernels which have size 5x7
                Arguments.of(dog, createKernel(5,7,0f,1f)),
                Arguments.of(dog, createKernel(5,7,1f,2f)),
                Arguments.of(dog, createKernel(5,7,-1f,3f)),
                Arguments.of(dog, createKernel(5,7,5f,0.2f)),
                Arguments.of(dog, createKernel(5,7,-5f,10f)),
                Arguments.of(landscape, createKernel(5,7,0f,1f)),
                Arguments.of(landscape, createKernel(5,7,1f,2f)),
                Arguments.of(landscape, createKernel(5,7,-1f,3f)),
                Arguments.of(landscape, createKernel(5,7,5f,0.2f)),
                Arguments.of(landscape, createKernel(5,7,-5f,10f)),
                // Kernels which gave size 7x5
                Arguments.of(dog, createKernel(7,5,0f,1f)),
                Arguments.of(dog, createKernel(7,5,1f,2f)),
                Arguments.of(dog, createKernel(7,5,-1f,3f)),
                Arguments.of(dog, createKernel(7,5,5f,0.2f)),
                Arguments.of(dog, createKernel(7,5,-5f,10f)),
                Arguments.of(landscape, createKernel(7,5,0f,1f)),
                Arguments.of(landscape, createKernel(7,5,1f,2f)),
                Arguments.of(landscape, createKernel(7,5,-1f,3f)),
                Arguments.of(landscape, createKernel(7,5,5f,0.2f)),
                Arguments.of(landscape, createKernel(7,5,-5f,10f)),
                // Kernels which have size 3x7
                Arguments.of(dog, createKernel(3,7,0f,1f)),
                Arguments.of(dog, createKernel(3,7,1f,2f)),
                Arguments.of(dog, createKernel(3,7,-1f,3f)),
                Arguments.of(dog, createKernel(3,7,5f,0.2f)),
                Arguments.of(dog, createKernel(3,7,-5f,10f)),
                Arguments.of(landscape, createKernel(3,7,0f,1f)),
                Arguments.of(landscape, createKernel(3,7,1f,2f)),
                Arguments.of(landscape, createKernel(3,7,-1f,3f)),
                Arguments.of(landscape, createKernel(3,7,5f,0.2f)),
                Arguments.of(landscape, createKernel(3,7,-5f,10f)),
                // Kernels which have size 7x3
                Arguments.of(dog, createKernel(7,3,0f,1f)),
                Arguments.of(dog, createKernel(7,3,1f,2f)),
                Arguments.of(dog, createKernel(7,3,-1f,3f)),
                Arguments.of(dog, createKernel(7,3,5f,0.2f)),
                Arguments.of(dog, createKernel(7,3,-5f,10f)),
                Arguments.of(landscape, createKernel(7,3,0f,1f)),
                Arguments.of(landscape, createKernel(7,3,1f,2f)),
                Arguments.of(landscape, createKernel(7,3,-1f,3f)),
                Arguments.of(landscape, createKernel(7,3,5f,0.2f)),
                Arguments.of(landscape, createKernel(7,3,-5f,10f)),
                // Kernels which have size 1x7
                Arguments.of(dog, createKernel(1,7,0f,1f)),
                Arguments.of(dog, createKernel(1,7,1f,2f)),
                Arguments.of(dog, createKernel(1,7,-1f,3f)),
                Arguments.of(dog, createKernel(1,7,5f,0.2f)),
                Arguments.of(dog, createKernel(1,7,-5f,10f)),
                Arguments.of(landscape, createKernel(1,7,0f,1f)),
                Arguments.of(landscape, createKernel(1,7,1f,2f)),
                Arguments.of(landscape, createKernel(1,7,-1f,3f)),
                Arguments.of(landscape, createKernel(1,7,5f,0.2f)),
                Arguments.of(landscape, createKernel(1,7,-5f,10f)),
                // Kernels which have size 7x1
                Arguments.of(dog, createKernel(7,1,0f,1f)),
                Arguments.of(dog, createKernel(7,1,1f,2f)),
                Arguments.of(dog, createKernel(7,1,-1f,3f)),
                Arguments.of(dog, createKernel(7,1,5f,0.2f)),
                Arguments.of(dog, createKernel(7,1,-5f,10f)),
                Arguments.of(landscape, createKernel(7,1,0f,1f)),
                Arguments.of(landscape, createKernel(7,1,1f,2f)),
                Arguments.of(landscape, createKernel(7,1,-1f,3f)),
                Arguments.of(landscape, createKernel(7,1,5f,0.2f)),
                Arguments.of(landscape, createKernel(7,1,-5f,10f)),
                // Kernels which have size 7x7
                Arguments.of(dog, createKernel(7,7,0f,1f)),
                Arguments.of(dog, createKernel(7,7,1f,2f)),
                Arguments.of(dog, createKernel(7,7,-1f,3f)),
                Arguments.of(dog, createKernel(7,7,5f,0.2f)),
                Arguments.of(dog, createKernel(7,7,-5f,10f)),
                Arguments.of(landscape, createKernel(7,7,0f,1f)),
                Arguments.of(landscape, createKernel(7,7,1f,2f)),
                Arguments.of(landscape, createKernel(7,7,-1f,3f)),
                Arguments.of(landscape, createKernel(7,7,5f,0.2f)),
                Arguments.of(landscape, createKernel(7,7,-5f,10f)),
                // Kernels which have size 9x3
                Arguments.of(dog, createKernel(9,3,0f,1f)),
                Arguments.of(dog, createKernel(9,3,1f,2f)),
                Arguments.of(dog, createKernel(9,3,-1f,3f)),
                Arguments.of(dog, createKernel(9,3,5f,0.2f)),
                Arguments.of(dog, createKernel(9,3,-5f,10f)),
                Arguments.of(landscape, createKernel(9,3,0f,1f)),
                Arguments.of(landscape, createKernel(9,3,1f,2f)),
                Arguments.of(landscape, createKernel(9,3,-1f,3f)),
                Arguments.of(landscape, createKernel(9,3,5f,0.2f)),
                Arguments.of(landscape, createKernel(9,3,-5f,10f)),
                // Kerels which have size 3x9
                Arguments.of(dog, createKernel(3,9,0f,1f)),
                Arguments.of(dog, createKernel(3,9,1f,2f)),
                Arguments.of(dog, createKernel(3,9,-1f,3f)),
                Arguments.of(dog, createKernel(3,9,5f,0.2f)),
                Arguments.of(dog, createKernel(3,9,-5f,10f)),
                Arguments.of(landscape, createKernel(3,9,0f,1f)),
                Arguments.of(landscape, createKernel(3,9,1f,2f)),
                Arguments.of(landscape, createKernel(3,9,-1f,3f)),
                Arguments.of(landscape, createKernel(3,9,5f,0.2f)),
                Arguments.of(landscape, createKernel(3,9,-5f,10f)),
                // Kernels which have size 5x9
                Arguments.of(dog, createKernel(5,9,0f,1f)),
                Arguments.of(dog, createKernel(5,9,1f,2f)),
                Arguments.of(dog, createKernel(5,9,-1f,3f)),
                Arguments.of(dog, createKernel(5,9,5f,0.2f)),
                Arguments.of(dog, createKernel(5,9,-5f,10f)),
                Arguments.of(landscape, createKernel(5,9,0f,1f)),
                Arguments.of(landscape, createKernel(5,9,1f,2f)),
                Arguments.of(landscape, createKernel(5,9,-1f,3f)),
                Arguments.of(landscape, createKernel(5,9,5f,0.2f)),
                Arguments.of(landscape, createKernel(5,9,-5f,10f)),
                // Kernels which have size 9x5
                Arguments.of(dog, createKernel(9,5,0f,1f)),
                Arguments.of(dog, createKernel(9,5,1f,2f)),
                Arguments.of(dog, createKernel(9,5,-1f,3f)),
                Arguments.of(dog, createKernel(9,5,5f,0.2f)),
                Arguments.of(dog, createKernel(9,5,-5f,10f)),
                Arguments.of(landscape, createKernel(9,5,0f,1f)),
                Arguments.of(landscape, createKernel(9,5,1f,2f)),
                Arguments.of(landscape, createKernel(9,5,-1f,3f)),
                Arguments.of(landscape, createKernel(9,5,5f,0.2f)),
                Arguments.of(landscape, createKernel(9,5,-5f,10f)),
                // Kernels which have size 7x9
                Arguments.of(dog, createKernel(7,9,0f,1f)),
                Arguments.of(dog, createKernel(7,9,1f,2f)),
                Arguments.of(dog, createKernel(7,9,-1f,3f)),
                Arguments.of(dog, createKernel(7,9,5f,0.2f)),
                Arguments.of(dog, createKernel(7,9,-5f,10f)),
                Arguments.of(landscape, createKernel(7,9,0f,1f)),
                Arguments.of(landscape, createKernel(7,9,1f,2f)),
                Arguments.of(landscape, createKernel(7,9,-1f,3f)),
                Arguments.of(landscape, createKernel(7,9,5f,0.2f)),
                Arguments.of(landscape, createKernel(7,9,-5f,10f)),
                // Kernels which have size 9x7
                Arguments.of(dog, createKernel(9,7,0f,1f)),
                Arguments.of(dog, createKernel(9,7,1f,2f)),
                Arguments.of(dog, createKernel(9,7,-1f,3f)),
                Arguments.of(dog, createKernel(9,7,5f,0.2f)),
                Arguments.of(dog, createKernel(9,7,-5f,10f)),
                Arguments.of(landscape, createKernel(9,7,0f,1f)),
                Arguments.of(landscape, createKernel(9,7,1f,2f)),
                Arguments.of(landscape, createKernel(9,7,-1f,3f)),
                Arguments.of(landscape, createKernel(9,7,5f,0.2f)),
                Arguments.of(landscape, createKernel(9,7,-5f,10f)),
                // Kernels which have size 1x9
                Arguments.of(dog, createKernel(1,9,0f,1f)),
                Arguments.of(dog, createKernel(1,9,1f,2f)),
                Arguments.of(dog, createKernel(1,9,-1f,3f)),
                Arguments.of(dog, createKernel(1,9,5f,0.2f)),
                Arguments.of(dog, createKernel(1,9,-5f,10f)),
                Arguments.of(landscape, createKernel(1,9,0f,1f)),
                Arguments.of(landscape, createKernel(1,9,1f,2f)),
                Arguments.of(landscape, createKernel(1,9,-1f,3f)),
                Arguments.of(landscape, createKernel(1,9,5f,0.2f)),
                Arguments.of(landscape, createKernel(1,9,-5f,10f)),
                // Kernels which have size 9x1
                Arguments.of(dog, createKernel(9,1,0f,1f)),
                Arguments.of(dog, createKernel(9,1,1f,2f)),
                Arguments.of(dog, createKernel(9,1,-1f,3f)),
                Arguments.of(dog, createKernel(9,1,5f,0.2f)),
                Arguments.of(dog, createKernel(9,1,-5f,10f)),
                Arguments.of(landscape, createKernel(9,1,0f,1f)),
                Arguments.of(landscape, createKernel(9,1,1f,2f)),
                Arguments.of(landscape, createKernel(9,1,-1f,3f)),
                Arguments.of(landscape, createKernel(9,1,5f,0.2f)),
                Arguments.of(landscape, createKernel(9,1,-5f,10f)),
                // Kernels which have size 9x9
                Arguments.of(dog, createKernel(9,9,0f,1f)),
                Arguments.of(dog, createKernel(9,9,1f,2f)),
                Arguments.of(dog, createKernel(9,9,-1f,3f)),
                Arguments.of(dog, createKernel(9,9,5f,0.2f)),
                Arguments.of(dog, createKernel(9,9,-5f,10f)),
                Arguments.of(landscape, createKernel(9,9,0f,1f)),
                Arguments.of(landscape, createKernel(9,9,1f,2f)),
                Arguments.of(landscape, createKernel(9,9,-1f,3f)),
                Arguments.of(landscape, createKernel(9,9,5f,0.2f)),
                Arguments.of(landscape, createKernel(9,9,-5f,10f)),
                // Kernels which have size 11x9
                Arguments.of(dog, createKernel(11,9,0f,1f)),
                Arguments.of(dog, createKernel(11,9,1f,2f)),
                Arguments.of(dog, createKernel(11,9,-1f,3f)),
                Arguments.of(dog, createKernel(11,9,5f,0.2f)),
                Arguments.of(dog, createKernel(11,9,-5f,10f)),
                Arguments.of(landscape, createKernel(11,9,0f,1f)),
                Arguments.of(landscape, createKernel(11,9,1f,2f)),
                Arguments.of(landscape, createKernel(11,9,-1f,3f)),
                Arguments.of(landscape, createKernel(11,9,5f,0.2f)),
                Arguments.of(landscape, createKernel(11,9,-5f,10f)),
                // Kernels which have size 9x11
                Arguments.of(dog, createKernel(9,11,0f,1f)),
                Arguments.of(dog, createKernel(9,11,1f,2f)),
                Arguments.of(dog, createKernel(9,11,-1f,3f)),
                Arguments.of(dog, createKernel(9,11,5f,0.2f)),
                Arguments.of(dog, createKernel(9,11,-5f,10f)),
                Arguments.of(landscape, createKernel(9,11,0f,1f)),
                Arguments.of(landscape, createKernel(9,11,1f,2f)),
                Arguments.of(landscape, createKernel(9,11,-1f,3f)),
                Arguments.of(landscape, createKernel(9,11,5f,0.2f)),
                Arguments.of(landscape, createKernel(9,11,-5f,10f)),
                // Kernels whoch have size 13x15
                Arguments.of(dog, createKernel(13,15,0f,1f)),
                Arguments.of(dog, createKernel(13,15,1f,2f)),
                Arguments.of(dog, createKernel(13,15,-1f,3f)),
                Arguments.of(dog, createKernel(13,15,5f,0.2f)),
                Arguments.of(dog, createKernel(13,15,-5f,10f)),
                Arguments.of(landscape, createKernel(13,15,0f,1f)),
                Arguments.of(landscape, createKernel(13,15,1f,2f)),
                Arguments.of(landscape, createKernel(13,15,-1f,3f)),
                Arguments.of(landscape, createKernel(13,15,5f,0.2f)),
                Arguments.of(landscape, createKernel(13,15,-5f,10f)),
                // Kernels which have size 17x15
                Arguments.of(dog, createKernel(17,15,0f,1f)),
                Arguments.of(dog, createKernel(17,15,1f,2f)),
                Arguments.of(dog, createKernel(17,15,-1f,3f)),
                Arguments.of(dog, createKernel(17,15,5f,0.2f)),
                Arguments.of(dog, createKernel(17,15,-5f,10f)),
                Arguments.of(landscape, createKernel(17,15,0f,1f)),
                Arguments.of(landscape, createKernel(17,15,1f,2f)),
                Arguments.of(landscape, createKernel(17,15,-1f,3f)),
                Arguments.of(landscape, createKernel(17,15,5f,0.2f)),
                Arguments.of(landscape, createKernel(17,15,-5f,10f)),
                // Kernels which have size 21x19
                Arguments.of(dog, createKernel(21,19,0f,1f)),
                Arguments.of(dog, createKernel(21,19,1f,2f)),
                Arguments.of(dog, createKernel(21,19,-1f,3f)),
                Arguments.of(dog, createKernel(21,19,5f,0.2f)),
                Arguments.of(dog, createKernel(21,19,-5f,10f)),
                Arguments.of(landscape, createKernel(21,19,0f,1f)),
                Arguments.of(landscape, createKernel(21,19,1f,2f)),
                Arguments.of(landscape, createKernel(21,19,-1f,3f)),
                Arguments.of(landscape, createKernel(21,19,5f,0.2f)),
                Arguments.of(landscape, createKernel(21,19,-5f,10f))


                );
    }

    Stream<Arguments> argumentsTestHybrid() throws IOException {

        String path = "C:\\Users\\tamaj\\OneDrive\\University of Southampton\\" +
                "University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\" +
                "OpenIMAJ-Tutorials\\images\\";

        MBFImage dog = ImageUtilities.readMBF(new File(path + "dog.bmp"));
        MBFImage cat = ImageUtilities.readMBF(new File(path + "cat.bmp"));
        MBFImage bicycle = ImageUtilities.readMBF(new File(path + "bicycle.bmp"));
        MBFImage motorcycle = ImageUtilities.readMBF(new File(path + "motorcycle.bmp"));
        MBFImage marylin = ImageUtilities.readMBF(new File(path + "marilyn.bmp"));
        MBFImage einstein = ImageUtilities.readMBF(new File(path + "einstein.bmp"));

        return Stream.of(
                // Dog and cat
                Arguments.of(dog, 0.2f, cat, 0.3f),
                Arguments.of(dog, 0.5f, cat, 0.1f),
                Arguments.of(dog, 0.8f, cat, 1f),
                Arguments.of(dog, 3f, cat, 5f),
                Arguments.of(dog, 1f, cat, 1f),
                Arguments.of(dog, 10f, cat, 10f),
                Arguments.of(dog, 0.2f, cat, 5f),
                Arguments.of(dog, 5f, cat, 5f),
                Arguments.of(dog, 3f, cat, 7f),
                Arguments.of(dog, 2.9f, cat, 11f),
                // Bicycle and motorcycle
                Arguments.of(bicycle, 0.2f, motorcycle, 0.3f),
                Arguments.of(bicycle, 0.5f, motorcycle, 0.1f),
                Arguments.of(bicycle, 0.8f, motorcycle, 1f),
                Arguments.of(bicycle, 3f, motorcycle, 5f),
                Arguments.of(bicycle, 1f, motorcycle, 1f),
                Arguments.of(bicycle, 10f, motorcycle, 10f),
                Arguments.of(bicycle, 0.2f, motorcycle, 5f),
                Arguments.of(bicycle, 5f, motorcycle, 5f),
                Arguments.of(bicycle, 3f, motorcycle, 7f),
                Arguments.of(bicycle, 2.9f, motorcycle, 11f),
                // Marylin and Einstein
                Arguments.of(marylin, 0.2f, einstein, 0.3f),
                Arguments.of(marylin, 0.5f, einstein, 0.1f),
                Arguments.of(marylin, 0.8f, einstein, 1f),
                Arguments.of(marylin, 3f, einstein, 5f),
                Arguments.of(marylin, 1f, einstein, 1f),
                Arguments.of(marylin, 10f, einstein, 10f),
                Arguments.of(marylin, 0.2f, einstein, 5f),
                Arguments.of(marylin, 5f, einstein, 5f),
                Arguments.of(marylin, 3f, einstein, 7f),
                Arguments.of(marylin, 2.9f, einstein, 11f)

        );
    }



    float[][] createKernel(int rows, int cols, float mean, float std){
        Random rand = new Random();
	    float[][] kernel = new float[rows][cols];
	    for(int i =0; i < rows; i++){
	        for(int c = 0; c < cols; c++ ){
	            kernel[i][c] = (float) rand.nextGaussian()*std + mean;
            }
        }
	    return kernel;
    }


    @Test
    void testConvolutionDebug() {
        float [][] kernel = {{1/3f},{1/3f},{1/3f}};
        FImage image = new FImage(new float[][]{{1.f,1.f,1.f,1.f}, {1f,1f,1f,1f}, {1f,1f,1f,1f}, {1f,1f,1f,1f}});
        FImage imageCopy = image.clone();
        FImage imageCopyCopy = image.clone();
        FImage image1 = image.process(new MyConvolution(kernel));
        FImage image2 = imageCopy.process(new ConvolutionGiovanni(kernel));
        FImage image3 = imageCopyCopy.process(new FConvolution(kernel));

        System.out.println(2);
    }



    @ParameterizedTest
    @MethodSource("argumentsTestGaussConvolution")
    void testGaussianConvolution(MBFImage image, float sigma) {
        MBFImage myGaussConv = MyHybridImages.gaussianConvolve(image, sigma);
        MBFImage actualGaussConv = image.process(new FGaussianConvolve(sigma));
        // Assert the size of the guassian convoluted image is the same as the original
        Assertions.assertEquals(myGaussConv.getWidth(), image.getWidth());
        Assertions.assertEquals(myGaussConv.getHeight(), image.getHeight());
        // Assert the convoluted image is convoluted correctly using the FGaussianConvolve class as reference
        Assertions.assertArrayEquals(myGaussConv.getBand(0).pixels, actualGaussConv.getBand(0).pixels);
        Assertions.assertArrayEquals(myGaussConv.getBand(1).pixels, actualGaussConv.getBand(1).pixels);
        Assertions.assertArrayEquals(myGaussConv.getBand(2).pixels, actualGaussConv.getBand(2).pixels);
        Assertions.assertEquals(actualGaussConv, myGaussConv);
    }

    Stream<Arguments> argumentsTestGaussConvolution() throws IOException {

        String path = "C:\\Users\\tamaj\\OneDrive\\University of Southampton\\" +
                "University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\" +
                "OpenIMAJ-Tutorials\\images\\";

        MBFImage dog = ImageUtilities.readMBF(new File(path + "dog.bmp"));
        MBFImage landscape = ImageUtilities.readMBF(new File(path + "landscape.jpg"));

        return Stream.of(
                //Arguments.of(dog, 0.1f),
                //Arguments.of(dog, 0.01f),
                Arguments.of(dog, 0.7f)
        );
    }

    @Test
    void testConvolutionGITHUB() {
        float[][] kernel = {{1/3f}, {1/3f}, {1/3f}};
        float[][] image = {{1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1}};
        FImage fimage = new FImage(image);
        float[][] output = Convolution.convolve(image,kernel);
        FImage outputF = fimage.process(new MyConvolution(kernel));
        System.out.println("hello");
    }
    void roundoff(int decimals, float[][] array){
	    float roundoff = 10^decimals;
	    for(int i=0; i < array.length; i++){
	        for(int c=0; c < array[0].length; c++){
	            array[i][c] = Math.round(array[i][c]*roundoff) / roundoff;
            }
        }
    }


}

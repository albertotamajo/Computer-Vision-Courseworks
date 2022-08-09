package uk.ac.soton.ecs.at2n19.hybridimages;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.FConvolution;

import java.io.File;
import java.io.IOException;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MyConvolutionTest {

    @ParameterizedTest
    @MethodSource("argumentsTestConvolution")
    void testConvolution(MBFImage image, float[][] kernel) {
        MBFImage myConv = image.process(new MyConvolution(kernel));
        MBFImage actualConv = image.process(new FConvolution(kernel));
        Assertions.assertEquals(actualConv, myConv);
    }

    @ParameterizedTest
    @MethodSource("argumentsTestConvolution")
    void testConvolutionGiovanni(MBFImage image, float[][] kernel) {
        MBFImage myConv = image.process(new MyConvolution(kernel));
        MBFImage actualConv = image.process(new ConvolutionGiovanni(kernel));
        Assertions.assertEquals(actualConv, myConv);
    }

    Stream<Arguments> argumentsTestConvolution() throws IOException {

        String path = "C:\\Users\\tamaj\\OneDrive\\University of Southampton\\" +
                "University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\" +
                "OpenIMAJ-Tutorials\\images\\";

        return Stream.of(
                Arguments.of(ImageUtilities.readMBF(new File(path + "dog.bmp")), new float[][]{{1f}})
        );
    }

}
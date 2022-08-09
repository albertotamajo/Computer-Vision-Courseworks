package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * A sliding window feature with the associated location
 * @param <T> feature vector
 */
public class SlidingWindowLocalFeature<T extends FeatureVector> implements LocalFeature<SpatialLocation,T> {

    private final SpatialLocation spaLocation;
    private final T featVector;

    /**
     *
     * @param x x ordinate of the left topmost pixel of the region inside the sliding window
     * @param y y ordinate of the left topmost pixel of the region inside the sliding window
     * @param featVector feature vector extracted from the region inside the sliding window
     */
    public SlidingWindowLocalFeature(float x, float y, T featVector) {
        this.spaLocation = new SpatialLocation(x,y);
        this.featVector = featVector;
    }

    @Override
    public T getFeatureVector() {
        return featVector;
    }

    @Override
    public SpatialLocation getLocation() {
        return spaLocation;
    }

    // not useful for the coursework
    @Override
    public void readASCII(Scanner in) throws IOException {

    }

    // not useful for the coursework
    @Override
    public void writeASCII(PrintWriter out) throws IOException {

    }

    // not useful for the coursework
    @Override
    public String asciiHeader() {
        return null;
    }

    // not useful for the coursework
    @Override
    public void readBinary(DataInput in) throws IOException {

    }

    // not useful for the coursework
    @Override
    public void writeBinary(DataOutput out) throws IOException {

    }

    // not useful for the coursework
    @Override
    public byte[] binaryHeader() {
        return new byte[0];
    }
}

package uk.ac.soton.ecs.at2n19.ch14;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.time.Timer;
import org.openimaj.util.function.Operation;
import org.openimaj.util.parallel.Parallel;
import org.openimaj.util.parallel.partition.RangePartitioner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Tutorial {
    public static void main(String[] args) throws IOException {

        // Parallel equivalent of for (int i=0; i<10; i++){System.out.println(i);}
        Parallel.forIndex(0, 10, 1, new Operation<Integer>() {
            public void perform(Integer i) {
                System.out.println(i);
            }
        });
        // Load CalTech 101 dataset
        VFSGroupDataset<MBFImage> allImages = Caltech101.getImages(ImageUtilities.MBFIMAGE_READER);
        // Dataset containing the first 8 categories of the CalTech101 dataset
        GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images = GroupSampler.sample(allImages, 8, false);

        // Build the average image for each group by looping through the images in the group, resampling and normalising
        // each image before drawing it in the centre of a white image, and then adding the result to an accumulator.
        // At the end of the loop, divide the accumulated image by the number of samples used to create it
        List<MBFImage> output = new ArrayList<>();
        final ResizeProcessor resize = new ResizeProcessor(200);
        Timer t1 = Timer.timer();
        for (ListDataset<MBFImage> clzImages : images.values()) {
            MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);

            for (MBFImage i : clzImages) {
                MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
                tmp.fill(RGBColour.WHITE);

                MBFImage small = i.process(resize).normalise();
                int x = (200 - small.getWidth()) / 2;
                int y = (200 - small.getHeight()) / 2;
                tmp.drawImage(small, x, y);

                current.addInplace(tmp);
            }
            current.divideInplace((float) clzImages.size());
            output.add(current);

        }
        // Print out the running time of the outer loop
        System.out.println("No parallelisation time: " + t1.duration() + "ms");
        // Display the average images
        DisplayUtilities.display("Images", output);


        // Build the average image for each group by looping through the images in the group, resampling and normalising
        // each image before drawing it in the centre of a white image, and then adding the result to an accumulator.
        // At the end of the loop, divide the accumulated image by the number of samples used to create it.
        // The inner loop is parallelized
        output = new ArrayList<>();
        t1 = Timer.timer();
        for (ListDataset<MBFImage> clzImages : images.values()) {
            MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);

            Parallel.forEach(clzImages, new Operation<MBFImage>() {
                public void perform(MBFImage i) {
                    final MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
                    tmp.fill(RGBColour.WHITE);

                    final MBFImage small = i.process(resize).normalise();
                    final int x = (200 - small.getWidth()) / 2;
                    final int y = (200 - small.getHeight()) / 2;
                    tmp.drawImage(small, x, y);

                    synchronized (current) {
                        current.addInplace(tmp);
                    }
                }
            });
            current.divideInplace((float) clzImages.size());
            output.add(current);

        }
        // Print out the running time of the outer loop
        System.out.println("Inner loop parallelised time: " + t1.duration() + "ms");
        // Display the average images
        DisplayUtilities.display("Images", output);


        // Build the average image for each group by looping through the images in the group, resampling and normalising
        // each image before drawing it in the centre of a white image, and then adding the result to an accumulator.
        // At the end of the loop, divide the accumulated image by the number of samples used to create it.
        // The inner loop is parallelized by feeding each thread a collection of images
        output = new ArrayList<>();
        t1 = Timer.timer();
        for (ListDataset<MBFImage> clzImages : images.values()) {
            MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);

            Parallel.forEachPartitioned(new RangePartitioner<MBFImage>(clzImages), new Operation<Iterator<MBFImage>>() {
                public void perform(Iterator<MBFImage> it) {
                    MBFImage tmpAccum = new MBFImage(200, 200, 3);
                    MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);

                    while (it.hasNext()) {
                        final MBFImage i = it.next();
                        tmp.fill(RGBColour.WHITE);

                        final MBFImage small = i.process(resize).normalise();
                        final int x = (200 - small.getWidth()) / 2;
                        final int y = (200 - small.getHeight()) / 2;
                        tmp.drawImage(small, x, y);
                        tmpAccum.addInplace(tmp);
                    }
                    synchronized (current) {
                        current.addInplace(tmpAccum);
                    }
                }
            });
            current.divideInplace((float) clzImages.size());
            output.add(current);

        }
        // Print out the running time of the outer loop
        System.out.println("Inner loop parallelized by feeding each thread a collection of images time: " + t1.duration() + "ms");
        // Display the average images
        DisplayUtilities.display("Images", output);

    }
}
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/*
    As we discussed earlier in the tutorial, there were three primary ways in which we could have approached the
    parallelisation of the image-averaging program. Instead of parallelising the inner loop, can you modify the code
    to parallelise the outer loop instead? Does this make the code faster? What are the pros and cons of doing this?
 */
public class Exercise1 {
    // The running time of the different parallelization techniques are listed below.
    //      Running time without using parallelization: 37349ms
    //      Running time with inner loop parallelization: 8111ms
    //      Running time with inner loop parallelization RangePartitioner: 8017ms
    //      Running time with outer loop parallelization: 16362ms
    //      Running time with inner and outer loop parallelization: 8293ms
    // As it is possible to notice, the advantage of using the outer loop parallelization is that
    // it makes it possible to complete the task in less than half of the time required without parallelization.
    // On the other hand, the outer loop parallelization performs poorly in comparison to both the inner loop
    // parallelizations. This is caused by the fact that the number of images inside each group is far larger than the
    // number of groups. Consequently, the inner loop parallelization performs more parallel operations than the outer
    // loop parallelization. The inner and outer loop parallelization need both to synchronize on the 'current' and
    // 'output' instances, respectively. However, while the inner loop has to synchronise on the 'current' instance at most
    // for a number of times which is equal to the number of images in a group, the outer loop at most needs to synchronise
    // on the 'output' instance only for a number of times which is equal to the number of groups in dataset. This is the
    // reason why inner loop parallelization with RangePartitioner has been proposed in the Tutorial so that to decrease
    // the number of times the inner loop needs to synchronise on the 'current' instance and consequently achieve a better
    // performance.
    public static void main(String[] args) throws IOException {

        // Load CalTech 101 dataset
        VFSGroupDataset<MBFImage> allImages = Caltech101.getImages(ImageUtilities.MBFIMAGE_READER);
        // Dataset containing the first 8 categories of the CalTech101 dataset
        GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images = GroupSampler.sample(allImages, 8, false);

        // Build the average image for each group by looping through the images in the group, resampling and normalising
        // each image before drawing it in the centre of a white image, and then adding the result to an accumulator.
        // At the end of the loop, divide the accumulated image by the number of samples used to create it.
        // The outer loop is parallelized
        List<MBFImage> output = new ArrayList<>();
        final ResizeProcessor resize = new ResizeProcessor(200);
        Timer t1 = Timer.timer();
        Parallel.forEach(images.values(), new Operation<ListDataset<MBFImage>>() {
            public void perform(ListDataset<MBFImage> clzImages) {
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
                synchronized (output){
                    output.add(current);
                }
            }
        });
        // Print out the running time of the outer loop
        System.out.println("Time: " + t1.duration() + "ms");
        // Display the average images
        DisplayUtilities.display("Images", output);


        // Build the average image for each group by looping through the images in the group, resampling and normalising
        // each image before drawing it in the centre of a white image, and then adding the result to an accumulator.
        // At the end of the loop, divide the accumulated image by the number of samples used to create it
        // The outer and inner loops are parallelized
        final List<MBFImage> output1 = new ArrayList<>();
        t1 = Timer.timer();
        Parallel.forEach(images.values(), new Operation<ListDataset<MBFImage>>() {
            public void perform(ListDataset<MBFImage> clzImages) {
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
                output1.add(current);
            }
        });
        // Print out the running time of the outer loop
        System.out.println("Time: " + t1.duration() + "ms");
        // Display the average images
        DisplayUtilities.display("Images", output1);

    }
}

package uk.ac.soton.ecs.at2n19.ch12;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
    Try running the code over all the classes in the Caltech 101 dataset. Also try increasing the number of visual words
    to 600, adding extra scales to the PyramidDenseSIFT (try [4, 6, 8, 10] and reduce the step-size of the DenseSIFT to
    3), and instead of using the BlockSpatialAggregator, try the PyramidSpatialAggregator with [2, 4] blocks.
    What level of classifier performance does this achieve?
 */
public class Exercise3 {
    // The performance of this classifier is provided below.
    // *********************** Overall Results ***********************
    //          Total instances: 1712.000
    //            Total correct: 575.000
    //          Total incorrect: 1137.000
    //                 Accuracy: 0.336
    //               Error Rate: 0.664
    //   Average Class Accuracy: 0.344
    // Average Class Error Rate: 0.656
    //
    // The feature extractor proposed in this exercise should be capable of extracting more significant features than
    // the ones that are extracted by the feature extractor in Exercise1. This holds because in this feature extractor
    // the step size of DenseSIFT is decreased to 3, the PyramidDenseSIFT is added with extra scales, a
    // PyramidSpatialAggregator is used instead of a BlockSpatialAggregator and the number of visual words is increased
    // to 600. However, the performance of this classifier is poor in comparison to the performance achieved in
    // Exercise1. There is a simple reason why this occurs and it is the fact that the number of images used for
    // learning the codebook is the same as in Exercise1, 30. In Exercise1, only 5 classes were used for the training and
    // testing of the classifier. 15 images were randomly selected from each class for creating the training dataset and
    // 30 of them were used for learning the codebook. Therefore, the codebook was learnt using 2/5 of the training
    // dataset. Instead, in this exercise, 15 images per each group in the dataset are randomly selected creating a
    // training dataset consisting of 15 x 101 = 1515 images. Only 1.9% of the training dataset (30 images) are used for
    // learning the codebook. Therefore, the codebook in this exercise is learnt on a very small percentage of the whole
    // training dataset. Also, given that only 30 images are used to learn the codebook, at least 71 classes have
    // no instances when the codebook is learnt. Thus, the codebook learns a set of visual words which have been extracted
    // using at most only 30 classes out of the 101 possible classes.
    public static void main(String[] args) throws IOException {

        // Caltech 101 dataset
        GroupedDataset<String, VFSListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> allData =
                Caltech101.getData(ImageUtilities.FIMAGE_READER);
        // Create a train dataset with 15 images per group, and a test dataset with 15 images per group
        GroupedRandomSplitter<String, Caltech101.Record<FImage>> splits =
                new GroupedRandomSplitter<String, Caltech101.Record<FImage>>(allData, 15, 0, 15);
        // Create a DenseSift object (just like SIFT but features are extracted on a regular grid across the image)
        DenseSIFT dsift = new DenseSIFT(3, 7); // step size decreased to 3
        // Create a PyramidDenseSIFT object (takes the DenseSIFT instance and applies it to multiple window of size
        // 4,6,8,10 respectively)
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4,6,8,10);

        // Path used to cache data
        String cachingPath = "C:\\Users\\tamaj\\OneDrive\\University of Southampton\\" +
                "University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\" +
                "OpenIMAJ-Tutorials\\images";
        HardAssigner<byte[], float[], IntFloatPair> assigner = null;
        // Path used to cache assigners
        File pathAssigner = new File(cachingPath + "/assigner");

        // If an assigner has been previously cached then fetch it
        if (Files.exists(Paths.get(pathAssigner.toString()))) {
            try {
                assigner = IOUtils.readFromFile(pathAssigner);
                System.out.println("Cached assigner is being used");
            } catch (IOException e) {
                e.printStackTrace();
                System.err.println("Cached assigner cannot be used");
            }
        }

        // If no assigner exists then train a new one
        if (assigner == null) {
            assigner = trainQuantiser(
                    GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);
            try {
                // Cache the assigner
                IOUtils.writeToFile(assigner, pathAssigner);
                System.out.println("Assigner is being cached");
            } catch (IOException e) {
                e.printStackTrace();
                System.err.println("Assigner cannot be cached");
            }
        }


        // Create an instance of the Homogeneous Kernel Map which transforms data into a compact linear representation
        // such that applying a linear classifier approximates, to a high degree of accuracy, the application of a
        // non-linear classifier over the original data.
        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        // Create an instance of the Pyramid Histogram of Words extractor
        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> phowExtractor = new Exercise1.PHOWExtractor(pdsift, assigner);
        // Wrap the homogenous kernel map on top of the Pyramid Histogram of Words extractor
        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> extractor = homogeneousKernelMap.createWrappedExtractor(phowExtractor);
        // Feature extractor that caches features on disk
        DiskCachingFeatureExtractor<DoubleFV, Caltech101.Record<FImage>> cachedExtractor = new DiskCachingFeatureExtractor<>(new File(cachingPath), extractor);
        // Construct and train a linear classifier
        LiblinearAnnotator<Caltech101.Record<FImage>, String> ann = new LiblinearAnnotator<Caltech101.Record<FImage>, String>(
                cachedExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(splits.getTrainingDataset());

        // Perform automated evaluation of the linear classifier accuracy and print out a summary report
        ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>> eval = new ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>>(
                ann, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        Map<Caltech101.Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());

    }

    // This method extracts the first 10000 dense SIFT features from the images in the dataset, and then clusters them
    // into 600 separate classes. The method then returns a HardAssigner which can be used to assign SIFT features to
    // identifiers. Thus, this method learns a codebook of 600 visual words and the assigner assigns
    // a visual word contained in the codebook to a given SIFT feature
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<Caltech101.Record<FImage>> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        for (Caltech101.Record<FImage> rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600);  // visual words increased to 600
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    // This class uses a PyramidSpatialAggregator together with a BagOfVisualWords to compute the histograms of an image.
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(Caltech101.Record<FImage> object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            // Changed to PyramidSpatialAggregator with blocks 2,4
            PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(bovw,2,4);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}

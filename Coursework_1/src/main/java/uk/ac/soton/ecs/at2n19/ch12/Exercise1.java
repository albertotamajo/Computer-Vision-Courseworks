package uk.ac.soton.ecs.at2n19.ch12;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
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
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
    A Homogeneous Kernel Map transforms data into a compact linear representation such that applying a linear
    classifier approximates, to a high degree of accuracy, the application of a non-linear classifier over
    the original data. Try using the HomogeneousKernelMap class with a KernelType.Chi2 kernel and WindowType.
    Rectangular window on top of the PHOWExtractor feature extractor. What effect does this have on performance?
 */
public class Exercise1 {
    // Using the Homogeneous Kernel Map leads to a better performance in comparison to the simple linear classifier
    // trained in the tutorial. Indeed, while this linear classifier achieves 74.7% accuracy on the test set, the linear
    // classifier in the Tutorial achieves only 64% accuracy on the test set.
    // In what follows, a detailed report on the performance of this linear classifier is provided.
    // *********************** Overall Results ***********************
    //          Total instances: 75.000
    //            Total correct: 56.000
    //          Total incorrect: 19.000
    //                 Accuracy: 0.747
    //               Error Rate: 0.253
    //   Average Class Accuracy: 0.747
    // Average Class Error Rate: 0.253
    //
    // ********************** Per Class Results **********************
    // Class	Class Accuracy	Class Error Rate	Actual Count	Predicted Count
    //  accordion	1.000	    0.000	15.000000	21.000000
    //        ant	0.733	    0.267	15.000000	20.000000
    //  airplanes	1.000	    0.000	15.000000	16.000000
    //     anchor	0.800	    0.200	15.000000	15.000000
    //BACKGROUND_Google	        0.200	0.800	    15.000000	     3.000000
    public static void main(String[] args) throws IOException {

        // Caltech 101 dataset
        GroupedDataset<String, VFSListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> allData =
                Caltech101.getData(ImageUtilities.FIMAGE_READER);
        // Create a new dataset from the first 5 classes of the Caltech 101 dataset
        GroupedDataset<String, ListDataset<Caltech101.Record<FImage>>, Caltech101.Record<FImage>> data =
                GroupSampler.sample(allData, 5, false);
        // Create a train dataset with 15 images per group, and a test dataset with 15 images per group
        GroupedRandomSplitter<String, Caltech101.Record<FImage>> splits =
                new GroupedRandomSplitter<String, Caltech101.Record<FImage>>(data, 15, 0, 15);
        // Create a DenseSift object (just like SIFT but features are extracted on a regular grid across the image)
        DenseSIFT dsift = new DenseSIFT(5, 7);
        // Create a PyramidDenseSIFT object (takes the DenseSIFT instance and applies it to a single window size of 7 pixels)
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        // Train the quantiser with a random sample of 30 images across all the groups of the training set
        HardAssigner<byte[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);

        // Create an instance of the Homogeneous Kernel Map which transforms data into a compact linear representation
        // such that applying a linear classifier approximates, to a high degree of accuracy, the application of a
        // non-linear classifier over the original data.
        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        // Create an instance of the Pyramid Histogram of Words extractor
        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> phowExtractor = new PHOWExtractor(pdsift, assigner);
        // Wrap the homogenous kernel map on top of the Pyramid Histogram of Words extractor
        FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> extractor = homogeneousKernelMap.createWrappedExtractor(phowExtractor);

        // Construct and train a linear classifier
        LiblinearAnnotator<Caltech101.Record<FImage>, String> ann = new LiblinearAnnotator<Caltech101.Record<FImage>, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(splits.getTrainingDataset());

        // Perform automated evaluation of the linear classifier accuracy and print out a summary report
        ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>> eval = new ClassificationEvaluator<CMResult<String>, String, Caltech101.Record<FImage>>(
                ann, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        Map<Caltech101.Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());

    }

    // This method extracts the first 10000 dense SIFT features from the images in the dataset, and then clusters them
    // into 300 separate classes. The method then returns a HardAssigner which can be used to assign SIFT features to
    // identifiers. Thus, this method learns a codebook of 300 visual words and the assigner assigns
    // a visual word contained in the codebook to a given SIFT feature.
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

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    // This class uses a BlockSpatialAggregator together with a BagOfVisualWords to compute 4 histograms across the image.
    // The BagOfVisualWords uses the HardAssigner to assign each Dense SIFT feature to a visual word and then compute the
    // histogram. The resultant spatial histograms are then appended together and normalised before being returned.
    // In other words, a feature vector is extracted from an image in the following way:
    //              the image is divided into 4 patches in a grid-like fashion and an histogram is computed for each of
    //              these patches. The feature vector of the image is computed by appending the 4 histograms and
    //              then normalising the result.
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

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}

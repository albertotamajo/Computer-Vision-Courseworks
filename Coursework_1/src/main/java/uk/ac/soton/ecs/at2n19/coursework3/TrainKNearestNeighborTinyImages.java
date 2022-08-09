package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.experiment.validation.cross.CrossValidationIterable;
import org.openimaj.experiment.validation.cross.GroupedKFold;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.ScoredAnnotation;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Utilities for training the KNearestNeighborTinyImages classifier and evaluating its performance
 */
public class TrainKNearestNeighborTinyImages {

    /**
     * Produces the run1.txt file with the parameters that have achieved the highest average accuracy (22.4%)
     * on the 5-fold cross-validation iterations. This code should replicate the annotations produced in the
     * submitted run1.txt file.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        KNearestNeighborTinyImages knnTiny = new KNearestNeighborTinyImages(5,196,16);
        GroupedDataset<String, VFSListDataset<FImage>, FImage> trainDataset = Dataset.getTrainDataset();
        knnTiny.train(trainDataset);
        classifyTestDataset(knnTiny,"run1.txt");
    }

    /**
     * Annotate the images on the test dataset with the given k-nearest neighbor classifier using the "tiny image"
     * feature extractor. The annotations are saved into a text file whose path is given as argument. The annotations
     * are saved into ascending order based on the name of the image.
     * @param knnTiny k-nearest neighbor classifier that uses the "tiny image" feature extractor
     * @param path path where to save the annotations
     * @throws IOException
     */
    public static void classifyTestDataset(KNearestNeighborTinyImages knnTiny, String path) throws IOException {
        List<Pair<Integer,String>> pairs = new ArrayList<>();
        // Create report file
        new File(path).createNewFile();
        VFSListDataset<FImage> testDataset = Dataset.getTestDataset();
        for (int i = 0; i < testDataset.size(); i++){
            FImage image = testDataset.get(i);
            String name = testDataset.getFileObject(i).getName().getBaseName();
            String annotation = classify(knnTiny.annotate(image));
            pairs.add(new Pair<>(Integer.parseInt(name.replace(".jpg","")),annotation));
        }
        pairs = pairs.stream().sorted(Comparator.comparing(x -> x.val1)).collect(Collectors.toList());
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        for (Pair<Integer,String> pair : pairs){
            pw.println(pair.val1 + ".jpg" + " " + pair.val2);
        }
        pw.close();
    }

    /**
     * Return the annotation of the class with the highest confidence from a list of scored annotations
     * @param scoredAnnotations list of scored annotations
     * @return annotation of the class with the highest confidence
     */
    public static String classify(List<ScoredAnnotation<String>> scoredAnnotations){
        float confidence = Float.MIN_VALUE;
        String annotation = null;
        for (ScoredAnnotation<String> scoredAnnotation : scoredAnnotations){
            if (scoredAnnotation.confidence > confidence){
                annotation = scoredAnnotation.annotation;
                confidence = scoredAnnotation.confidence;
            }
        }
        return annotation;
    }


    /**
     * Perform K-fold cross validation on a given dataset for several models with different parameters.
     * The performance of each model is appended to a report.
     * @param minCropWidth minimum width of the image cropped about the centre
     * @param maxCropWidth maximum width of the image cropped about the centre
     * @param stepCropWidth incrementing step for the crop width
     * @param mink minimum number of neighbours
     * @param maxk maximum number of neighbours
     * @param stepk incrementing step for the number of neighbours
     * @param resWidth width to be used to resize the cropped image
     * @param nfolds number of folds
     * @param dataset dataset
     * @param filePath path to the report file
     * @throws IOException
     */
    public static void evaluateModelsCrossValidation(int minCropWidth, int maxCropWidth, int stepCropWidth, int mink,
                                                     int maxk, int stepk, int resWidth, int nfolds,
                                                     GroupedDataset<String, ListDataset<FImage>, FImage> dataset,
                                                     String filePath) throws IOException {
        // Create report file
        new File(filePath).createNewFile();
        PrintWriter pw = new PrintWriter(new FileWriter(filePath));
        // Print the header on the report file
        pw.println("CropWidth,K,Accuracy,Step CropWidth,Step K,ResWidth,Folds");
        pw.close();
        for(int cropWidth = minCropWidth; cropWidth <= maxCropWidth; cropWidth+=stepCropWidth) {
            for (int k = mink; k <= maxk; k += stepk) {
                // Perform cross validation on the given model
                double avgAccuracy = crossValidation(k, cropWidth, resWidth, nfolds, dataset);
                pw = new PrintWriter(new FileWriter(filePath,true));
                // Append to the report file the performance of the given model
                pw.println(cropWidth + "," + k + "," + avgAccuracy + "," + stepCropWidth + "," + stepk + "," +
                        resWidth + "," + nfolds);
                pw.close();
            }
        }
    }


    /**
     * Train a model with the given parameters on the given train dataset.
     * The performance of the trained model is evaluated on the given validation dataset.
     * @param k number of neighbours
     * @param cropWidth width of the image cropped about the centre
     * @param resWidth width to be used to resize the cropped image
     * @param trainDataset train dataset
     * @param valDataset validation dataset
     * @return accuracy on the validation dataset
     */
    public static double evaluateModel(int k, int cropWidth, int resWidth,
                                       GroupedDataset<String, ListDataset<FImage>, FImage> trainDataset,
                                       GroupedDataset<String, ListDataset<FImage>, FImage> valDataset){

        KNearestNeighborTinyImages knnTiny = new KNearestNeighborTinyImages(k,cropWidth,resWidth);
        // Train the model on the train dataset
        knnTiny.train(trainDataset);
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(knnTiny,
                valDataset, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        // Evaluate the model on the validation dataset
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
        // Return the accuracy on the validation dataset
        return result.getMatrix().getAccuracy();
    }


    /**
     * Perform K-fold cross validation on a given dataset for a model with the given parameters
     * @param k number of neighbours
     * @param cropWidth width of the image cropped about the centre
     * @param resWidth width to be used to resize the cropped image
     * @param nfolds number of folds
     * @param dataset dataset
     * @return average accuracy among the K-fold cross validation iterations
     */
    public static double crossValidation(int k, int cropWidth, int resWidth, int nfolds,
                                         GroupedDataset<String, ListDataset<FImage>, FImage> dataset){

        GroupedKFold<String,FImage> kfold = new GroupedKFold<>(nfolds);
        CrossValidationIterable<GroupedDataset<String,ListDataset<FImage>,FImage>> iterator =
                kfold.createIterable(dataset);
        double accum = 0;
        // Evaluate the model for each iteration of the K-fold cross validation process
        for (var data : iterator){
            accum += evaluateModel(k, cropWidth, resWidth, data.getTrainingDataset(), data.getValidationDataset());
        }
        // return avg accuracy among the K-fold cross validation iterations
        return accum / iterator.numberIterations();
    }
}
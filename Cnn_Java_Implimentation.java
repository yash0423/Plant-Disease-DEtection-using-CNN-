package it.rcpvision.dl4j.workbench;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;     //used to make a Multilayered Neural Network
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;      
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;     //Used for pooling 
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.datavec.api.io.labels.ParentPathLabelGenerator;   //Returns as label the base name of the parent file of the path (the directory).
import org.datavec.api.split.FileSplit;                      //Use to SPlit files
import org.datavec.image.loader.NativeImageLoader;           //Uses JavaCV to load images.
import org.datavec.image.recordreader.ImageRecordReader;     //Reads a local file system and parses images of a given height and width

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;       //Keep track of the previous layer's gradient and use it as a way of updating the gradient.
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.File;
import java.util.Random;

public class Cnn_Java_Implimentation {
	
	//private static final String DATA_PATH="C:\\Users\\Gabbar\\Desktop\\Minor Dataset";
	  public static void main(String[] args) throws Exception {
		  
	    /*
	    image information
	    64 * 64 RGB images
	    RGB implies Three channels
	    */
	    int height = 64;
	    int width = 64;
	    int rngseed = 123;
	    Random randNumGen = new Random(rngseed);
	    int batchSize = 15;
	    int outputNum = 2;
	    int numEpochs = 100;
	    int channels = 3;
	    
        // Define the File Paths where data set is stored
	    File trainData = new File("C:\\Users\\Gabbar\\Desktop\\Minor Dataset\\train_ds");
	    File testData = new File("C:\\Users\\Gabbar\\Desktop\\Minor Dataset\\test_ds");
	    
	    // Define the FileSplit(PATH, ALLOWED FORMATS,random)
	    FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
	    FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

	    // Extract the parent path as the image label
	    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

	    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

	    // Initialize the record reader
	    // add a listener, to extract the name
	    recordReader.initialize(train);

	    // The LogRecordListener will log the path of each image read
	    // used here for information purposes,
	    // If the whole dataset was ingested this would place 60,000
	    // lines in our logs
	    
	    // DataSet Iterator
	    DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

	    // Scale pixel values to 0-1
	    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	    scaler.fit(dataIter);
	    dataIter.setPreProcessor(scaler);
	    
	 // Build Our Neural Network
	    System.out.println("########## BUILDING MODEL ##########");
		
	    
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs( 0.005, 0.9
                		//new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)
                		))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(40)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(5, 5)
                    .stride(2, 2)
                    .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.RELU)
                    .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(5, 5)
                    .stride(2, 2)
                    .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                .build();

	    MultiLayerNetwork model = new MultiLayerNetwork(conf);

	    // The Score iteration Listener will log
	    // output to show how well the network is training
	    model.setListeners(new ScoreIterationListener(1));
	    

	    System.out.println("########## TRAINING MODEL ##########");
	    for (int i = 0; i < numEpochs; i++) {
	    	System.out.println("Epoch==>"+i);
	      model.fit(dataIter);  //Model is fiited to our DataSet
	    }
	    
	    System.out.println("########## SAVING THE  TRAINED MODEL ##########");
	    // Where to save model
	    File locationToSave = new File( "Saved_model.zip");

	    // boolean save Updater
	    boolean saveUpdater = false;

	    // ModelSerializer needs modelname, saveUpdater, Location
	      //noinspection ConstantConditions
	      model.save(locationToSave, saveUpdater);
	    
	    System.out.println("########## EVALUATING MODEL ##########");
	    recordReader.reset();

	    // The model trained on the training dataset split
	    // now that it has trained we evaluate against the
	    // test data of images the network has not seen

	    recordReader.initialize(test);
	    DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
	    scaler.fit(testIter);
	    testIter.setPreProcessor(scaler);

	    /*
	    log the order of the labels for later use
	    In previous versions the label order was consistent, but random
	    In current verions label order is lexicographic
	    preserving the RecordReader Labels order is no
	    longer needed left in for demonstration
	    purposes
	    */
	    System.out.println(recordReader.getLabels().toString());

	    // Create Eval object with 10 possible classes
	    Evaluation eval = new Evaluation(outputNum);

	    // Evaluate the network
	    while (testIter.hasNext()) {
	      DataSet next = testIter.next();
	      INDArray output = model.output(next.getFeatures());
	      // Compare the Feature Matrix from the model
	      // with the labels from the RecordReader
	      eval.eval(next.getLabels(), output);
	    }

	    System.out.println(eval.stats());

}
}

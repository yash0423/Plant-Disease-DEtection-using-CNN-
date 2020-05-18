package it.rcpvision.dl4j.workbench;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.text.DecimalFormat;

public class Single_prediction {

	 /* Create a popup window to allow you to chose an image file to test against the
	  trained Neural Network
	  Chosen images will be automatically
	  scaled to 64*64 RGB 
	  */
	  private static String fileChose() {
	    JFileChooser fc = new JFileChooser();
	    int ret = fc.showOpenDialog(null);
	    if (ret == JFileChooser.APPROVE_OPTION) {
	      File file = fc.getSelectedFile();
	        return file.getAbsolutePath();
	    } else {
	      return null;
	    }
	  }
	  private static DecimalFormat df2 = new DecimalFormat("#.##");
	  
	  public static void main(String[] args) throws Exception {
	    int height = 64;
	    int width = 64;
	    int channels = 3;

	    // recordReader.getLabels()
	    // In this version Labels are always in order
	    // So this is no longer needed
	    //List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);
	    List<Integer> labelList = Arrays.asList(0, 1);

	    // pop up file chooser
	    String filechose = fileChose();

	    //LOAD NEURAL NETWORK

	    // Where to save model
	    File locationToSave = new File("Saved_Model.zip");
	    // Check for presence of saved model
	    if (locationToSave.exists()) {
	    	System.out.println("Saved Model Found!");
	    } else {
	    	System.out.println("File not found!");
	    	System.out.println("This example depends on running Cnn_Java_Implimentation, run that example first");
	      System.exit(0);
	    }

	    MultiLayerNetwork model = MultiLayerNetwork.load(locationToSave, true);

	    System.out.println("TESTING YOUR IMAGE AGAINST SAVED NETWORK..........");
	    // FileChose is a string we will need a file
	    File file = new File(Objects.requireNonNull(filechose));

	    // Use NativeImageLoader to convert to numerical matrix
	    NativeImageLoader loader = new NativeImageLoader(height, width, channels);

	    // Get the image into an INDarray
	    INDArray image = loader.asMatrix(file);

	    // 0-255
	    // 0-1
	    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	    scaler.transform(image);

	    // Pass through to neural Net
	    INDArray output = model.output(image);

	    System.out.println("The file chosen was " + filechose);
	    System.out.println("The neural nets prediction (list of probabilities per label)");
	    //log.info("## List of Labels in Order## ");
	    // In new versions labels are always in order
	    System.out.println(output.toString());
	    System.out.println(labelList.toString());
	    if(output.getDouble(0)>output.getDouble(1)) {
	    	System.out.println("-------------------------------------------------------------");
	    	System.out.println("Model is "+ df2.format(output.getDouble(0)*100) + "% sure that the Crop is HEALTHY");
	    	System.out.println("-------------------------------------------------------------");
	    }
	    else {
	    	System.out.println("-------------------------------------------------------------");
	    	System.out.println("Model is "+ df2.format(output.getDouble(1)*100) + "% sure that the Crop is INFECTED");
	    	System.out.println("-------------------------------------------------------------");
	    }
	    
	  }

}

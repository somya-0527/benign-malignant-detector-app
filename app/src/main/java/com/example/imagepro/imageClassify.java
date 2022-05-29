package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

//import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class imageClassify {
    //close to 0 benign
    //close to 1 - malignant
    //using this interpreter we will load model and predict frame
private Interpreter interpreter;
private  int INPUT_SIZE;
private int PIXEL_SIZE = 3; //for rgb
private  int IMAGE_MEAN=0;
    //for scaling image from 0-255 to 0-1
private float IMAGE_STD=255.0f;
//used to initialize gpu in interpreter
private GpuDelegate gpuDelegate;
 private int height=0;
 private int width=0;
 //red for malignant and green for benign [r g b alpha ]
 private Scalar red=new Scalar(255,0,0,50);
 private  Scalar green=new Scalar(0,255,0,50);

 imageClassify(AssetManager assetManager,String modelpath,int inputSize)throws IOException{
     INPUT_SIZE=inputSize;
     //used to set gpu or number of threads
     Interpreter.Options options=new Interpreter.Options();
     gpuDelegate=new GpuDelegate();
     options.addDelegate(gpuDelegate);
     //set num of threads according to ur phone
     options.setNumThreads(6);
     interpreter=new Interpreter(loadModelFiles(assetManager,modelpath),options);

 }
 public Mat recognizeImg(Mat matimg){

     //input image is in landscape mode
     //convert into portrait mode
     //to convert it rotate image by 90 deg
     Mat rot_img=new Mat();
     Core.flip(matimg.t(),rot_img,1);

     //define height and width of original bitmap
     height=rot_img.height();
     width=rot_img.width();

     //now draw rect around the image of size(400x400)
     //before that crop that part of img
     Rect roi_cropped=new Rect((width-400)/2,(height-400)/2,400,400);
     Mat cropped=new Mat(rot_img,roi_cropped);
     //for prediction use this image



     //convert mat image to bitmap image
     Bitmap bitmap=null ;
     bitmap=Bitmap.createBitmap(cropped.cols(),cropped.rows(),Bitmap.Config.ARGB_8888);
     Utils.matToBitmap(cropped,bitmap);


     //resize bitmap image to original size

     Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
     //convert scaled bitmap to Bytebuffer
     ByteBuffer byteBuffer=convertBitmapTOByteBuffer(scaledBitmap);
     //create an input and output for interpreter
     float[][] output=new float[1][1];
     Object[] out=new Object[1];
     out[0]=output;

     Object[] input=new Object[1];
     input[0]=byteBuffer;

     //now pass it to interpreter
     interpreter.run(byteBuffer,output);
     Log.d("imageClassify","out"+ Arrays.deepToString(output));

     //extract value from output
     float val_pred=(float)Array.get(Array.get(output,0),0);
     //set threshold
     if(val_pred>0.4){
         //put text for malignant [red]
         //Imgproc.putText(rot_img,new Point((width-400)/2,(height-400)/2),new Point((width+400)/2,(height+400)/2),);

         Imgproc.putText(rot_img,"Malignant is detected",new Point(((width-400)/2+30),80),3,1,red,2);
         Imgproc.rectangle(rot_img,new Point((width-400)/2,(height-400)/2),new Point((width+400)/2,(height+400)/2),red,2);
     }
     else {

         Imgproc.putText(rot_img,"Benign is detected",new Point(((width-400)/2+30),80),3,1,green,2);
         Imgproc.rectangle(rot_img,new Point((width-400)/2,(height-400)/2),new Point((width+400)/2,(height+400)/2),green,2);

     }

     //return image is landscape
     //rotate image by 90 deg
     Core.flip(rot_img.t(),rot_img,0);
     return  rot_img;
 }

    private ByteBuffer convertBitmapTOByteBuffer(Bitmap scaledBitmap) {
     ByteBuffer byteBuffer;
     byteBuffer=ByteBuffer.allocateDirect(4*INPUT_SIZE*INPUT_SIZE*PIXEL_SIZE);
     byteBuffer.order(ByteOrder.nativeOrder());
     int[] intval=new int[INPUT_SIZE*INPUT_SIZE];
     scaledBitmap.getPixels(intval,0,scaledBitmap.getWidth(),0,0,scaledBitmap.getWidth(),scaledBitmap.getHeight());
     int pixel=0;
     for(int i=0;i<INPUT_SIZE;++i){
         for (int j=0;j<INPUT_SIZE;++j){
             final int val = intval[pixel++];
             //set values of byte buffer
             //image mean and std used to convert pixel from 0-255 to 0-1 or 0-255 to -1-1
             byteBuffer.putFloat((((val>>16) & 0xFF)- IMAGE_MEAN)/IMAGE_STD);
             byteBuffer.putFloat((((val>>8) & 0xFF)- IMAGE_MEAN)/IMAGE_STD);
             byteBuffer.putFloat((((val) & 0xFF)- IMAGE_MEAN)/IMAGE_STD);

         }
     }
     return byteBuffer;
    }

    //this is used to load model
    private MappedByteBuffer loadModelFiles(AssetManager assetManager, String modelpath) throws  IOException {

        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelpath);
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getLength();


    return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
 }


}

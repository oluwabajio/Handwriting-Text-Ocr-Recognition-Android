package handwriting.text.converter.ocr;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import handwriting.text.converter.ocr.databinding.ActivityBigSmallAlphabetBinding;
import handwriting.text.converter.ocr.databinding.ActivityLettersBinding;


//Source = https://www.kaggle.com/code/sankalpsrivastava26/lower-and-upper-case-alphabet-recognition/notebook
//input is [1,300,300,3] because its 3 bytes color
//input shape is the desired size of the image 300x300 with 3 bytes color


public class BigSmallAlphabetActivity extends AppCompatActivity {


    ActivityBigSmallAlphabetBinding binding;
    private String MODEL_FILE = "alpharecognition.tflite";
    private int inputImageWidth = 0;
    private int inputImageHeight = 0;
    private int modelInputSize = 0;
    private boolean isInitialized = false;
    Interpreter interpreter = null;
    private static final String TAG = "DigitActivity";
    String labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; //output is 52

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityBigSmallAlphabetBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initListeners();
        initInterpreter();
    }

    private void initInterpreter() {

        // Load the TF Lite model
        try {
            ByteBuffer model = loadModelFile();

            // Initialize TF Lite Interpreter with NNAPI enabled
            Interpreter.Options options = new Interpreter.Options();

            Interpreter interpreter = new Interpreter(model, options);

            // Read input shape from model file
            int[] inputShape = interpreter.getInputTensor(0).shape();
            inputImageWidth = inputShape[1];
            inputImageHeight = inputShape[2];
            modelInputSize = 4 * inputImageWidth * inputImageHeight;

            // Finish interpreter initialization
            this.interpreter = interpreter;
            isInitialized = true;
            Log.e(TAG, "Initialized TFLite interpreter.");

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "initInterpreter: " + e.getMessage());
        }
    }


    private ByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void initListeners() {

        binding.drawView.setStrokeWidth(70.0f);
        binding.drawView.setColor(Color.WHITE);
        binding.drawView.setBackgroundColor(Color.BLACK);

        binding.clearButton.setOnClickListener(v -> {
            binding.drawView.clearCanvas();
            binding.predictedText.setText("Please Draw");
        });

        binding.drawView.setOnTouchListener((view, motionEvent) -> {
            binding.drawView.onTouchEvent(motionEvent);


            // Then if user finished a touch event, run classification
            if (motionEvent.getAction() == MotionEvent.ACTION_UP) {
                String output = classifyDrawing(binding.drawView.getBitmap());
                binding.predictedText.setText(output);
            }

            return true;
        });


    }

    private String classifyDrawing(Bitmap bitmap) {
        if (!isInitialized) {
            throw new IllegalStateException("TF Lite Interpreter is not initialized yet.");
        }

        long startTime;
        long elapsedTime;

        // Preprocessing: resize the input
        startTime = System.nanoTime();
        Bitmap resizedImage = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true);
        Bitmap btmp = resizedImage.copy(Bitmap.Config.ARGB_8888, true);

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedImage);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.e(TAG, "Preprocessing time = " + elapsedTime + "ms");


        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new CastOp(DataType.FLOAT32))
                        .build();

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

// Analysis code for every frame
// Preprocess the image
        tensorImage.load(btmp);
        tensorImage = imageProcessor.process(tensorImage);


        startTime = System.nanoTime();
        float[][] result = new float[1][52]; // Array of size 1 because there is only 1 output. Size of output is 10, then it means the content of the array is of size 10
        interpreter.run(tensorImage.getBuffer(), result);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.e(TAG, "Inference time = " + elapsedTime + "ms");

        return getOutputString(result[0]);
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelInputSize); // modelInputSize = 28 * 28 * 4 = because a float size is 4
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight()); //extract image pixels

        for (int pixelValue : pixels) {  //loop through 28 * 28 times and putfloat
            int r = (pixelValue >> 16 & 0xFF);
            int g = (pixelValue >> 8 & 0xFF);
            int b = (pixelValue & 0xFF);

            // Convert RGB to grayscale and normalize pixel value to [0..1]
            float normalizedPixelValue = (r + g + b) / 3.0f / 255.0f;
            byteBuffer.putFloat(normalizedPixelValue);

        }

        return byteBuffer;
    }

    private String getOutputString(float[] array) {
//        int maxIndex = output.indices.maxByOrNull {
//            output[it]
//        } ?:-1


        if (array.length <= 0)
            throw new IllegalArgumentException("The array is empty");

        int maxIndex = -1;
        for (int i = 0; i < array.length; i++) {
            Log.e(TAG, "getOutputString: char =  " + labelNames.charAt(i) + " value = " + array[i]);
            if (array[i] > maxIndex) {
                maxIndex = i;
            }
        }

        if (maxIndex == -1) {
            return "no result found";
        } else {
            return "Prediction Result: index = " + maxIndex + " result = " + array[maxIndex] + " value = " + labelNames.charAt(maxIndex);
        }
    }

    private void initViews() {

    }


}
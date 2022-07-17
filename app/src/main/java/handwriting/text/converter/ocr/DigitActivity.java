package handwriting.text.converter.ocr;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;

import com.divyanshu.draw.widget.DrawView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import handwriting.text.converter.ocr.databinding.ActivityDigitBinding;

public class DigitActivity extends AppCompatActivity {

    ActivityDigitBinding binding;
    private String MODEL_FILE = "mnist.tflite";
    private int inputImageWidth = 0;
    private int inputImageHeight = 0;
    private int modelInputSize = 0;
    private boolean isInitialized = false;
    Interpreter interpreter = null;
    private static final String TAG = "DigitActivity";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityDigitBinding.inflate(getLayoutInflater());
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
            Log.e(TAG, "initInterpreter: "+ e.getMessage() );
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
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedImage);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.e(TAG, "Preprocessing time = " + elapsedTime + "ms");

        startTime = System.nanoTime();
        float[][] result = new float[1][10]; // Array of size 1 because there is only 1 output. Size of output is 10, then it means the content of the array is of size 10
        interpreter.run(byteBuffer, result);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.e(TAG, "Inference time = " + elapsedTime + "ms");

        return getOutputString(result[0]);
    }



    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelInputSize); // modelInputSize = 28 * 28 * 4 = because a float size is 4
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels =new int[inputImageWidth * inputImageHeight];
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
            for (int i = 1; i < array.length; i++) {
                if (array[i] > maxIndex) {
                    maxIndex = i;
                }
            }


        return "Prediction Result: index = " +maxIndex + " result = "+array[maxIndex];
    }

    private void initViews() {

    }
}
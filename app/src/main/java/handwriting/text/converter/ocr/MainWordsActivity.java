package handwriting.text.converter.ocr;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_TREE;

import static java.lang.Double.max;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

import handwriting.text.converter.ocr.databinding.ActivityMainWordsBinding;

public class MainWordsActivity extends AppCompatActivity {

    ActivityMainWordsBinding binding;
    private String MAIN_MODEL_FILE = "emnist.tflite";
    private int inputImageWidth = 0;
    private int inputImageHeight = 0;
    private int modelInputSize = 0;
    private boolean isInitialized = false;
    Interpreter interpreter = null;
    private static final String TAG = "DigitActivity";
    String labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";


    static {
        if (OpenCVLoader.initDebug()) {
            Log.e("Check", "OpenCv configured successfully");
        } else {
            Log.d("Check", "OpenCv doesn’t configured successfully");
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainWordsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        initViews();
        initListeners();
        // initInterpreter();
        initInterpreter2();
    }

    private void initInterpreter2() {

        // Load the TF Lite model
        try {
            ByteBuffer model = loadModelFile(MAIN_MODEL_FILE);

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

//    private void initInterpreter() {
//
//        // Load the TF Lite model
//        try {
//            ByteBuffer model = loadModelFile(MODEL_FILE);
//
//            // Initialize TF Lite Interpreter with NNAPI enabled
//            Interpreter.Options options = new Interpreter.Options();
//
//            Interpreter interpreter = new Interpreter(model, options);
//
//            // Read input shape from model file
//            int[] inputShape = interpreter.getInputTensor(0).shape();
//            inputImageWidth = inputShape[1];
//            inputImageHeight = inputShape[2];
//            modelInputSize = 4 * inputImageWidth * inputImageHeight;
//
//            // Finish interpreter initialization
//            this.interpreter = interpreter;
//            isInitialized = true;
//            Log.e(TAG, "Initialized TFLite interpreter.");
//
//        } catch (IOException e) {
//            e.printStackTrace();
//            Log.e(TAG, "initInterpreter: " + e.getMessage());
//        }
//    }


    private ByteBuffer loadModelFile(String model) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(model);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void initListeners() {

        binding.btnProcess.setOnClickListener(v -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                processImage();
            }
        });


    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void processImage() {
        try {


            InputStream istr = getAssets().open("hello_world.png");
            Bitmap bitmap = BitmapFactory.decodeStream(istr);
            istr.close();


//             load the input image from disk, convert it to grayscale, and blur  it to reduce noise
//            image = cv2.imread(args["image"])
//            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            //Reading the image
            Mat src = new Mat();
            Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true); //todo
            Utils.bitmapToMat(bmp32, src);  // or Mat src = Imgcodecs.imread(input);

            Mat srcc = src.clone();

            //Creating the empty destination matrix
            Mat greyScaledImage = new Mat();

            //Converting the image to gray sacle and saving it in the dst matrix
            Imgproc.cvtColor(src, greyScaledImage, Imgproc.COLOR_RGB2GRAY);


            //
            // Creating an empty matrix to store the result


            // Applying GaussianBlur on the Image
            //  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            Mat dstblur = new Mat();
            Imgproc.GaussianBlur(src, dstblur, new Size(5, 5), 0);

//# perform edge detection, find contours in the edge map, and sort the
//# resulting contours from left-to-right
//            edged = cv2.Canny(blurred, 30, 150)
            Mat edges = new Mat();
            Imgproc.Canny(dstblur, edges, 30, 150);

//            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
//                    cv2.CHAIN_APPROX_SIMPLE)
            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            ;
            Mat hierarchy = new Mat();
            Imgproc.findContours(edges.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


//            cnts = imutils.grab_contours(cnts)
//            cnts = sort_contours(cnts, method="left-to-right")[0]
//# initialize the list of contour bounding boxes and associated
//# characters that we'll be OCR'ing
//            chars = []

//            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++)
//            {
//                Log.e(TAG, "processImage: contour = "+contourIdx );
//                Imgproc.drawContours(source, contours, contourIdx, new Scalar(0,0,255), 2);
//            }
            //   drawContour(src, contours); //todo

            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
                // Minimum size allowed for consideration
                MatOfPoint2f approxCurve = new MatOfPoint2f();

                MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(contourIdx).toArray());

                //Processing on mMOP2f1 which is in type MatOfPoint2f
                double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;

                //Detect contours
                Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

                //Convert back to MatOfPoint
                MatOfPoint points = new MatOfPoint(approxCurve.toArray());

                // Get bounding rect of contour
                Rect rect = Imgproc.boundingRect(points);

                if ((rect.width >= 5 && rect.width <= 150) && (rect.height >= 15 && rect.height <= 120)) {
                    Log.e(TAG, "processImage: Satisfied");
//                    # extract the character and threshold it to make the character
//		# appear as *white* (foreground) on a *black* background, then
//		# grab the width and height of the thresholded image
//                            roi = gray[y:y + h, x:x + w]

                    Rect ROI = rect;
// deep copy ROI to new image
                    Mat croppedImage = (new Mat(greyScaledImage, rect)).clone();
//                    displayROI(croppedImage);


                    //Apply threshold to the greyscaled cropped roi (region of interest) - threshold basically set image to be pure white and black with no greyed area
//                    thresh = cv2.threshold(roi, 0, 255,
//                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
//                    (tH, tW) = thresh.shape
                    Mat croppedImageThreshold = new Mat();
                    Imgproc.threshold(croppedImage, croppedImageThreshold, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);
                    // displayROI(croppedImageThreshold);
                    Log.e(TAG, "processImage: Size of threshold = width " + croppedImageThreshold.width() + " height  = " + croppedImageThreshold.height());


                    //		# if the width is greater than the height, resize along the
//		# width dimension
//                    if tW > tH:
//                    thresh = imutils.resize(thresh, width=32)
//
//		# otherwise, resize along the height
//		else:
//                    thresh = imutils.resize(thresh, height=32)

                    Mat croppedImageThresholdResized = new Mat();
                    if (croppedImageThreshold.width() > croppedImageThreshold.height()) {
//                          # calculate the ratio of the width and construct the
//        # dimensions
                        float r = (float) 32 / (float) croppedImageThreshold.width();
//                        dim = (width, int(h * r))
                        Size size = new Size(32, croppedImageThreshold.height() * r);
//                                resized = cv2.resize(image, dim, interpolation = inter)
                        Imgproc.resize(croppedImageThreshold, croppedImageThresholdResized, size);
                    } else {
//                          # calculate the ratio of the width and construct the
//        # dimensions
                        float r = (float) 32 / (float) croppedImageThreshold.height();
                        Log.e(TAG, "processImage: --r = " + r + "   --cith = " + croppedImageThreshold.height());
//                        dim = (width, int(h * r))
                        Size size = new Size(croppedImageThreshold.width() * r, 32);
//                                resized = cv2.resize(image, dim, interpolation = inter)
                        Log.e(TAG, "processImage: --w = " + size.width + "   --h = " + size.height);
                        Imgproc.resize(croppedImageThreshold, croppedImageThresholdResized, size);
                    }
                    Log.e(TAG, "processImage: new Resized Size is = width " + croppedImageThresholdResized.width() + " height  = " + croppedImageThresholdResized.height());

                    //    displayROI(croppedImageThresholdResized);

//
//		# re-grab the image dimensions (now that its been resized)
//		# and then determine how much we need to pad the width and
//		# height such that our image will be 32x32
//                            (tH, tW) = thresh.shape
//                    dX = int(max(0, 32 - tW) / 2.0)
//                    dY = int(max(0, 32 - tH) / 2.0)


                    Size sizeb = croppedImageThresholdResized.size();
                    int dX = (int) (max(0, 32 - sizeb.width) / 2.0);
                    int dY = (int) (max(0, 32 - sizeb.height) / 2.0);


//
//		# pad the image and force 32x32 dimensions
//                    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
//                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
//                            value=(0, 0, 0))
//                    padded = cv2.resize(padded, (32, 32))
                    Mat croppedImageThresholdResizedPadded = new Mat();
                    Core.copyMakeBorder(croppedImageThresholdResized, croppedImageThresholdResizedPadded, dY, dY, dX, dX, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
                    Imgproc.resize(croppedImageThresholdResizedPadded, croppedImageThresholdResizedPadded, new Size(32, 32));
                    Log.e(TAG, "processImage: final Resized Size is = width " + croppedImageThresholdResizedPadded.width() + " height  = " + croppedImageThresholdResizedPadded.height());
                    //displayROI(croppedImageThresholdResizedPadded);


                    //loge


                    displayROI(croppedImageThresholdResizedPadded);
                    char letter = classifyBitmap(getBitmap(croppedImageThresholdResizedPadded));
                    Log.e(TAG, "processImage: RESULT = " + classifyDrawing(getBitmap(croppedImageThresholdResizedPadded)));


                    //Draw rectangle for the detected countour on the original image
//                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
//                    cv2.putText(image, label, (x - 10, y - 10),
//                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    Imgproc.rectangle(srcc, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(255, 0, 255, 0), 3);
                    Imgproc.putText(srcc, ""+letter, new Point(rect.x -10, rect.y-10), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2,
                            new Scalar( 0, 255, 0), 3);


//		# prepare the padded image for classification via our
//		# handwriting OCR model
//                            padded = padded.astype("float32") / 255.0
//                    padded = np.expand_dims(padded, axis=-1)
//
//		# update our list of characters that will be OCR'd
//                    chars.append((padded, (x, y, w, h)))
                } else {
                    Log.e(TAG, "processImage: not Satisfied");
                }

            }

            Bitmap bmp = Bitmap.createBitmap(srcc.cols(), srcc.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(srcc, bmp);
            binding.imgImage.setImageBitmap(bmp);

        } catch (IOException ex) {
            return;
        }
    }

    private Bitmap displayROI(Mat croppedImage) {
        // create bitmap
        Bitmap bmp = null;
// create a new 4 channel Mat because bitmap is ARGB
        Mat tmp = new Mat(croppedImage.rows(), croppedImage.cols(), CvType.CV_8U, new Scalar(4));
// convert ROI image from single channel to 4 channel
        Imgproc.cvtColor(croppedImage, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
// Initialize bitmap
        bmp = Bitmap.createBitmap(croppedImage.cols(), croppedImage.rows(), Bitmap.Config.ARGB_8888);
// convert Mat to bitmap
        Utils.matToBitmap(tmp, bmp);

        binding.imgImage.setImageBitmap(bmp);
        return bmp;
    }

    private Bitmap getBitmap(Mat croppedImage) {
        // create bitmap
        Bitmap bmp = null;
// create a new 4 channel Mat because bitmap is ARGB
        Mat tmp = new Mat(croppedImage.rows(), croppedImage.cols(), CvType.CV_8U, new Scalar(4));
// convert ROI image from single channel to 4 channel
        Imgproc.cvtColor(croppedImage, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
// Initialize bitmap
        bmp = Bitmap.createBitmap(croppedImage.cols(), croppedImage.rows(), Bitmap.Config.ARGB_8888);
// convert Mat to bitmap
        Utils.matToBitmap(tmp, bmp);

        return bmp;
    }

    private void drawContour(Mat src, List<MatOfPoint> contours) {

        Mat srcc = src.clone();

        // For each detected contour(closed shape) Find the co-ordinates of the countour and draw a rectangle on the input image.
        //TODO: For each Contour detected, below logic can be changed to detect countours having only 4 edges(rectangle/square).
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            // Minimum size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();

            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(contourIdx).toArray());

            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;

            //Detect contours
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint(approxCurve.toArray());

            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(points);

            //Draw rectangle for the detected countour on the original image
            Imgproc.rectangle(srcc, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 255, 255), 3);
        }

        Bitmap bmp = Bitmap.createBitmap(srcc.cols(), srcc.rows(), Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(srcc, bmp);
        binding.imgImage.setImageBitmap(bmp);
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
        float[][] result = new float[1][36]; // Array of size 1 because there is only 1 output. Size of output is 10, then it means the content of the array is of size 10
        interpreter.run(byteBuffer, result);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.e(TAG, "Inference time = " + elapsedTime + "ms");

        return getOutputString(result[0]);
    }

    private char classifyBitmap(Bitmap bitmap) {
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
        float[][] result = new float[1][36]; // Array of size 1 because there is only 1 output. Size of output is 10, then it means the content of the array is of size 10
        interpreter.run(byteBuffer, result);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.e(TAG, "Inference time = " + elapsedTime + "ms");

        float[] array = result[0];
        if (array.length <= 0)
            throw new IllegalArgumentException("The array is empty");

        int maxIndex = -1;
        float maxResult = -1;
        for (int i = 0; i < array.length; i++) {
            Log.e(TAG, "getOutputString: char =  " + labelNames.charAt(i) + " value = " + array[i]);
            if (array[i] > maxResult) {
                maxResult = array[i];
                maxIndex = i;

            }
        }

        if (maxIndex == -1) {
            return '-';
        } else {
            return  labelNames.charAt(maxIndex);
        }
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
        float maxResult = -1;
        for (int i = 0; i < array.length; i++) {
            Log.e(TAG, "getOutputString: char =  " + labelNames.charAt(i) + " value = " + array[i]);
            if (array[i] > maxResult) {
                maxResult = array[i];
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

    private static Bitmap convertMatToBitMap(Mat input) {
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_GRAY2RGB);

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        } catch (CvException e) {
            Log.d("Exception", e.getMessage());
        }
        return bmp;
    }
}
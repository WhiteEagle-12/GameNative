package app.gamenative;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class QuickSrTestActivity extends Activity {
    private static final String TAG = "QuickSrTest";
    private static final String MODEL_ASSET = "models/quicksrnetsmall.tflite";

    private final AtomicBoolean running = new AtomicBoolean(false);

    private ExecutorService executor;
    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private boolean usingGpu;
    private String delegateLabel = "Not initialized";

    private TextView statusView;
    private ImageView inputView;
    private ImageView outputView;
    private Button toggleButton;

    private int inputWidth;
    private int inputHeight;
    private int outputWidth;
    private int outputHeight;
    private int frameIndex;
    private double averageMs;

    private Bitmap inputBitmap;
    private Bitmap outputBitmap;
    private Canvas inputCanvas;
    private Paint paint;
    private int[] inputPixels;
    private int[] outputPixels;
    private ByteBuffer inputTensorBuffer;
    private ByteBuffer outputTensorBuffer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        executor = Executors.newSingleThreadExecutor();
        buildUi();

        executor.execute(() -> {
            try {
                initializeModel();
                runOnUiThread(this::startLoop);
            } catch (Throwable throwable) {
                Log.e(TAG, "QuickSRNet test failed to initialize", throwable);
                runOnUiThread(() -> setStatus("Initialization failed:\n" + throwable));
            }
        });
    }

    @Override
    protected void onDestroy() {
        running.set(false);
        if (executor != null) {
            executor.shutdownNow();
        }
        if (interpreter != null) {
            interpreter.close();
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }
        super.onDestroy();
    }

    private void buildUi() {
        ScrollView scrollView = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(18), dp(18), dp(18));
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        scrollView.addView(root);

        TextView title = new TextView(this);
        title.setText("QuickSRNetSmall GPU Upscaler Test");
        title.setTextSize(22f);
        title.setTextColor(Color.WHITE);
        title.setGravity(Gravity.CENTER);
        root.addView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        statusView = new TextView(this);
        statusView.setTextColor(Color.rgb(224, 229, 236));
        statusView.setTextSize(14f);
        statusView.setPadding(0, dp(12), 0, dp(12));
        statusView.setText("Loading model...");
        root.addView(statusView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        toggleButton = new Button(this);
        toggleButton.setText("Stop");
        toggleButton.setEnabled(false);
        toggleButton.setOnClickListener(view -> {
            if (running.get()) {
                running.set(false);
                toggleButton.setText("Start");
            } else {
                startLoop();
            }
        });
        root.addView(toggleButton, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView inputLabel = label("Low-resolution source frame");
        root.addView(inputLabel);
        inputView = imageView();
        root.addView(inputView, imageParams(220));

        TextView outputLabel = label("QuickSRNetSmall output");
        root.addView(outputLabel);
        outputView = imageView();
        root.addView(outputView, imageParams(420));

        setContentView(scrollView);
    }

    private TextView label(String value) {
        TextView label = new TextView(this);
        label.setText(value);
        label.setTextColor(Color.rgb(179, 187, 200));
        label.setTextSize(13f);
        label.setGravity(Gravity.CENTER);
        label.setPadding(0, dp(18), 0, dp(6));
        return label;
    }

    private ImageView imageView() {
        ImageView view = new ImageView(this);
        view.setAdjustViewBounds(true);
        view.setBackgroundColor(Color.rgb(10, 13, 18));
        view.setScaleType(ImageView.ScaleType.FIT_CENTER);
        return view;
    }

    private LinearLayout.LayoutParams imageParams(int heightDp) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(heightDp));
        params.setMargins(0, 0, 0, dp(4));
        return params;
    }

    private void initializeModel() throws IOException {
        ByteBuffer modelBuffer = loadAsset(MODEL_ASSET);

        Interpreter.Options gpuOptions = new Interpreter.Options();
        gpuOptions.setNumThreads(4);
        try {
            gpuDelegate = new GpuDelegate();
            gpuOptions.addDelegate(gpuDelegate);
            interpreter = new Interpreter(modelBuffer, gpuOptions);
            usingGpu = true;
            delegateLabel = "TensorFlow Lite GPU delegate";
            Log.i(TAG, "Initialized with TensorFlow Lite GPU delegate");
        } catch (Throwable gpuError) {
            Log.w(TAG, "GPU delegate initialization failed; retrying on CPU", gpuError);
            if (gpuDelegate != null) {
                gpuDelegate.close();
                gpuDelegate = null;
            }
            modelBuffer.rewind();
            Interpreter.Options cpuOptions = new Interpreter.Options();
            cpuOptions.setNumThreads(4);
            interpreter = new Interpreter(modelBuffer, cpuOptions);
            usingGpu = false;
            delegateLabel = "CPU fallback - GPU delegate failed: " + gpuError.getClass().getSimpleName();
        }

        Tensor input = interpreter.getInputTensor(0);
        Tensor output = interpreter.getOutputTensor(0);
        int[] inputShape = input.shape();
        int[] outputShape = output.shape();
        Log.i(TAG, "Model input shape=" + Arrays.toString(inputShape)
                + " output shape=" + Arrays.toString(outputShape));
        if (input.dataType() != DataType.FLOAT32 || output.dataType() != DataType.FLOAT32) {
            throw new IllegalStateException("This prototype expects a FLOAT32 model. Input="
                    + input.dataType() + " output=" + output.dataType());
        }
        if (!isNhwcRgb(inputShape) || !isNhwcRgb(outputShape)) {
            throw new IllegalStateException("This prototype expects NHWC RGB tensors. Input="
                    + Arrays.toString(inputShape) + " output=" + Arrays.toString(outputShape));
        }

        inputHeight = inputShape[1];
        inputWidth = inputShape[2];
        outputHeight = outputShape[1];
        outputWidth = outputShape[2];

        inputBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888);
        outputBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
        inputCanvas = new Canvas(inputBitmap);
        paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        inputPixels = new int[inputWidth * inputHeight];
        outputPixels = new int[outputWidth * outputHeight];
        inputTensorBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        outputTensorBuffer = ByteBuffer.allocateDirect(outputWidth * outputHeight * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());

        String initialized = "Model: " + MODEL_ASSET
                + "\nDelegate: " + delegateLabel
                + "\nInput: " + Arrays.toString(inputShape)
                + "\nOutput: " + Arrays.toString(outputShape)
                + "\nRunning synthetic real-time frames...";
        runOnUiThread(() -> setStatus(initialized));
    }

    private boolean isNhwcRgb(int[] shape) {
        return shape.length == 4 && shape[0] == 1 && shape[3] == 3 && shape[1] > 0 && shape[2] > 0;
    }

    private ByteBuffer loadAsset(String assetPath) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[16 * 1024];
        try (InputStream input = getAssets().open(assetPath)) {
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        byte[] bytes = output.toByteArray();
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bytes.length);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.put(bytes);
        byteBuffer.rewind();
        return byteBuffer;
    }

    private void startLoop() {
        if (!running.compareAndSet(false, true)) {
            return;
        }
        toggleButton.setText("Stop");
        toggleButton.setEnabled(true);
        executor.execute(() -> {
            while (running.get() && !Thread.currentThread().isInterrupted()) {
                try {
                    runFrame();
                } catch (Throwable throwable) {
                    Log.e(TAG, "Inference loop failed", throwable);
                    running.set(false);
                    runOnUiThread(() -> {
                        toggleButton.setText("Start");
                        setStatus("Inference failed:\n" + throwable);
                    });
                }
            }
        });
    }

    private void runFrame() {
        long frameStartNanos = System.nanoTime();
        int frame = frameIndex++;
        drawSyntheticFrame(frame);

        long inputStartNanos = System.nanoTime();
        bitmapToTensor();
        double inputMs = (System.nanoTime() - inputStartNanos) / 1_000_000.0;

        long startNanos = System.nanoTime();
        outputTensorBuffer.rewind();
        interpreter.run(inputTensorBuffer, outputTensorBuffer);
        double elapsedMs = (System.nanoTime() - startNanos) / 1_000_000.0;
        averageMs = averageMs == 0.0 ? elapsedMs : averageMs * 0.90 + elapsedMs * 0.10;

        long outputStartNanos = System.nanoTime();
        boolean updatePreview = frame % 3 == 0;
        if (updatePreview) {
            tensorToBitmap();
        }
        double outputMs = (System.nanoTime() - outputStartNanos) / 1_000_000.0;
        double frameMs = (System.nanoTime() - frameStartNanos) / 1_000_000.0;

        double fps = averageMs > 0.0 ? 1000.0 / averageMs : 0.0;
        String status = String.format(Locale.US,
                "Delegate: %s%nInput: %dx%d  Output: %dx%d%nInput prep: %.2f ms%nTFLite run: %.2f ms%nOutput preview: %.2f ms%s%nWhole loop: %.2f ms%nSmoothed run: %.2f ms / %.1f FPS%nFrame: %d",
                delegateLabel,
                inputWidth,
                inputHeight,
                outputWidth,
                outputHeight,
                inputMs,
                elapsedMs,
                outputMs,
                updatePreview ? "" : " (skipped this frame)",
                frameMs,
                averageMs,
                fps,
                frameIndex);

        runOnUiThread(() -> {
            inputView.setImageBitmap(inputBitmap);
            if (updatePreview) {
                outputView.setImageBitmap(outputBitmap);
            }
            setStatus(status);
        });

        if (frame % 60 == 0) {
            Log.i(TAG, String.format(Locale.US,
                    "frame=%d inputPrep=%.2fms tfliteRun=%.2fms outputPreview=%.2fms wholeLoop=%.2fms delegate=%s",
                    frame,
                    inputMs,
                    elapsedMs,
                    outputMs,
                    frameMs,
                    delegateLabel));
        }

        long targetFrameMs = usingGpu ? 1L : 8L;
        SystemClock.sleep(targetFrameMs);
    }

    private void drawSyntheticFrame(int frame) {
        inputCanvas.drawColor(Color.rgb(14, 17, 24));

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(Math.max(1f, inputWidth / 128f));
        paint.setColor(Color.rgb(47, 60, 78));
        int grid = Math.max(8, inputWidth / 8);
        for (int x = 0; x < inputWidth; x += grid) {
            inputCanvas.drawLine(x, 0, x, inputHeight, paint);
        }
        for (int y = 0; y < inputHeight; y += grid) {
            inputCanvas.drawLine(0, y, inputWidth, y, paint);
        }

        float t = (frame % 180) / 180f;
        float centerX = inputWidth * (0.12f + 0.76f * t);
        float centerY = inputHeight * (0.56f + 0.22f * (float) Math.sin(frame * 0.11f));

        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.rgb(255, 202, 77));
        inputCanvas.drawCircle(centerX, centerY, inputWidth * 0.075f, paint);

        paint.setColor(Color.rgb(76, 217, 245));
        RectF player = new RectF(
                inputWidth * 0.12f,
                inputHeight * 0.20f,
                inputWidth * 0.36f,
                inputHeight * 0.42f);
        inputCanvas.drawRoundRect(player, inputWidth * 0.03f, inputWidth * 0.03f, paint);

        paint.setColor(Color.rgb(250, 88, 124));
        float enemyX = inputWidth * (0.76f - 0.16f * (float) Math.sin(frame * 0.07f));
        inputCanvas.drawRect(
                enemyX,
                inputHeight * 0.18f,
                enemyX + inputWidth * 0.11f,
                inputHeight * 0.42f,
                paint);

        paint.setColor(Color.rgb(159, 245, 92));
        paint.setTextSize(Math.max(10f, inputWidth / 10f));
        paint.setFakeBoldText(true);
        inputCanvas.drawText("SR", inputWidth * 0.08f, inputHeight * 0.86f, paint);
        paint.setFakeBoldText(false);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(Math.max(2f, inputWidth / 52f));
        paint.setColor(Color.rgb(241, 245, 249));
        inputCanvas.drawLine(
                inputWidth * 0.45f,
                inputHeight * 0.78f,
                inputWidth * 0.90f,
                inputHeight * 0.66f,
                paint);
    }

    private void bitmapToTensor() {
        inputBitmap.getPixels(inputPixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);
        inputTensorBuffer.rewind();
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                int color = inputPixels[y * inputWidth + x];
                inputTensorBuffer.putFloat(Color.red(color) / 255f);
                inputTensorBuffer.putFloat(Color.green(color) / 255f);
                inputTensorBuffer.putFloat(Color.blue(color) / 255f);
            }
        }
        inputTensorBuffer.rewind();
    }

    private void tensorToBitmap() {
        outputTensorBuffer.rewind();
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                outputPixels[y * outputWidth + x] = Color.rgb(
                        toByte(outputTensorBuffer.getFloat()),
                        toByte(outputTensorBuffer.getFloat()),
                        toByte(outputTensorBuffer.getFloat()));
            }
        }
        outputTensorBuffer.rewind();
        outputBitmap.setPixels(outputPixels, 0, outputWidth, 0, 0, outputWidth, outputHeight);
    }

    private int toByte(float value) {
        if (Float.isNaN(value)) {
            return 0;
        }
        return Math.max(0, Math.min(255, Math.round(value * 255f)));
    }

    private void setStatus(String status) {
        statusView.setText(status);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
